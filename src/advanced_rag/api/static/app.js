/* Chat UI for the arXiv RAG. No framework, no build step. Model output is untrusted, so every node is
 * built from text (never parsed as markup). The parsing helpers are pure so they can be unit-tested. */

/* ====================== pure helpers (no DOM) ====================== */

const INLINE_RE = /(\*\*[^*\n]+\*\*|`[^`\n]+`|\[\d+\])/g;

/** Split a line of model text into inline nodes. Citation markers that do not match a real source are dropped. */
function parseInline(text, valid) {
  const out = [];
  let last = 0;
  // Normalise narrow no-break spaces; and drop <mark> tags that PDF extraction leaves around quoted
  // tokens (cosmetic only: any other markup stays inert text because nothing is ever parsed as HTML).
  const s = String(text).replace(/\u202f/g, " ").replace(/<\/?mark>/gi, "");
  for (const m of s.matchAll(INLINE_RE)) {
    if (m.index > last) out.push({ t: "text", v: s.slice(last, m.index) });
    const tok = m[0];
    if (tok.startsWith("**")) out.push({ t: "b", v: tok.slice(2, -2) });
    else if (tok.startsWith("`")) out.push({ t: "code", v: tok.slice(1, -1) });
    else {
      const n = parseInt(tok.slice(1, -1), 10);
      if (valid.has(n)) out.push({ t: "cite", n });
    }
    last = m.index + tok.length;
  }
  if (last < s.length) out.push({ t: "text", v: s.slice(last) });
  return out;
}

/** Tiny markdown subset -> block list: paragraphs, bullet/numbered lists, preformatted tables. */
function parseBlocks(text, valid) {
  const blocks = [];
  let para = [];
  let list = null;
  let pre = null;
  const flush = () => {
    if (para.length) blocks.push({ type: "p", inline: parseInline(para.join(" "), valid) });
    para = [];
    if (list) blocks.push(list);
    list = null;
    if (pre) blocks.push({ type: "pre", text: pre.join("\n") });
    pre = null;
  };
  for (const raw of String(text).replace(/\r/g, "").split("\n")) {
    const line = raw.trimEnd();
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    const num = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (!line.trim()) {
      flush();
    } else if (line.trimStart().startsWith("|")) {
      if (!pre) { flush(); pre = []; }
      pre.push(line);
    } else if (bullet || num) {
      const type = bullet ? "ul" : "ol";
      if (!list || list.type !== type) { flush(); list = { type, items: [] }; }
      list.items.push(parseInline((bullet || num)[1], valid));
    } else {
      if (list || pre) flush();
      para.push(line.replace(/^#{1,6}\s+/, ""));
    }
  }
  flush();
  return blocks;
}

const DECLINE_WHY = {
  no_relevant_sources: "Nothing in the indexed papers matched closely enough, even after rephrasing the search.",
  could_not_ground_answer: "I drafted an answer but could not verify it against the sources, so I am not showing it.",
  model_found_insufficient_context: "The papers I found do not contain enough to answer this.",
};

function sectionLabel(section) {
  // PDF-extracted headings sometimes keep markdown emphasis markers (e.g. "_C. Adaptive Retrieval_").
  const parts = String(section || "").split(" > ").map((p) => p.replace(/^[_*\s]+|[_*\s]+$/g, ""));
  return parts.length > 1 ? parts.slice(1).join(" › ") : parts[0];
}

/** Only https://arxiv.org/abs|pdf/<id> links are ever rendered as hrefs (blocks javascript:, lookalike hosts, etc.). */
function safeArxivUrl(url) {
  return /^https:\/\/arxiv\.org\/(abs|pdf)\/[A-Za-z0-9._\/-]+$/.test(String(url)) ? String(url) : null;
}

/** Turn the server's per-node trace into human-readable steps for the "How this answer was produced" panel. */
function buildSteps(trace) {
  const steps = [];
  for (const e of trace || []) {
    const ms = e.ms;
    switch (e.node) {
      case "condense_question":
        steps.push({ text: "Rewrote your follow-up as a standalone question", quote: e.standalone, ms });
        break;
      case "retrieve": {
        const score = typeof e.top_score === "number" ? ` · top re-rank score ${e.top_score.toFixed(1)}` : "";
        steps.push({ text: "Searched the papers (dense + keyword), then re-ranked" + score, quote: e.query, ms });
        break;
      }
      case "rewrite_query":
        steps.push({ text: "Matches were weak, so the search query was rewritten", quote: e.new_query, ms, cls: "bad" });
        break;
      case "generate":
        steps.push({ text: "Drafted an answer with citations", ms });
        break;
      case "check_groundedness":
        if (e.verdict === true) steps.push({ text: "Self-check passed: every claim is supported by the sources", ms, cls: "good" });
        else if (e.verdict === "model_declined") steps.push({ text: "The model reported the sources were insufficient", ms });
        else if (e.verdict === "no_citations") steps.push({ text: "Draft rejected: it cited no sources (no model call needed)", ms, cls: "bad" });
        else if (e.verdict === "judge_error") steps.push({ text: "Self-check was unavailable, so the answer failed closed", ms, cls: "bad" });
        else steps.push({ text: `Self-check failed: ${e.unsupported ?? 0} unsupported claim(s), regenerating`, ms, cls: "bad" });
        break;
      case "classify_scope":
        // Only worth showing when it changed the path; a "no" just continues the normal flow.
        if (e.scope === true) steps.push({ text: "Recognised a question about the collection itself, not about one paper", ms });
        break;
      case "corpus_overview":
        steps.push({ text: `Answered from the index catalog (${e.papers} papers) instead of searching passages`, ms, cls: "good" });
        break;
      case "decline":
        steps.push({ text: "Declined instead of guessing", ms, cls: "bad" });
        break;
      default:
        break; // finalize etc. are not interesting to a reader
    }
  }
  return steps;
}

if (typeof module !== "undefined") {
  module.exports = { parseInline, parseBlocks, buildSteps, sectionLabel, safeArxivUrl, DECLINE_WHY };
}

/* ====================== DOM + app logic (browser only) ====================== */

if (typeof document !== "undefined") {
  (function () {
    const STORE_KEY = "arxiv-rag-chat-v1";
    const SLOW_MS = 8000;
    const $ = (id) => document.getElementById(id);
    const thread = $("thread"), empty = $("empty"), form = $("composer"), input = $("q");
    const sendBtn = $("send"), newChat = $("new-chat");
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let messages = []; // {id, role:"user", text} | {id, role:"assistant", data} | {id, role:"assistant", error}
    let nextId = 1;
    let busy = false;

    const el = (tag, cls, text) => {
      const n = document.createElement(tag);
      if (cls) n.className = cls;
      if (text !== undefined) n.textContent = text;
      return n;
    };

    /* ---------- persistence (best effort) ---------- */
    function save() {
      try { sessionStorage.setItem(STORE_KEY, JSON.stringify(messages.filter((m) => !m.pending))); } catch (_) { /* private mode */ }
    }
    function load() {
      try {
        const raw = sessionStorage.getItem(STORE_KEY);
        if (raw) messages = JSON.parse(raw);
        nextId = messages.reduce((mx, m) => Math.max(mx, m.id), 0) + 1;
      } catch (_) { messages = []; }
    }

    /* ---------- rendering ---------- */
    function renderInline(parent, nodes, msgId) {
      for (const n of nodes) {
        if (n.t === "text") parent.append(document.createTextNode(n.v));
        else if (n.t === "b") parent.append(el("strong", "", n.v));
        else if (n.t === "code") parent.append(el("code", "", n.v));
        else if (n.t === "cite") {
          const b = el("button", "cite", String(n.n));
          b.type = "button";
          b.setAttribute("aria-label", `Go to source ${n.n}`);
          b.addEventListener("click", () => {
            const target = document.getElementById(`src-${msgId}-${n.n}`);
            if (!target) return;
            target.scrollIntoView({ block: "center", behavior: reduceMotion ? "auto" : "smooth" });
            target.classList.add("flash");
            setTimeout(() => target.classList.remove("flash"), 1400);
          });
          parent.append(b);
        }
      }
    }

    function renderAnswer(parent, text, valid, msgId) {
      for (const b of parseBlocks(text, valid)) {
        if (b.type === "p") { const p = el("p"); renderInline(p, b.inline, msgId); parent.append(p); }
        else if (b.type === "pre") parent.append(el("pre", "", b.text));
        else {
          const list = el(b.type);
          for (const item of b.items) { const li = el("li"); renderInline(li, item, msgId); list.append(li); }
          parent.append(list);
        }
      }
    }

    function renderSources(parent, citations, msgId, title) {
      if (!citations.length) return;
      parent.append(el("div", "sources-title", title));
      const ol = el("ol", "sources");
      for (const c of citations) {
        const li = el("li", "source");
        li.id = `src-${msgId}-${c.index}`;
        li.title = c.snippet || "";
        li.append(el("span", "n", String(c.index)));
        const meta = el("div", "meta");
        meta.append(el("div", "t", c.title), el("div", "s", sectionLabel(c.section)));
        li.append(meta);
        const href = safeArxivUrl(c.abs_url);
        if (href) {
          const a = el("a", "", "arXiv ↗");
          a.href = href; a.target = "_blank"; a.rel = "noopener noreferrer";
          li.append(a);
        }
        ol.append(li);
      }
      parent.append(ol);
    }

    function renderHow(parent, data) {
      const steps = buildSteps(data.trace);
      if (!steps.length) return;
      const d = el("details", "how");
      d.append(el("summary", "", "How this answer was produced"));
      const ul = el("ul", "steps");
      for (const s of steps) {
        const li = el("li", "step" + (s.cls ? " " + s.cls : ""));
        li.append(document.createTextNode(s.text));
        if (s.quote) { li.append(document.createTextNode(" "), el("span", "q", `“${s.quote}”`)); }
        if (typeof s.ms === "number") li.append(el("span", "ms", `${s.ms} ms`));
        ul.append(li);
      }
      d.append(ul);
      const secs = (data.latency_ms / 1000).toFixed(1);
      d.append(el("div", "totals", `${data.llm_calls} model call${data.llm_calls === 1 ? "" : "s"} · ${secs}s total`));
      parent.append(d);
    }

    function renderMessage(m) {
      const wrap = el("article", "msg " + (m.role === "user" ? "msg-user" : "msg-assistant"));
      wrap.dataset.id = String(m.id);
      if (m.role === "user") {
        wrap.append(el("div", "bubble", m.text));
      } else if (m.pending) {
        const p = el("div", "pending");
        const dots = el("span", "dots");
        dots.append(el("span"), el("span"), el("span"));
        p.append(dots, el("span", "status", "Searching the papers…"));
        wrap.append(p);
      } else if (m.error) {
        wrap.append(el("div", "error", m.error));
      } else {
        const d = m.data;
        const valid = new Set((d.citations || []).map((c) => c.index));
        if (d.decision === "declined") {
          const card = el("div", "declined");
          card.append(el("p", "", "I can’t answer that from the indexed papers."));
          card.append(el("p", "why", DECLINE_WHY[d.decline_reason] || DECLINE_WHY.model_found_insufficient_context));
          wrap.append(card);
          renderSources(wrap, d.citations || [], m.id, "Closest papers searched");
        } else {
          const a = el("div", "answer");
          renderAnswer(a, d.answer, valid, m.id);
          wrap.append(a);
          renderSources(wrap, d.citations || [], m.id, "Sources");
        }
        renderHow(wrap, d);
      }
      return wrap;
    }

    function syncChrome() {
      const started = messages.length > 0;
      empty.hidden = started;
      newChat.hidden = !started;
      input.placeholder = started ? "Ask a follow-up…" : "Ask about retrieval, RAG, ColBERT, GraphRAG…";
    }

    function append(m) {
      const node = renderMessage(m);
      thread.append(node);
      return node;
    }

    function scrollTo(node) {
      node.scrollIntoView({ block: "start", behavior: reduceMotion ? "auto" : "smooth" });
    }

    /* ---------- talking to the API ---------- */
    function historyFor() {
      const turns = [];
      for (const m of messages) {
        if (m.role === "user") turns.push({ role: "user", content: m.text });
        else if (m.data) turns.push({ role: "assistant", content: m.data.answer });
      }
      return turns.slice(-6);
    }

    async function ask(question) {
      question = question.trim();
      if (busy || question.length < 3) return;
      busy = true;
      sendBtn.disabled = true;
      input.value = "";
      autosize();

      const history = historyFor();
      const userMsg = { id: nextId++, role: "user", text: question };
      messages.push(userMsg);
      const pending = { id: nextId++, role: "assistant", pending: true };
      messages.push(pending);
      syncChrome();
      const userNode = append(userMsg);
      const pendingNode = append(pending);
      scrollTo(userNode);

      const slow = setTimeout(() => {
        const s = pendingNode.querySelector(".status");
        if (s) s.textContent = "Waking the server (free tier scales to zero, so the first request can take up to a minute)…";
      }, SLOW_MS);

      let result;
      try {
        const ctrl = new AbortController();
        const kill = setTimeout(() => ctrl.abort(), 150000);
        const res = await fetch("/query", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ question, history, debug: true }),
          signal: ctrl.signal,
        });
        clearTimeout(kill);
        if (res.ok) result = { data: await res.json() };
        else if (res.status === 429) result = { error: "The free tier allows 3 questions per minute. Please wait a few seconds and try again." };
        else if (res.status === 503) result = { error: "The free AI quota is busy or used up for now. Please try again in a minute." };
        else if (res.status === 422) result = { error: "That question could not be processed. Try rephrasing it." };
        else result = { error: "Something went wrong upstream. Please try again in a moment." };
      } catch (_) {
        result = { error: "Can’t reach the server right now. Please try again." };
      }
      clearTimeout(slow);

      const done = { id: pending.id, role: "assistant", ...result };
      messages[messages.indexOf(pending)] = done;
      pendingNode.replaceWith(append(done));
      save();
      busy = false;
      sendBtn.disabled = false;
      input.focus({ preventScroll: true });
    }

    /* ---------- wiring ---------- */
    function autosize() {
      input.style.height = "auto";
      input.style.height = Math.min(input.scrollHeight, 160) + "px";
    }

    form.addEventListener("submit", (e) => { e.preventDefault(); ask(input.value); });
    input.addEventListener("input", autosize);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); }
    });
    document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => ask(c.dataset.q)));
    newChat.addEventListener("click", () => {
      if (busy) return;
      messages = [];
      try { sessionStorage.removeItem(STORE_KEY); } catch (_) { /* ignore */ }
      thread.querySelectorAll(".msg").forEach((n) => n.remove());
      syncChrome();
      input.focus();
    });

    load();
    // Drop anything that cannot be re-rendered (e.g. an interrupted pending message).
    messages = messages.filter((m) => m.role === "user" || m.data || m.error);
    for (const m of messages) append(m);
    syncChrome();
    autosize();
    input.focus({ preventScroll: true });

    // Pre-warm the scale-to-zero instance while the visitor is still reading the page.
    fetch("/health").catch(() => {});
  })();
}
