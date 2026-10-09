#!/usr/bin/env bash
# One-time GCP setup for the free-tier deployment. Idempotent where possible; safe to re-run.
#
#   PROJECT_ID=my-project BILLING_ACCOUNT=XXXXXX-XXXXXX-XXXXXX GITHUB_REPO=msaiprathyush/advanced-rag \
#     ./scripts/gcp_bootstrap.sh
#
# Prereqs: `gcloud auth login`, a project with billing linked (required even for free tier).
# Secret VALUES are read from your local .env and piped straight to Secret Manager; they are never
# printed or written to disk.
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
: "${BILLING_ACCOUNT:?set BILLING_ACCOUNT (for the budget alert)}"
: "${GITHUB_REPO:?set GITHUB_REPO, e.g. msaiprathyush/advanced-rag}"
REGION="${REGION:-us-central1}"   # free-tier eligible region
AR_REPO="${AR_REPO:-advanced-rag}"
BUDGET_AMOUNT="${BUDGET_AMOUNT:-1}"   # in the BILLING ACCOUNT's currency (see below)

gcloud config set project "$PROJECT_ID" >/dev/null
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"

# A budget must use the billing account's own currency or the API returns INVALID_ARGUMENT.
CURRENCY="$(gcloud billing accounts describe "$BILLING_ACCOUNT" --format='value(currencyCode)')"
if [ "$CURRENCY" = "INR" ] && [ "$BUDGET_AMOUNT" = "1" ]; then BUDGET_AMOUNT=100; fi   # ~ $1.20
echo "==> Budget alert (${BUDGET_AMOUNT} ${CURRENCY}) -- the guard for the free-tier-only constraint"
gcloud services enable billingbudgets.googleapis.com
if gcloud billing budgets list --billing-account="$BILLING_ACCOUNT" --format='value(displayName)' | grep -qx "advanced-rag guard"; then
  echo "   budget already exists"
else
  gcloud billing budgets create \
  --billing-account="$BILLING_ACCOUNT" \
  --display-name="advanced-rag guard" \
  --budget-amount="${BUDGET_AMOUNT}${CURRENCY}" \
  --threshold-rule=percent=0.5 --threshold-rule=percent=1.0 \
  --filter-projects="projects/${PROJECT_NUMBER}"
fi

echo "==> Enabling APIs"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com cloudscheduler.googleapis.com iamcredentials.googleapis.com \
  cloudbuild.googleapis.com

echo "==> Artifact Registry (keep only the 2 newest images: stays under the 0.5 GB free tier)"
gcloud artifacts repositories describe "$AR_REPO" --location="$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$AR_REPO" --repository-format=docker --location="$REGION"
cat >/tmp/ar-cleanup.json <<'EOF'
[{"name":"keep-recent","action":{"type":"Keep"},"mostRecentVersions":{"keepCount":2}},
 {"name":"delete-rest","action":{"type":"Delete"},"condition":{"tagState":"any"}}]
EOF
gcloud artifacts repositories set-cleanup-policies "$AR_REPO" --location="$REGION" \
  --policy=/tmp/ar-cleanup.json --no-dry-run

echo "==> Secrets (values come from .env; INGEST_API_KEY included)"
set -a; source .env; set +a
put_secret() {
  local name="$1" value="$2"
  [ -n "$value" ] || { echo "   skipping $name (empty in .env)"; return; }
  gcloud secrets describe "$name" >/dev/null 2>&1 || gcloud secrets create "$name" --replication-policy=automatic
  printf '%s' "$value" | gcloud secrets versions add "$name" --data-file=-
}
put_secret groq-api-key   "${GROQ_API_KEY:-}"
put_secret qdrant-url     "${QDRANT_URL:-}"
put_secret qdrant-api-key "${QDRANT_API_KEY:-}"
put_secret ingest-api-key "${INGEST_API_KEY:-}"

echo "==> Service accounts"
for sa in rag-runtime rag-deployer rag-scheduler; do
  gcloud iam service-accounts describe "$sa@${PROJECT_ID}.iam.gserviceaccount.com" >/dev/null 2>&1 || \
    gcloud iam service-accounts create "$sa"
done
RUNTIME="rag-runtime@${PROJECT_ID}.iam.gserviceaccount.com"
DEPLOYER="rag-deployer@${PROJECT_ID}.iam.gserviceaccount.com"
SCHED="rag-scheduler@${PROJECT_ID}.iam.gserviceaccount.com"
bind() { gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:$1" --role="$2" --condition=None >/dev/null; }
bind "$RUNTIME"  roles/secretmanager.secretAccessor
bind "$RUNTIME"  roles/logging.logWriter
bind "$DEPLOYER" roles/run.admin
bind "$DEPLOYER" roles/artifactregistry.writer
bind "$DEPLOYER" roles/secretmanager.viewer
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME" \
  --member="serviceAccount:$DEPLOYER" --role=roles/iam.serviceAccountUser >/dev/null

echo "==> Workload Identity Federation for GitHub Actions (no JSON keys)"
gcloud iam workload-identity-pools describe github --location=global >/dev/null 2>&1 || \
  gcloud iam workload-identity-pools create github --location=global --display-name="GitHub Actions"
gcloud iam workload-identity-pools providers describe github-oidc \
  --location=global --workload-identity-pool=github >/dev/null 2>&1 || \
  gcloud iam workload-identity-pools providers create-oidc github-oidc \
    --location=global --workload-identity-pool=github \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --attribute-condition="assertion.repository=='${GITHUB_REPO}'"
POOL="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github"
gcloud iam service-accounts add-iam-policy-binding "$DEPLOYER" \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/${POOL}/attribute.repository/${GITHUB_REPO}" >/dev/null

cat <<EOF

Done. Set these as GitHub *repository variables* (Settings > Secrets and variables > Actions > Variables):
  GCP_PROJECT_ID   = ${PROJECT_ID}
  GCP_REGION       = ${REGION}
  AR_REPO          = ${AR_REPO}
  GCP_DEPLOY_SA    = ${DEPLOYER}
  GCP_WIF_PROVIDER = ${POOL}/providers/github-oidc
And one repository *secret* for the eval workflow:  GROQ_API_KEY

After the first deploy, schedule the nightly re-index (1 of 3 free Scheduler jobs):
  gcloud run jobs add-iam-policy-binding arxiv-reindex --region=${REGION} \\
    --member=serviceAccount:${SCHED} --role=roles/run.invoker
  gcloud scheduler jobs create http arxiv-reindex-daily --location=${REGION} --schedule="0 6 * * *" \\
    --uri="https://run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/arxiv-reindex:run" \\
    --http-method=POST --oauth-service-account-email=${SCHED}
EOF
