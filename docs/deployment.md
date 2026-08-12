# Azure Container Apps deployment

The deployment uses two Container Apps (API and UI), Azure Container Registry, and the
checksummed Phase VI release already stored in Google Cloud Storage.

## 1. Bootstrap Azure resources

Authenticate with Azure CLI, set the variables, and run:

```bash
export AZURE_SUBSCRIPTION_ID="..."
export AZURE_ACR_NAME="globallyuniquename"
bash infra/bootstrap-azure.sh
```

The workflow can create the API and UI Container Apps in this environment. Both use external
ingress: target port `8000` for the API and `8080` for the UI. After the first deployment,
configure HTTP startup/readiness probes on `/health/ready` for the API and a liveness probe
on `/health/live`. DINOv2 initialization can take time, so give the startup probe at least
three minutes.

After the first deployment, assign each Container App a system-managed identity, grant it
`AcrPull` on the registry, and configure the app to pull from ACR using that identity. This
is the recommended credential model for Azure Container Apps.

## 2. Configure GitHub

Repository variables:

- `AZURE_RESOURCE_GROUP`
- `AZURE_ACR_NAME`
- `AZURE_CONTAINERAPPS_ENV`
- `AZURE_API_APP_NAME`
- `AZURE_UI_APP_NAME`

Repository/environment secrets:

- `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` for Azure OIDC
- `GCP_CREDENTIALS`, restricted to read the single Phase VI release object

Protect the `production` GitHub environment with manual approval if desired.

## 3. Deploy

Run **Deploy Azure Container Apps** manually, or let it start automatically only after the **CI** workflow succeeds on `main`. Each image is tagged with
the immutable Git commit SHA. The workflow downloads the locked release, builds the API,
resolves its public URL, injects that URL at frontend build time, and deploys both apps.

The API is intentionally limited to one replica because causal EMA state is held in memory.
Horizontal scaling requires moving sequence state to a shared store such as Redis.

## 4. Verify

```bash
curl https://API_FQDN/health/live
curl https://API_FQDN/health/ready
curl https://API_FQDN/api/v1/model
curl https://API_FQDN/metrics
```

Never interpret successful deployment as additional scientific validation. Phase VI model
results remain limited to Virtual KITTI 2 and oracle semantic-sky masks.


## 5. Rollback

See `docs/rollback.md`. Deployments use immutable Git-SHA image tags, and model releases must pass SHA-256 validation before deployment.
