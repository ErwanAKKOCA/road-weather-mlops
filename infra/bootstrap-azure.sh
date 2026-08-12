#!/usr/bin/env bash
set -euo pipefail

: "${AZURE_SUBSCRIPTION_ID:?Set AZURE_SUBSCRIPTION_ID}"
: "${AZURE_LOCATION:=francecentral}"
: "${AZURE_RESOURCE_GROUP:=phase7-weather-rg}"
: "${AZURE_CONTAINERAPPS_ENV:=phase7-weather-env}"
: "${AZURE_ACR_NAME:?Set a globally unique AZURE_ACR_NAME}"

az account set --subscription "$AZURE_SUBSCRIPTION_ID"
az extension add --name containerapp --upgrade
az group create --name "$AZURE_RESOURCE_GROUP" --location "$AZURE_LOCATION"
az acr create \
  --name "$AZURE_ACR_NAME" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --sku Basic \
  --admin-enabled false
az containerapp env create \
  --name "$AZURE_CONTAINERAPPS_ENV" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --location "$AZURE_LOCATION"

echo "Azure base resources are ready. Add the repository variables and OIDC secrets from docs/deployment.md, then run the deployment workflow."
