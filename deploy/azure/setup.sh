#!/usr/bin/env bash
#
# CPQAI — Azure infrastructure setup
#
# Creates:  Resource Group, VNet, Azure Container Registry,
#           Azure OpenAI Service (GPT-4.1 + text-embedding-3-large),
#           VM (PostgreSQL + AGE + pgvector), Key Vault, Managed Identity,
#           Container Apps Environment + Container App,
#           and optionally API Management (APIM).
#
# Prerequisites:
#   - Azure CLI (`az`) installed and logged in (`az login`)
#   - An Azure subscription with sufficient quota
#   - Owner or Contributor role on the subscription
#   - Azure OpenAI access approved (https://aka.ms/oai/access)
#
# Usage:
#   chmod +x deploy/azure/setup.sh
#   ./deploy/azure/setup.sh              # interactive prompts
#   ./deploy/azure/setup.sh --defaults   # use all defaults (eastus)

set -euo pipefail

# ── Colour helpers ──────────────────────────────────────────────────
info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()    { printf '\033[1;32m[OK]\033[0m    %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
err()   { printf '\033[1;31m[ERR]\033[0m   %s\n' "$*" >&2; }

# ── Prerequisites check ────────────────────────────────────────────
if ! command -v az &>/dev/null; then
  err "Azure CLI not found. Install: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli"
  exit 1
fi

SUBSCRIPTION_ID=$(az account show --query id --output tsv 2>/dev/null) || {
  err "Azure CLI not logged in. Run: az login"
  exit 1
}

SUBSCRIPTION_NAME=$(az account show --query name --output tsv 2>/dev/null)

# ── Configuration ───────────────────────────────────────────────────
if [[ "${1:-}" == "--defaults" ]]; then
  LOCATION="eastus"
  RESOURCE_GROUP="cpqai-rg"
  SERVICE_NAME="cpqai"
  DB_VM_NAME="cpqai-db"
  DB_NAME="cpqai"
  DB_USER="cpqai"
  DB_VM_SIZE="Standard_B2s"          # 2 vCPU, 4 GB RAM — ~$30/month
  DB_DISK_SIZE="64"                  # GB Premium SSD for PostgreSQL data
  ACR_NAME="cpqaiacr"               # Must be globally unique, lowercase alphanumeric
  OPENAI_NAME="cpqai-openai"
  VNET_NAME="cpqai-vnet"
  KV_NAME="cpqai-kv"                # Must be globally unique
  IDENTITY_NAME="cpqai-identity"
  ENABLE_APIM=false
else
  read -rp "Location [eastus]: "                      LOCATION;       LOCATION=${LOCATION:-eastus}
  read -rp "Resource group [cpqai-rg]: "              RESOURCE_GROUP; RESOURCE_GROUP=${RESOURCE_GROUP:-cpqai-rg}
  read -rp "Container App name [cpqai]: "             SERVICE_NAME;   SERVICE_NAME=${SERVICE_NAME:-cpqai}
  read -rp "DB VM name [cpqai-db]: "                  DB_VM_NAME;     DB_VM_NAME=${DB_VM_NAME:-cpqai-db}
  read -rp "Database name [cpqai]: "                  DB_NAME;        DB_NAME=${DB_NAME:-cpqai}
  read -rp "Database user [cpqai]: "                  DB_USER;        DB_USER=${DB_USER:-cpqai}
  read -rp "DB VM size [Standard_B2s]: "              DB_VM_SIZE;     DB_VM_SIZE=${DB_VM_SIZE:-Standard_B2s}
  read -rp "DB disk size GB [64]: "                   DB_DISK_SIZE;   DB_DISK_SIZE=${DB_DISK_SIZE:-64}
  read -rp "Container Registry name [cpqaiacr]: "     ACR_NAME;       ACR_NAME=${ACR_NAME:-cpqaiacr}
  read -rp "Azure OpenAI name [cpqai-openai]: "       OPENAI_NAME;    OPENAI_NAME=${OPENAI_NAME:-cpqai-openai}
  read -rp "VNet name [cpqai-vnet]: "                 VNET_NAME;      VNET_NAME=${VNET_NAME:-cpqai-vnet}
  read -rp "Key Vault name [cpqai-kv]: "              KV_NAME;        KV_NAME=${KV_NAME:-cpqai-kv}
  read -rp "Managed Identity name [cpqai-identity]: " IDENTITY_NAME;  IDENTITY_NAME=${IDENTITY_NAME:-cpqai-identity}
  read -rp "Enable API Management? [n]: "             ENABLE_APIM_IN; ENABLE_APIM_IN=${ENABLE_APIM_IN:-n}
  [[ "$ENABLE_APIM_IN" =~ ^[yY] ]] && ENABLE_APIM=true || ENABLE_APIM=false
fi

ENV_NAME="${SERVICE_NAME}-env"
NSG_NAME="${DB_VM_NAME}-nsg"
CA_SUBNET="container-apps-subnet"
VM_SUBNET="vm-subnet"

info "Subscription:   $SUBSCRIPTION_NAME ($SUBSCRIPTION_ID)"
info "Location:       $LOCATION"
info "Resource Group: $RESOURCE_GROUP"
info "Service:        $SERVICE_NAME"
info "DB VM:          $DB_VM_NAME ($DB_VM_SIZE, ${DB_DISK_SIZE}GB Premium SSD)"
info "ACR:            $ACR_NAME"
info "Azure OpenAI:   $OPENAI_NAME"
info "VNet:           $VNET_NAME"
info "Key Vault:      $KV_NAME"
info "APIM:           $ENABLE_APIM"
echo ""

# ── 1. Resource Group ──────────────────────────────────────────────
info "Creating resource group..."
if az group show --name "$RESOURCE_GROUP" &>/dev/null; then
  ok "Resource group '$RESOURCE_GROUP' already exists"
else
  az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none
  ok "Resource group '$RESOURCE_GROUP' created"
fi

# ── 2. VNet + Subnets ─────────────────────────────────────────────
info "Creating VNet and subnets..."
if az network vnet show --name "$VNET_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
  ok "VNet '$VNET_NAME' already exists"
else
  az network vnet create \
    --name "$VNET_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --address-prefix 10.0.0.0/16 \
    --output none
  ok "VNet created"
fi

# Container Apps subnet (minimum /23 required by Azure Container Apps)
if az network vnet subnet show --name "$CA_SUBNET" --vnet-name "$VNET_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
  ok "Subnet '$CA_SUBNET' already exists"
else
  az network vnet subnet create \
    --name "$CA_SUBNET" \
    --vnet-name "$VNET_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --address-prefix 10.0.0.0/23 \
    --delegations Microsoft.App/environments \
    --output none
  ok "Container Apps subnet created (10.0.0.0/23)"
fi

# VM subnet
if az network vnet subnet show --name "$VM_SUBNET" --vnet-name "$VNET_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
  ok "Subnet '$VM_SUBNET' already exists"
else
  az network vnet subnet create \
    --name "$VM_SUBNET" \
    --vnet-name "$VNET_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --address-prefix 10.0.2.0/24 \
    --output none
  ok "VM subnet created (10.0.2.0/24)"
fi

# ── 3. Network Security Group ──────────────────────────────────────
info "Creating NSG for PostgreSQL VM..."
if az network nsg show --name "$NSG_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
  ok "NSG '$NSG_NAME' already exists"
else
  az network nsg create \
    --name "$NSG_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none

  # Allow PostgreSQL from Container Apps subnet only
  az network nsg rule create \
    --nsg-name "$NSG_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --name AllowPostgreSQL \
    --priority 100 \
    --direction Inbound \
    --access Allow \
    --protocol Tcp \
    --destination-port-ranges 5432 \
    --source-address-prefixes 10.0.0.0/23 \
    --output none

  ok "NSG created with PostgreSQL rule"
fi

# Associate NSG with VM subnet
az network vnet subnet update \
  --name "$VM_SUBNET" \
  --vnet-name "$VNET_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --network-security-group "$NSG_NAME" \
  --output none &>/dev/null

# ── 4. User-Assigned Managed Identity ──────────────────────────────
info "Creating managed identity..."
if az identity show --name "$IDENTITY_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
  ok "Managed identity '$IDENTITY_NAME' already exists"
else
  az identity create \
    --name "$IDENTITY_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none
  ok "Managed identity created"
fi

IDENTITY_ID=$(az identity show --name "$IDENTITY_NAME" --resource-group "$RESOURCE_GROUP" \
  --query id --output tsv)
IDENTITY_PRINCIPAL_ID=$(az identity show --name "$IDENTITY_NAME" --resource-group "$RESOURCE_GROUP" \
  --query principalId --output tsv)
IDENTITY_CLIENT_ID=$(az identity show --name "$IDENTITY_NAME" --resource-group "$RESOURCE_GROUP" \
  --query clientId --output tsv)
ok "Identity: $IDENTITY_CLIENT_ID"

# ── 5. Azure Container Registry ───────────────────────────────────
info "Creating Azure Container Registry..."
if az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
  ok "ACR '$ACR_NAME' already exists"
else
  az acr create \
    --name "$ACR_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --sku Basic \
    --output none
  ok "ACR '$ACR_NAME' created"
fi

ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" \
  --query loginServer --output tsv)

# Enable admin credentials for image pulls (no role assignments needed)
info "Enabling ACR admin credentials..."
az acr update --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" \
  --admin-enabled true --output none
ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" \
  --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" \
  --query "passwords[0].value" --output tsv)
ok "ACR admin credentials enabled"

# ── 6. Azure OpenAI Service ───────────────────────────────────────
info "Creating Azure OpenAI resource..."
if az cognitiveservices account show --name "$OPENAI_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
  ok "Azure OpenAI '$OPENAI_NAME' already exists"
else
  # Try creating; if it fails because of a soft-deleted resource, purge it first
  if ! az cognitiveservices account create \
    --name "$OPENAI_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --kind OpenAI \
    --sku S0 \
    --location "$LOCATION" \
    --yes \
    --output none 2>/dev/null; then
    info "Resource was soft-deleted, purging before re-creating..."
    az cognitiveservices account purge \
      --name "$OPENAI_NAME" \
      --resource-group "$RESOURCE_GROUP" \
      --location "$LOCATION" \
      --output none
    az cognitiveservices account create \
      --name "$OPENAI_NAME" \
      --resource-group "$RESOURCE_GROUP" \
      --kind OpenAI \
      --sku S0 \
      --location "$LOCATION" \
      --yes \
      --output none
    ok "Azure OpenAI resource created (after purging soft-deleted resource)"
  else
    ok "Azure OpenAI resource created"
  fi
fi

AOAI_ENDPOINT=$(az cognitiveservices account show \
  --name "$OPENAI_NAME" --resource-group "$RESOURCE_GROUP" \
  --query properties.endpoint --output tsv)
AOAI_KEY=$(az cognitiveservices account keys list \
  --name "$OPENAI_NAME" --resource-group "$RESOURCE_GROUP" \
  --query key1 --output tsv)
ok "Azure OpenAI endpoint: $AOAI_ENDPOINT"

# Create LLM deployment (GPT-4.1)
info "Creating GPT-4.1 deployment..."
if az cognitiveservices account deployment show \
    --name "$OPENAI_NAME" --resource-group "$RESOURCE_GROUP" \
    --deployment-name gpt-4.1 &>/dev/null; then
  ok "GPT-4.1 deployment already exists"
else
  az cognitiveservices account deployment create \
    --name "$OPENAI_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --deployment-name gpt-4.1 \
    --model-name gpt-4.1 \
    --model-version "2025-04-14" \
    --model-format OpenAI \
    --sku-name Standard \
    --sku-capacity 10 \
    --output none
  ok "GPT-4.1 deployment created (10K TPM)"
fi

# Create embedding deployment (text-embedding-3-large)
info "Creating text-embedding-3-large deployment..."
if az cognitiveservices account deployment show \
    --name "$OPENAI_NAME" --resource-group "$RESOURCE_GROUP" \
    --deployment-name text-embedding-3-large &>/dev/null; then
  ok "text-embedding-3-large deployment already exists"
else
  az cognitiveservices account deployment create \
    --name "$OPENAI_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --deployment-name text-embedding-3-large \
    --model-name text-embedding-3-large \
    --model-version "1" \
    --model-format OpenAI \
    --sku-name Standard \
    --sku-capacity 10 \
    --output none
  ok "text-embedding-3-large deployment created (10K TPM)"
fi

# ── 7. Generate DB password + API key ──────────────────────────────
DB_PASSWORD=$(openssl rand -base64 24 | tr -d '/+=' | head -c 32)
API_KEY=$(openssl rand -base64 32 | tr -d '/+=' | head -c 40)

# ── 8. Key Vault + Secrets ────────────────────────────────────────
# Try to store secrets in Key Vault for production use. If the current user
# lacks Key Vault permissions (common with Contributor-only roles), fall back
# to passing secrets directly to the Container App.
USE_KEYVAULT=false

info "Creating Key Vault..."
if az keyvault show --name "$KV_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
  ok "Key Vault '$KV_NAME' already exists"
else
  if ! az keyvault create \
    --name "$KV_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none 2>/dev/null; then
    info "Key Vault was soft-deleted, purging before re-creating..."
    az keyvault purge --name "$KV_NAME" --location "$LOCATION" --output none
    az keyvault create \
      --name "$KV_NAME" \
      --resource-group "$RESOURCE_GROUP" \
      --location "$LOCATION" \
      --output none
    ok "Key Vault '$KV_NAME' created (after purging soft-deleted vault)"
  else
    ok "Key Vault '$KV_NAME' created"
  fi
fi

KV_ID=$(az keyvault show --name "$KV_NAME" --resource-group "$RESOURCE_GROUP" \
  --query id --output tsv)
VAULT_URI=$(az keyvault show --name "$KV_NAME" --resource-group "$RESOURCE_GROUP" \
  --query properties.vaultUri --output tsv)

# Try to switch to access-policy mode (creator gets automatic permissions).
# This may fail on existing RBAC vaults if the user lacks authorization/write.
if az keyvault update --name "$KV_NAME" --resource-group "$RESOURCE_GROUP" \
    --enable-rbac-authorization false --output none 2>/dev/null; then
  # Grant current user a secret-set access policy (covers existing vaults
  # where the default policy may have been removed)
  CURRENT_USER_OID=$(az ad signed-in-user show --query id --output tsv 2>/dev/null) || true
  if [[ -n "$CURRENT_USER_OID" ]]; then
    az keyvault set-policy --name "$KV_NAME" --resource-group "$RESOURCE_GROUP" \
      --object-id "$CURRENT_USER_OID" \
      --secret-permissions set --output none 2>/dev/null || true
  fi
fi

# Try to store secrets (works via access policy or existing RBAC role)
info "Storing secrets in Key Vault..."
if az keyvault secret set --vault-name "$KV_NAME" --name cpqai-db-password \
    --value "$DB_PASSWORD" --output none 2>/dev/null; then
  ok "Secret 'cpqai-db-password' stored"

  az keyvault secret set --vault-name "$KV_NAME" --name cpqai-api-key \
    --value "$API_KEY" --output none
  ok "Secret 'cpqai-api-key' stored"

  az keyvault secret set --vault-name "$KV_NAME" --name cpqai-aoai-key \
    --value "$AOAI_KEY" --output none
  ok "Secret 'cpqai-aoai-key' stored"

  # Enable RBAC + grant managed identity access for runtime
  az keyvault update --name "$KV_NAME" --resource-group "$RESOURCE_GROUP" \
    --enable-rbac-authorization true --output none 2>/dev/null || true
  az role assignment create \
    --assignee-object-id "$IDENTITY_PRINCIPAL_ID" \
    --assignee-principal-type ServicePrincipal \
    --role "Key Vault Secrets User" \
    --scope "$KV_ID" \
    --output none 2>/dev/null || true
  ok "Key Vault configured for runtime access"
  USE_KEYVAULT=true
else
  warn "Cannot write to Key Vault (insufficient permissions)"
  warn "Secrets will be passed directly to the Container App"
  warn "To use Key Vault, ask your admin to grant you 'Key Vault Secrets Officer' on $KV_NAME"
fi

# ── 9. PostgreSQL VM ──────────────────────────────────────────────
info "Creating PostgreSQL VM..."
DB_VM_CREATED=false

if az vm show --name "$DB_VM_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
  ok "VM '$DB_VM_NAME' already exists"
else
  # Cloud-init script: install Docker, mount data disk, run PostgreSQL
  CLOUD_INIT_FILE=$(mktemp)
  cat > "$CLOUD_INIT_FILE" <<'CLOUDINIT'
#cloud-config
package_upgrade: true
packages:
  - docker.io
runcmd:
  - systemctl enable docker
  - systemctl start docker
  - |
    # Mount data disk (LUN 0 = first data disk attached to the VM)
    DATA_DEV=$(readlink -f /dev/disk/azure/scsi1/lun0 2>/dev/null || echo "")
    if [ -n "$DATA_DEV" ] && [ -b "$DATA_DEV" ]; then
      if ! blkid "$DATA_DEV" &>/dev/null; then
        mkfs.ext4 "$DATA_DEV"
      fi
      mkdir -p /pgdata
      mount "$DATA_DEV" /pgdata
      echo "$DATA_DEV /pgdata ext4 defaults,nofail 0 2" >> /etc/fstab
    else
      mkdir -p /pgdata
    fi
    chmod 777 /pgdata
  - docker pull gzdaniel/postgres-for-rag:16.6
  - docker run -d --name cpqai-postgres --restart always -p 5432:5432 -v /pgdata:/var/lib/postgresql/data gzdaniel/postgres-for-rag:16.6
CLOUDINIT

  az vm create \
    --name "$DB_VM_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --image Canonical:ubuntu-24_04-lts:server:latest \
    --size "$DB_VM_SIZE" \
    --vnet-name "$VNET_NAME" \
    --subnet "$VM_SUBNET" \
    --nsg "$NSG_NAME" \
    --public-ip-address "${DB_VM_NAME}-pip" \
    --data-disk-sizes-gb "$DB_DISK_SIZE" \
    --data-disk-caching ReadWrite \
    --storage-sku Premium_LRS \
    --admin-username azureuser \
    --generate-ssh-keys \
    --custom-data "$CLOUD_INIT_FILE" \
    --output none

  rm -f "$CLOUD_INIT_FILE"
  DB_VM_CREATED=true
  ok "VM created"
fi

# Get the VM's private IP
DB_PRIVATE_IP=$(az vm show \
  --name "$DB_VM_NAME" --resource-group "$RESOURCE_GROUP" \
  --show-details --query privateIps --output tsv)
ok "DB VM private IP: $DB_PRIVATE_IP"

# Wait for PostgreSQL and create user/database
if [[ "$DB_VM_CREATED" == "true" ]]; then
  info "Waiting for PostgreSQL to start (cloud-init + Docker pull)..."

  # Wait for PostgreSQL to be ready via az vm run-command (no SSH needed)
  for i in $(seq 1 40); do
    RESULT=$(az vm run-command invoke \
      --resource-group "$RESOURCE_GROUP" \
      --name "$DB_VM_NAME" \
      --command-id RunShellScript \
      --scripts "docker exec cpqai-postgres pg_isready 2>/dev/null && echo PGREADY || echo PGNOTREADY" \
      --query 'value[0].message' --output tsv 2>/dev/null) || true

    if [[ "$RESULT" == *"PGREADY"* ]]; then
      ok "PostgreSQL is ready"
      break
    fi

    if [[ $i -eq 40 ]]; then
      warn "PostgreSQL readiness check timed out. Cloud-init may still be running."
      warn "Check: az vm run-command invoke --resource-group $RESOURCE_GROUP --name $DB_VM_NAME --command-id RunShellScript --scripts 'cloud-init status; docker logs cpqai-postgres 2>&1 | tail -20'"
    fi
    sleep 10
  done

  info "Creating database user '$DB_USER' and database '$DB_NAME'..."
  SETUP_RESULT=$(az vm run-command invoke \
    --resource-group "$RESOURCE_GROUP" \
    --name "$DB_VM_NAME" \
    --command-id RunShellScript \
    --scripts "
      docker exec -e PGPASSWORD=postgres cpqai-postgres psql -h 127.0.0.1 -U postgres \
        -c \"CREATE USER ${DB_USER} WITH SUPERUSER CREATEDB CREATEROLE PASSWORD '${DB_PASSWORD}';\" && \
      docker exec -e PGPASSWORD=postgres cpqai-postgres psql -h 127.0.0.1 -U postgres \
        -c \"CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};\" && \
      docker exec -e PGPASSWORD=postgres cpqai-postgres psql -h 127.0.0.1 -U postgres -d ${DB_NAME} \
        -c \"CREATE EXTENSION IF NOT EXISTS age; CREATE EXTENSION IF NOT EXISTS vector;\" && \
      echo DBSETUP_OK
    " --query 'value[0].message' --output tsv 2>/dev/null) || true

  if [[ "$SETUP_RESULT" == *"DBSETUP_OK"* ]]; then
    ok "Database user, database, and extensions created"
  else
    warn "Could not create DB user/database. Create manually:"
    warn "  az vm run-command invoke --resource-group $RESOURCE_GROUP --name $DB_VM_NAME --command-id RunShellScript --scripts 'docker exec -e PGPASSWORD=postgres cpqai-postgres psql -h 127.0.0.1 -U postgres -c \"CREATE USER ${DB_USER} WITH SUPERUSER PASSWORD \\\"${DB_PASSWORD}\\\";\"'"
  fi
fi

# ── 10. Build and push Docker image ────────────────────────────────
BUILD_TAG=$(date +%Y%m%d-%H%M%S)
info "Building container image with ACR Build (tag: $BUILD_TAG)..."

az acr build \
  --registry "$ACR_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --image "${SERVICE_NAME}:${BUILD_TAG}" \
  --image "${SERVICE_NAME}:latest" \
  --file Dockerfile.containerapp \
  --timeout 1800 \
  .

ok "Container image built and pushed to ACR"

# ── 11. Container Apps Environment ─────────────────────────────────
info "Creating Container Apps Environment..."
CA_SUBNET_ID=$(az network vnet subnet show \
  --name "$CA_SUBNET" \
  --vnet-name "$VNET_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query id --output tsv)

if az containerapp env show --name "$ENV_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
  ok "Container Apps Environment '$ENV_NAME' already exists"
else
  az containerapp env create \
    --name "$ENV_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --infrastructure-subnet-resource-id "$CA_SUBNET_ID" \
    --output none
  ok "Container Apps Environment created"
fi

# ── 12. Container App ─────────────────────────────────────────────
info "Creating Container App..."

if az containerapp show --name "$SERVICE_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
  ok "Container App '$SERVICE_NAME' already exists — updating"
  # Ensure registry uses admin credentials (may have been created with managed identity)
  az containerapp registry set \
    --name "$SERVICE_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --server "$ACR_LOGIN_SERVER" \
    --username "$ACR_USERNAME" \
    --password "$ACR_PASSWORD" \
    --output none
  az containerapp update \
    --name "$SERVICE_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --image "${ACR_LOGIN_SERVER}/${SERVICE_NAME}:${BUILD_TAG}" \
    --output none
else
  # Build secrets args: Key Vault references if available, direct values otherwise
  if [[ "$USE_KEYVAULT" == "true" ]]; then
    CA_SECRETS=(
      "db-password=keyvaultref:${VAULT_URI}secrets/cpqai-db-password,identityref:${IDENTITY_ID}"
      "api-key=keyvaultref:${VAULT_URI}secrets/cpqai-api-key,identityref:${IDENTITY_ID}"
      "aoai-key=keyvaultref:${VAULT_URI}secrets/cpqai-aoai-key,identityref:${IDENTITY_ID}"
    )
  else
    CA_SECRETS=(
      "db-password=$DB_PASSWORD"
      "api-key=$API_KEY"
      "aoai-key=$AOAI_KEY"
    )
  fi

  az containerapp create \
    --name "$SERVICE_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --environment "$ENV_NAME" \
    --image "${ACR_LOGIN_SERVER}/${SERVICE_NAME}:latest" \
    --registry-server "$ACR_LOGIN_SERVER" \
    --registry-username "$ACR_USERNAME" \
    --registry-password "$ACR_PASSWORD" \
    --user-assigned "$IDENTITY_ID" \
    --target-port 8080 \
    --ingress external \
    --min-replicas 0 \
    --max-replicas 10 \
    --cpu 2 \
    --memory 4Gi \
    --secrets "${CA_SECRETS[@]}" \
    --env-vars \
      "POSTGRES_HOST=${DB_PRIVATE_IP}" \
      "POSTGRES_PORT=5432" \
      "POSTGRES_USER=${DB_USER}" \
      "POSTGRES_DATABASE=${DB_NAME}" \
      "POSTGRES_PASSWORD=secretref:db-password" \
      "LIGHTRAG_API_KEY=secretref:api-key" \
      "AZURE_OPENAI_API_KEY=secretref:aoai-key" \
      "AZURE_OPENAI_ENDPOINT=${AOAI_ENDPOINT}" \
      "AZURE_OPENAI_API_VERSION=2024-08-01-preview" \
      "AZURE_OPENAI_DEPLOYMENT=gpt-4.1" \
      "AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large" \
      "AZURE_EMBEDDING_API_VERSION=2023-05-15" \
    --output none

  ok "Container App created"
fi

# Wait for Container App to be ready
info "Waiting for Container App to be ready..."
for i in $(seq 1 30); do
  STATUS=$(az containerapp show \
    --name "$SERVICE_NAME" --resource-group "$RESOURCE_GROUP" \
    --query "properties.provisioningState" --output tsv 2>/dev/null) || true

  if [[ "$STATUS" == "Succeeded" ]]; then
    ok "Container App is ready"
    break
  elif [[ "$STATUS" == "Failed" ]]; then
    err "Container App deployment failed"
    err "Check: az containerapp show --name $SERVICE_NAME --resource-group $RESOURCE_GROUP"
    exit 1
  fi

  if [[ $i -eq 30 ]]; then
    warn "Timed out waiting for Container App. Current status: $STATUS"
    warn "Check: az containerapp show --name $SERVICE_NAME --resource-group $RESOURCE_GROUP"
  fi
  sleep 10
done

SERVICE_URL=$(az containerapp show \
  --name "$SERVICE_NAME" --resource-group "$RESOURCE_GROUP" \
  --query "properties.configuration.ingress.fqdn" --output tsv 2>/dev/null || echo "(pending)")

# ── 13. API Management (optional) ──────────────────────────────────
if [[ "$ENABLE_APIM" == "true" ]]; then
  APIM_NAME="${SERVICE_NAME}-apim"
  info "Creating API Management instance (Consumption tier)..."
  info "Note: APIM provisioning can take 15-30 minutes"

  if az apim show --name "$APIM_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
    ok "APIM '$APIM_NAME' already exists"
  else
    PUBLISHER_EMAIL=$(az ad signed-in-user show --query mail --output tsv 2>/dev/null || echo "admin@example.com")
    az apim create \
      --name "$APIM_NAME" \
      --resource-group "$RESOURCE_GROUP" \
      --location "$LOCATION" \
      --publisher-name "CPQAI" \
      --publisher-email "${PUBLISHER_EMAIL:-admin@example.com}" \
      --sku-name Consumption \
      --output none
    ok "APIM created"
  fi

  # Import Container App as APIM backend
  APIM_STATE=$(az apim show --name "$APIM_NAME" --resource-group "$RESOURCE_GROUP" \
    --query "provisioningState" --output tsv 2>/dev/null) || true

  if [[ "$APIM_STATE" == "Succeeded" ]]; then
    info "Configuring APIM backend..."

    az apim api create \
      --resource-group "$RESOURCE_GROUP" \
      --service-name "$APIM_NAME" \
      --api-id cpqai-api \
      --display-name "CPQAI API" \
      --path "" \
      --service-url "https://${SERVICE_URL}" \
      --protocols https \
      --output none 2>/dev/null || true

    APIM_URL=$(az apim show --name "$APIM_NAME" --resource-group "$RESOURCE_GROUP" \
      --query "gatewayUrl" --output tsv 2>/dev/null || echo "(pending)")
    ok "APIM configured — gateway URL: $APIM_URL"
  else
    info "APIM is still provisioning ($APIM_STATE). Configure the backend API manually after provisioning completes."
  fi
fi

# ── 14. Print summary ─────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  CPQAI deployed on Microsoft Azure"
echo "=============================================="
echo ""
echo "  Service URL:    https://${SERVICE_URL}"
echo "  DB VM:          $DB_VM_NAME ($DB_PRIVATE_IP:5432)"
echo "  Database:       $DB_NAME"
echo "  DB User:        $DB_USER"
echo "  API Key:        $API_KEY"
echo ""
echo "  Azure OpenAI:   $AOAI_ENDPOINT"
echo "  LLM Model:      gpt-4.1"
echo "  Embed Model:    text-embedding-3-large"
echo ""
echo "  ACR:            $ACR_LOGIN_SERVER"
echo "  Image:          ${ACR_LOGIN_SERVER}/${SERVICE_NAME}:latest"
echo ""
echo "  Test:"
echo "    curl -H 'X-API-Key: ${API_KEY}' https://${SERVICE_URL}/health"
echo ""
if [[ "$USE_KEYVAULT" == "true" ]]; then
  echo "  Secrets stored in Key Vault ($KV_NAME):"
  echo "    - cpqai-db-password"
  echo "    - cpqai-api-key"
  echo "    - cpqai-aoai-key"
else
  echo "  Secrets:        passed directly to Container App"
fi
echo ""
echo "  DB VM management:"
echo "    az vm start --name $DB_VM_NAME --resource-group $RESOURCE_GROUP"
echo "    az vm stop  --name $DB_VM_NAME --resource-group $RESOURCE_GROUP --no-wait"
echo "    az vm run-command invoke --resource-group $RESOURCE_GROUP --name $DB_VM_NAME \\"
echo "      --command-id RunShellScript --scripts 'docker logs cpqai-postgres 2>&1 | tail -20'"
echo ""
echo "  Redeploy:"
echo "    az acr build --registry $ACR_NAME --image ${SERVICE_NAME}:latest -f Dockerfile.containerapp ."
echo "    az containerapp update --name $SERVICE_NAME --resource-group $RESOURCE_GROUP \\"
echo "      --image ${ACR_LOGIN_SERVER}/${SERVICE_NAME}:latest"
echo ""
if [[ "$ENABLE_APIM" == "true" ]]; then
  echo "  APIM gateway:   ${APIM_URL:-check Azure portal}"
  echo ""
fi
if [[ "$USE_KEYVAULT" == "true" ]]; then
  warn "Save the API key above — it is also in Key Vault (cpqai-api-key)"
else
  warn "Save the API key above — it is NOT stored in Key Vault (insufficient permissions)"
fi
echo ""
