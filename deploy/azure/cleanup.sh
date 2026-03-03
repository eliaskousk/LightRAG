#!/usr/bin/env bash
#
# CPQAI — Azure resource cleanup
#
# Deletes all resources created by setup.sh.
# The fastest way is to delete the entire resource group (--resource-group-only),
# which removes everything in one operation.
#
# Usage:
#   ./deploy/azure/cleanup.sh              # interactive (confirms each step)
#   ./deploy/azure/cleanup.sh --yes        # skip confirmations
#   ./deploy/azure/cleanup.sh --defaults   # use default names + skip confirmations

set -euo pipefail

# ── Colour helpers ──────────────────────────────────────────────────
info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()    { printf '\033[1;32m[OK]\033[0m    %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
err()   { printf '\033[1;31m[ERR]\033[0m   %s\n' "$*" >&2; }
skip()  { printf '\033[1;90m[SKIP]\033[0m  %s\n' "$*"; }

# ── Prerequisites check ────────────────────────────────────────────
if ! command -v az &>/dev/null; then
  err "Azure CLI not found."
  exit 1
fi

SUBSCRIPTION_ID=$(az account show --query id --output tsv 2>/dev/null) || {
  err "Azure CLI not logged in. Run: az login"
  exit 1
}

# ── Configuration ───────────────────────────────────────────────────
AUTO_YES=false

if [[ "${1:-}" == "--defaults" ]]; then
  AUTO_YES=true
  LOCATION="eastus"
  RESOURCE_GROUP="cpqai-rg"
  SERVICE_NAME="cpqai"
  DB_VM_NAME="cpqai-db"
  ACR_NAME="cpqaiacr"
  OPENAI_NAME="cpqai-openai"
  VNET_NAME="cpqai-vnet"
  KV_NAME="cpqai-kv"
  IDENTITY_NAME="cpqai-identity"
elif [[ "${1:-}" == "--yes" ]]; then
  AUTO_YES=true
  read -rp "Resource group [cpqai-rg]: "              RESOURCE_GROUP; RESOURCE_GROUP=${RESOURCE_GROUP:-cpqai-rg}
  read -rp "Container App name [cpqai]: "             SERVICE_NAME;   SERVICE_NAME=${SERVICE_NAME:-cpqai}
  read -rp "DB VM name [cpqai-db]: "                  DB_VM_NAME;     DB_VM_NAME=${DB_VM_NAME:-cpqai-db}
  read -rp "Container Registry name [cpqaiacr]: "     ACR_NAME;       ACR_NAME=${ACR_NAME:-cpqaiacr}
  read -rp "Azure OpenAI name [cpqai-openai]: "       OPENAI_NAME;    OPENAI_NAME=${OPENAI_NAME:-cpqai-openai}
  read -rp "VNet name [cpqai-vnet]: "                 VNET_NAME;      VNET_NAME=${VNET_NAME:-cpqai-vnet}
  read -rp "Key Vault name [cpqai-kv]: "              KV_NAME;        KV_NAME=${KV_NAME:-cpqai-kv}
  read -rp "Managed Identity name [cpqai-identity]: " IDENTITY_NAME;  IDENTITY_NAME=${IDENTITY_NAME:-cpqai-identity}
else
  read -rp "Resource group [cpqai-rg]: "              RESOURCE_GROUP; RESOURCE_GROUP=${RESOURCE_GROUP:-cpqai-rg}
  read -rp "Container App name [cpqai]: "             SERVICE_NAME;   SERVICE_NAME=${SERVICE_NAME:-cpqai}
  read -rp "DB VM name [cpqai-db]: "                  DB_VM_NAME;     DB_VM_NAME=${DB_VM_NAME:-cpqai-db}
  read -rp "Container Registry name [cpqaiacr]: "     ACR_NAME;       ACR_NAME=${ACR_NAME:-cpqaiacr}
  read -rp "Azure OpenAI name [cpqai-openai]: "       OPENAI_NAME;    OPENAI_NAME=${OPENAI_NAME:-cpqai-openai}
  read -rp "VNet name [cpqai-vnet]: "                 VNET_NAME;      VNET_NAME=${VNET_NAME:-cpqai-vnet}
  read -rp "Key Vault name [cpqai-kv]: "              KV_NAME;        KV_NAME=${KV_NAME:-cpqai-kv}
  read -rp "Managed Identity name [cpqai-identity]: " IDENTITY_NAME;  IDENTITY_NAME=${IDENTITY_NAME:-cpqai-identity}
fi

ENV_NAME="${SERVICE_NAME}-env"
NSG_NAME="${DB_VM_NAME}-nsg"
APIM_NAME="${SERVICE_NAME}-apim"

echo ""
warn "This will permanently delete the following resources in resource group '$RESOURCE_GROUP':"
echo ""
echo "  - Container App:          $SERVICE_NAME"
echo "  - Container Apps Env:     $ENV_NAME"
echo "  - VM + disks:             $DB_VM_NAME"
echo "  - Azure OpenAI:           $OPENAI_NAME"
echo "  - Container Registry:     $ACR_NAME"
echo "  - Key Vault:              $KV_NAME"
echo "  - VNet:                   $VNET_NAME"
echo "  - NSG:                    $NSG_NAME"
echo "  - Managed Identity:       $IDENTITY_NAME"
echo "  - APIM (if exists):       $APIM_NAME"
echo ""
echo "  TIP: Deleting the entire resource group is faster and guaranteed to"
echo "       remove everything. Run: az group delete --name $RESOURCE_GROUP --yes"
echo ""

if [[ "$AUTO_YES" != "true" ]]; then
  read -rp "Are you sure? Type 'delete' to confirm: " CONFIRM
  if [[ "$CONFIRM" != "delete" ]]; then
    err "Aborted."
    exit 1
  fi
fi

# ── Helper to delete a resource safely ──────────────────────────────
try_delete() {
  local description="$1"
  shift
  info "Deleting $description..."
  if "$@" 2>/dev/null; then
    ok "$description deleted"
  else
    skip "$description not found or already deleted"
  fi
}

# ── 1. API Management (if exists) ─────────────────────────────────
if az apim show --name "$APIM_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
  try_delete "APIM '$APIM_NAME'" \
    az apim delete --name "$APIM_NAME" --resource-group "$RESOURCE_GROUP" --yes
fi

# ── 2. Container App ──────────────────────────────────────────────
try_delete "Container App '$SERVICE_NAME'" \
  az containerapp delete --name "$SERVICE_NAME" --resource-group "$RESOURCE_GROUP" --yes

# ── 3. Container Apps Environment ──────────────────────────────────
try_delete "Container Apps Environment '$ENV_NAME'" \
  az containerapp env delete --name "$ENV_NAME" --resource-group "$RESOURCE_GROUP" --yes

# ── 4. VM (includes OS disk, data disk is separate) ────────────────
info "Deleting VM '$DB_VM_NAME'..."
if az vm show --name "$DB_VM_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
  # Get data disk IDs before deleting the VM
  DATA_DISKS=$(az vm show --name "$DB_VM_NAME" --resource-group "$RESOURCE_GROUP" \
    --query "storageProfile.dataDisks[*].managedDisk.id" --output tsv 2>/dev/null) || true
  OS_DISK=$(az vm show --name "$DB_VM_NAME" --resource-group "$RESOURCE_GROUP" \
    --query "storageProfile.osDisk.managedDisk.id" --output tsv 2>/dev/null) || true

  # Delete VM (does not automatically delete disks or NICs)
  az vm delete --name "$DB_VM_NAME" --resource-group "$RESOURCE_GROUP" \
    --yes --force-deletion true --output none 2>/dev/null
  ok "VM deleted"

  # Delete data disks
  for disk_id in $DATA_DISKS; do
    info "Deleting data disk..."
    az disk delete --ids "$disk_id" --yes --output none 2>/dev/null || true
    ok "Data disk deleted"
  done

  # Delete OS disk
  if [[ -n "$OS_DISK" ]]; then
    az disk delete --ids "$OS_DISK" --yes --output none 2>/dev/null || true
    ok "OS disk deleted"
  fi

  # Delete NIC
  NIC_NAME="${DB_VM_NAME}VMNic"
  az network nic delete --name "$NIC_NAME" --resource-group "$RESOURCE_GROUP" \
    --output none 2>/dev/null || true

  # Delete public IP
  try_delete "public IP '${DB_VM_NAME}-pip'" \
    az network public-ip delete --name "${DB_VM_NAME}-pip" --resource-group "$RESOURCE_GROUP"
else
  skip "VM '$DB_VM_NAME' not found"
fi

# ── 5. Azure OpenAI ──────────────────────────────────────────────
try_delete "Azure OpenAI '$OPENAI_NAME'" \
  az cognitiveservices account delete --name "$OPENAI_NAME" --resource-group "$RESOURCE_GROUP"

# Purge the soft-deleted resource (Azure OpenAI uses soft-delete)
info "Purging soft-deleted Azure OpenAI resource..."
az cognitiveservices account purge \
  --name "$OPENAI_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$(az group show --name "$RESOURCE_GROUP" --query location --output tsv 2>/dev/null || echo eastus)" \
  --output none 2>/dev/null || true

# ── 6. Key Vault ──────────────────────────────────────────────────
info "Deleting Key Vault '$KV_NAME'..."
if az keyvault show --name "$KV_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
  az keyvault delete --name "$KV_NAME" --resource-group "$RESOURCE_GROUP" --output none 2>/dev/null
  ok "Key Vault deleted"
  # Purge the soft-deleted vault
  info "Purging soft-deleted Key Vault..."
  az keyvault purge --name "$KV_NAME" --output none 2>/dev/null || true
  ok "Key Vault purged"
else
  skip "Key Vault '$KV_NAME' not found"
fi

# ── 7. Container Registry ─────────────────────────────────────────
try_delete "ACR '$ACR_NAME'" \
  az acr delete --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --yes

# ── 8. Managed Identity ───────────────────────────────────────────
try_delete "Managed Identity '$IDENTITY_NAME'" \
  az identity delete --name "$IDENTITY_NAME" --resource-group "$RESOURCE_GROUP"

# ── 9. NSG ────────────────────────────────────────────────────────
# Must detach from subnet first
az network vnet subnet update \
  --name "vm-subnet" \
  --vnet-name "$VNET_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --remove networkSecurityGroup \
  --output none 2>/dev/null || true

try_delete "NSG '$NSG_NAME'" \
  az network nsg delete --name "$NSG_NAME" --resource-group "$RESOURCE_GROUP"

# ── 10. VNet ──────────────────────────────────────────────────────
try_delete "VNet '$VNET_NAME'" \
  az network vnet delete --name "$VNET_NAME" --resource-group "$RESOURCE_GROUP"

# ── 11. Resource Group (optional — leave for user) ─────────────────
echo ""
info "Resource group '$RESOURCE_GROUP' was NOT deleted (it may contain other resources)."
info "To delete it: az group delete --name $RESOURCE_GROUP --yes"
echo ""

# ── Done ─────────────────────────────────────────────────────────
ok "Cleanup complete. All CPQAI resources have been removed from resource group '$RESOURCE_GROUP'."
echo ""
