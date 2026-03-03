# CPQAI on Microsoft Azure

Deploy CPQAI as a serverless Container App backed by a self-managed PostgreSQL VM (with Apache AGE + pgvector) and Azure OpenAI Service (GPT-4.1 + text-embedding-3-large).

## Architecture

```
                    API Management (optional gateway)
                            |
Internet  ──>  Container Apps (CPQAI container, HTTPS)
                   |              |
                   |              └── Azure OpenAI Service
                   |                   ├── GPT-4.1 (LLM)
                   |                   └── text-embedding-3-large (embeddings)
                   |
              VNet integration (Container Apps → VM subnet)
                   |
              Azure VM (internal IP)
              gzdaniel/postgres-for-rag:16.6
              PostgreSQL 16 + AGE + pgvector
                   ├── KV Storage
                   ├── Vector Storage (pgvector HNSW)
                   ├── Graph Storage  (Apache AGE)
                   └── Doc Status Storage
              [Premium SSD attached]

Azure Container Registry ── container images
Key Vault                ── DB password, API key, Azure OpenAI key
Managed Identity         ── ACR pull + Key Vault access
```

## Why a VM instead of Azure Database for PostgreSQL?

LightRAG's `PGGraphStorage` uses the **Apache AGE** extension for graph operations (Cypher queries, vertex/edge storage). Azure Database for PostgreSQL Flexible Server does not support AGE — it only allows a curated set of extensions.

A self-managed PostgreSQL container (`gzdaniel/postgres-for-rag:16.6`) ships with AGE, pgvector, and other RAG-relevant extensions pre-installed, at a fraction of the cost:

| Option | Monthly cost |
|---|---|
| Azure DB for PostgreSQL `GP_Standard_D2s_v3` (HA) | ~$130 |
| Azure VM `Standard_B2s` + 64GB Premium SSD | **~$35** |

## Prerequisites

1. **Azure CLI** installed and logged in (`az login`)
2. **Azure subscription** with sufficient quota (VM cores, OpenAI)
3. **Owner or Contributor** role on the subscription
4. **Azure OpenAI access** approved ([Request access](https://aka.ms/oai/access))
5. GPT-4.1 and text-embedding-3-large must be available in your chosen region (eastus recommended)

## Quick start

### Automated setup

```bash
# Interactive (prompts for region, names, VM size, etc.)
./deploy/azure/setup.sh

# Or accept all defaults (eastus, Standard_B2s, 64GB Premium SSD)
./deploy/azure/setup.sh --defaults
```

This creates:
- Resource group with all resources
- VNet with Container Apps + VM subnets
- NSG restricting PostgreSQL to Container Apps subnet
- User-assigned managed identity (ACR pull + Key Vault access)
- Azure Container Registry
- Azure OpenAI resource with GPT-4.1 and text-embedding-3-large deployments
- Key Vault with DB password, API key, and Azure OpenAI key
- VM running `gzdaniel/postgres-for-rag:16.6` with Premium SSD
- Container Apps Environment with VNet integration
- Container App with Key Vault secret references
- (Optional) API Management gateway

### Manual / step-by-step

#### 1. Create resource group and VNet

```bash
az group create --name cpqai-rg --location eastus

az network vnet create \
  --name cpqai-vnet --resource-group cpqai-rg \
  --address-prefix 10.0.0.0/16

# Container Apps subnet (minimum /23)
az network vnet subnet create \
  --name container-apps-subnet --vnet-name cpqai-vnet \
  --resource-group cpqai-rg --address-prefix 10.0.0.0/23 \
  --delegations Microsoft.App/environments

# VM subnet
az network vnet subnet create \
  --name vm-subnet --vnet-name cpqai-vnet \
  --resource-group cpqai-rg --address-prefix 10.0.2.0/24
```

#### 2. Create Azure OpenAI resource

```bash
az cognitiveservices account create \
  --name cpqai-openai --resource-group cpqai-rg \
  --kind OpenAI --sku S0 --location eastus --yes

# Deploy models
az cognitiveservices account deployment create \
  --name cpqai-openai --resource-group cpqai-rg \
  --deployment-name gpt-4.1 --model-name gpt-4.1 \
  --model-version "2025-04-14" --model-format OpenAI \
  --sku-name Standard --sku-capacity 10

az cognitiveservices account deployment create \
  --name cpqai-openai --resource-group cpqai-rg \
  --deployment-name text-embedding-3-large --model-name text-embedding-3-large \
  --model-version "1" --model-format OpenAI \
  --sku-name Standard --sku-capacity 10

# Get endpoint and key
AOAI_ENDPOINT=$(az cognitiveservices account show --name cpqai-openai \
  --resource-group cpqai-rg --query properties.endpoint --output tsv)
AOAI_KEY=$(az cognitiveservices account keys list --name cpqai-openai \
  --resource-group cpqai-rg --query key1 --output tsv)
```

#### 3. Create the PostgreSQL VM

```bash
DB_PASSWORD=$(openssl rand -base64 24 | tr -d '/+=' | head -c 32)

# NSG allowing PostgreSQL from Container Apps subnet
az network nsg create --name cpqai-db-nsg --resource-group cpqai-rg
az network nsg rule create --nsg-name cpqai-db-nsg --resource-group cpqai-rg \
  --name AllowPostgreSQL --priority 100 --direction Inbound --access Allow \
  --protocol Tcp --destination-port-ranges 5432 --source-address-prefixes 10.0.0.0/23

# Create VM with data disk
az vm create \
  --name cpqai-db --resource-group cpqai-rg \
  --image Canonical:ubuntu-24_04-lts:server:latest \
  --size Standard_B2s \
  --vnet-name cpqai-vnet --subnet vm-subnet --nsg cpqai-db-nsg \
  --public-ip-address cpqai-db-pip \
  --data-disk-sizes-gb 64 --storage-sku Premium_LRS \
  --admin-username azureuser --generate-ssh-keys \
  --custom-data cloud-init.yaml

DB_IP=$(az vm show --name cpqai-db --resource-group cpqai-rg \
  --show-details --query privateIps --output tsv)
echo "DB IP: $DB_IP"
```

#### 4. Store secrets in Key Vault

```bash
az keyvault create --name cpqai-kv --resource-group cpqai-rg \
  --enable-rbac-authorization true

az keyvault secret set --vault-name cpqai-kv --name cpqai-db-password --value "$DB_PASSWORD"
az keyvault secret set --vault-name cpqai-kv --name cpqai-api-key --value "YOUR_API_KEY"
az keyvault secret set --vault-name cpqai-kv --name cpqai-aoai-key --value "$AOAI_KEY"
```

#### 5. Build and deploy

```bash
# Build with ACR Build (no local Docker needed)
az acr build --registry cpqaiacr --image cpqai:latest -f Dockerfile.containerapp .

# Or build locally and push
az acr login --name cpqaiacr
docker build -f Dockerfile.containerapp -t cpqaiacr.azurecr.io/cpqai:latest .
docker push cpqaiacr.azurecr.io/cpqai:latest
```

## Files

| File | Purpose |
|------|---------|
| `Dockerfile.containerapp` | Multi-stage build optimized for Container Apps (port 8080, PG storage defaults, Azure OpenAI) |
| `deploy/azure/setup.sh` | One-command infrastructure provisioning script |
| `deploy/azure/cleanup.sh` | One-command resource teardown script |
| `deploy/azure/.env.azure.example` | Reference environment variable configuration |

## Configuration

### Azure OpenAI authentication

The Container App uses an **API key** stored in Key Vault and injected via managed identity at runtime. No hardcoded credentials.

Key environment variables:
- `LLM_BINDING=azure_openai` — selects the Azure OpenAI LLM provider
- `LLM_MODEL=gpt-4.1` — model name (matches deployment name)
- `AZURE_OPENAI_ENDPOINT` — Azure OpenAI resource endpoint
- `AZURE_OPENAI_DEPLOYMENT=gpt-4.1` — LLM deployment name
- `AZURE_OPENAI_API_VERSION=2024-08-01-preview` — API version
- `EMBEDDING_BINDING=azure_openai` — selects the Azure OpenAI embedding provider
- `EMBEDDING_MODEL=text-embedding-3-large` — embedding model (1536 dimensions)
- `AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large` — embedding deployment name

### PostgreSQL connection

Container Apps connects to the PostgreSQL VM via **VNet integration** — the Container Apps Environment is deployed in a subnet within the same VNet as the VM.

```
POSTGRES_HOST=10.0.2.4    # VM's private IP
POSTGRES_PORT=5432
```

An NSG restricts port 5432 to the Container Apps subnet (`10.0.0.0/23`).

### Secrets

Passwords and API keys are stored in **Key Vault** and injected as environment variables at runtime via managed identity.

| Key Vault secret | Injected as |
|---|---|
| `cpqai-db-password` | `POSTGRES_PASSWORD` |
| `cpqai-api-key` | `LIGHTRAG_API_KEY` |
| `cpqai-aoai-key` | `AZURE_OPENAI_API_KEY` |

### Scaling

Default Container App configuration:

| Setting | Value | Notes |
|---|---|---|
| Min replicas | 0 | Scale to zero when idle |
| Max replicas | 10 | Adjust based on load |
| CPU | 2 | Minimum for concurrent LLM calls |
| Memory | 4 Gi | Handles graph operations and large contexts |
| Ingress | External | HTTPS with auto-TLS |

### API Management (optional)

To add APIM as a gateway (rate limiting, analytics, developer portal):

```bash
# Consumption tier (~$3.50 per million calls)
az apim create \
  --name cpqai-apim --resource-group cpqai-rg \
  --publisher-name "CPQAI" --publisher-email admin@example.com \
  --sku-name Consumption

# Import the Container App as a backend
az apim api create \
  --resource-group cpqai-rg --service-name cpqai-apim \
  --api-id cpqai-api --display-name "CPQAI API" \
  --path "" --service-url "https://YOUR_CONTAINER_APP_FQDN" \
  --protocols https
```

## DB VM management

```bash
# Run commands on the VM (no SSH needed)
az vm run-command invoke --resource-group cpqai-rg --name cpqai-db \
  --command-id RunShellScript --scripts 'docker logs cpqai-postgres 2>&1 | tail -20'

# Connect to PostgreSQL from inside the VM
az vm run-command invoke --resource-group cpqai-rg --name cpqai-db \
  --command-id RunShellScript --scripts 'docker exec cpqai-postgres psql -U cpqai -d cpqai -c "\dt"'

# Stop / start the VM (data persists on the Premium SSD)
az vm stop  --name cpqai-db --resource-group cpqai-rg --no-wait
az vm start --name cpqai-db --resource-group cpqai-rg

# Resize the VM (deallocate first)
az vm deallocate --name cpqai-db --resource-group cpqai-rg
az vm resize --name cpqai-db --resource-group cpqai-rg --size Standard_B4ms
az vm start --name cpqai-db --resource-group cpqai-rg
```

### Backups

The PostgreSQL data lives on a separate managed disk (Premium SSD). For backups:

```bash
# Get the data disk ID
DATA_DISK=$(az vm show --name cpqai-db --resource-group cpqai-rg \
  --query "storageProfile.dataDisks[0].managedDisk.id" --output tsv)

# Create a snapshot
az snapshot create \
  --name cpqai-db-backup-$(date +%Y%m%d) \
  --resource-group cpqai-rg \
  --source "$DATA_DISK"

# For automated backups, use Azure Backup
az backup vault create --name cpqai-backup --resource-group cpqai-rg --location eastus
az backup protection enable-for-vm \
  --resource-group cpqai-rg --vault-name cpqai-backup \
  --vm cpqai-db --policy-name DefaultPolicy
```

## Monitoring

```bash
# Container App logs
az containerapp logs show --name cpqai --resource-group cpqai-rg --follow

# Container App revisions and replicas
az containerapp revision list --name cpqai --resource-group cpqai-rg --output table

# VM serial console (boot logs)
az vm boot-diagnostics get-boot-log --name cpqai-db --resource-group cpqai-rg
```

## Cost estimates

| Component | Approximate monthly cost |
|---|---|
| VM `Standard_B2s` + 64GB Premium SSD | ~$35 |
| Container Apps (scale-to-zero, moderate traffic) | ~$5–50 |
| Azure OpenAI (GPT-4.1 + text-embedding-3-large, moderate) | ~$10–50 |
| ACR Basic | ~$5 |
| Key Vault | < $1 |
| VNet | Free |
| APIM Consumption (optional) | ~$3.50/M calls |
| **Total** | **~$55–140** |

Costs vary with document ingestion volume and query frequency. The DB VM runs 24/7 — stop it when not in use to save costs, or use [VM auto-shutdown](https://learn.microsoft.com/en-us/azure/virtual-machines/auto-shutdown-vm) for off-hours.

## Cleanup

```bash
# Interactive (confirms each resource)
./deploy/azure/cleanup.sh

# Skip confirmations
./deploy/azure/cleanup.sh --yes

# Use defaults + skip confirmations
./deploy/azure/cleanup.sh --defaults

# Fastest: delete the entire resource group
az group delete --name cpqai-rg --yes --no-wait
```
