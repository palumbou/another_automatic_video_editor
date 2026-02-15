# Guida all'Infrastruttura

Guida completa all'infrastruttura AWS per Another Automatic Video Editor.

> **Lingue disponibili**: [English](INFRASTRUCTURE.md) | [Italiano (corrente)](INFRASTRUCTURE.it.md)

## Architettura

Il sistema utilizza un'architettura serverless su AWS:

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   S3 Input      │────▶ │  Step Functions  │────▶│   ECS Fargate   │
│   Bucket        │      │  State Machine   │      │   (FFmpeg)      │
└─────────────────┘      └──────────────────┘      └─────────────────┘
                                                          │
                        ┌──────────────────┐              │
                        │   S3 Output      │◀─────────────┘
                        │   Bucket         │
                        └──────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐    ┌───────────────────┐    ┌───────────────────┐
│  Rekognition  │    │     Bedrock       │    │    Transcribe     │
│  (Labels)     │    │  (AI Planning)    │    │  (Speech-to-Text) │
└───────────────┘    └───────────────────┘    └───────────────────┘
```

## Componenti AWS

### Networking

- **VPC**: VPC dedicata con CIDR configurabile (default: 10.20.0.0/16)
- **Subnet**: Due subnet pubbliche in AZ diverse per alta disponibilità
- **Internet Gateway**: Per accesso a internet dai task Fargate
- **Security Group**: Solo traffico outbound per API AWS

### Storage

- **Input Bucket**: Upload job e codice applicazione
- **Output Bucket**: Video renderizzati e metadati

### Compute

- **ECS Cluster**: Cluster Fargate per task di rendering
- **Task Definition**: Container Python con FFmpeg e ImageMagick
- **Step Functions**: Orchestrazione del workflow di rendering

### Servizi AI

- **Amazon Rekognition**: Analisi immagini e frame video
- **Amazon Bedrock**: Generazione piano video e SEO
- **Amazon Transcribe**: Speech-to-text (opzionale)

### Monitoring

- **CloudWatch Logs**: Log centralizzati con retention configurabile
- **Cost Tags**: Tutte le risorse taggate con `CostCenter`

## Deployment

### Quick Start

```bash
# Deploy con defaults
./another_automatic_video_editor.sh create --region eu-west-1

# Verifica stato
./another_automatic_video_editor.sh status

# Esegui un job
./another_automatic_video_editor.sh run --job-dir ./examples/job
```

### Parametri CloudFormation

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `VpcCidr` | 10.20.0.0/16 | CIDR block per la VPC |
| `FargateCpu` | 2048 | CPU units (1024-16384) |
| `FargateMemory` | 4096 | Memoria in MiB |
| `EphemeralStorageGiB` | 50 | Storage effimero (20-200) |
| `BedrockModelId` | amazon.nova-lite-v1:0 | Modello Bedrock |
| `BedrockRegion` | (regione stack) | Override regione Bedrock |
| `EnableTranscribe` | true | Abilita speech-to-text |
| `LogRetentionDays` | 14 | Retention log CloudWatch |

## Costi

### Stima per job tipico (video 3 min, 50 file media)

| Servizio | Costo stimato |
|----------|---------------|
| ECS Fargate (10 min) | ~$0.05 |
| S3 Storage | ~$0.01 |
| Rekognition | ~$0.10 |
| Bedrock | ~$0.02 |
| Transcribe | ~$0.05 |
| **Totale** | **~$0.23/job** |

### Ottimizzazione costi

- Disabilita Transcribe se non necessario (`--enable-transcribe false`)
- Usa modelli Bedrock più economici
- Riduci CPU/memoria se i job sono semplici

## Cleanup

```bash
# Elimina tutte le risorse
./another_automatic_video_editor.sh delete --yes
```

⚠️ Questo elimina tutti i bucket S3 e i dati contenuti.

## Troubleshooting

### Task Fargate fallisce

1. Controlla i log CloudWatch: `/aws/ecs/<stack-name>/worker`
2. Verifica che il manifest.json sia valido
3. Controlla che i media siano nei formati supportati

### Bedrock non risponde

1. Verifica che il modello sia abilitato nella regione
2. Usa `--bedrock-region` per specificare una regione diversa
3. Controlla i permessi IAM

### Out of memory

Aumenta la memoria Fargate:
```bash
./another_automatic_video_editor.sh create --memory 8192
```
