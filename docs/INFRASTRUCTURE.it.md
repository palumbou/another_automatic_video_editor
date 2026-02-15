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
- **Amazon Bedrock**: Generazione piano video e SEO (con verifica connettività)
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
| `BedrockModelId` | us.amazon.nova-lite-v1:0 | Modello Bedrock o inference profile |
| `BedrockRegion` | (regione stack) | Override regione Bedrock |
| `EnableTranscribe` | true | Abilita speech-to-text |
| `LogRetentionDays` | 14 | Retention log CloudWatch |

### Configurazione Modello Bedrock

Il modello di default usa un inference profile (`us.amazon.nova-lite-v1:0`) per inferenza cross-region. Profile disponibili:

| Regione | Inference Profile |
|---------|-------------------|
| Regioni US | `us.amazon.nova-lite-v1:0` |
| Regioni EU | `eu.amazon.nova-lite-v1:0` |

Puoi anche usare model ID diretti se hai accesso on-demand abilitato:
- `anthropic.claude-3-haiku-20240307-v1:0`
- `anthropic.claude-3-sonnet-20240229-v1:0`

## Comportamento AI

### Verifica Connettività

Prima di elaborare i media, il runner testa la connettività Bedrock:

1. **Test passa**: La pianificazione AI procede normalmente
2. **Test fallisce (interattivo)**: L'utente sceglie se continuare con fallback o interrompere
3. **Test fallisce (non-interattivo)**: Usa automaticamente il piano di fallback

### Opzioni Run

| Opzione | Comportamento |
|---------|---------------|
| (default) | Testa AI, chiede all'utente se non disponibile |
| `--skip-ai-check` | Salta test, fallback silenzioso se AI fallisce |
| `--require-ai` | Fallisce immediatamente se AI non disponibile |

### Piani di Fallback

Quando l'AI non è disponibile:
- **modalità aftermovie**: Usa piano euristico (cronologico, scalato alla durata target)
- **modalità longform**: Usa piano deterministico (include tutti i media, basato su capitoli)

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
- Usa `--skip-ai-check` per evitare overhead del test AI

## Cleanup

Il comando delete svuota automaticamente i bucket S3 prima dell'eliminazione:

```bash
# Elimina tutte le risorse (con conferma)
./another_automatic_video_editor.sh delete

# Elimina senza conferma
./another_automatic_video_editor.sh delete --yes
```

Il processo di cleanup:
1. Recupera i nomi dei bucket dagli output dello stack
2. Svuota il bucket di input (`aws s3 rm --recursive`)
3. Svuota il bucket di output (`aws s3 rm --recursive`)
4. Elimina lo stack CloudFormation
5. Attende il completamento dell'eliminazione

## Troubleshooting

### AI non disponibile

Se vedi "Bedrock AI is NOT available":

1. **Verifica accesso al modello**: Assicurati che il modello sia abilitato nel tuo account AWS (console Bedrock → Model access)
2. **Usa inference profile**: Cambia a `us.amazon.nova-lite-v1:0` o `eu.amazon.nova-lite-v1:0`
3. **Verifica permessi IAM**: Assicurati che `bedrock:Converse` sia permesso
4. **Prova regione diversa**: Usa `--bedrock-region` per specificare una regione con accesso al modello

### Task Fargate fallisce

1. Controlla i log CloudWatch: `/aws/ecs/<stack-name>/worker`
2. Verifica che il manifest.json sia JSON valido
3. Controlla che i media siano nei formati supportati (jpg, png, mp4, mov, ecc.)
4. Controlla `render.log` nell'output per errori dettagliati

### Out of memory

Aumenta la memoria Fargate:
```bash
./another_automatic_video_editor.sh create --memory 8192
```

### Eliminazione stack fallisce

Se l'eliminazione si blocca, i bucket potrebbero non essere vuoti. Lo script ora gestisce questo automaticamente, ma se persistono problemi:

```bash
# Svuota manualmente i bucket
aws s3 rm s3://<input-bucket> --recursive
aws s3 rm s3://<output-bucket> --recursive

# Riprova eliminazione
./another_automatic_video_editor.sh delete --yes
```

### Volume musica troppo basso

Aumenta `style.music.volume` nel manifest.json (consigliato: 0.40-0.50):

```json
"music": {
  "enabled": true,
  "duck": true,
  "volume": 0.45
}
```
