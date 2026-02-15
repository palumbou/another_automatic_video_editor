# Another Automatic Video Editor

> **Lingue disponibili**: [English](README.md) | [Italiano (corrente)](README.it.md)

Mini-infrastruttura AWS (CloudFormation) + uno script di lancio che trasforma una cartella di **immagini + video** in un **MP4 pronto per YouTube**, con:

- **Pianificazione AI** (AWS Bedrock) per sequenziamento intelligente del video
- **Analisi AI** di immagini / frame di video (AWS Rekognition)
- Opzionale **speech-to-text** (AWS Transcribe) per migliorare capitoli e descrizione
- Opzionale **estrazione "Best moments"** (shot scoring leggero) per creare un capitolo di highlights
- **Verifica connettività AI** prima dell'elaborazione con opzioni di fallback interattive
- Generazione automatica di:
  - `chapters.txt`
  - `description.md` (SEO-friendly, template rigido)
  - `title.txt`

Tu fornisci un `manifest.json` (scaletta / vincoli / link YouTube). Il sistema produce `final.mp4` pronto da caricare.

> Modello Bedrock di default: **Amazon Nova Lite** (`us.amazon.nova-lite-v1:0` inference profile).

## Architettura

Sistema serverless su AWS con i seguenti componenti:

### Componenti AWS
- **S3 Buckets**: Input (upload job) e Output (video renderizzati)
- **Step Functions**: Orchestrazione workflow
- **ECS Fargate**: Rendering video containerizzato con FFmpeg
- **Amazon Rekognition**: Analisi immagini e frame video
- **Amazon Bedrock**: Pianificazione AI e generazione SEO
- **Amazon Transcribe**: Speech-to-text (opzionale)
- **CloudWatch**: Logging e monitoring

### Componenti Codice
- `src/config.py`: Gestione configurazione
- `src/logging_config.py`: Logging strutturato
- `src/aws_clients.py`: Utility client AWS
- `src/models.py`: Modelli dati
- `src/utils.py`: Funzioni utility
- `app/runner.py`: Pipeline di rendering principale

## Struttura Progetto

```
another_automatic_video_editor/
├── app/                    # Codice applicazione principale
│   └── runner.py          # Pipeline rendering video
├── src/                    # Codice sorgente modulare
│   ├── __init__.py
│   ├── config.py
│   ├── logging_config.py
│   ├── aws_clients.py
│   ├── models.py
│   └── utils.py
├── infrastructure/         # Template CloudFormation
│   └── template.yaml
├── docs/                  # Documentazione
│   └── INFRASTRUCTURE.md
├── examples/              # Job di esempio
│   ├── job/
│   └── job_longform/
├── requirements.txt       # Dipendenze Python
└── another_automatic_video_editor.sh  # CLI principale
```

## Due Modalità

Imposta `style.mode` nel `manifest.json`:

- `aftermovie` (default): video corto tipo highlights (es. 2–4 minuti)
- `longform`: video **lungo** (es. 30–120+ minuti)

Entrambe le modalità usano l'AI (Bedrock) per la pianificazione quando disponibile, con fallback automatico alla pianificazione deterministica se l'AI non è disponibile.

## Quick Start

### Prerequisiti

- AWS CLI installato e configurato
- Python 3.11+
- Credenziali AWS valide con permessi appropriati
- Accesso al modello Bedrock abilitato (Amazon Nova Lite consigliato)

### 1) Crea lo stack AWS

```bash
./another_automatic_video_editor.sh create \
  --region eu-west-1 \
  --bedrock-model-id us.amazon.nova-lite-v1:0
```

### 2) Esegui un job di esempio

#### Esempio A — Aftermovie (corto)

```bash
./another_automatic_video_editor.sh run \
  --job-dir ./examples/job
```

#### Esempio B — Longform (video lungo + intro)

Metti i media qui:

- `./examples/job_longform/media/`
  - `intro.jpg` o `intro.mp4` (prima clip)
  - registrazioni talk complete (es. `talk1.mp4`, `talk2.mp4`)
  - foto

Poi lancia:

```bash
./another_automatic_video_editor.sh run \
  --job-dir ./examples/job_longform
```

### 3) Output

Il comando `run` aspetta la fine del job e scarica i risultati in locale.

Output locale di default: `./another_automatic_video_editor_output/<JOB_ID>/`.

## Riferimento CLI

```bash
./another_automatic_video_editor.sh <comando> [opzioni]
```

### Comandi

| Comando | Descrizione |
|---------|-------------|
| `create` | Deploy (o aggiorna) lo stack CloudFormation |
| `status` | Mostra stato stack e output importanti |
| `run` | Carica una cartella job ed esegue il workflow |
| `delete` | Elimina lo stack CloudFormation (svuota i bucket prima) |
| `help` | Mostra aiuto |

### Opzioni Comuni

| Opzione | Descrizione | Default |
|---------|-------------|---------|
| `--region <region>` | Regione AWS | `eu-west-1` |
| `--name <stack-name>` | Nome stack CloudFormation | auto-generato |
| `--yes` | Salta conferme | `false` |

### Opzioni Create

| Opzione | Descrizione | Default |
|---------|-------------|---------|
| `--bedrock-model-id <id>` | Modello Bedrock o inference profile | `us.amazon.nova-lite-v1:0` |
| `--bedrock-region <region>` | Override regione Bedrock | regione stack |
| `--cpu <units>` | CPU Fargate (1024-16384) | `2048` |
| `--memory <MiB>` | Memoria Fargate | `4096` |
| `--ephemeral-gib <GiB>` | Storage Fargate (20-200) | `50` |
| `--enable-transcribe <bool>` | Abilita Amazon Transcribe | `true` |

### Opzioni Run

| Opzione | Descrizione | Default |
|---------|-------------|---------|
| `--job-dir <path>` | Cartella job con manifest.json + media/ | richiesto |
| `--out-dir <path>` | Cartella output locale | `./another_automatic_video_editor_output` |
| `--skip-ai-check` | Salta test connettività AI | `false` |
| `--require-ai` | Fallisce se AI non disponibile (no fallback) | `false` |

### Opzioni Comportamento AI

Il runner esegue un test di connettività AI prima di elaborare i media:

- **Default**: Testa Bedrock, chiede all'utente se non disponibile (continua con fallback o interrompi)
- `--skip-ai-check`: Salta il test, usa silenziosamente il fallback se l'AI fallisce dopo
- `--require-ai`: Fallisce immediatamente se l'AI non è disponibile

## Configurazione Manifest

### Intro (prima clip)

Supporta sia immagini (jpg, png, ecc.) che video:

```json
"intro": {
  "file": "intro.jpg",
  "duration_seconds": 5,
  "caption": "AWS User Group Salerno"
}
```

### Musica

```json
"music": {
  "enabled": true,
  "duck": true,
  "volume": 0.45
}
```

- `volume`: 0.0-1.0 (consigliato: 0.40-0.50)
- `duck`: Abbassa il volume della musica quando il video ha parlato

### Best Moments (shot scoring)

```json
"best_moments": {
  "enabled": true,
  "insert_in_timeline": true,
  "max_clips_total": 12,
  "max_moments_per_video": 2,
  "clip_duration_seconds": 8,
  "samples_per_video": 10,
  "min_gap_seconds": 25
}
```

### Override SEO

```json
"seo": {
  "enabled": true,
  "cta": "👍 Like + iscriviti!",
  "hashtags": ["#AWS", "#Serverless", "#Meetup"]
}
```

### Configurazione AI

```json
"ai": {
  "enabled": true,
  "enable_transcribe": true
}
```

## Output

La cartella output contiene:

| File | Descrizione |
|------|-------------|
| `final.mp4` | Video finale renderizzato |
| `video_no_music.mp4` | Video senza traccia musicale |
| `chapters.txt` | Capitoli YouTube (timestamp + titoli) |
| `description.md` | Descrizione YouTube SEO-friendly |
| `title.txt` | Titolo YouTube suggerito |
| `catalog.json` | Catalogo media con analisi |
| `plan.json` | Piano video (AI o fallback) |
| `render_meta.json` | Metadati render |
| `best_moments_top.json` | Clip con punteggio più alto (se abilitato) |
| `render.log` | Log elaborazione |

## Documentazione

- [Guida Infrastruttura](docs/INFRASTRUCTURE.it.md) - Setup completo infrastruttura ([English](docs/INFRASTRUCTURE.md))

## Pulizia

Il comando delete svuota automaticamente i bucket S3 prima dell'eliminazione:

```bash
./another_automatic_video_editor.sh delete --yes
```

Senza `--yes`, verrà chiesta conferma.

## Troubleshooting

### AI non disponibile

Se vedi "Bedrock AI is NOT available", verifica:
1. L'accesso al modello è abilitato nel tuo account AWS
2. Stai usando l'inference profile corretto (es. `us.amazon.nova-lite-v1:0` per US, `eu.amazon.nova-lite-v1:0` per EU)
3. I permessi IAM includono `bedrock:Converse`

### Volume musica troppo basso

Aumenta `style.music.volume` nel manifest (consigliato: 0.40-0.50).

### Non tutti i media usati

Per la modalità `aftermovie`, i media vengono scalati per rientrare in `target_duration_seconds`. Aumenta questo valore o usa la modalità `longform`.

## Licenza

CC BY-NC 4.0 — vedi [LICENSE](LICENSE).
