# Another Automatic Video Editor

> **Lingue disponibili**: [English](README.md) | [Italiano (corrente)](README.it.md)

Mini-infrastruttura AWS (CloudFormation) + uno script di lancio che trasforma una cartella di **immagini + video** in un **MP4 pronto per YouTube**, con:

- **Analisi AI** di immagini / frame di video (AWS Rekognition)
- Opzionale **speech-to-text** (AWS Transcribe) per migliorare capitoli e descrizione
- Opzionale **estrazione “Best moments”** (shot scoring leggero) per creare un capitolo di highlights
- Generazione automatica di:
  - `chapters.txt`
  - `description.md` (SEO-friendly, template rigido)
  - `title.txt`

Tu fornisci un `manifest.json` (scaletta / vincoli / link YouTube). Il sistema produce `output.mp4` pronto da caricare.

> Modello Bedrock di default: **Amazon Nova 2 Lite** (`amazon.nova-lite-v1:0`).

---

## Cosa viene creato su AWS

CloudFormation crea:

- Bucket S3 di input (upload del job)
- Bucket S3 di output (MP4 renderizzato + capitoli/descrizione)
- Step Functions state machine
- Task ECS Fargate (render FFmpeg)
- Ruoli IAM + CloudWatch Logs

---

## Due modalità

Imposta `style.mode` nel `manifest.json`:

- `aftermovie` (default): video corto tipo highlights (es. 2–4 minuti)
- `longform`: video **lungo** (es. 30–120+ minuti)

La modalità longform è costruita in modo **deterministico** (così include sempre i video principali), mentre Bedrock viene usato per **SEO/titolo/descrizione**.

---

## Quick start (script)

### 1) Crea lo stack AWS

```bash
./another_automatic_video_editor.sh create \
  --region eu-west-1 \
  --bedrock-region eu-west-1
```

### 2) Esegui un job di esempio

#### Esempio A — Aftermovie (corto)

```bash
./another_automatic_video_editor.sh run \
  --region eu-west-1 \
  --job-dir ./examples/job
```

#### Esempio B — Longform (video lungo + intro come prima clip)

Metti i media qui:

- `./examples/job_longform/media/`
  - `intro.mp4` (prima clip)
  - registrazioni talk complete (es. `talk1.mp4`, `talk2.mp4`)
  - foto

Poi lancia:

```bash
./another_automatic_video_editor.sh run \
  --region eu-west-1 \
  --job-dir ./examples/job_longform
```

### 3) Output

Il comando `run` aspetta la fine del job e scarica i risultati in locale.

Output locale di default: `./another_automatic_video_editor_output/<JOB_ID>/`.

---

## Estratti utili dal manifest

### Forzare l’intro come prima clip

```json
"intro": {
  "file": "intro.mp4",
  "duration_seconds": 8,
  "caption": "AWS User Group Salerno"
}
```

### Abilitare “Best moments” (shot scoring)

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

### Template descrizione YouTube (override manuale)

```json
"seo": {
  "enabled": true,
  "cta": "👍 Like + iscriviti!",
  "hashtags": ["#AWS", "#Serverless", "#Meetup"]
}
```

---

## Output

Nel bucket di output (e nella cartella di download locale) trovi:

- `output.mp4` — video finale
- `chapters.txt` — capitoli YouTube (timestamp + titolo)
- `description.md` — descrizione SEO-friendly (template + hashtag + link)
- `title.txt` — titolo suggerito
- `catalog.json`, `plan.json`, `render_meta.json` — artefatti debug
- `best_moments_top.json` — highlights “scorati” (se abilitato)

---

## Clean up

```bash
./another_automatic_video_editor.sh delete --yes
```

---

## Licenza

CC BY-NC 4.0 — vedi [LICENSE](LICENSE).
