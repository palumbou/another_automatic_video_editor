# Another Automatic Video Editor

> **Available languages**: [English (current)](README.md) | [Italiano](README.it.md)

AWS mini-infrastructure (CloudFormation) + a single launcher script that turns a folder of **images + videos** into a **YouTube-ready MP4**, with:

- **AI-driven analysis** (AWS Rekognition) of images / sampled video frames
- Optional **speech-to-text** (AWS Transcribe) to improve chapter titles / description
- Optional **"Best moments" extraction** (lightweight shot scoring) to build a highlights chapter
- Auto-generated:
  - `chapters.txt`
  - `description.md` (SEO-friendly, rigid template)
  - `title.txt`

You provide a `manifest.json` (outline / constraints / YouTube links). The system outputs a final `output.mp4` you can upload directly to YouTube.

> Default Bedrock model: **Amazon Nova Lite** (`amazon.nova-lite-v1:0`).

## Architecture

Serverless system on AWS with the following components:

### AWS Components
- **S3 Buckets**: Input (job uploads) and Output (rendered videos)
- **Step Functions**: Workflow orchestration
- **ECS Fargate**: Container-based video rendering with FFmpeg
- **Amazon Rekognition**: Image and video frame analysis
- **Amazon Bedrock**: AI-powered video planning and SEO generation
- **Amazon Transcribe**: Speech-to-text (optional)
- **CloudWatch**: Logging and monitoring

### Code Components
- `src/config.py`: Configuration management
- `src/logging_config.py`: Structured logging
- `src/aws_clients.py`: AWS client utilities
- `src/models.py`: Data models
- `src/utils.py`: Utility functions
- `app/runner.py`: Main rendering pipeline

## Project Structure

```
another_automatic_video_editor/
├── app/                    # Main application code
│   └── runner.py          # Video rendering pipeline
├── src/                    # Modular source code
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── logging_config.py  # Structured logging
│   ├── aws_clients.py     # AWS client utilities
│   ├── models.py          # Data models
│   └── utils.py           # Utility functions
├── infrastructure/         # CloudFormation templates
│   └── template.yaml
├── scripts/               # Deployment scripts
│   ├── deploy.sh
│   └── README.md
├── docs/                  # Documentation
│   └── INFRASTRUCTURE.md
├── examples/              # Example jobs
│   ├── job/
│   └── job_longform/
├── requirements.txt       # Python dependencies
└── another_automatic_video_editor.sh  # Main CLI
```

## Two Modes

Set `style.mode` in your `manifest.json`:

- `aftermovie` (default): short highlight-style video (e.g. 2–4 minutes)
- `longform`: **full-length** event video (e.g. 30–120+ minutes)

The longform mode is built **deterministically** (so it reliably includes your main videos), while Bedrock is used for **SEO/title/description**.

## Quick Start

### Prerequisites

- AWS CLI installed and configured
- Python 3.11+
- Valid AWS credentials with appropriate permissions

### 1) Create the AWS stack

```bash
./another_automatic_video_editor.sh create \
  --region eu-west-1 \
  --bedrock-region eu-west-1
```

Or use the deploy script:

```bash
./scripts/deploy.sh --region eu-west-1
```

### 2) Run an example job

#### Example A — Aftermovie (short)

```bash
./another_automatic_video_editor.sh run \
  --region eu-west-1 \
  --job-dir ./examples/job
```

#### Example B — Longform (full video, intro first)

Put your media into:

- `./examples/job_longform/media/`
  - `intro.mp4` (first segment)
  - your full talk recordings (e.g. `talk1.mp4`, `talk2.mp4`)
  - photos

Then run:

```bash
./another_automatic_video_editor.sh run \
  --region eu-west-1 \
  --job-dir ./examples/job_longform
```

### 3) Outputs

The `run` command already waits for completion and downloads results locally.

By default outputs go to `./another_automatic_video_editor_output/<JOB_ID>/`.

## Manifest Configuration

### Force an intro as the first segment

```json
"intro": {
  "file": "intro.mp4",
  "duration_seconds": 8,
  "caption": "AWS User Group Salerno"
}
```

### Enable "Best moments" (shot scoring)

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

### SEO-friendly rigid template (override AI)

```json
"seo": {
  "enabled": true,
  "cta": "Like + subscribe!",
  "hashtags": ["#AWS", "#Serverless", "#Meetup"]
}
```

## Outputs

The output bucket (and local download folder) contains:

- `output.mp4` — final video
- `chapters.txt` — YouTube chapters (timestamps + titles)
- `description.md` — SEO-friendly description (template + hashtags + links)
- `title.txt` — suggested YouTube title
- `catalog.json`, `plan.json`, `render_meta.json` — debug artifacts
- `best_moments_top.json` — top scored highlight clips (if enabled)

## Documentation

- [Infrastructure Guide](docs/INFRASTRUCTURE.md) - Complete infrastructure setup ([Italiano](docs/INFRASTRUCTURE.it.md))
- [Scripts Documentation](scripts/README.md) - Script usage ([Italiano](scripts/README.it.md))

## Clean up

```bash
./another_automatic_video_editor.sh delete --yes
```

Or:

```bash
./scripts/deploy.sh --cleanup --yes
```

## License

CC BY-NC 4.0 — see [LICENSE](LICENSE).
