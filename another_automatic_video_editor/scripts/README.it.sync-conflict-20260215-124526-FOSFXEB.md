# Scripts

Script di automazione per Another Automatic Video Editor.

> **Lingue disponibili**: [English](README.md) | [Italiano (corrente)](README.it.md)

## Script Principale

Lo script principale `another_automatic_video_editor.sh` nella root del progetto gestisce tutte le operazioni:

```bash
# Deploy dell'infrastruttura
./another_automatic_video_editor.sh create --region eu-west-1

# Verifica stato
./another_automatic_video_editor.sh status

# Esegui un job
./another_automatic_video_editor.sh run --job-dir ./examples/job

# Elimina tutte le risorse
./another_automatic_video_editor.sh delete --yes
```

### Supporto NixOS

Lo script rileva automaticamente se i tool richiesti (`aws`, `jq`, `python3`) non sono installati e, se in esecuzione su NixOS, entra in una nix-shell temporanea con i pacchetti necessari.

### Comandi

| Comando | Descrizione |
|---------|-------------|
| `create` | Deploy o aggiornamento dello stack CloudFormation |
| `status` | Mostra stato e output dello stack |
| `run` | Carica un job ed esegue il workflow di rendering |
| `delete` | Elimina tutte le risorse AWS |
| `help` | Mostra messaggio di aiuto |

### Opzioni

| Opzione | Descrizione | Default |
|---------|-------------|---------|
| `--region` | Regione AWS | `eu-west-1` |
| `--name` | Nome stack CloudFormation | auto-generato |
| `--bedrock-model-id` | ID modello Bedrock | `amazon.nova-lite-v1:0` |
| `--bedrock-region` | Override regione Bedrock | regione stack |
| `--cpu` | CPU Fargate (1024-16384) | `2048` |
| `--memory` | Memoria Fargate (MiB) | `4096` |
| `--ephemeral-gib` | Storage effimero (20-200 GiB) | `50` |
| `--enable-transcribe` | Abilita speech-to-text | `true` |
| `--job-dir` | Percorso cartella job (per `run`) | - |
| `--out-dir` | Cartella output locale | `./another_automatic_video_editor_output` |
| `--yes` | Salta conferme | - |
