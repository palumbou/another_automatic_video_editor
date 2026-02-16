#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fnmatch
import math
import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.config import Config as BotoConfig
from PIL import Image, ExifTags, ImageFilter, ImageStat


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif", ".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mts", ".m2ts"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


# -----------------------------
# Small helpers
# -----------------------------
def now_utc() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def sha1_short(s: str, n: int = 8) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def run_cmd(cmd: List[str], *, cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    logging.info("RUN: %s", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check, text=True, capture_output=False)


def run_cmd_capture(cmd: List[str], *, cwd: Optional[Path] = None) -> str:
    logging.info("RUN: %s", " ".join(shlex.quote(c) for c in cmd))
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, text=True, capture_output=True)
    return p.stdout.strip()


def which(cmd: str) -> Optional[str]:
    for p in os.environ.get("PATH", "").split(os.pathsep):
        cand = os.path.join(p, cmd)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_json_dump(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def format_ts(seconds: int) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def escape_drawtext(text: str) -> str:
    # ffmpeg drawtext escaping: escape ':' and '\' and "'"
    # We also replace newlines.
    t = text.replace("\\", "\\\\")
    t = t.replace(":", "\\:")
    t = t.replace("'", "\\'")
    t = t.replace("\n", " ")
    t = t.replace("\r", " ")
    return t


def detect_language_code(manifest_language: str) -> Tuple[bool, str]:
    '''
    Returns (use_identify_language, language_code_or_empty).

    If we can map to a specific Transcribe LanguageCode, return that.
    Otherwise, use IdentifyLanguage.
    '''
    lang = (manifest_language or "").strip().lower()
    if not lang:
        return True, ""

    # Common mappings
    mapping = {
        "it": "it-IT",
        "it-it": "it-IT",
        "italian": "it-IT",
        "en": "en-US",
        "en-us": "en-US",
        "english": "en-US",
    }
    if lang in mapping:
        return False, mapping[lang]

    # If already looks like xx-XX (Transcribe style), use it.
    if re.fullmatch(r"[a-z]{2}-[A-Z]{2}", manifest_language.strip()):
        return False, manifest_language.strip()

    return True, ""


# -----------------------------
# Media probing
# -----------------------------
@dataclasses.dataclass
class VideoInfo:
    duration_s: float
    width: int
    height: int
    has_audio: bool


def ffprobe_video(path: Path) -> VideoInfo:
    # Duration
    dur_str = run_cmd_capture([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    duration_s = float(dur_str) if dur_str else 0.0

    # Streams
    streams_json = run_cmd_capture([
        "ffprobe", "-v", "error",
        "-show_streams",
        "-of", "json",
        str(path),
    ])
    data = json.loads(streams_json)
    width = 0
    height = 0
    has_audio = False
    for st in data.get("streams", []):
        if st.get("codec_type") == "video" and width == 0 and height == 0:
            width = int(st.get("width") or 0)
            height = int(st.get("height") or 0)
        if st.get("codec_type") == "audio":
            has_audio = True
    return VideoInfo(duration_s=duration_s, width=width, height=height, has_audio=has_audio)


def extract_video_frame(video: Path, out_jpg: Path, at_s: float) -> None:
    # -ss before -i is faster but less accurate; fine for thumbnails.
    run_cmd([
        "ffmpeg", "-y",
        "-ss", str(max(0.0, at_s)),
        "-i", str(video),
        "-frames:v", "1",
        "-q:v", "2",
        str(out_jpg),
    ])


def image_exif_datetime(path: Path) -> Optional[str]:
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return None
            # Find DateTimeOriginal
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag in ("DateTimeOriginal", "DateTime"):
                    if isinstance(value, bytes):
                        try:
                            value = value.decode("utf-8", "ignore")
                        except Exception:
                            pass
                    if isinstance(value, str) and value.strip():
                        return value.strip()
    except Exception:
        return None
    return None


# -----------------------------
# AWS clients
# -----------------------------
def boto3_config() -> BotoConfig:
    # Retries help with transient Bedrock/Rekognition issues.
    return BotoConfig(
        retries={"max_attempts": 10, "mode": "standard"},
        connect_timeout=10,
        read_timeout=300,
    )


def rekognition_client(region: str):
    return boto3.client("rekognition", region_name=region, config=boto3_config())


def transcribe_client(region: str):
    return boto3.client("transcribe", region_name=region, config=boto3_config())


def bedrock_runtime_client(region: str):
    return boto3.client("bedrock-runtime", region_name=region, config=boto3_config())


def s3_client(region: str):
    return boto3.client("s3", region_name=region, config=boto3_config())


# -----------------------------
# Rekognition analysis
# -----------------------------
def rekognition_labels_for_image_bytes(reko, image_bytes: bytes, *, max_labels: int = 10, min_conf: float = 70.0) -> List[str]:
    resp = reko.detect_labels(
        Image={"Bytes": image_bytes},
        MaxLabels=max_labels,
        MinConfidence=min_conf,
    )
    labels = []
    for lab in resp.get("Labels", []):
        name = lab.get("Name")
        if name:
            labels.append(str(name))
    return labels


def analyze_image(reko, image_path: Path) -> Dict[str, Any]:
    b = image_path.read_bytes()
    labels = rekognition_labels_for_image_bytes(reko, b)
    exif_dt = image_exif_datetime(image_path)
    return {
        "labels": labels,
        "exif_datetime": exif_dt,
    }


def analyze_video(reko, video_path: Path, tmp_dir: Path, samples: int = 3) -> Dict[str, Any]:
    info = ffprobe_video(video_path)
    labels_count: Dict[str, int] = {}

    # Sample frames at 10%, 50%, 90% (or fewer if very short)
    dur = max(0.1, info.duration_s)
    sample_points = [dur * 0.1, dur * 0.5, dur * 0.9]
    sample_points = sample_points[:samples]

    ensure_dir(tmp_dir)

    for i, t in enumerate(sample_points):
        frame = tmp_dir / f"{video_path.stem}_frame{i}.jpg"
        extract_video_frame(video_path, frame, t)
        try:
            lbs = rekognition_labels_for_image_bytes(reko, frame.read_bytes())
            for lb in lbs:
                labels_count[lb] = labels_count.get(lb, 0) + 1
        except Exception as e:
            logging.warning("Rekognition failed on %s: %s", frame, e)

    top_labels = sorted(labels_count.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    return {
        "duration_s": info.duration_s,
        "width": info.width,
        "height": info.height,
        "has_audio": info.has_audio,
        "labels": [k for k, _ in top_labels],
    }


# -----------------------------
# Best moments (shot scoring)
# -----------------------------

def image_metrics_for_scoring(image_path: Path) -> Dict[str, float]:
    """Return lightweight quality metrics for scoring frames.

    - brightness: 0..255 (mean luma)
    - edge_var: variance of edges (proxy for sharpness)
    """
    try:
        with Image.open(image_path) as im:
            im_l = im.convert("L")
            brightness = float(ImageStat.Stat(im_l).mean[0])
            edges = im_l.filter(ImageFilter.FIND_EDGES)
            edge_var = float(ImageStat.Stat(edges).var[0])
            return {"brightness": brightness, "edge_var": edge_var}
    except Exception:
        return {"brightness": 0.0, "edge_var": 0.0}


def score_frame_for_best_moments(labels: List[str], metrics: Dict[str, float]) -> Tuple[float, List[str]]:
    """Score a frame based on Rekognition labels + simple quality metrics.

    Returns (score, reasons).
    """
    label_set = {str(l).lower() for l in (labels or [])}

    score = 0.0
    reasons: List[str] = []

    def has_any(keys: List[str]) -> bool:
        return any(k.lower() in label_set for k in keys)

    if has_any(["audience", "crowd", "group"]):
        score += 3.0
        reasons.append("audience")
    if has_any(["person", "human", "people", "speaker"]):
        score += 2.0
        reasons.append("people")
    if has_any(["stage", "presentation", "conference", "microphone"]):
        score += 1.5
        reasons.append("stage")

    # Sharpness proxy
    edge_var = float(metrics.get("edge_var", 0.0) or 0.0)
    sharp = min(edge_var / 700.0, 2.0)
    score += sharp
    if sharp > 0.8:
        reasons.append("sharp")

    # Prefer mid brightness
    brightness = float(metrics.get("brightness", 0.0) or 0.0)
    bright_score = max(0.0, 1.0 - abs(brightness - 120.0) / 120.0)
    score += bright_score
    if bright_score < 0.35:
        reasons.append("dark/bright")

    return score, reasons


def best_moments_for_video(
    reko,
    video_path: Path,
    tmp_dir: Path,
    *,
    samples_per_video: int = 8,
    clip_duration_s: float = 8.0,
    max_moments: int = 3,
    min_gap_s: float = 20.0,
) -> List[Dict[str, Any]]:
    """Extract 'best moments' candidates by sampling frames and scoring them.

    This is intentionally lightweight/cost-aware: it samples a small number of frames
    and uses Rekognition DetectLabels + simple image metrics.

    Returns a list of moments with start_s/duration_s/score and debug fields.
    """
    info = ffprobe_video(video_path)
    dur = float(info.duration_s or 0.0)

    clip_duration_s = max(2.0, float(clip_duration_s))
    if dur < clip_duration_s + 2.0:
        return []

    samples = max(3, min(int(samples_per_video), 20))
    ensure_dir(tmp_dir)

    # Avoid the very beginning/end (often shaky or fade)
    pad = min(3.0, dur / 10.0)
    usable = max(0.1, dur - 2 * pad)
    times = [pad + usable * (i + 1) / (samples + 1) for i in range(samples)]

    candidates: List[Dict[str, Any]] = []

    for i, t in enumerate(times):
        frame = tmp_dir / f"{video_path.stem}_bm_{i:03d}.jpg"
        try:
            extract_video_frame(video_path, frame, t)
        except Exception:
            continue

        try:
            labels = rekognition_labels_for_image_bytes(reko, frame.read_bytes(), max_labels=10, min_conf=70.0)
        except Exception:
            labels = []

        metrics = image_metrics_for_scoring(frame)
        score, reasons = score_frame_for_best_moments(labels, metrics)

        candidates.append({
            "at_s": float(t),
            "score": float(score),
            "reasons": reasons,
            "labels": labels[:10],
            "metrics": {
                "brightness": round(float(metrics.get("brightness", 0.0)), 1),
                "edge_var": round(float(metrics.get("edge_var", 0.0)), 1),
            },
        })

    candidates.sort(key=lambda x: (-x["score"], x["at_s"]))

    selected: List[Dict[str, Any]] = []
    for cand in candidates:
        if len(selected) >= max(0, int(max_moments)):
            break

        at_s = float(cand["at_s"])
        if any(abs(at_s - float(s["at_s"])) < float(min_gap_s) for s in selected):
            continue

        start = max(0.0, at_s - (clip_duration_s / 2.0))
        start = min(start, max(0.0, dur - clip_duration_s))

        out = dict(cand)
        out["start_s"] = round(float(start), 3)
        out["duration_s"] = float(clip_duration_s)
        out["score"] = round(float(out["score"]), 3)
        selected.append(out)

    selected.sort(key=lambda x: x["start_s"])
    return selected


# -----------------------------
# Transcribe
# -----------------------------
def transcribe_video_audio(
    s3,
    transcribe,
    *,
    input_bucket: str,
    job_id: str,
    video_path: Path,
    tmp_dir: Path,
    language: str,
    max_wait_s: int = 3600,
) -> Optional[str]:
    info = ffprobe_video(video_path)
    if not info.has_audio:
        return None

    ensure_dir(tmp_dir)

    audio_path = tmp_dir / f"{video_path.stem}_audio.mp3"
    # Reasonable Transcribe-friendly audio
    run_cmd([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-b:a", "64k",
        str(audio_path),
    ])

    key = f"jobs/{job_id}/tmp/transcribe/{audio_path.name}"
    logging.info("Uploading audio for Transcribe: s3://%s/%s", input_bucket, key)
    s3.upload_file(str(audio_path), input_bucket, key)

    use_identify, lang_code = detect_language_code(language)

    job_name = f"another-automatic-video-editor-{job_id}-{sha1_short(video_path.name, 10)}-{sha1_short(str(time.time()), 6)}"

    media_uri = f"s3://{input_bucket}/{key}"

    args: Dict[str, Any] = {
        "TranscriptionJobName": job_name,
        "Media": {"MediaFileUri": media_uri},
        # No OutputBucketName => service-managed bucket
    }

    if use_identify:
        args["IdentifyLanguage"] = True
    else:
        args["LanguageCode"] = lang_code

    logging.info("Starting Transcribe job: %s", job_name)
    transcribe.start_transcription_job(**args)

    start = time.time()
    while True:
        resp = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        job = resp.get("TranscriptionJob", {})
        status = job.get("TranscriptionJobStatus")
        if status in ("COMPLETED", "FAILED"):
            if status == "FAILED":
                reason = job.get("FailureReason")
                logging.warning("Transcribe FAILED for %s: %s", video_path.name, reason)
                try:
                    transcribe.delete_transcription_job(TranscriptionJobName=job_name)
                except Exception:
                    pass
                return None

            uri = job.get("Transcript", {}).get("TranscriptFileUri")
            if not uri:
                logging.warning("Transcribe completed but TranscriptFileUri missing.")
                return None

            logging.info("Downloading transcript JSON from: %s", uri)
            with urllib.request.urlopen(uri) as f:
                data = json.loads(f.read().decode("utf-8"))

            # Extract transcript text
            transcript = None
            try:
                transcript = data["results"]["transcripts"][0]["transcript"]
            except Exception:
                transcript = None

            try:
                transcribe.delete_transcription_job(TranscriptionJobName=job_name)
            except Exception:
                pass

            return transcript

        if time.time() - start > max_wait_s:
            logging.warning("Transcribe timed out for %s", video_path.name)
            try:
                transcribe.delete_transcription_job(TranscriptionJobName=job_name)
            except Exception:
                pass
            return None

        time.sleep(10)


# -----------------------------
# Bedrock (plan generation)
# -----------------------------
def bedrock_chat_text(
    brt,
    *,
    model_id: str,
    system: str,
    user: str,
    max_tokens: int = 2048,
    temperature: float = 0.2,
) -> str:
    # Prefer Converse API (model-agnostic). Fallback to InvokeModel for a few common families.
    if hasattr(brt, "converse"):
        resp = brt.converse(
            modelId=model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
        content = resp.get("output", {}).get("message", {}).get("content", [])
        parts = []
        for block in content:
            if "text" in block:
                parts.append(block["text"])
        return "\n".join(parts).strip()

    # Fallback: InvokeModel
    body: Dict[str, Any]
    accept = "application/json"
    content_type = "application/json"

    if model_id.startswith("anthropic."):
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{system}\n\n{user}"}
                    ],
                }
            ],
        }
        resp = brt.invoke_model(
            modelId=model_id,
            accept=accept,
            contentType=content_type,
            body=json.dumps(body).encode("utf-8"),
        )
        payload = json.loads(resp["body"].read().decode("utf-8"))
        parts = []
        for block in payload.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts).strip()

    if model_id.startswith("amazon.titan-text") or model_id.startswith("amazon.titan"):
        body = {
            "inputText": f"{system}\n\n{user}",
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
                "topP": 0.9,
            },
        }
        resp = brt.invoke_model(
            modelId=model_id,
            accept=accept,
            contentType=content_type,
            body=json.dumps(body).encode("utf-8"),
        )
        payload = json.loads(resp["body"].read().decode("utf-8"))
        results = payload.get("results", [])
        if results and isinstance(results[0], dict):
            return str(results[0].get("outputText", "")).strip()
        return json.dumps(payload)

    if "llama" in model_id or model_id.startswith("meta."):
        body = {
            "prompt": f"{system}\n\n{user}",
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        }
        resp = brt.invoke_model(
            modelId=model_id,
            accept=accept,
            contentType=content_type,
            body=json.dumps(body).encode("utf-8"),
        )
        payload = json.loads(resp["body"].read().decode("utf-8"))
        if "generation" in payload:
            return str(payload["generation"]).strip()
        outs = payload.get("outputs", [])
        if outs and isinstance(outs[0], dict) and "text" in outs[0]:
            return str(outs[0]["text"]).strip()
        return json.dumps(payload)

    raise RuntimeError("Bedrock Converse API not available, and model family not recognized for InvokeModel fallback.")


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    # Find first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# -----------------------------
# Rendering
# -----------------------------
@dataclasses.dataclass
class Segment:
    source_id: str
    type: str  # image|video
    duration: float
    chapter: int
    caption: str = ""
    in_seconds: float = 0.0  # for video only


def scale_pad_filter(target_w: int = 1920, target_h: int = 1080) -> str:
    # Keep aspect ratio, pad to target
    return (
        f"scale=w={target_w}:h={target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2"
    )


def make_image_clip(
    image_path: Path,
    out_path: Path,
    *,
    duration: float,
    caption: str,
    fade_s: float,
    fps: int,
    target_w: int,
    target_h: int,
    enable_captions: bool,
) -> None:
    dur = max(0.5, float(duration))
    fade = min(fade_s, dur / 3.0)

    # Ken Burns-ish zoompan
    zoom_expr = "min(zoom+0.0015,1.10)"
    zoompan = f"zoompan=z='{zoom_expr}':d={int(dur*fps)}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"

    vf_parts = [
        "format=yuv420p",
        scale_pad_filter(target_w, target_h),
        zoompan,
        f"fps={fps}",
        f"fade=t=in:st=0:d={fade}",
        f"fade=t=out:st={max(0.0, dur - fade)}:d={fade}",
    ]

    if enable_captions and caption.strip():
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        text = escape_drawtext(caption.strip())
        vf_parts.append(
            f"drawtext=fontfile={font}:text='{text}':x=(w-text_w)/2:y=h-(text_h*2.5):"
            f"fontsize=36:box=1:boxborderw=18:boxcolor=black@0.45:fontcolor=white"
        )

    vf = ",".join(vf_parts)

    # Add silent audio track for concat compatibility
    run_cmd([
        "ffmpeg", "-y",
        "-loop", "1",
        "-t", str(dur),
        "-i", str(image_path),
        "-f", "lavfi",
        "-t", str(dur),
        "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        str(out_path),
    ])


def make_video_clip(
    video_path: Path,
    out_path: Path,
    *,
    start_s: float,
    duration: float,
    caption: str,
    fade_s: float,
    fps: int,
    target_w: int,
    target_h: int,
    enable_captions: bool,
    normalize_audio: bool = True,
) -> None:
    dur = max(0.5, float(duration))
    fade = min(fade_s, dur / 3.0)

    vf_parts = [
        "format=yuv420p",
        scale_pad_filter(target_w, target_h),
        f"fps={fps}",
        f"fade=t=in:st=0:d={fade}",
        f"fade=t=out:st={max(0.0, dur - fade)}:d={fade}",
    ]

    if enable_captions and caption.strip():
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        text = escape_drawtext(caption.strip())
        vf_parts.append(
            f"drawtext=fontfile={font}:text='{text}':x=(w-text_w)/2:y=h-(text_h*2.5):"
            f"fontsize=36:box=1:boxborderw=18:boxcolor=black@0.45:fontcolor=white"
        )

    vf = ",".join(vf_parts)

    # Audio filter: fade + optional loudnorm normalization
    af_parts = [
        f"afade=t=in:st=0:d={fade}",
        f"afade=t=out:st={max(0.0, dur - fade)}:d={fade}",
    ]
    if normalize_audio:
        # loudnorm normalizes audio to -14 LUFS (YouTube standard)
        af_parts.append("loudnorm=I=-14:TP=-1:LRA=11")
    
    af = ",".join(af_parts)

    info = ffprobe_video(video_path)

    if info.has_audio:
        run_cmd([
            "ffmpeg", "-y",
            "-ss", str(max(0.0, start_s)),
            "-i", str(video_path),
            "-t", str(dur),
            "-vf", vf,
            "-af", af,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(out_path),
        ])
    else:
        run_cmd([
            "ffmpeg", "-y",
            "-ss", str(max(0.0, start_s)),
            "-i", str(video_path),
            "-t", str(dur),
            "-f", "lavfi",
            "-t", str(dur),
            "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            "-movflags", "+faststart",
            str(out_path),
        ])


def concat_clips(clips: List[Path], out_path: Path) -> None:
    list_file = out_path.parent / "concat_list.txt"
    lines = []
    for c in clips:
        # FFmpeg concat format requires: file 'path/to/file.mp4'
        # Escape single quotes in path by replacing ' with '\''
        escaped_path = str(c).replace("'", "'\\''")
        lines.append(f"file '{escaped_path}'")
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    run_cmd([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out_path),
    ])


def add_final_fade_out(
    input_video: Path,
    out_path: Path,
    *,
    fade_duration_s: float = 3.0,
) -> None:
    """Add fade to black at the end of the video."""
    info = ffprobe_video(input_video)
    fade_start = max(0, info.duration_s - fade_duration_s)
    
    vf = f"fade=t=out:st={fade_start}:d={fade_duration_s}:color=black"
    af = f"afade=t=out:st={fade_start}:d={fade_duration_s}"
    
    run_cmd([
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-vf", vf,
        "-af", af,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_path),
    ])


def mix_music(
    main_video: Path,
    music_audio: Path,
    out_path: Path,
    *,
    music_volume: float = 0.20,
    duck: bool = True,
    duck_amount: float = 0.15,
    loop: bool = True,
    video_duration_s: float = 0.0,
    fade_out_s: float = 3.0,
) -> None:
    """Mix background music with video audio.
    
    Args:
        music_volume: Base volume for music (0.0-1.0)
        duck: Enable ducking (lower music when speech detected)
        duck_amount: Volume multiplier during ducking (0.0-1.0, lower = more ducking)
        loop: Loop music if shorter than video
        fade_out_s: Fade out duration at end
    """
    # Get video duration if not provided
    if video_duration_s <= 0:
        info = ffprobe_video(main_video)
        video_duration_s = info.duration_s
    
    fade_start = max(0, video_duration_s - fade_out_s)
    
    # Music input handling (loop or not)
    if loop:
        music_input = f"[1:a]aloop=loop=-1:size=2e+09,atrim=0:{video_duration_s},volume={music_volume}[bgvol]"
        stream_loop_args = ["-stream_loop", "-1"]
    else:
        music_input = f"[1:a]volume={music_volume}[bgvol]"
        stream_loop_args = []
    
    if duck:
        # Sidechain compress: music ducks when video audio is loud
        # - threshold=0.015: trigger ducking at low audio levels (more sensitive)
        # - ratio=12: stronger compression ratio for more noticeable ducking
        # - attack=30: faster response to speech
        # - release=600: smooth return after speech
        # - level_sc controls how much the sidechain affects compression
        # duck_amount controls the makeup gain (lower = quieter during speech)
        fc = (
            f"{music_input};"
            f"[bgvol][0:a]sidechaincompress=threshold=0.015:ratio=12:attack=30:release=600:level_in=1:level_sc=1.5:makeup={duck_amount}[bgduck];"
            f"[0:a][bgduck]amix=inputs=2:duration=first:weights=1 0.7:dropout_transition=0[mixed];"
            f"[mixed]afade=t=out:st={fade_start}:d={fade_out_s}[aout]"
        )
    else:
        fc = (
            f"{music_input};"
            f"[0:a][bgvol]amix=inputs=2:duration=first:weights=1 0.6:dropout_transition=0[mixed];"
            f"[mixed]afade=t=out:st={fade_start}:d={fade_out_s}[aout]"
        )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(main_video),
    ] + stream_loop_args + [
        "-i", str(music_audio),
        "-filter_complex", fc,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-t", str(video_duration_s),
        "-movflags", "+faststart",
        str(out_path),
    ]
    
    run_cmd(cmd)


def generate_placeholder_music(out_path: Path, *, duration_s: float) -> None:
    dur = max(5.0, float(duration_s))
    run_cmd([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"sine=frequency=220:duration={dur}",
        "-f", "lavfi",
        "-i", f"sine=frequency=277:duration={dur}",
        "-f", "lavfi",
        "-i", f"sine=frequency=329:duration={dur}",
        "-f", "lavfi",
        "-i", f"anoisesrc=color=white:amplitude=0.02:duration={dur}",
        "-filter_complex",
        "[0:a][1:a][2:a]amix=inputs=3,volume=0.10[chord];"
        "[chord][3:a]amix=inputs=2,lowpass=f=1400,volume=0.35[aout]",
        "-map", "[aout]",
        "-c:a", "mp3",
        "-q:a", "6",
        str(out_path),
    ])


# -----------------------------
# Planning defaults/fallback
# -----------------------------
def default_manifest() -> Dict[str, Any]:
    return {
        "project": {
            "title": "My Event Video",
            "date": "",
            "location": "",
            "language": "it",
        },
        "style": {
            "mode": "aftermovie",
            "target_duration_seconds": 180,
            "fade_seconds": 0.5,
            "fps": 30,
            "resolution": {"w": 1920, "h": 1080},
            "captions": {"enabled": True},
            "music": {"enabled": True, "duck": True, "volume": 0.20},
        },
        "outline": [
            {"title": "Intro", "intent": "venue, setup, community"},
            {"title": "Highlights", "intent": "talks, speakers, audience"},
            {"title": "Networking", "intent": "people, swag, food"},
            {"title": "Closing", "intent": "call to action"},
        ],
        "best_moments": {
            "enabled": false
        },
        "ai": {
            "enabled": True,
        },
    }


def load_manifest(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_manifest()


def heuristic_plan(
    *,
    manifest: Dict[str, Any],
    catalog: Dict[str, Any],
) -> Dict[str, Any]:
    target = int(manifest.get("style", {}).get("target_duration_seconds", 180))
    outline = manifest.get("outline", []) or [{"title": "Video", "intent": ""}]
    chapter_titles = [str(x.get("title", f"Chapter {i+1}")) for i, x in enumerate(outline)]
    if not chapter_titles:
        chapter_titles = ["Video"]

    items = catalog.get("items", [])
    def sort_key(it: Dict[str, Any]):
        dt_str = it.get("exif_datetime") or ""
        return (0 if dt_str else 1, dt_str, it.get("filename", ""))
    items = sorted(items, key=sort_key)

    # Count media to calculate per-item duration
    n_images = sum(1 for it in items if it.get("type") == "image")
    n_videos = sum(1 for it in items if it.get("type") == "video")
    
    # Default durations
    default_img_dur = 4.0
    default_vid_dur = 8.0
    
    # Estimate total duration with defaults
    estimated_total = (n_images * default_img_dur) + (n_videos * default_vid_dur)
    
    # If estimated exceeds target, scale down durations (min 2s for images, 4s for videos)
    # If estimated is less than target, use defaults (don't stretch unnecessarily)
    scale = 1.0
    if estimated_total > target and estimated_total > 0:
        scale = target / estimated_total
    
    img_dur = max(2.0, default_img_dur * scale)
    vid_dur_factor = max(0.5, scale)  # Scale factor for video duration

    segments: List[Dict[str, Any]] = []
    elapsed = 0.0
    chapter = 0

    for it in items:
        typ = it["type"]
        src_id = it["id"]
        if typ == "image":
            dur = img_dur
            segments.append({
                "source_id": src_id,
                "type": "image",
                "duration": dur,
                "chapter": chapter,
                "caption": "",
            })
            elapsed += dur
        elif typ == "video":
            vdur = float(it.get("duration_s") or 0.0)
            dur = min(10.0, max(4.0, vdur * 0.15 * vid_dur_factor)) if vdur > 0 else default_vid_dur * vid_dur_factor
            dur = max(4.0, dur)  # Minimum 4s for videos
            start_s = max(0.0, (vdur / 2.0) - (dur / 2.0))
            segments.append({
                "source_id": src_id,
                "type": "video",
                "in_seconds": start_s,
                "duration": dur,
                "chapter": chapter,
                "caption": "",
            })
            elapsed += dur

        if len(chapter_titles) > 1:
            total_items = len(items)
            items_per_chapter = max(1, total_items // len(chapter_titles))
            chapter = min(len(chapter_titles) - 1, len(segments) // items_per_chapter)

    return {
        "video_title": manifest.get("project", {}).get("title", "Aftermovie"),
        "youtube_description": f"{manifest.get('project', {}).get('title', 'Aftermovie')}\n\nGenerated with Another Automatic Video Editor.\n",
        "chapters": [{"title": t} for t in chapter_titles],
        "segments": segments,
        "notes": "heuristic_fallback",
    }


def get_style_mode(manifest: Dict[str, Any]) -> str:
    style = manifest.get("style", {}) or {}
    mode = str(style.get("mode", "aftermovie") or "aftermovie").strip().lower()
    if mode in {"long", "longform", "full", "full_length", "full-length", "event"}:
        return "longform"
    return "aftermovie"


def resolve_source_id_by_filename(catalog: Dict[str, Any], filename_or_path: str) -> Optional[str]:
    if not filename_or_path:
        return None
    fname = Path(filename_or_path).name.lower()
    for it in catalog.get("items", []):
        if str(it.get("filename", "")).lower() == fname:
            return str(it.get("id"))
    return None


def collect_best_moments_global(catalog: Dict[str, Any], *, max_total: int = 12) -> List[Dict[str, Any]]:
    moments: List[Dict[str, Any]] = []
    for it in catalog.get("items", []):
        if it.get("type") != "video":
            continue
        vid = str(it.get("id"))
        for m in (it.get("best_moments") or []):
            try:
                moments.append({
                    "source_id": vid,
                    "type": "video",
                    "in_seconds": float(m.get("start_s", 0.0) or 0.0),
                    "duration": float(m.get("duration_s", 8.0) or 8.0),
                    "score": float(m.get("score", 0.0) or 0.0),
                    "caption": "",
                })
            except Exception:
                continue

    moments.sort(key=lambda x: (-x.get("score", 0.0), x.get("source_id", ""), x.get("in_seconds", 0.0)))
    return moments[: max(0, int(max_total))]


def match_items_by_patterns(catalog: Dict[str, Any], patterns: List[str], *, types: Optional[List[str]] = None) -> List[str]:
    pats = [p for p in (patterns or []) if str(p).strip()]
    if not pats:
        return []

    out: List[str] = []
    for it in catalog.get("items", []):
        if types and it.get("type") not in types:
            continue
        fn = str(it.get("filename", ""))
        fn_l = fn.lower()
        for pat in pats:
            pat_l = str(pat).lower()
            if fnmatch.fnmatch(fn_l, pat_l):
                out.append(str(it.get("id")))
                break
    return out


def longform_plan(
    *,
    manifest: Dict[str, Any],
    catalog: Dict[str, Any],
) -> Dict[str, Any]:
    """Deterministic longform plan (full-length video).

    The LLM is still used for titles/SEO, but the timeline is assembled
    deterministically so it reliably includes the main assets.
    """
    project = manifest.get("project", {}) or {}
    style = manifest.get("style", {}) or {}
    lf = style.get("longform", {}) or {}

    image_dur = float(lf.get("image_duration_seconds", 3.5) or 3.5)
    max_images_total = int(lf.get("max_images_total", 80) or 80)
    include_full_videos = bool(lf.get("include_full_videos", True))
    trim_head = float(lf.get("trim_head_seconds", 0.0) or 0.0)
    trim_tail = float(lf.get("trim_tail_seconds", 0.0) or 0.0)

    outline = manifest.get("outline", []) or []
    if not outline:
        # Fallback: one chapter per video
        outline = []
        for it in catalog.get("items", []):
            if it.get("type") == "video":
                outline.append({"title": it.get("filename", "Video"), "intent": ""})
        if not outline:
            outline = [{"title": "Video", "intent": ""}]

    # Intro config
    intro_conf = manifest.get("intro", {}) or {}
    intro_file = str(intro_conf.get("file") or intro_conf.get("asset") or "").strip()
    intro_id = resolve_source_id_by_filename(catalog, intro_file) if intro_file else None
    intro_duration = float(intro_conf.get("duration_seconds", 0.0) or 0.0)
    intro_caption = str(intro_conf.get("caption", "") or "").strip()

    # Best moments config
    best_conf = manifest.get("best_moments", {}) or {}
    best_enabled = bool(best_conf.get("enabled", False))
    insert_best = bool(best_conf.get("insert_in_timeline", True))
    max_best_total = int(best_conf.get("max_clips_total", 12) or 12)

    # Identify a chapter that should host highlights (if any)
    highlight_idx: Optional[int] = None
    if best_enabled and insert_best:
        for i, ch in enumerate(outline):
            t = str(ch.get("title", "")).lower()
            if any(k in t for k in ["highlight", "best", "momenti", "highlights"]):
                highlight_idx = i
                break

    chapters: List[Dict[str, Any]] = []
    for ch in outline:
        chapters.append({"title": str(ch.get("title", "Chapter")).strip() or "Chapter", "intent": str(ch.get("intent", "") or "").strip(), "include": ch.get("include") or ch.get("includes") or []})

    # If best moments enabled and no explicit highlights chapter, insert one after the first chapter
    if best_enabled and insert_best and highlight_idx is None:
        insert_at = 1 if len(chapters) >= 1 else 0
        chapters.insert(insert_at, {"title": "Best moments", "intent": "auto highlights", "include": [], "_special": "best_moments"})
        highlight_idx = insert_at

    # Pre-select images (cap total)
    images = [it for it in catalog.get("items", []) if it.get("type") == "image"]

    def sort_key_img(it: Dict[str, Any]):
        dt_str = it.get("exif_datetime") or ""
        return (0 if dt_str else 1, dt_str, it.get("filename", ""))

    images = sorted(images, key=sort_key_img)
    if len(images) > max_images_total > 0:
        # even sampling
        step = max(1, len(images) // max_images_total)
        images = images[::step][:max_images_total]

    videos = [it for it in catalog.get("items", []) if it.get("type") == "video"]
    videos = sorted(videos, key=lambda it: str(it.get("filename", "")).lower())

    used: set[str] = set()
    segments: List[Dict[str, Any]] = []

    # Helper to add a segment with per-chapter caption on the first segment
    def add_seg(seg: Dict[str, Any], *, chapter_idx: int) -> None:
        if seg.get("caption") is None:
            seg["caption"] = ""
        seg["chapter"] = int(chapter_idx)
        segments.append(seg)

    # Insert intro as the very first segment (chapter 0)
    if intro_id:
        intro_item = next((it for it in catalog.get("items", []) if it.get("id") == intro_id), None)
        if intro_item:
            if intro_item.get("type") == "video":
                vdur = float(intro_item.get("duration_s") or 0.0)
                dur = intro_duration if intro_duration > 0 else min(10.0, vdur if vdur > 0 else 10.0)
                add_seg({
                    "source_id": intro_id,
                    "type": "video",
                    "in_seconds": 0.0,
                    "duration": float(dur),
                    "caption": intro_caption or chapters[0]["title"],
                }, chapter_idx=0)
            else:
                dur = intro_duration if intro_duration > 0 else 5.0
                add_seg({
                    "source_id": intro_id,
                    "type": "image",
                    "duration": float(dur),
                    "caption": intro_caption or chapters[0]["title"],
                }, chapter_idx=0)
            used.add(intro_id)

    # Prepare global best moments if needed
    global_bm: List[Dict[str, Any]] = []
    if best_enabled and insert_best and highlight_idx is not None:
        global_bm = collect_best_moments_global(catalog, max_total=max_best_total)

    img_ptr = 0
    vid_ptr = 0

    for c_idx, ch in enumerate(chapters):
        title = str(ch.get("title", "Chapter")).strip() or "Chapter"
        include_patterns = ch.get("include") or []
        special = ch.get("_special")

        # If this is the highlight chapter, fill from best moments
        if special == "best_moments" or (highlight_idx == c_idx and global_bm):
            first = True
            for bm in global_bm:
                seg = {
                    "source_id": bm["source_id"],
                    "type": "video",
                    "in_seconds": float(bm.get("in_seconds", 0.0) or 0.0),
                    "duration": float(bm.get("duration", 8.0) or 8.0),
                    "caption": title if first else "",
                }
                first = False
                add_seg(seg, chapter_idx=c_idx)
            continue

        # Explicit includes (filename globs)
        included_ids: List[str] = []
        if include_patterns:
            included_ids = match_items_by_patterns(catalog, list(include_patterns))

        # If no includes, auto-pick based on title
        if not included_ids:
            tl = title.lower()
            wants_images = any(k in tl for k in ["foto", "photo", "gallery", "galleria", "networking", "backstage"])
            if wants_images and images:
                # pick a slice of images
                take = min(20, max(5, len(images) // max(1, len(chapters))))
                picked = images[img_ptr: img_ptr + take]
                img_ptr = (img_ptr + take) % max(1, len(images))
                included_ids = [str(it.get("id")) for it in picked]
            else:
                # pick a video
                if vid_ptr < len(videos):
                    included_ids = [str(videos[vid_ptr].get("id"))]
                    vid_ptr += 1
                else:
                    # fallback: some images
                    take = min(10, len(images) - img_ptr) if images else 0
                    picked = images[img_ptr: img_ptr + take]
                    img_ptr += take
                    included_ids = [str(it.get("id")) for it in picked]

        # Build segments for included assets
        first = True
        for sid in included_ids:
            item = next((it for it in catalog.get("items", []) if str(it.get("id")) == str(sid)), None)
            if not item:
                continue
            if item.get("type") == "image":
                add_seg({
                    "source_id": str(sid),
                    "type": "image",
                    "duration": float(image_dur),
                    "caption": title if first else "",
                }, chapter_idx=c_idx)
                first = False
                used.add(str(sid))
            else:
                vdur = float(item.get("duration_s") or 0.0)
                if include_full_videos and vdur > 0:
                    start_s = max(0.0, trim_head)
                    dur = max(0.0, vdur - trim_head - trim_tail)
                    if dur <= 0.5:
                        continue
                    add_seg({
                        "source_id": str(sid),
                        "type": "video",
                        "in_seconds": float(start_s),
                        "duration": float(dur),
                        "caption": title if first else "",
                    }, chapter_idx=c_idx)
                else:
                    # fallback: short excerpt
                    dur = min(30.0, max(8.0, vdur * 0.10)) if vdur > 0 else 12.0
                    start_s = max(0.0, (vdur / 2.0) - (dur / 2.0))
                    add_seg({
                        "source_id": str(sid),
                        "type": "video",
                        "in_seconds": float(start_s),
                        "duration": float(dur),
                        "caption": title if first else "",
                    }, chapter_idx=c_idx)
                first = False
                used.add(str(sid))

    # Ensure we included all videos at least once (append missing as extra chapters)
    all_video_ids = [str(it.get("id")) for it in videos]
    missing_videos = [vid for vid in all_video_ids if vid not in used]
    if missing_videos:
        base_idx = len(chapters)
        for j, vid in enumerate(missing_videos):
            item = next((it for it in catalog.get("items", []) if str(it.get("id")) == vid), None)
            if not item:
                continue
            chapters.append({"title": f"Extra: {item.get('filename','Video')}", "intent": "", "include": []})
            vdur = float(item.get("duration_s") or 0.0)
            dur = vdur if vdur > 0 else 60.0
            add_seg({
                "source_id": vid,
                "type": "video",
                "in_seconds": float(trim_head),
                "duration": float(max(1.0, dur - trim_head - trim_tail)),
                "caption": chapters[base_idx + j]["title"],
            }, chapter_idx=base_idx + j)

    return {
        "video_title": project.get("title") or "Video",
        "youtube_description": "",
        "chapters": [{"title": ch["title"]} for ch in chapters],
        "segments": segments,
        "notes": "longform_deterministic",
    }


def build_seo_fallback(manifest: Dict[str, Any]) -> Dict[str, Any]:
    yt = (manifest.get("project", {}) or {}).get("youtube", {}) or {}
    tags = yt.get("tags") or []
    hashtags = []
    for t in tags:
        t = str(t).strip()
        if not t:
            continue
        ht = "#" + re.sub(r"[^A-Za-z0-9_]+", "", t.replace(" ", ""))
        if len(ht) > 1:
            hashtags.append(ht)
    hashtags = hashtags[:12]

    return {
        "hook": "",
        "summary_bullets": [],
        "hashtags": hashtags,
        "cta": "Iscriviti al canale e attiva la campanella per non perdere i prossimi eventi!",
    }


def bedrock_generate_seo(
    *,
    brt,
    model_id: str,
    manifest: Dict[str, Any],
    catalog: Dict[str, Any],
    chapters: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Generate SEO-friendly metadata (hook, bullets, hashtags, CTA).

    Returns JSON or None.
    """
    project = manifest.get("project", {}) or {}
    yt = project.get("youtube", {}) or {}

    # Lightweight media summary for the model
    videos = []
    images = []
    for it in catalog.get("items", []):
        if it.get("type") == "video":
            videos.append({
                "filename": it.get("filename"),
                "labels": (it.get("labels") or [])[:8],
                "duration_s": round(float(it.get("duration_s") or 0.0), 1),
                "transcript_excerpt": (it.get("transcript_excerpt") or "")[:400],
            })
        elif it.get("type") == "image":
            images.append({
                "filename": it.get("filename"),
                "labels": (it.get("labels") or [])[:8],
            })

    user_obj = {
        "task": "Create SEO-friendly YouTube metadata in Italian.",
        "project": project,
        "youtube": {
            "channel_name": yt.get("channel_name"),
            "tags": yt.get("tags", []),
            "links": yt.get("links", {}),
        },
        "chapters": [{"title": c.get("title"), "start": format_ts(int(c.get("start_seconds", 0)))} for c in chapters],
        "media_summary": {
            "videos": videos[:12],
            "images": images[:12],
            "counts": catalog.get("counts", {}),
        },
        "required_output_schema": {
            "video_title": "string",
            "hook": "string (1-2 sentences)",
            "summary_bullets": ["string"],
            "cta": "string",
            "hashtags": ["string starting with #"],
        },
    }

    system = (
        "You are a YouTube copywriter. Return ONLY valid JSON (no markdown). "
        "Follow the required_output_schema exactly."
    )

    prompt = json.dumps(user_obj, ensure_ascii=False)

    try:
        text = bedrock_chat_text(
            brt,
            model_id=model_id,
            system=system,
            user=prompt,
            max_tokens=1024,
            temperature=0.4,
        )
    except Exception as e:
        logging.warning("Bedrock SEO call failed: %s", e)
        return None

    obj = extract_json_from_text(text)
    if not isinstance(obj, dict):
        return None

    # Minimal validation
    if not isinstance(obj.get("hashtags", []), list):
        obj["hashtags"] = []
    if not isinstance(obj.get("summary_bullets", []), list):
        obj["summary_bullets"] = []

    return obj


def build_youtube_description(
    *,
    manifest: Dict[str, Any],
    plan: Dict[str, Any],
    chapters_txt: str,
    seo: Dict[str, Any],
    music_info: Dict[str, Any],
) -> str:
    project = manifest.get("project", {}) or {}
    yt = project.get("youtube", {}) or {}
    links = (yt.get("links") or {})

    title = str(seo.get("video_title") or plan.get("video_title") or project.get("title") or "Video").strip()
    hook = str(seo.get("hook") or "").strip()
    bullets = [str(x).strip() for x in (seo.get("summary_bullets") or []) if str(x).strip()]
    cta = str(seo.get("cta") or "").strip()
    hashtags = [str(x).strip() for x in (seo.get("hashtags") or []) if str(x).strip()]

    date = str(project.get("date") or "").strip()
    location = str(project.get("location") or "").strip()
    channel_name = str(yt.get("channel_name") or "").strip()

    # Music credits
    music_line = ""
    if music_info.get("music_path"):
        name = Path(str(music_info.get("music_path"))).name
        music_line = f"🎵 Musica: {name}"
        if music_info.get("music_generated"):
            music_line += " (placeholder generata automaticamente)"

    # Links section (only include non-empty)
    link_lines = []
    for k, v in links.items():
        if v:
            link_lines.append(f"- {k}: {v}")

    # Bullet fallback
    if not bullets:
        bullets = []
        if project.get("title"):
            bullets.append(f"Highlights e contenuti dall'evento: {project.get('title')}")
        if location:
            bullets.append(f"Community e networking a {location}")
        bullets.append("Talk, demo e momenti chiave (capitoli inclusi)")

    # Hashtags fallback
    if not hashtags:
        hashtags = build_seo_fallback(manifest).get("hashtags", [])

    parts: List[str] = []
    if hook:
        parts.append(hook)

    header = f"🎬 {title}"
    if location or date:
        header += " — " + " | ".join([x for x in [location, date] if x])
    parts.append(header)

    if channel_name:
        parts.append(f"Canale: {channel_name}")

    parts.append("\n✅ In questo video:")
    parts.extend([f"- {b}" for b in bullets[:8]])

    if cta:
        parts.append("\n👉 " + cta)

    if link_lines:
        parts.append("\n🔗 Link utili:")
        parts.extend(link_lines)

    if music_line:
        parts.append("\n" + music_line)

    parts.append("\n🕒 Chapters:")
    parts.append(chapters_txt.strip())

    if hashtags:
        parts.append("\n" + " ".join(hashtags[:15]))

    return "\n".join(parts).rstrip() + "\n"


# -----------------------------
# Main pipeline
# -----------------------------
def build_catalog(
    *,
    reko_region: str,
    enable_transcribe: bool,
    input_bucket: str,
    job_id: str,
    job_dir: Path,
    tmp_dir: Path,
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    reko = rekognition_client(reko_region)
    s3 = s3_client(reko_region)
    tr = transcribe_client(reko_region)

    best_conf = manifest.get("best_moments", {}) or {}
    best_enabled = bool(best_conf.get("enabled", False))
    bm_samples = int(best_conf.get("samples_per_video", 8) or 8)
    bm_clip_dur = float(best_conf.get("clip_duration_seconds", 8.0) or 8.0)
    bm_max_per_video = int(best_conf.get("max_moments_per_video", 3) or 3)
    bm_min_gap = float(best_conf.get("min_gap_seconds", 20.0) or 20.0)

    items: List[Dict[str, Any]] = []

    media_dir = job_dir / "media"
    if not media_dir.exists():
        raise RuntimeError(f"Missing media directory: {media_dir}")

    normalized_dir = tmp_dir / "normalized"
    images_norm = normalized_dir / "images"
    ensure_dir(images_norm)

    frames_dir = tmp_dir / "frames"
    ensure_dir(frames_dir)

    magick = which("magick") or which("convert")
    if not magick:
        raise RuntimeError("ImageMagick not found (magick/convert).")

    # Collect media files
    media_files: List[Path] = []
    for pth in media_dir.rglob("*"):
        if not pth.is_file():
            continue
        ext = pth.suffix.lower()
        if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
            media_files.append(pth)

    # Stable order: by filename
    media_files.sort(key=lambda p: p.name.lower())

    # --------------------
    # Normalize + analyze images
    # --------------------
    img_count = 0
    for pth in media_files:
        if pth.suffix.lower() not in IMAGE_EXTS:
            continue
        img_count += 1
        out_jpg = images_norm / f"img_{img_count:04d}.jpg"

        # Keep EXIF datetime from the original *before* stripping metadata
        exif_dt = image_exif_datetime(pth)

        cmd = [
            magick,
            str(pth),
            "-auto-orient",
            "-strip",
            "-colorspace", "sRGB",
            "-resize", "1920x1920>",
            "-quality", "92",
            str(out_jpg),
        ]

        analysis_path = out_jpg
        try:
            run_cmd(cmd)
        except Exception as e:
            logging.warning("Image conversion failed for %s: %s (using original)", pth, e)
            analysis_path = pth

        try:
            labels = rekognition_labels_for_image_bytes(reko, Path(analysis_path).read_bytes(), max_labels=15, min_conf=70.0)
        except Exception as e:
            logging.warning("Rekognition image analysis failed for %s: %s", analysis_path, e)
            labels = []

        items.append({
            "id": f"img_{img_count:04d}",
            "type": "image",
            "filename": pth.name,
            "local_path": str(analysis_path),
            "size_bytes": Path(analysis_path).stat().st_size,
            "labels": labels,
            "exif_datetime": exif_dt,
        })

    # --------------------
    # Analyze videos (+ optional Transcribe + best moments)
    # --------------------
    v_count = 0
    lang = str(manifest.get("project", {}).get("language", ""))

    for pth in media_files:
        if pth.suffix.lower() not in VIDEO_EXTS:
            continue
        v_count += 1
        vid_id = f"vid_{v_count:04d}"

        try:
            v_analysis = analyze_video(reko, pth, tmp_dir=frames_dir, samples=3)
        except Exception as e:
            logging.warning("Video analysis failed for %s: %s", pth, e)
            v_analysis = {"duration_s": 0.0, "width": 0, "height": 0, "has_audio": False, "labels": []}

        transcript_excerpt = None
        if enable_transcribe and float(v_analysis.get("duration_s") or 0.0) >= 20.0:
            try:
                tr_text = transcribe_video_audio(
                    s3, tr,
                    input_bucket=input_bucket,
                    job_id=job_id,
                    video_path=pth,
                    tmp_dir=tmp_dir / "transcribe",
                    language=lang,
                )
                if tr_text:
                    transcript_excerpt = tr_text[:1200]
            except Exception as e:
                logging.warning("Transcribe failed for %s: %s", pth, e)

        best_moments = []
        if best_enabled and float(v_analysis.get("duration_s") or 0.0) >= bm_clip_dur + 2.0:
            try:
                best_moments = best_moments_for_video(
                    reko,
                    pth,
                    tmp_dir=tmp_dir / "best_moments" / vid_id,
                    samples_per_video=bm_samples,
                    clip_duration_s=bm_clip_dur,
                    max_moments=bm_max_per_video,
                    min_gap_s=bm_min_gap,
                )
            except Exception as e:
                logging.warning("Best moments scoring failed for %s: %s", pth, e)
                best_moments = []

        items.append({
            "id": vid_id,
            "type": "video",
            "filename": pth.name,
            "local_path": str(pth),
            "size_bytes": pth.stat().st_size,
            "duration_s": v_analysis.get("duration_s", 0.0),
            "width": v_analysis.get("width", 0),
            "height": v_analysis.get("height", 0),
            "has_audio": v_analysis.get("has_audio", False),
            "labels": v_analysis.get("labels", []),
            "transcript_excerpt": transcript_excerpt,
            "best_moments": best_moments,
        })

    return {
        "generated_at": now_utc(),
        "items": items,
        "counts": {
            "images": img_count,
            "videos": v_count,
        },
        "best_moments_enabled": best_enabled,
    }
def bedrock_plan(
    *,
    brt,
    model_id: str,
    manifest: Dict[str, Any],
    catalog: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    outline = manifest.get("outline", [])
    project = manifest.get("project", {})
    style = manifest.get("style", {})
    target = int(style.get("target_duration_seconds", 180))

    items = catalog.get("items", [])
    item_summaries = []
    for it in items:
        if it.get("type") == "image":
            item_summaries.append({
                "id": it["id"],
                "type": "image",
                "filename": it.get("filename"),
                "labels": it.get("labels", [])[:7],
                "exif_datetime": it.get("exif_datetime"),
            })
        else:
            item_summaries.append({
                "id": it["id"],
                "type": "video",
                "filename": it.get("filename"),
                "duration_s": round(float(it.get("duration_s") or 0.0), 2),
                "labels": it.get("labels", [])[:7],
                "transcript_excerpt": (it.get("transcript_excerpt") or "")[:300],
                "exif_datetime": it.get("exif_datetime"),
            })

    # Get AI prompt configuration from manifest (or use defaults)
    ai_conf = manifest.get("ai", {}) or {}
    
    # System prompt (configurable)
    default_system = (
        "You are an expert video editor assistant. "
        "Return ONLY valid JSON (no markdown, no commentary). "
        "Do not include extra keys. "
        "CRITICAL: You MUST include ALL media items from the catalog in your segments. Do not skip any. "
        "CRITICAL: Order segments CHRONOLOGICALLY based on exif_datetime or filename. "
        "Mix images and videos naturally based on their timestamps - do NOT group all images together."
    )
    system = str(ai_conf.get("system_prompt", default_system) or default_system)

    # Calculate realistic duration based on media count
    n_images = sum(1 for it in items if it.get("type") == "image")
    n_videos = sum(1 for it in items if it.get("type") == "video")
    total_video_dur = sum(float(it.get("duration_s") or 0) for it in items if it.get("type") == "video")
    
    # Estimate: 4s per image, 10s per video clip (or full if longform)
    estimated_duration = (n_images * 4) + min(total_video_dur, n_videos * 10)
    
    # Use the larger of target or estimated
    effective_target = max(target, int(estimated_duration * 0.8))

    # Default task prompt (configurable)
    default_task = "Create a storyboard/timeline for an event video / slideshow."
    task_prompt = str(ai_conf.get("task_prompt", default_task) or default_task)
    
    # Additional instructions from manifest
    extra_instructions = str(ai_conf.get("extra_instructions", "") or "")

    user_obj = {
        "task": task_prompt,
        "extra_instructions": extra_instructions if extra_instructions else None,
        "critical_requirements": {
            "MUST_USE_ALL_MEDIA": True,
            "CHRONOLOGICAL_ORDER": "Order segments by exif_datetime or filename timestamp. Mix images and videos naturally.",
            "total_media_items": len(items),
            "images_count": n_images,
            "videos_count": n_videos,
            "minimum_segments_required": len(items),
        },
        "constraints": {
            "target_total_duration_seconds": effective_target,
            "video_format": "YouTube-ready 1080p 30fps",
            "chapters_requirements": "At least 3 chapters, first starts at 0 seconds.",
            "no_duplicate_segments": True,
            "segment_duration_limits": {
                "image_seconds": [3, 5],
                "video_seconds": [5, 12],
            },
        },
        "project": project,
        "outline": outline,
        "media_catalog": item_summaries,
        "required_output_schema": {
            "video_title": "string",
            "youtube_description": "string",
            "chapters": [{"title": "string"}],
            "segments": [
                {
                    "source_id": "string (must match a media id from catalog)",
                    "type": "image|video",
                    "duration": "number (seconds)",
                    "chapter": "integer (index into chapters)",
                    "caption": "string (short, can be empty)",
                    "in_seconds": "number (seconds, only for video; for images omit or set 0)"
                }
            ]
        }
    }
    
    # Remove None values
    user_obj = {k: v for k, v in user_obj.items() if v is not None}

    prompt = json.dumps(user_obj, ensure_ascii=False)

    # Increase max_tokens for large catalogs
    max_tokens = min(16000, max(4096, len(items) * 100))

    try:
        text = bedrock_chat_text(
            brt,
            model_id=model_id,
            system=system,
            user=prompt,
            max_tokens=max_tokens,
            temperature=0.2,
        )
    except Exception as e:
        logging.warning("Bedrock call failed: %s", e)
        return None

    plan = extract_json_from_text(text)
    if not plan or not isinstance(plan, dict):
        logging.warning("Bedrock returned non-JSON or unparsable output.")
        logging.debug("Bedrock raw output: %s", text)
        return None

    if "segments" not in plan or "chapters" not in plan:
        return None
    if not isinstance(plan.get("segments"), list) or not isinstance(plan.get("chapters"), list):
        return None
    
    # Validate that AI used most of the media (at least 70%)
    used_ids = set(s.get("source_id") for s in plan.get("segments", []))
    all_ids = set(it.get("id") for it in items)
    coverage = len(used_ids & all_ids) / max(1, len(all_ids))
    
    if coverage < 0.7:
        logging.warning("AI plan only used %.0f%% of media (%d/%d). Will use fallback.", 
                       coverage * 100, len(used_ids & all_ids), len(all_ids))
        return None
    
    logging.info("AI plan uses %.0f%% of media (%d/%d segments)", 
                coverage * 100, len(used_ids & all_ids), len(all_ids))
    
    return plan


def pick_music(job_dir: Path, tmp_dir: Path, total_duration_s: float) -> Tuple[Path, bool]:
    '''
    Returns (music_path, is_generated).
    '''
    music_dir = job_dir / "music"
    if music_dir.exists():
        for p in sorted(music_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                return p, False

    gen = tmp_dir / "generated_music.mp3"
    generate_placeholder_music(gen, duration_s=total_duration_s)
    return gen, True


def render_from_plan(
    *,
    manifest: Dict[str, Any],
    catalog: Dict[str, Any],
    plan: Dict[str, Any],
    job_dir: Path,
    out_dir: Path,
    tmp_dir: Path,
) -> Dict[str, Any]:
    style = manifest.get("style", {})
    fade_s = float(style.get("fade_seconds", 0.5))
    fps = int(style.get("fps", 30))
    res = style.get("resolution", {}) or {}
    target_w = int(res.get("w", 1920))
    target_h = int(res.get("h", 1080))
    captions_enabled = bool(style.get("captions", {}).get("enabled", True))
    normalize_audio = bool(style.get("normalize_audio", True))

    clips_dir = tmp_dir / "clips"
    ensure_dir(clips_dir)

    items_by_id = {it["id"]: it for it in catalog.get("items", [])}

    segments_raw = plan.get("segments", [])
    segments: List[Segment] = []
    for seg in segments_raw:
        try:
            source_id = str(seg["source_id"])
            typ = str(seg["type"])
            dur = float(seg["duration"])
            chap = int(seg.get("chapter", 0))
            caption = str(seg.get("caption", "") or "")
            in_s = float(seg.get("in_seconds", 0.0) or 0.0)
        except Exception:
            continue
        if source_id not in items_by_id:
            continue
        if typ not in ("image", "video"):
            continue
        if dur <= 0:
            continue
        segments.append(Segment(source_id=source_id, type=typ, duration=dur, chapter=chap, caption=caption, in_seconds=in_s))

    if not segments:
        raise RuntimeError("Plan had zero valid segments.")

    clips: List[Path] = []
    for idx, seg in enumerate(segments, start=1):
        item = items_by_id[seg.source_id]
        src = Path(item["local_path"])
        clip = clips_dir / f"clip_{idx:04d}.mp4"

        if seg.type == "image":
            make_image_clip(
                src, clip,
                duration=seg.duration,
                caption=seg.caption,
                fade_s=fade_s,
                fps=fps,
                target_w=target_w,
                target_h=target_h,
                enable_captions=captions_enabled,
            )
        else:
            make_video_clip(
                src, clip,
                start_s=seg.in_seconds,
                duration=seg.duration,
                caption=seg.caption,
                fade_s=fade_s,
                fps=fps,
                target_w=target_w,
                target_h=target_h,
                enable_captions=captions_enabled,
                normalize_audio=normalize_audio,
            )

        clips.append(clip)

    # Generate output filename prefix from project title
    project_title = manifest.get("project", {}).get("title", "video") or "video"
    # Sanitize for filename: replace spaces and special chars
    safe_title = re.sub(r'[^\w\-]', '_', project_title)
    safe_title = re.sub(r'_+', '_', safe_title).strip('_')[:50]  # Limit length
    
    no_music = out_dir / f"{safe_title}_no_music.mp4"
    concat_clips(clips, no_music)

    music_conf = style.get("music", {}) or {}
    music_enabled = bool(music_conf.get("enabled", True))
    music_duck = bool(music_conf.get("duck", True))
    music_duck_amount = float(music_conf.get("duck_amount", 0.15) or 0.15)
    music_loop = bool(music_conf.get("loop", True))
    music_vol = float(music_conf.get("volume", 0.20))
    
    # Fade out configuration
    fade_out_s = float(style.get("fade_out_seconds", 3.0) or 3.0)

    final = out_dir / f"{safe_title}_final.mp4"
    total_dur = sum(s.duration for s in segments)

    if music_enabled:
        music_path, generated = pick_music(job_dir, tmp_dir, total_dur)
        # Mix music with loop, ducking, and fade out
        mix_music(
            no_music, music_path, final,
            music_volume=music_vol,
            duck=music_duck,
            duck_amount=music_duck_amount,
            loop=music_loop,
            video_duration_s=total_dur,
            fade_out_s=fade_out_s,
        )
        return {
            "final_video": str(final),
            "no_music_video": str(no_music),
            "music_path": str(music_path),
            "music_generated": generated,
            "segments_count": len(segments),
            "total_duration_s": total_dur,
        }

    # No music - just add fade out
    final_with_fade = out_dir / f"{safe_title}_final.mp4"
    add_final_fade_out(no_music, final_with_fade, fade_duration_s=fade_out_s)
    return {
        "final_video": str(final_with_fade),
        "no_music_video": str(no_music),
        "music_path": None,
        "music_generated": False,
        "segments_count": len(segments),
        "total_duration_s": total_dur,
    }


def compute_chapters_from_segments(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    chapters = plan.get("chapters", [])
    segs = plan.get("segments", [])

    chapter_titles = []
    for ch in chapters:
        if isinstance(ch, dict) and "title" in ch:
            chapter_titles.append(str(ch["title"]))
        elif isinstance(ch, str):
            chapter_titles.append(ch)
    if not chapter_titles:
        chapter_titles = ["Video"]

    starts = {0: 0.0}
    t = 0.0
    for seg in segs:
        try:
            chap = int(seg.get("chapter", 0))
            dur = float(seg.get("duration", 0.0))
        except Exception:
            continue
        if chap not in starts:
            starts[chap] = t
        t += max(0.0, dur)

    out = []
    for idx, title in enumerate(chapter_titles):
        start = int(round(starts.get(idx, 0.0)))
        out.append({"title": title, "start_seconds": start})

    out.sort(key=lambda x: x["start_seconds"])
    if out and out[0]["start_seconds"] != 0:
        out[0]["start_seconds"] = 0
    return out


def write_chapters_txt(chapters: List[Dict[str, Any]], path: Path) -> None:
    lines = []
    for ch in chapters:
        ts = format_ts(int(ch["start_seconds"]))
        title = str(ch["title"]).strip() or "Chapter"
        lines.append(f"{ts} {title}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_description_has_chapters(desc: str, chapters_txt: str) -> str:
    if re.search(r"(^|\n)\d{1,2}:\d{2}\s+", desc):
        return desc
    return desc.rstrip() + "\n\nChapters:\n" + chapters_txt.strip() + "\n"


def upload_outputs(
    s3,
    *,
    output_bucket: str,
    job_id: str,
    out_dir: Path,
) -> None:
    prefix = f"jobs/{job_id}/output/"
    for p in sorted(out_dir.iterdir()):
        if not p.is_file():
            continue
        key = prefix + p.name
        logging.info("Uploading output: s3://%s/%s", output_bucket, key)
        s3.upload_file(str(p), output_bucket, key)


def test_bedrock_connectivity(brt, model_id: str) -> Tuple[bool, str]:
    """Test Bedrock connectivity with a simple prompt.
    
    Returns (success, error_message).
    """
    try:
        resp = brt.converse(
            modelId=model_id,
            system=[{"text": "You are a test assistant."}],
            messages=[{"role": "user", "content": [{"text": "Reply with exactly: OK"}]}],
            inferenceConfig={"maxTokens": 10, "temperature": 0.0},
        )
        content = resp.get("output", {}).get("message", {}).get("content", [])
        if content:
            return True, ""
        return False, "Empty response from Bedrock"
    except Exception as e:
        return False, str(e)


def main() -> int:
    parser = argparse.ArgumentParser(description="Another Automatic Video Editor runner (ECS worker).")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tmp-dir", required=True)
    parser.add_argument("--input-bucket", required=True)
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument("--bedrock-model-id", required=True)
    parser.add_argument("--bedrock-region", required=True)
    parser.add_argument("--enable-transcribe", required=True)
    parser.add_argument("--skip-ai-check", action="store_true", help="Skip AI connectivity check and use fallback if AI fails")
    parser.add_argument("--require-ai", action="store_true", help="Fail immediately if AI is not available (no fallback)")

    args = parser.parse_args()

    job_id = args.job_id
    job_dir = Path(args.job_dir)
    out_dir = Path(args.out_dir)
    tmp_dir = Path(args.tmp_dir)

    ensure_dir(out_dir)
    ensure_dir(tmp_dir)

    log_path = out_dir / "render.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )

    logging.info("Job started: %s", job_id)
    logging.info("Job dir: %s", job_dir)
    logging.info("Out dir: %s", out_dir)
    logging.info("Tmp dir: %s", tmp_dir)

    manifest_path = job_dir / "manifest.json"
    manifest = load_manifest(manifest_path)

    enable_transcribe = str(args.enable_transcribe).lower() == "true"

    stack_region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or args.bedrock_region
    bedrock_region = args.bedrock_region or stack_region

    # --------------------
    # AI connectivity check (before heavy processing)
    # --------------------
    ai_enabled = bool(manifest.get("ai", {}).get("enabled", True))
    ai_available = False
    
    if ai_enabled and not args.skip_ai_check:
        logging.info("Testing Bedrock connectivity (model: %s, region: %s)...", args.bedrock_model_id, bedrock_region)
        try:
            brt_test = bedrock_runtime_client(bedrock_region)
            ai_available, ai_error = test_bedrock_connectivity(brt_test, args.bedrock_model_id)
        except Exception as e:
            ai_available = False
            ai_error = str(e)
        
        if ai_available:
            logging.info("✓ Bedrock AI is available and working")
        else:
            logging.warning("✗ Bedrock AI is NOT available: %s", ai_error)
            
            if args.require_ai:
                logging.error("AI is required (--require-ai flag) but not available. Aborting.")
                return 1
            
            # Interactive prompt (only if running in a terminal)
            if sys.stdin.isatty():
                print("\n" + "=" * 60)
                print("⚠️  WARNING: AI (Bedrock) is not available!")
                print(f"   Error: {ai_error}")
                print("=" * 60)
                print("\nOptions:")
                print("  [c] Continue with fallback (deterministic plan, no AI)")
                print("  [a] Abort and fix the issue")
                print("")
                try:
                    choice = input("Your choice [c/a]: ").strip().lower()
                    if choice == 'a':
                        logging.info("User chose to abort.")
                        return 1
                    logging.info("User chose to continue with fallback.")
                except (EOFError, KeyboardInterrupt):
                    logging.info("No input received, continuing with fallback.")
            else:
                logging.info("Non-interactive mode: continuing with fallback plan.")
            
            ai_enabled = False  # Disable AI for this run
    elif args.skip_ai_check:
        logging.info("Skipping AI connectivity check (--skip-ai-check flag)")
        ai_available = True  # Assume available, will fallback if it fails later

    catalog = build_catalog(
        reko_region=stack_region,
        enable_transcribe=enable_transcribe,
        input_bucket=args.input_bucket,
        job_id=job_id,
        job_dir=job_dir,
        tmp_dir=tmp_dir,
        manifest=manifest,
    )
    safe_json_dump(catalog, out_dir / "catalog.json")

    plan: Optional[Dict[str, Any]] = None

    mode = get_style_mode(manifest)
    logging.info("Style mode: %s", mode)

    # Emit best moments (top) for debugging, if enabled
    if bool(catalog.get("best_moments_enabled")):
        bm_conf = manifest.get("best_moments", {}) or {}
        bm_top = collect_best_moments_global(catalog, max_total=int(bm_conf.get("max_clips_total", 12) or 12))
        safe_json_dump(bm_top, out_dir / "best_moments_top.json")

    # --------------------
    # Build plan
    # --------------------
    # Always try AI first if enabled, regardless of mode
    if ai_enabled:
        try:
            brt = bedrock_runtime_client(bedrock_region)
            plan = bedrock_plan(
                brt=brt,
                model_id=args.bedrock_model_id,
                manifest=manifest,
                catalog=catalog,
            )
            if plan:
                logging.info("AI planning succeeded")
        except Exception as e:
            logging.warning("AI planning failed: %s", e)

    # Fallback to deterministic plan if AI failed or disabled
    if not plan:
        if mode == "longform":
            logging.info("Using deterministic longform plan (AI fallback)")
            plan = longform_plan(manifest=manifest, catalog=catalog)
        else:
            logging.info("Using heuristic plan (AI fallback)")
            plan = heuristic_plan(manifest=manifest, catalog=catalog)

    # Always force intro segment at the beginning (if configured)
    intro_conf = manifest.get("intro", {}) or {}
    intro_file = str(intro_conf.get("file") or intro_conf.get("asset") or "").strip()
    intro_id = resolve_source_id_by_filename(catalog, intro_file) if intro_file else None
    
    if intro_id:
        intro_item = next((it for it in catalog.get("items", []) if it.get("id") == intro_id), None)
        if intro_item:
            # Check if intro is already the first segment
            existing_segments = plan.get("segments", []) or []
            first_seg_id = existing_segments[0].get("source_id") if existing_segments else None
            
            if first_seg_id != intro_id:
                logging.info("Adding intro segment: %s", intro_file)
                chapters = plan.get("chapters", []) or []
                
                # Only add Intro chapter if not already present
                if not chapters or str(chapters[0].get("title", "")).lower() != "intro":
                    plan["chapters"] = [{"title": "Intro"}] + chapters
                    # Shift existing segment chapters by +1
                    for s in existing_segments:
                        try:
                            s["chapter"] = int(s.get("chapter", 0)) + 1
                        except Exception:
                            s["chapter"] = 1

                intro_dur = float(intro_conf.get("duration_seconds", 0.0) or 0.0)
                intro_caption = str(intro_conf.get("caption", "") or "").strip()
                
                if intro_item.get("type") == "video":
                    vdur = float(intro_item.get("duration_s") or 0.0)
                    dur = intro_dur if intro_dur > 0 else min(10.0, vdur if vdur > 0 else 10.0)
                    intro_seg = {
                        "source_id": intro_id,
                        "type": "video",
                        "in_seconds": 0.0,
                        "duration": float(dur),
                        "chapter": 0,
                        "caption": intro_caption or "Intro",
                    }
                else:
                    dur = intro_dur if intro_dur > 0 else 5.0
                    intro_seg = {
                        "source_id": intro_id,
                        "type": "image",
                        "duration": float(dur),
                        "chapter": 0,
                        "caption": intro_caption or "Intro",
                    }

                # Remove intro from existing segments if present elsewhere
                plan["segments"] = [intro_seg] + [s for s in existing_segments if s.get("source_id") != intro_id]
            else:
                logging.info("Intro already first segment: %s", intro_file)

    safe_json_dump(plan, out_dir / "plan.json")

    # --------------------
    # Render
    # --------------------
    render_meta = render_from_plan(
        manifest=manifest,
        catalog=catalog,
        plan=plan,
        job_dir=job_dir,
        out_dir=out_dir,
        tmp_dir=tmp_dir,
    )
    safe_json_dump(render_meta, out_dir / "render_meta.json")

    # --------------------
    # Chapters
    # --------------------
    chapters = compute_chapters_from_segments(plan)
    write_chapters_txt(chapters, out_dir / "chapters.txt")
    chapters_txt = (out_dir / "chapters.txt").read_text(encoding="utf-8")

    # --------------------
    # SEO-friendly YouTube description (rigid template)
    # --------------------
    seo_conf = manifest.get("seo", {}) or {}
    seo_enabled = bool(seo_conf.get("enabled", True))

    seo_obj: Dict[str, Any] = {}

    # Start from AI-generated SEO, if enabled
    if seo_enabled and ai_enabled:
        try:
            brt = bedrock_runtime_client(bedrock_region)
            ai_seo = bedrock_generate_seo(
                brt=brt,
                model_id=args.bedrock_model_id,
                manifest=manifest,
                catalog=catalog,
                chapters=chapters,
            )
            if isinstance(ai_seo, dict):
                seo_obj.update(ai_seo)
        except Exception as e:
            logging.warning("SEO generation failed (fallback): %s", e)

    # Fallback
    if not seo_obj:
        seo_obj.update(build_seo_fallback(manifest))
        seo_obj["video_title"] = plan.get("video_title") or (manifest.get("project", {}) or {}).get("title") or "Video"

    # User overrides from manifest.seo
    if isinstance(seo_conf.get("video_title"), str) and seo_conf.get("video_title").strip():
        seo_obj["video_title"] = seo_conf["video_title"].strip()
    if isinstance(seo_conf.get("hook"), str) and seo_conf.get("hook").strip():
        seo_obj["hook"] = seo_conf["hook"].strip()
    if isinstance(seo_conf.get("cta"), str) and seo_conf.get("cta").strip():
        seo_obj["cta"] = seo_conf["cta"].strip()
    if isinstance(seo_conf.get("hashtags"), list) and seo_conf.get("hashtags"):
        seo_obj["hashtags"] = seo_conf.get("hashtags")

    # Persist title for convenience
    (out_dir / "title.txt").write_text(str(seo_obj.get("video_title", "")).strip() + "\n", encoding="utf-8")

    desc = build_youtube_description(
        manifest=manifest,
        plan=plan,
        chapters_txt=chapters_txt,
        seo=seo_obj,
        music_info=render_meta,
    )
    (out_dir / "description.md").write_text(desc, encoding="utf-8")

    # Upload outputs to S3
    s3 = s3_client(stack_region)
    upload_outputs(s3, output_bucket=args.output_bucket, job_id=job_id, out_dir=out_dir)

    logging.info("Job completed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
