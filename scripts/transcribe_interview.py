#!/usr/bin/env python3
"""Prepare and diarize German interview audio with the OpenAI Transcriptions API.

The script has no third-party Python dependencies. It expects ffmpeg/ffprobe and
reads the API key only from OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import unicodedata
import urllib.error
import urllib.request
import uuid


MODEL = "gpt-4o-transcribe-diarize"
LANGUAGE = "de"
RESPONSE_FORMAT = "diarized_json"
CHUNKING_STRATEGY = "auto"
API_URL = "https://api.openai.com/v1/audio/transcriptions"
API_UPLOAD_LIMIT_BYTES = 25_000_000
TARGET_UPLOAD_BYTES = 23_500_000
BITRATE_CHOICES_KBPS = (96, 80, 72, 64, 56, 48, 40)
MODEL_MAX_AUDIO_SECONDS = 1400.0
LOCAL_CHUNK_TARGET_SECONDS = 1200.0
LOCAL_CHUNK_OVERLAP_SECONDS = 10.0
LOCAL_CHUNK_SAFETY_SECONDS = 20.0
SILENCE_SEARCH_WINDOW_SECONDS = 45.0
SILENCE_THRESHOLD_DB = -35
SILENCE_MIN_SECONDS = 0.45
SPEAKER_REFERENCE_SECONDS = 5.0
ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = ROOT / "data" / "work"
RESULTS_ROOT = ROOT / "data" / "results"


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"Error: {message}")


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=capture,
        )
    except FileNotFoundError:
        fail(f"required executable not found: {command[0]}")
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        fail(f"command failed ({' '.join(command[:2])}): {details}")


def slugify(path: Path) -> str:
    value = unicodedata.normalize("NFKD", path.stem)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return value or "interview"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        capture=True,
    )
    return json.loads(result.stdout)


def audio_streams(info: dict) -> list[dict]:
    return [item for item in info.get("streams", []) if item.get("codec_type") == "audio"]


def duration_seconds(info: dict) -> float:
    try:
        return float(info["format"]["duration"])
    except (KeyError, TypeError, ValueError):
        fail("ffprobe did not return the recording duration")


def format_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def inspect(path: Path) -> dict:
    info = probe(path)
    streams = audio_streams(info)
    if not streams:
        fail("the file has no audio stream")
    summary = {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "duration_seconds": duration_seconds(info),
        "container": info.get("format", {}).get("format_name"),
        "audio_streams": [
            {
                "index": stream.get("index"),
                "codec": stream.get("codec_name"),
                "profile": stream.get("profile"),
                "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
                "channels": stream.get("channels"),
                "channel_layout": stream.get("channel_layout"),
                "bit_rate": int(stream["bit_rate"]) if stream.get("bit_rate") else None,
                "duration_seconds": float(stream["duration"]) if stream.get("duration") else None,
            }
            for stream in streams
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def choose_bitrate(duration: float) -> int:
    maximum_kbps = TARGET_UPLOAD_BYTES * 8 / duration / 1000
    for bitrate in BITRATE_CHOICES_KBPS:
        if bitrate <= maximum_kbps * 0.96:
            return bitrate
    fail(
        "the recording is too long for a single sufficiently high-quality file under 25 MB; "
        "splitting into multiple parts is required"
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_audio(
    source: Path,
    *,
    kind: str,
    start: float,
    sample_duration: float,
    force: bool,
) -> tuple[Path, dict]:
    info = probe(source)
    if not audio_streams(info):
        fail("the source has no audio stream")
    source_duration = duration_seconds(info)
    slug = slugify(source)
    audio_dir = WORK_ROOT / slug / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    if kind == "sample":
        if start < 0 or sample_duration <= 0 or start >= source_duration:
            fail("invalid sample start time or duration")
        actual_duration = min(sample_duration, source_duration - start)
        end = start + actual_duration
        output = audio_dir / (
            f"sample_{format_timestamp(start).replace(':', '').replace('.', '-')}_"
            f"{format_timestamp(end).replace(':', '').replace('.', '-')}.m4a"
        )
        bitrate = 96
        ffmpeg_args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y" if force else "-n",
            "-i",
            str(source),
            "-ss",
            str(start),
            "-t",
            str(actual_duration),
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "aac",
            "-b:a",
            f"{bitrate}k",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-map_metadata",
            "-1",
            "-movflags",
            "+faststart",
            str(output),
        ]
        offset = start
    else:
        actual_duration = source_duration
        bitrate = choose_bitrate(source_duration)
        output = audio_dir / f"full_{bitrate}k.m4a"
        ffmpeg_args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y" if force else "-n",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "aac",
            "-b:a",
            f"{bitrate}k",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-map_metadata",
            "-1",
            "-movflags",
            "+faststart",
            str(output),
        ]
        offset = 0.0

    if output.exists() and not force:
        print(f"Prepared audio already exists: {output}")
    else:
        run(ffmpeg_args)

    prepared_info = probe(output)
    if output.stat().st_size >= API_UPLOAD_LIMIT_BYTES:
        fail(
            f"the prepared file is {output.stat().st_size} bytes and exceeds the API upload limit"
        )

    manifest = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            "path": str(source.resolve()),
            "sha256": sha256(source),
            "size_bytes": source.stat().st_size,
            "duration_seconds": source_duration,
            "ffprobe": info,
        },
        "prepared_audio": {
            "kind": kind,
            "path": str(output.resolve()),
            "sha256": sha256(output),
            "size_bytes": output.stat().st_size,
            "offset_seconds": offset,
            "duration_seconds": duration_seconds(prepared_info),
            "codec": "aac",
            "bitrate_kbps_requested": bitrate,
            "channels": 1,
            "sample_rate_hz": 48000,
            "metadata_removed": True,
        },
        "transcription_settings": {
            "endpoint": API_URL,
            "model": MODEL,
            "language": LANGUAGE,
            "response_format": RESPONSE_FORMAT,
            "chunking_strategy": CHUNKING_STRATEGY,
            "stream": True,
        },
    }
    manifest_path = output.with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    print(f"Prepared audio: {output} ({output.stat().st_size / 1_000_000:.2f} MB)")
    print(f"Preparation manifest: {manifest_path}")
    return output, manifest


def compact_timestamp(seconds: float) -> str:
    return format_timestamp(seconds).replace(":", "").replace(".", "-")


def silence_midpoints(source: Path) -> list[float]:
    result = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-af",
            f"silencedetect=noise={SILENCE_THRESHOLD_DB}dB:d={SILENCE_MIN_SECONDS}",
            "-f",
            "null",
            "-",
        ],
        capture=True,
    )
    pattern = re.compile(
        r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)"
    )
    midpoints: list[float] = []
    for end_text, duration_text in pattern.findall(result.stderr or ""):
        end = float(end_text)
        silence_duration = float(duration_text)
        midpoints.append(end - silence_duration / 2.0)
    return midpoints


def choose_chunk_boundaries(duration: float, silence_points: list[float]) -> list[float]:
    chunk_count = max(1, math.ceil(duration / LOCAL_CHUNK_TARGET_SECONDS))
    if chunk_count == 1:
        return [0.0, duration]

    maximum_core_duration = (
        MODEL_MAX_AUDIO_SECONDS
        - 2 * LOCAL_CHUNK_OVERLAP_SECONDS
        - LOCAL_CHUNK_SAFETY_SECONDS
    )
    boundaries = [0.0]
    for index in range(1, chunk_count):
        ideal = duration * index / chunk_count
        candidates = [
            point
            for point in silence_points
            if abs(point - ideal) <= SILENCE_SEARCH_WINDOW_SECONDS
            and point - boundaries[-1] <= maximum_core_duration
        ]
        boundaries.append(
            min(candidates, key=lambda point: abs(point - ideal)) if candidates else ideal
        )
    boundaries.append(duration)

    if any(
        end - start > maximum_core_duration
        for start, end in zip(boundaries, boundaries[1:])
    ):
        boundaries = [duration * index / chunk_count for index in range(chunk_count + 1)]
    return boundaries


def prepare_chunked_audio(source: Path, *, force: bool) -> tuple[list[dict], dict]:
    info = probe(source)
    if not audio_streams(info):
        fail("the source has no audio stream")
    source_duration = duration_seconds(info)
    slug = slugify(source)
    audio_dir = WORK_ROOT / slug / "audio" / "full_chunks"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print("Finding quiet locations for local chunk boundaries...")
    boundaries = choose_chunk_boundaries(source_duration, silence_midpoints(source))
    chunks: list[dict] = []
    for index, (core_start, core_end) in enumerate(zip(boundaries, boundaries[1:]), start=1):
        upload_start = max(
            0.0,
            core_start - (LOCAL_CHUNK_OVERLAP_SECONDS if index > 1 else 0.0),
        )
        upload_end = min(
            source_duration,
            core_end
            + (LOCAL_CHUNK_OVERLAP_SECONDS if index < len(boundaries) - 1 else 0.0),
        )
        requested_duration = upload_end - upload_start
        output = audio_dir / (
            f"chunk_{index:03d}_{compact_timestamp(upload_start)}_"
            f"{compact_timestamp(upload_end)}.m4a"
        )
        if requested_duration >= MODEL_MAX_AUDIO_SECONDS - LOCAL_CHUNK_SAFETY_SECONDS:
            fail(
                f"planned chunk {index} is {requested_duration:.3f} seconds, too close to "
                f"the model maximum of {MODEL_MAX_AUDIO_SECONDS:.0f} seconds"
            )
        if output.exists() and not force:
            print(f"Prepared chunk already exists: {output}")
        else:
            run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "warning",
                    "-y" if force else "-n",
                    "-ss",
                    str(upload_start),
                    "-i",
                    str(source),
                    "-t",
                    str(requested_duration),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "96k",
                    "-ac",
                    "1",
                    "-ar",
                    "48000",
                    "-map_metadata",
                    "-1",
                    "-movflags",
                    "+faststart",
                    str(output),
                ]
            )
        prepared_info = probe(output)
        prepared_duration = duration_seconds(prepared_info)
        if output.stat().st_size >= API_UPLOAD_LIMIT_BYTES:
            fail(f"prepared chunk {index} exceeds the API upload limit")
        if prepared_duration >= MODEL_MAX_AUDIO_SECONDS:
            fail(
                f"prepared chunk {index} is {prepared_duration:.3f} seconds and exceeds "
                f"the model maximum of {MODEL_MAX_AUDIO_SECONDS:.0f} seconds"
            )
        chunks.append(
            {
                "index": index,
                "path": str(output.resolve()),
                "sha256": sha256(output),
                "size_bytes": output.stat().st_size,
                "duration_seconds": prepared_duration,
                "core_start_seconds": core_start,
                "core_end_seconds": core_end,
                "upload_start_seconds": upload_start,
                "upload_end_seconds": upload_end,
            }
        )

    manifest = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            "path": str(source.resolve()),
            "sha256": sha256(source),
            "size_bytes": source.stat().st_size,
            "duration_seconds": source_duration,
            "ffprobe": info,
        },
        "prepared_audio": {
            "kind": "full",
            "path": str(audio_dir.resolve()),
            "offset_seconds": 0.0,
            "duration_seconds": source_duration,
            "chunked": True,
            "chunk_target_seconds": LOCAL_CHUNK_TARGET_SECONDS,
            "overlap_seconds": LOCAL_CHUNK_OVERLAP_SECONDS,
            "boundary_method": "silence detection near evenly spaced targets",
            "codec": "aac",
            "bitrate_kbps_requested": 96,
            "channels": 1,
            "sample_rate_hz": 48000,
            "metadata_removed": True,
            "chunks": chunks,
        },
        "transcription_settings": {
            "endpoint": API_URL,
            "model": MODEL,
            "language": LANGUAGE,
            "response_format": RESPONSE_FORMAT,
            "chunking_strategy": CHUNKING_STRATEGY,
            "stream": True,
            "model_max_audio_seconds_observed": MODEL_MAX_AUDIO_SECONDS,
        },
    }
    manifest_path = audio_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(f"Prepared {len(chunks)} local chunks:")
    for chunk in chunks:
        print(
            f"  {chunk['index']}/{len(chunks)}: "
            f"{format_timestamp(chunk['upload_start_seconds'])}–"
            f"{format_timestamp(chunk['upload_end_seconds'])} "
            f"({chunk['size_bytes'] / 1_000_000:.2f} MB)"
        )
    print(f"Preparation manifest: {manifest_path}")
    return chunks, manifest


def multipart_body(fields: list[tuple[str, str]], audio_path: Path) -> tuple[bytes, str]:
    boundary = f"codex-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    safe_name = audio_path.name.replace('"', "")
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{safe_name}"\r\n'.encode(),
            b"Content-Type: audio/mp4\r\n\r\n",
            audio_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), boundary


def collect_streamed_transcription(
    response: object,
    total_duration: float,
    *,
    progress_label: str = "",
    original_offset: float = 0.0,
    original_duration: float | None = None,
    core_start: float | None = None,
    core_end: float | None = None,
) -> dict:
    """Collect a server-sent event stream into a diarized response object."""
    events: list[dict] = []
    segments: list[dict] = []
    done_event: dict | None = None
    event_name: str | None = None
    data_lines: list[str] = []

    def process_event() -> None:
        nonlocal done_event, event_name, data_lines
        if not data_lines:
            event_name = None
            return
        payload = "\n".join(data_lines)
        current_event_name = event_name
        event_name = None
        data_lines = []
        if payload == "[DONE]":
            return
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            fail(f"the API returned an invalid streaming event: {exc}")
        if not isinstance(event, dict):
            fail("the API returned a non-object streaming event")
        event_type = str(event.get("type") or current_event_name or "")
        events.append(event)
        if event_type == "transcript.text.segment":
            segments.append(event)
            try:
                completed_seconds = float(event["end"])
                progress = format_timestamp(completed_seconds)
                chunk_percent = min(100.0, completed_seconds / total_duration * 100)
                if original_duration is None:
                    remaining = format_timestamp(max(0.0, total_duration - completed_seconds))
                    progress_details = (
                        f"{chunk_percent:5.1f}% complete, {remaining} of audio remaining"
                    )
                else:
                    original_completed = original_offset + completed_seconds
                    if core_start is not None:
                        original_completed = max(core_start, original_completed)
                    if core_end is not None:
                        original_completed = min(core_end, original_completed)
                    overall_percent = min(100.0, original_completed / original_duration * 100)
                    overall_remaining = format_timestamp(
                        max(0.0, original_duration - original_completed)
                    )
                    progress_details = (
                        f"chunk {chunk_percent:5.1f}%; total {overall_percent:5.1f}%; "
                        f"{overall_remaining} of original audio remaining"
                    )
            except (KeyError, TypeError, ValueError):
                progress = "unknown time"
                progress_details = "progress unavailable"
            print(
                f"{progress_label}Received {len(segments)} segment(s), through {progress} "
                f"({progress_details})",
                end="\r",
                flush=True,
            )
        elif event_type == "transcript.text.done":
            done_event = event
        elif event_type in {"error", "transcript.error"}:
            fail(f"OpenAI API streaming error: {event}")

    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            process_event()
        elif line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    process_event()
    if segments:
        print()
    if done_event is None:
        fail("the API stream ended before transcript.text.done was received")
    if not segments:
        fail("the completed API stream contains no diarized speaker segments")
    return {
        "task": "transcribe",
        "duration": max(float(item.get("end", 0)) for item in segments),
        "text": str(done_event.get("text") or " ".join(item.get("text", "") for item in segments)),
        "segments": segments,
        "usage": done_event.get("usage"),
        "streamed": True,
        "stream_events": events,
    }


def call_api(
    audio_path: Path,
    total_duration: float,
    *,
    known_speakers: list[tuple[str, Path]] | None = None,
    progress_label: str = "",
    original_offset: float = 0.0,
    original_duration: float | None = None,
    core_start: float | None = None,
    core_end: float | None = None,
) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        fail(
            "OPENAI_API_KEY is not set. Configure it in the local environment; "
            "do not put it in chat or a project file."
        )
    fields = [
        ("model", MODEL),
        ("language", LANGUAGE),
        ("response_format", RESPONSE_FORMAT),
        ("chunking_strategy", CHUNKING_STRATEGY),
        ("stream", "true"),
    ]
    for speaker_name, reference_path in known_speakers or []:
        encoded_reference = base64.b64encode(reference_path.read_bytes()).decode("ascii")
        fields.append(("known_speaker_names[]", speaker_name))
        fields.append(
            ("known_speaker_references[]", f"data:audio/wav;base64,{encoded_reference}")
        )
    body, boundary = multipart_body(fields, audio_path)
    request = urllib.request.Request(
        API_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "interview-transcription/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            return collect_streamed_transcription(
                response,
                total_duration,
                progress_label=progress_label,
                original_offset=original_offset,
                original_duration=original_duration,
                core_start=core_start,
                core_end=core_end,
            )
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        fail(f"OpenAI API returned HTTP {exc.code}: {details}")
    except urllib.error.URLError as exc:
        fail(f"could not connect to the OpenAI API: {exc.reason}")
    except (TimeoutError, socket.timeout):
        fail(
            "the OpenAI API stream timed out before completion. The request may still have "
            "been billed; check the API usage dashboard before retrying."
        )


def speaker_sort_key(label: str) -> tuple[int, str]:
    match = re.fullmatch(r"S(\d+)", label)
    return (int(match.group(1)), label) if match else (10_000, label)


def next_speaker_label(existing: set[str]) -> str:
    number = 1
    while f"S{number}" in existing:
        number += 1
    return f"S{number}"


def absolute_chunk_segments(raw: dict, chunk: dict) -> list[dict]:
    raw_segments = raw.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        fail(f"chunk {chunk['index']} contains no diarized segments")
    offset = float(chunk["upload_start_seconds"])
    output: list[dict] = []
    for index, segment in enumerate(raw_segments):
        try:
            start = offset + float(segment["start"])
            end = offset + float(segment["end"])
        except (KeyError, TypeError, ValueError):
            fail(f"chunk {chunk['index']} segment {index} has invalid timestamps")
        output.append(
            {
                "id": segment.get("id", f"segment-{index + 1}"),
                "start": start,
                "end": end,
                "speaker_raw": str(segment.get("speaker", "unknown")),
                "text": str(segment.get("text", "")).strip(),
            }
        )
    return output


def initial_speaker_mapping(segments: list[dict]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for segment in segments:
        raw_speaker = segment["speaker_raw"]
        if raw_speaker not in mapping:
            mapping[raw_speaker] = f"S{len(mapping) + 1}"
    return mapping


def reconcile_speaker_mapping(
    current: list[dict], previous: list[dict], existing_speakers: set[str]
) -> tuple[dict[str, str], dict[str, str]]:
    local_speakers = list(dict.fromkeys(segment["speaker_raw"] for segment in current))
    mapping: dict[str, str] = {}
    evidence: dict[str, str] = {}

    for local in local_speakers:
        if local in existing_speakers:
            mapping[local] = local
            evidence[local] = "known speaker reference returned by the API"

    for local in local_speakers:
        if local in mapping:
            continue
        scores: list[tuple[float, str]] = []
        for global_speaker in sorted(existing_speakers, key=speaker_sort_key):
            score = 0.0
            for current_segment in current:
                if current_segment["speaker_raw"] != local:
                    continue
                for previous_segment in previous:
                    if previous_segment["speaker"] != global_speaker:
                        continue
                    score += max(
                        0.0,
                        min(current_segment["end"], previous_segment["end"])
                        - max(current_segment["start"], previous_segment["start"]),
                    )
            scores.append((score, global_speaker))
        if scores:
            score, global_speaker = max(scores)
            if score > 0.1:
                mapping[local] = global_speaker
                evidence[local] = f"{score:.3f}s matching speech in the local overlap"

    remaining_local = [speaker for speaker in local_speakers if speaker not in mapping]
    remaining_global = [
        speaker
        for speaker in sorted(existing_speakers, key=speaker_sort_key)
        if speaker not in mapping.values()
    ]
    if len(remaining_local) == len(remaining_global) == 1:
        mapping[remaining_local[0]] = remaining_global[0]
        evidence[remaining_local[0]] = "inferred from the only unmatched existing speaker"

    for local in local_speakers:
        if local in mapping:
            continue
        new_label = next_speaker_label(existing_speakers | set(mapping.values()))
        mapping[local] = new_label
        evidence[local] = "new or unresolved speaker; manual continuity review required"
    return mapping, evidence


def map_speakers_from_sample(
    current: list[dict], sample_segments: list[dict]
) -> tuple[dict[str, str], dict[str, str]]:
    sample_speakers = sorted(
        {str(segment["speaker"]) for segment in sample_segments}, key=speaker_sort_key
    )
    mapping: dict[str, str] = {}
    evidence: dict[str, str] = {}
    for local in dict.fromkeys(segment["speaker_raw"] for segment in current):
        if local in sample_speakers:
            mapping[local] = local
            evidence[local] = "speaker label already matches the validation sample"
            continue
        scores: list[tuple[float, str]] = []
        for sample_speaker in sample_speakers:
            score = 0.0
            for current_segment in current:
                if current_segment["speaker_raw"] != local:
                    continue
                for sample_segment in sample_segments:
                    if sample_segment["speaker"] != sample_speaker:
                        continue
                    score += max(
                        0.0,
                        min(current_segment["end"], sample_segment["end"])
                        - max(current_segment["start"], sample_segment["start"]),
                    )
            scores.append((score, sample_speaker))
        if scores:
            score, sample_speaker = max(scores)
            if score > 0.05:
                mapping[local] = sample_speaker
                evidence[local] = (
                    f"{score:.3f}s matching speech in the validated sample transcript"
                )

    assigned = set(sample_speakers) | set(mapping.values())
    for local in dict.fromkeys(segment["speaker_raw"] for segment in current):
        if local in mapping:
            continue
        new_label = next_speaker_label(assigned)
        mapping[local] = new_label
        assigned.add(new_label)
        evidence[local] = "not present in the validation sample; assigned as a new speaker"
    return mapping, evidence


def apply_speaker_mapping(
    segments: list[dict], mapping: dict[str, str], chunk_index: int
) -> list[dict]:
    output: list[dict] = []
    for index, segment in enumerate(segments, start=1):
        output.append(
            {
                **segment,
                "id": f"chunk-{chunk_index:03d}-{segment['id'] or index}",
                "speaker": mapping[segment["speaker_raw"]],
                "chunk_index": chunk_index,
            }
        )
    return output


def select_core_segments(segments: list[dict], chunk: dict, is_last: bool) -> list[dict]:
    core_start = float(chunk["core_start_seconds"])
    core_end = float(chunk["core_end_seconds"])
    selected: list[dict] = []
    for segment in segments:
        midpoint = (segment["start"] + segment["end"]) / 2.0
        if midpoint < core_start:
            continue
        if midpoint >= core_end and not is_last:
            continue
        selected.append(segment)
    return selected


def build_speaker_references(
    source: Path, segments: list[dict], reference_dir: Path
) -> list[tuple[str, Path]]:
    reference_dir.mkdir(parents=True, exist_ok=True)
    references: list[tuple[str, Path]] = []
    speakers = sorted({segment["speaker"] for segment in segments}, key=speaker_sort_key)[:4]
    for speaker in speakers:
        candidates: list[tuple[float, float, dict, float, float]] = []
        for segment in segments:
            if segment["speaker"] != speaker:
                continue
            segment_duration = segment["end"] - segment["start"]
            clip_duration = min(SPEAKER_REFERENCE_SECONDS, segment_duration - 0.4)
            if clip_duration < 2.0:
                continue
            clip_start = segment["start"] + (segment_duration - clip_duration) / 2.0
            clip_end = clip_start + clip_duration
            other_speaker_overlap = sum(
                max(0.0, min(clip_end, other["end"]) - max(clip_start, other["start"]))
                for other in segments
                if other["speaker"] != speaker
            )
            candidates.append(
                (other_speaker_overlap, -clip_duration, segment, clip_start, clip_duration)
            )
        if not candidates:
            print(f"Warning: no clean 2–10 second reference found for {speaker}")
            continue
        _, _, _, clip_start, clip_duration = min(candidates, key=lambda item: item[:2])
        output = reference_dir / f"{speaker}.wav"
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-y",
                "-ss",
                str(clip_start),
                "-i",
                str(source),
                "-t",
                str(clip_duration),
                "-map",
                "0:a:0",
                "-vn",
                "-c:a",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-map_metadata",
                "-1",
                str(output),
            ]
        )
        references.append((speaker, output))
    return references


def chunk_request_signature(chunk: dict, references: list[tuple[str, Path]]) -> dict:
    return {
        "audio_sha256": chunk["sha256"],
        "model": MODEL,
        "language": LANGUAGE,
        "response_format": RESPONSE_FORMAT,
        "chunking_strategy": CHUNKING_STRATEGY,
        "speaker_references": [
            {"name": name, "sha256": sha256(path)} for name, path in references
        ],
    }


def load_cached_chunk(path: Path, signature: dict) -> dict | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cached_signature = raw.get("_local_request")
    if not isinstance(cached_signature, dict) or not raw.get("segments"):
        return None
    essential_keys = {
        "audio_sha256",
        "model",
        "language",
        "response_format",
        "chunking_strategy",
    }
    if any(cached_signature.get(key) != signature.get(key) for key in essential_keys):
        return None
    return raw


def load_sample_segments(manifest: dict) -> tuple[list[dict], Path | None]:
    slug = slugify(Path(manifest["source"]["path"]))
    sample_path = RESULTS_ROOT / slug / "sample" / "transcript.json"
    if not sample_path.exists():
        return [], None
    try:
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print(f"Warning: could not read validation sample: {sample_path}")
        return [], None
    if sample.get("source", {}).get("sha256") != manifest["source"]["sha256"]:
        print(f"Warning: validation sample does not match the current source: {sample_path}")
        return [], None
    segments = sample.get("segments")
    if not isinstance(segments, list) or not segments:
        print(f"Warning: validation sample contains no segments: {sample_path}")
        return [], None
    normalized: list[dict] = []
    for segment in segments:
        try:
            normalized.append(
                {
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                    "speaker": str(segment["speaker"]),
                    "text": str(segment.get("text", "")),
                }
            )
        except (KeyError, TypeError, ValueError):
            print(f"Warning: validation sample has an invalid segment: {sample_path}")
            return [], None
    return normalized, sample_path


def transcribe_chunked(source: Path, *, force: bool) -> None:
    chunks, manifest = prepare_chunked_audio(source, force=force)

    slug = slugify(source)
    output_dir = RESULTS_ROOT / slug / "full"
    chunk_output_dir = output_dir / "chunks"
    chunk_output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = WORK_ROOT / slug / "audio" / "speaker_references"
    original_duration = float(manifest["source"]["duration_seconds"])
    sample_segments, sample_path = load_sample_segments(manifest)
    if sample_segments:
        print(f"Using validation sample for speaker continuity: {sample_path}")
    merged_segments: list[dict] = []
    previous_segments: list[dict] = []
    existing_speakers: set[str] = {
        segment["speaker"] for segment in sample_segments
    }
    references: list[tuple[str, Path]] = []
    chunk_records: list[dict] = []
    merged_events: list[dict] = []

    for chunk in chunks:
        index = int(chunk["index"])
        is_first = index == 1
        is_last = index == len(chunks)
        request_references = [] if is_first else references
        signature = chunk_request_signature(chunk, request_references)
        raw_path = chunk_output_dir / f"chunk_{index:03d}_api_raw.json"
        raw = None if force else load_cached_chunk(raw_path, signature)
        if raw is None:
            print(
                f"Sending chunk {index}/{len(chunks)} "
                f"({Path(chunk['path']).name}) to {MODEL}..."
            )
            raw = call_api(
                Path(chunk["path"]),
                float(chunk["duration_seconds"]),
                known_speakers=request_references,
                progress_label=f"Chunk {index}/{len(chunks)} — ",
                original_offset=float(chunk["upload_start_seconds"]),
                original_duration=original_duration,
                core_start=float(chunk["core_start_seconds"]),
                core_end=float(chunk["core_end_seconds"]),
            )
            raw["_local_request"] = signature
            write_json(raw_path, raw)
            print(f"Saved completed chunk immediately: {raw_path}")
        else:
            print(f"Reusing completed chunk {index}/{len(chunks)}: {raw_path}")

        absolute_segments = absolute_chunk_segments(raw, chunk)
        if is_first and sample_segments:
            speaker_mapping, mapping_evidence = map_speakers_from_sample(
                absolute_segments, sample_segments
            )
        elif is_first:
            speaker_mapping = initial_speaker_mapping(absolute_segments)
            mapping_evidence = {
                raw_speaker: "assigned by first appearance in the first chunk"
                for raw_speaker in speaker_mapping
            }
        else:
            speaker_mapping, mapping_evidence = reconcile_speaker_mapping(
                absolute_segments, previous_segments, existing_speakers
            )
        global_segments = apply_speaker_mapping(absolute_segments, speaker_mapping, index)
        existing_speakers.update(segment["speaker"] for segment in global_segments)
        merged_segments.extend(select_core_segments(global_segments, chunk, is_last))
        previous_segments = global_segments

        if is_first:
            references = build_speaker_references(source, global_segments, reference_dir)
            if references:
                print(
                    "Prepared speaker references for later chunks: "
                    + ", ".join(name for name, _ in references)
                )
            else:
                print("Warning: continuing without speaker references")

        for event in raw.get("stream_events", []):
            merged_events.append({"chunk_index": index, **event})
        chunk_records.append(
            {
                **chunk,
                "api_raw_json": str(raw_path.resolve()),
                "speaker_mapping": speaker_mapping,
                "speaker_mapping_evidence": mapping_evidence,
                "usage": raw.get("usage"),
            }
        )

    merged_segments.sort(key=lambda segment: (segment["start"], segment["end"]))
    merged_raw = {
        "task": "transcribe",
        "duration": original_duration,
        "text": " ".join(segment["text"] for segment in merged_segments if segment["text"]),
        "segments": merged_segments,
        "usage": [record.get("usage") for record in chunk_records],
        "streamed": True,
        "locally_chunked": True,
        "chunks": chunk_records,
        "stream_events": merged_events,
    }
    manifest["prepared_audio"]["chunks"] = chunk_records
    outputs = render_results(merged_raw, manifest)
    for label, path in outputs.items():
        print(f"{label}: {path}")


def normalized_segments(raw: dict, offset: float) -> tuple[list[dict], dict[str, str]]:
    raw_segments = raw.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        fail("the API response contains no diarized segments")
    speaker_map: dict[str, str] = {}
    used_labels: set[str] = set()
    normalized: list[dict] = []
    for index, segment in enumerate(raw_segments):
        raw_speaker = str(segment.get("speaker", "unknown"))
        if raw_speaker not in speaker_map:
            if re.fullmatch(r"S\d+", raw_speaker) and raw_speaker not in used_labels:
                speaker_map[raw_speaker] = raw_speaker
            else:
                speaker_map[raw_speaker] = next_speaker_label(used_labels)
            used_labels.add(speaker_map[raw_speaker])
        try:
            start = float(segment["start"]) + offset
            end = float(segment["end"]) + offset
        except (KeyError, TypeError, ValueError):
            fail(f"segment {index} has invalid timestamps")
        normalized.append(
            {
                "id": segment.get("id", f"segment-{index + 1}"),
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": speaker_map[raw_speaker],
                "speaker_raw": raw_speaker,
                "text": str(segment.get("text", "")).strip(),
            }
        )
    return normalized, speaker_map


def paragraphs(segments: list[dict]) -> list[dict]:
    output: list[dict] = []
    for segment in segments:
        if not segment["text"]:
            continue
        if (
            output
            and output[-1]["speaker"] == segment["speaker"]
            and segment["start"] - output[-1]["end"] <= 2.0
            and segment["end"] - output[-1]["start"] <= 90.0
        ):
            output[-1]["end"] = segment["end"]
            output[-1]["text"] = f"{output[-1]['text']} {segment['text']}"
        else:
            output.append(
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": segment["speaker"],
                    "text": segment["text"],
                }
            )
    return output


def review_candidates(
    segments: list[dict], prepared_end: float, chunk_boundaries: list[float] | None = None
) -> list[dict]:
    candidates: list[dict] = []
    uncertain = re.compile(
        r"\b(?:unverständlich|unverstaendlich|inaudible)\b|\[.*?\]|\(.*?\)", re.I
    )
    previous: dict | None = None
    for segment in segments:
        duration = max(0.001, segment["end"] - segment["start"])
        words = segment["text"].split()
        reasons: list[str] = []
        if uncertain.search(segment["text"]):
            reasons.append("a marker or parenthetical may indicate uncertain transcription")
        if len(words) >= 8 and len(words) / duration > 5.5:
            reasons.append("extremely high estimated speech rate")
        if previous and segment["start"] < previous["end"] - 0.2:
            reasons.append("overlapping speaker segments")
        if reasons:
            candidates.append(
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": segment["speaker"],
                    "reasons": reasons,
                    "text": segment["text"],
                }
            )
        previous = segment
    if segments and prepared_end - segments[-1]["end"] > 15:
        candidates.append(
            {
                "start": segments[-1]["end"],
                "end": prepared_end,
                "speaker": "-",
                "reasons": [
                    "more than 15 seconds between the final segment and the end of the audio"
                ],
                "text": "",
            }
        )
    for boundary in chunk_boundaries or []:
        start = max(0.0, boundary - 5.0)
        end = min(prepared_end, boundary + 5.0)
        nearby_text = " ".join(
            segment["text"]
            for segment in segments
            if segment["end"] >= start and segment["start"] <= end and segment["text"]
        )
        candidates.append(
            {
                "start": start,
                "end": end,
                "speaker": "-",
                "reasons": ["local chunk join; verify text and speaker continuity"],
                "text": nearby_text,
            }
        )
    return sorted(candidates, key=lambda item: (item["start"], item["end"]))


def render_results(raw: dict, manifest: dict) -> dict[str, Path]:
    prepared = manifest["prepared_audio"]
    kind = prepared["kind"]
    slug = slugify(Path(manifest["source"]["path"]))
    output_dir = RESULTS_ROOT / slug / kind
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "api_raw.json"
    # Preserve the provider response even if a later normalization check fails.
    write_json(raw_path, raw)
    offset = float(prepared["offset_seconds"])
    segments, speaker_map = normalized_segments(raw, offset)
    prepared_end = offset + float(prepared["duration_seconds"])
    chunk_boundaries = [
        float(chunk["core_end_seconds"])
        for chunk in prepared.get("chunks", [])[:-1]
    ]
    review = review_candidates(segments, prepared_end, chunk_boundaries)

    normalized_path = output_dir / "transcript.json"
    txt_path = output_dir / "transcript.txt"
    review_path = output_dir / "manual_review.txt"
    settings_path = output_dir / "run_manifest.json"

    normalized = {
        "schema_version": 1,
        "source": manifest["source"],
        "prepared_audio": prepared,
        "model": MODEL,
        "language": LANGUAGE,
        "speaker_map": speaker_map,
        "speaker_roles": {},
        "segments": segments,
        "text": " ".join(item["text"] for item in segments if item["text"]),
        "notice": (
            "Automated draft. The model does not return confidence/logprobs with "
            "diarized_json; confirm [unverständlich] passages by listening."
        ),
    }
    write_json(normalized_path, normalized)

    header = [
        "AUTOMATED TRANSCRIPT DRAFT — AUDIO REVIEW REQUIRED",
        f"Model: {MODEL}",
        f"Language: {LANGUAGE}",
        "Speakers: provisional labels S1, S2, etc.; roles have not been assigned automatically.",
        "",
    ]
    lines = header + [
        f"[{format_timestamp(item['start'])}–{format_timestamp(item['end'])}] "
        f"{item['speaker']}: {item['text']}"
        for item in paragraphs(segments)
    ]
    txt_path.write_text("\n\n".join(lines).rstrip() + "\n", encoding="utf-8")

    review_lines = [
        "LOCATIONS FOR MANUAL REVIEW",
        "",
        (
            "The diarization model does not return confidence/logprobs. This list is "
            "heuristic and does not replace sample validation or final audio review."
        ),
        "",
    ]
    if not review:
        review_lines.append("The automated heuristics did not flag any locations.")
    else:
        for item in review:
            reason = "; ".join(item["reasons"])
            review_lines.append(
                f"[{format_timestamp(item['start'])}–{format_timestamp(item['end'])}] "
                f"{item['speaker']} — {reason}\n{item['text']}"
            )
    review_path.write_text("\n\n".join(review_lines).rstrip() + "\n", encoding="utf-8")

    run_manifest = {
        **manifest,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "outputs": {
            "api_raw_json": str(raw_path.resolve()),
            "normalized_json": str(normalized_path.resolve()),
            "transcript_txt": str(txt_path.resolve()),
            "manual_review_txt": str(review_path.resolve()),
        },
        "speaker_map": speaker_map,
        "segment_count": len(segments),
        "review_candidate_count": len(review),
    }
    write_json(settings_path, run_manifest)
    return {
        "raw": raw_path,
        "json": normalized_path,
        "txt": txt_path,
        "review": review_path,
        "manifest": settings_path,
    }


def transcribe(source: Path, kind: str, start: float, sample_duration: float, force: bool) -> None:
    if (
        kind == "full"
        and duration_seconds(probe(source)) >= MODEL_MAX_AUDIO_SECONDS - LOCAL_CHUNK_SAFETY_SECONDS
    ):
        transcribe_chunked(source, force=force)
        return
    audio_path, manifest = prepare_audio(
        source,
        kind=kind,
        start=start,
        sample_duration=sample_duration,
        force=force,
    )
    if not os.environ.get("OPENAI_API_KEY"):
        fail(
            "OPENAI_API_KEY is not set. Configure it in the local environment; "
            "do not put it in chat or a project file."
        )
    print(f"Sending {audio_path.name} to {MODEL}...")
    raw = call_api(audio_path, float(manifest["prepared_audio"]["duration_seconds"]))
    outputs = render_results(raw, manifest)
    for label, path in outputs.items():
        print(f"{label}: {path}")


def existing_manifest(audio_path: Path) -> dict:
    path = audio_path.with_suffix(".manifest.json")
    if not path.exists():
        fail(f"preparation manifest is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and diarize sociological interview recordings."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser("inspect", help="show source media details")
    inspect_parser.add_argument("source", type=Path)

    prepare_parser = commands.add_parser("prepare", help="prepare audio without an API call")
    prepare_parser.add_argument("source", type=Path)
    prepare_parser.add_argument("--kind", choices=("sample", "full"), default="sample")
    prepare_parser.add_argument("--start", type=float, default=0.0, help="sample start in seconds")
    prepare_parser.add_argument("--duration", type=float, default=480.0, help="sample duration in seconds")
    prepare_parser.add_argument("--force", action="store_true")

    for name, help_text in (
        ("sample", "prepare and transcribe a validation sample"),
        ("full", "prepare and transcribe the full interview"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("source", type=Path)
        command.add_argument("--start", type=float, default=0.0)
        command.add_argument("--duration", type=float, default=480.0)
        command.add_argument("--force", action="store_true")

    render_parser = commands.add_parser("render", help="regenerate outputs from raw API JSON")
    render_parser.add_argument("raw_json", type=Path)
    render_parser.add_argument("prepared_audio", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "inspect":
        inspect(args.source.resolve())
        return
    if args.command == "render":
        raw = json.loads(args.raw_json.read_text(encoding="utf-8"))
        outputs = render_results(raw, existing_manifest(args.prepared_audio.resolve()))
        for label, path in outputs.items():
            print(f"{label}: {path}")
        return
    source = args.source.resolve()
    if not source.is_file():
        fail(f"source file does not exist: {source}")
    kind = args.kind if args.command == "prepare" else args.command
    if args.command == "prepare":
        if (
            kind == "full"
            and duration_seconds(probe(source))
            >= MODEL_MAX_AUDIO_SECONDS - LOCAL_CHUNK_SAFETY_SECONDS
        ):
            prepare_chunked_audio(source, force=args.force)
        else:
            prepare_audio(
                source,
                kind=kind,
                start=args.start,
                sample_duration=args.duration,
                force=args.force,
            )
    else:
        transcribe(source, kind, args.start, args.duration, args.force)


if __name__ == "__main__":
    main()
