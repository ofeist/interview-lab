#!/usr/bin/env python3
"""Prepare and diarize German interview audio with the OpenAI Transcriptions API.

The script has no third-party Python dependencies. It expects ffmpeg/ffprobe and
reads the API key only from OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
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


def multipart_body(fields: dict[str, str], audio_path: Path) -> tuple[bytes, str]:
    boundary = f"codex-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
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


def collect_streamed_transcription(response: object) -> dict:
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
                progress = format_timestamp(float(event["end"]))
            except (KeyError, TypeError, ValueError):
                progress = "unknown time"
            print(
                f"Received {len(segments)} completed speaker segment(s), through {progress}",
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


def call_api(audio_path: Path) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        fail(
            "OPENAI_API_KEY is not set. Configure it in the local environment; "
            "do not put it in chat or a project file."
        )
    fields = {
        "model": MODEL,
        "language": LANGUAGE,
        "response_format": RESPONSE_FORMAT,
        "chunking_strategy": CHUNKING_STRATEGY,
        "stream": "true",
    }
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
            return collect_streamed_transcription(response)
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


def normalized_segments(raw: dict, offset: float) -> tuple[list[dict], dict[str, str]]:
    raw_segments = raw.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        fail("the API response contains no diarized segments")
    speaker_map: dict[str, str] = {}
    normalized: list[dict] = []
    for index, segment in enumerate(raw_segments):
        raw_speaker = str(segment.get("speaker", "unknown"))
        if raw_speaker not in speaker_map:
            speaker_map[raw_speaker] = f"S{len(speaker_map) + 1}"
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


def review_candidates(segments: list[dict], prepared_end: float) -> list[dict]:
    candidates: list[dict] = []
    uncertain = re.compile(r"\b(?:unverständlich|unverstaendlich|inaudible)\b|\[.*?\]|\(.*?\)", re.I)
    previous: dict | None = None
    for segment in segments:
        duration = max(0.001, segment["end"] - segment["start"])
        words = segment["text"].split()
        reasons: list[str] = []
        if uncertain.search(segment["text"]):
            reasons.append("a marker or parenthetical may indicate uncertain transcription")
        if len(words) >= 6 and len(words) / duration > 4.8:
            reasons.append("unusually high estimated speech rate")
        if duration < 0.35 and words:
            reasons.append("very short speech segment")
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
                "reasons": ["more than 15 seconds between the final segment and the end of the audio"],
                "text": "",
            }
        )
    return candidates


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
    review = review_candidates(segments, prepared_end)

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
    raw = call_api(audio_path)
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
