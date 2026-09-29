# Interview Transcription Procedure

`scripts/transcribe_interview.py` keeps the source recording in `data/raw/`,
creates prepared audio in `data/work/`, and writes transcripts to
`data/results/`. It reads the API key only from the `OPENAI_API_KEY` environment
variable. The key is never written to project files, run manifests, or command
arguments.

## Current configuration

- Model: `gpt-4o-transcribe-diarize`.
- Input language parameter: `de`, matching the current interview collection.
  This is currently a script setting, not a command-line option.
- Response format: `diarized_json`.
- API-side speech segmentation: `chunking_strategy=auto` (voice activity
  detection). This is separate from the local splitting described below.
- Response delivery: `stream=true`. The script reports completed speech
  segments and progress while the API works.
- Speakers: neutral labels such as `S1` and `S2`. Roles are assigned only after
  human review.
- Local splitting: chunks of approximately 20 minutes, boundaries near silence,
  and 10 seconds of overlap on each side of an internal boundary. The API
  returned a 1,400-second maximum for this model when given the first full
  interview.
- Speaker continuity: when a validation sample exists for the same source
  recording, its checked speaker labels help map the first full chunk. The
  script then creates short local voice references for up to four speakers and
  sends them with later chunks. It also compares speakers in the overlap.
- Merging: timestamps are restored to the original recording's timeline.
  Segments are selected by their midpoint in each non-overlapping core area to
  avoid duplicate text from the overlaps.
- Recovery: each completed API chunk is saved immediately and reused after a
  connection failure or interrupted process.

Official API documentation:

- https://developers.openai.com/api/docs/guides/speech-to-text
- https://developers.openai.com/api/docs/models/gpt-4o-transcribe-diarize
- https://developers.openai.com/api/docs/pricing

## Commands

Inspect a recording without changing it or calling the API:

```bash
python3 scripts/transcribe_interview.py inspect "data/raw/INTERVIEW.mp4"
```

Prepare a validation sample without calling the API:

```bash
python3 scripts/transcribe_interview.py prepare \
  "data/raw/INTERVIEW.mp4" --kind sample
```

Transcribe the validation sample:

```bash
read -rsp 'OpenAI API key: ' OPENAI_API_KEY
echo
export OPENAI_API_KEY
python3 scripts/transcribe_interview.py sample "data/raw/INTERVIEW.mp4"
unset OPENAI_API_KEY
```

The sample defaults to the first 480 seconds. Use `--start SECONDS` and
`--duration SECONDS` to select a different passage and length. Review the
sample's wording and speaker labels before processing the full recording.

Transcribe the full interview:

```bash
read -rsp 'OpenAI API key: ' OPENAI_API_KEY
echo
export OPENAI_API_KEY
python3 scripts/transcribe_interview.py full "data/raw/INTERVIEW.mp4"
unset OPENAI_API_KEY
```

For a recording longer than the safe per-request model limit, `full` prepares
and processes local chunks automatically. No separate merge command is needed.
Rerun the same command after an interruption to reuse completed chunks. Do not
use `--force` when resuming: it re-prepares the audio and resends every chunk.

If a secret manager already sets `OPENAI_API_KEY`, skip the `read`, `export`, and
`unset` lines. Do not paste the key into chat, source code, or shell history.

To regenerate the MAXQDA and plain TXT files from an existing transcript, with
no API key and no audio upload:

```bash
python3 scripts/transcribe_interview.py export \
  "data/results/INTERVIEW_SLUG/full/transcript.json"
```

The `export` command writes the TXT files beside `transcript.json` and updates
their paths in `run_manifest.json`. It does not alter `transcript.json` or the
existing `transcript.txt`.

## Outputs

Each `sample` or `full` run writes these files under
`data/results/<interview>/<sample-or-full>/`:

- `api_raw.json`: combined streaming response, including original API events
  in `stream_events`.
- `transcript.json`: normalized segments, timestamps relative to the original
  recording, and speaker labels.
- `transcript.txt`: readable paragraphs with precise timestamp ranges.
- `transcript_maxqda.txt`: one `[hh:mm:ss]` timestamp at the end of each
  paragraph.
- `transcript_maxqda_precise.txt`: one `(h:mm:ss.xx)` timestamp at the end of
  each paragraph.
- `transcript_plain.txt`: the same paragraphs without timestamps.
- `manual_review.txt`: a heuristic list of passages to replay.
- `run_manifest.json`: model, settings, checksums, source information, and
  output paths.

Locally split interviews also have a `chunks/` directory containing each
completed raw API response. `manual_review.txt` includes the local join points.

The diarization model does not return word-level confidence scores with
`diarized_json`. Treat the transcript as a draft. The review list is not
exhaustive; mark unclear passages as `[unverständlich]` only after listening.
The script does not summarize or polish the API's transcription text.

## MAXQDA import

First, select **Import → Transcripts → Transcript with Timestamps** and import
`transcript_maxqda.txt`. When MAXQDA asks for the media file, choose the local
recording corresponding to the transcript. For the first interview in this
project, `data/work/interview-19-established-no5-7-mar-2026/audio/full_56k.m4a`
is available. Test whether clicking a timestamp opens the correct audio
position.

Import `transcript_maxqda_precise.txt` separately if you want to test the
more precise format. Use `transcript_plain.txt` for an import without timestamp
links. Both timestamped formats use paragraph end times. The whole-second
format drops the fraction; the precise format rounds to a hundredth of a
second.

MAXQDA documentation: https://www.maxqda.com/help/import/transcripts

## Cost and limits (information checked on 2026-09-20)

The API documentation specifies a 25 MB file limit. The model page listed
audio input at USD 2.50 and text output at USD 10.00 per million tokens. The
public pricing page did not give a separate per-minute rate for the diarized
model. An earlier planning estimate used the approximately USD 0.006/minute
rate for `gpt-4o-transcribe` as a proxy; actual token-based charges can differ.

The public documentation did not list a maximum duration for one request. The
API returned a 1,400-second limit for `gpt-4o-transcribe-diarize` when this
project submitted its first full interview, so the script leaves a safety
margin below that observed limit.

The initial proxy estimate was about USD 0.30 for the first recording, USD
5.40 for 15 hours, and USD 0.05 for its validation sample. The observed charge
for that initial 480-second diarization request was USD 0.10; the client timed
out before receiving the result. If that observed rate is representative,
rough planning figures are USD 0.63 for the first recording and USD 11.25 for
15 hours. Failed or repeated requests may add charges. Check current pricing
and actual project usage before processing a larger collection.

## Research-data note

Before sending research recordings to an external service, confirm that the
participants' consent, ethics approval, data-processing agreement, and
institutional rules allow the chosen provider. Prepared audio and transcripts
may contain personal data and need the same protection as the originals.
