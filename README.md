# Interview Lab

Transcribe interviews with speaker labels and timestamps, then export
formats for MAXQDA. The script uses `gpt-4o-transcribe-diarize` through the
OpenAI Transcriptions API. Transcripts are drafts that need audio review.

## Requirements

- Python 3.10 or newer; no third-party Python packages
- `ffmpeg` and `ffprobe` on `PATH`
- An OpenAI API key for `sample` and `full` commands

Run commands from the repository root. Place recordings in `data/raw/`.
The `data/raw/`, `data/work/`, `data/results/`, and `.env` paths are ignored by Git.
Do not put the API key in the repository or in a command argument.

## Transcribe an interview

Inspect a recording without calling the API:

```bash
python3 scripts/transcribe_interview.py inspect "data/raw/INTERVIEW.mp4"
```

Set the key in the current shell without displaying it or saving it in shell
history:

```bash
read -rsp "OpenAI API key: " OPENAI_API_KEY
echo
export OPENAI_API_KEY
```

Transcribe a validation sample and check its speakers and wording:

```bash
python3 scripts/transcribe_interview.py sample "data/raw/INTERVIEW.mp4"
```

Use `--start` and `--duration` to select a different section of the recording.

Then transcribe the full interview:

```bash
python3 scripts/transcribe_interview.py full "data/raw/INTERVIEW.mp4"
```

Long recordings are split near silences into API-sized chunks with a short
overlap. Each completed chunk is saved immediately. If a run is interrupted,
run the same `full` command again to reuse completed chunks. Do not add
`--force` when resuming. For long runs, a `tmux` session keeps the process
running after you disconnect from the terminal.

When finished, remove the key from the current shell if you entered it there:

```bash
unset OPENAI_API_KEY
```

## Files created

The original recording stays in `data/raw/`. Prepared audio, local chunks, and
speaker reference clips go into `data/work/<interview>/audio/`. Each `sample`
or `full` run creates `data/results/<interview>/<sample-or-full>/` containing:

| File | Purpose |
| --- | --- |
| `api_raw.json` | Combined API response and event record. |
| `transcript.json` | Segments with original-recording timestamps and `S1`, `S2`, etc. speaker labels. |
| `transcript.txt` | Readable draft with timestamp ranges. |
| `transcript_maxqda.txt` | One `[hh:mm:ss]` timestamp at the end of each paragraph. |
| `transcript_maxqda_precise.txt` | One `(h:mm:ss.xx)` timestamp at the end of each paragraph. |
| `transcript_plain.txt` | The same paragraphs without timestamps. |
| `manual_review.txt` | Locations flagged for audio review, including chunk joins. |
| `run_manifest.json` | Source, model settings, mappings, counts, and output paths. |

Long interviews also have a `chunks/` subdirectory with the saved raw API
response for each completed request. These files contain interview data and
remain outside Git.

The `transcript.json` file is the source for the three MAXQDA/plain exports.
To recreate those TXT files locally without an API key or transcription cost:

```bash
python3 scripts/transcribe_interview.py export \
  "data/results/INTERVIEW_SLUG/full/transcript.json"
```

## Import into MAXQDA

Use **Import → Transcripts → Transcript with Timestamps** and select
`transcript_maxqda.txt`. When asked, select the corresponding local recording.
For the first interview in this project, the prepared
`data/work/interview-19-established-no5-7-mar-2026/audio/full_56k.m4a` is
available. Check that clicking a timestamp opens the expected audio position.

You can separately test `transcript_maxqda_precise.txt` for more precise
timestamps. Use `transcript_plain.txt` when you want to import text without
timestamp links. MAXQDA's [transcript import guide](https://www.maxqda.com/help/import/transcripts)
lists its recognized timestamp formats.

## More detail

[Transcription procedure](docs/TRANSCRIPTION.md) covers preparation,
recovery, cost estimates, and research-data considerations.

Run the local checks with:

```bash
python3 -m unittest discover -s tests -v
```
