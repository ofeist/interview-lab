import io
import unittest
from contextlib import redirect_stdout

from scripts import transcribe_interview as transcription


class ChunkingTests(unittest.TestCase):
    def test_boundaries_stay_below_model_limit(self) -> None:
        duration = 3024.507
        boundaries = transcription.choose_chunk_boundaries(
            duration,
            [1003.0, 1013.0, 2005.0, 2018.0],
        )

        self.assertEqual(boundaries[0], 0.0)
        self.assertEqual(boundaries[-1], duration)
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            upload_start = max(
                0.0,
                start - (transcription.LOCAL_CHUNK_OVERLAP_SECONDS if index else 0.0),
            )
            upload_end = min(
                duration,
                end
                + (
                    transcription.LOCAL_CHUNK_OVERLAP_SECONDS
                    if index < len(boundaries) - 2
                    else 0.0
                ),
            )
            self.assertLess(upload_end - upload_start, transcription.MODEL_MAX_AUDIO_SECONDS)

    def test_overlap_reconciles_speaker_labels(self) -> None:
        previous = [
            {"start": 90.0, "end": 96.0, "speaker": "S1"},
            {"start": 97.0, "end": 104.0, "speaker": "S2"},
        ]
        current = [
            {"start": 90.2, "end": 96.1, "speaker_raw": "B"},
            {"start": 97.1, "end": 104.2, "speaker_raw": "A"},
        ]

        mapping, _ = transcription.reconcile_speaker_mapping(
            current, previous, {"S1", "S2"}
        )

        self.assertEqual(mapping, {"B": "S1", "A": "S2"})

    def test_overlap_can_collapse_spurious_labels(self) -> None:
        previous = [{"start": 90.0, "end": 100.0, "speaker": "S1"}]
        current = [
            {"start": 90.0, "end": 95.0, "speaker_raw": "A"},
            {"start": 95.0, "end": 100.0, "speaker_raw": "B"},
        ]

        mapping, _ = transcription.reconcile_speaker_mapping(
            current, previous, {"S1"}
        )

        self.assertEqual(mapping, {"A": "S1", "B": "S1"})

    def test_validation_sample_collapses_spurious_labels(self) -> None:
        sample = [
            {"start": 0.0, "end": 5.0, "speaker": "S1"},
            {"start": 5.0, "end": 10.0, "speaker": "S2"},
        ]
        current = [
            {"start": 0.0, "end": 4.0, "speaker_raw": "A"},
            {"start": 4.0, "end": 5.0, "speaker_raw": "C"},
            {"start": 5.0, "end": 10.0, "speaker_raw": "B"},
        ]

        mapping, _ = transcription.map_speakers_from_sample(current, sample)

        self.assertEqual(mapping, {"A": "S1", "C": "S1", "B": "S2"})

    def test_core_selection_removes_overlap_duplicates(self) -> None:
        segments = [
            {"start": 95.0, "end": 99.0},
            {"start": 99.0, "end": 101.0},
            {"start": 101.0, "end": 105.0},
        ]
        chunk = {"core_start_seconds": 100.0, "core_end_seconds": 200.0}

        selected = transcription.select_core_segments(segments, chunk, False)

        self.assertEqual(selected, segments[1:])


class OutputTests(unittest.TestCase):
    def test_short_backchannels_are_not_review_candidates(self) -> None:
        segments = [
            {"start": 0.0, "end": 0.2, "speaker": "S1", "text": "Ja."},
            {"start": 1.0, "end": 1.2, "speaker": "S2", "text": "Mhm."},
        ]

        candidates = transcription.review_candidates(segments, 1.2)

        self.assertEqual(candidates, [])

    def test_normalization_preserves_global_speaker_labels(self) -> None:
        raw = {
            "segments": [
                {"start": 0, "end": 1, "speaker": "S1", "text": "Hallo"},
                {"start": 1, "end": 2, "speaker": "S2", "text": "Servus"},
            ]
        }

        segments, mapping = transcription.normalized_segments(raw, 0.0)

        self.assertEqual(mapping, {"S1": "S1", "S2": "S2"})
        self.assertEqual([segment["speaker"] for segment in segments], ["S1", "S2"])

    def test_stream_progress_uses_original_timeline(self) -> None:
        stream = io.BytesIO(
            b'data: {"type":"transcript.text.segment","id":"1","start":0,'
            b'"end":10,"speaker":"A","text":"Test"}\n\n'
            b'data: {"type":"transcript.text.done","text":"Test"}\n\n'
        )
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            result = transcription.collect_streamed_transcription(
                stream,
                100.0,
                progress_label="Chunk 2/3 - ",
                original_offset=90.0,
                original_duration=300.0,
                core_start=100.0,
                core_end=200.0,
            )

        self.assertEqual(result["segments"][0]["speaker"], "A")
        self.assertIn("total  33.3%", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
