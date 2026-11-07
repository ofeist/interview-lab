import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

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
        selected = transcription.select_core_segments(segments, 100.0, 200.0)

        self.assertEqual(selected, segments[1:])


class OutputTests(unittest.TestCase):
    def test_maxqda_exports_use_paragraph_end_times(self) -> None:
        segments = [
            {"start": 950.0, "end": 952.0, "speaker": "S1", "text": "Pseudonym"},
            {"start": 952.1, "end": 954.684, "speaker": "S1", "text": "hast du gemeint."},
            {"start": 955.0, "end": 962.23, "speaker": "S2", "text": "Ja, genau."},
        ]

        self.assertEqual(
            transcription.render_maxqda_text(segments, precise=False),
            "S1: Pseudonym hast du gemeint. [00:15:54]\n\n"
            "S2: Ja, genau. [00:16:02]\n",
        )
        self.assertEqual(
            transcription.render_maxqda_text(segments, precise=True),
            "S1: Pseudonym hast du gemeint. (0:15:54.68)\n\n"
            "S2: Ja, genau. (0:16:02.23)\n",
        )
        self.assertEqual(
            transcription.render_maxqda_text(segments, precise=None),
            "S1: Pseudonym hast du gemeint.\n\nS2: Ja, genau.\n",
        )

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
        self.assertIn("00:03:20.000 left", stdout.getvalue())


def sse(*events: str) -> io.BytesIO:
    return io.BytesIO("".join(f"{event}\n\n" for event in events).encode("utf-8"))


def collect_quietly(stream: io.BytesIO, total_duration: float = 10.0) -> dict:
    with redirect_stdout(io.StringIO()):
        return transcription.collect_streamed_transcription(stream, total_duration)


class StreamParsingTests(unittest.TestCase):
    def test_event_name_and_multiline_data_are_combined(self) -> None:
        stream = io.BytesIO(
            b"event: transcript.text.segment\r\n"
            b'data: {"start":0,"end":4,\r\n'
            b'data: "speaker":"A","text":"Hallo"}\r\n'
            b"\r\n"
            b'data: {"type":"transcript.text.done","text":"Hallo",'
            b'"usage":{"seconds":4}}\r\n'
            b"\r\n"
            b"data: [DONE]\r\n"
            b"\r\n"
        )

        result = collect_quietly(stream)

        self.assertEqual(len(result["segments"]), 1)
        self.assertEqual(result["segments"][0]["text"], "Hallo")
        self.assertEqual(result["duration"], 4.0)
        self.assertEqual(result["usage"], {"seconds": 4})
        self.assertEqual(len(result["stream_events"]), 2)

    def test_final_event_without_trailing_blank_line_is_processed(self) -> None:
        stream = io.BytesIO(
            b'data: {"type":"transcript.text.segment","start":0,"end":2,'
            b'"speaker":"A","text":"Ja"}\n\n'
            b'data: {"type":"transcript.text.done","text":""}'
        )

        result = collect_quietly(stream)

        self.assertEqual(result["text"], "Ja")

    def test_progress_without_original_timeline(self) -> None:
        stdout = io.StringIO()
        stream = sse(
            'data: {"type":"transcript.text.segment","start":0,"end":5,'
            '"speaker":"A","text":"Test"}',
            'data: {"type":"transcript.text.done","text":"Test"}',
        )

        with redirect_stdout(stdout):
            transcription.collect_streamed_transcription(stream, 10.0, progress_label="S ")

        self.assertIn("S 1 seg |  50.0% | 00:00:05.000 audio left", stdout.getvalue())

    def test_stream_failures_stop_the_run(self) -> None:
        segment = (
            'data: {"type":"transcript.text.segment","start":0,"end":1,'
            '"speaker":"A","text":"x"}'
        )
        cases = {
            "invalid streaming event": sse("data: {not json"),
            "non-object streaming event": sse("data: [1, 2]"),
            "streaming error": sse('data: {"type":"error","message":"boom"}'),
            "ended before transcript.text.done": sse(segment),
            "no diarized speaker segments": sse(
                'data: {"type":"transcript.text.done","text":""}'
            ),
        }
        for message, stream in cases.items():
            with self.subTest(message=message):
                with self.assertRaises(SystemExit) as raised:
                    collect_quietly(stream)
                self.assertIn(message, str(raised.exception))


class BoundaryTests(unittest.TestCase):
    def test_short_recording_is_one_chunk(self) -> None:
        self.assertEqual(transcription.choose_chunk_boundaries(600.0, [300.0]), [0.0, 600.0])

    def test_nearest_silence_inside_window_is_chosen(self) -> None:
        boundaries = transcription.choose_chunk_boundaries(2400.0, [1150.0, 1190.0, 1230.0])

        self.assertEqual(boundaries, [0.0, 1190.0, 2400.0])

    def test_without_nearby_silence_the_ideal_split_is_used(self) -> None:
        boundaries = transcription.choose_chunk_boundaries(2400.0, [500.0, 1900.0])

        self.assertEqual(boundaries, [0.0, 1200.0, 2400.0])


class SpeakerLabelTests(unittest.TestCase):
    def test_labels_sort_numerically_and_fill_gaps(self) -> None:
        labels = sorted(["S10", "unknown", "S2", "S1"], key=transcription.speaker_sort_key)

        self.assertEqual(labels, ["S1", "S2", "S10", "unknown"])
        self.assertEqual(transcription.next_speaker_label({"S1", "S3"}), "S2")
        self.assertEqual(transcription.next_speaker_label(set()), "S1")

    def test_initial_mapping_follows_first_appearance(self) -> None:
        segments = [{"speaker_raw": label} for label in ("B", "A", "B", "C")]

        self.assertEqual(
            transcription.initial_speaker_mapping(segments),
            {"B": "S1", "A": "S2", "C": "S3"},
        )

    def test_known_speaker_returned_by_api_keeps_its_label(self) -> None:
        current = [{"start": 0.0, "end": 5.0, "speaker_raw": "S2"}]

        mapping, evidence = transcription.reconcile_speaker_mapping(current, [], {"S1", "S2"})

        self.assertEqual(mapping, {"S2": "S2"})
        self.assertIn("known speaker", evidence["S2"])

    def test_single_unmatched_speaker_is_inferred(self) -> None:
        previous = [{"start": 90.0, "end": 100.0, "speaker": "S1"}]
        current = [
            {"start": 90.0, "end": 100.0, "speaker_raw": "A"},
            {"start": 200.0, "end": 210.0, "speaker_raw": "B"},
        ]

        mapping, evidence = transcription.reconcile_speaker_mapping(
            current, previous, {"S1", "S2"}
        )

        self.assertEqual(mapping, {"A": "S1", "B": "S2"})
        self.assertIn("only unmatched", evidence["B"])

    def test_unresolved_speakers_get_new_labels(self) -> None:
        current = [
            {"start": 200.0, "end": 210.0, "speaker_raw": "A"},
            {"start": 210.0, "end": 220.0, "speaker_raw": "B"},
        ]

        mapping, evidence = transcription.reconcile_speaker_mapping(
            current, [], {"S1", "S2"}
        )

        self.assertEqual(mapping, {"A": "S3", "B": "S4"})
        self.assertIn("manual continuity review", evidence["A"])

    def test_tiny_overlap_is_not_evidence(self) -> None:
        previous = [{"start": 90.0, "end": 100.0, "speaker": "S1"}]
        current = [
            {"start": 99.95, "end": 110.0, "speaker_raw": "A"},
            {"start": 110.0, "end": 120.0, "speaker_raw": "B"},
        ]

        mapping, _ = transcription.reconcile_speaker_mapping(current, previous, {"S1"})

        self.assertEqual(mapping, {"A": "S2", "B": "S3"})

    def test_speaker_missing_from_sample_gets_new_label(self) -> None:
        sample = [{"start": 0.0, "end": 5.0, "speaker": "S1"}]
        current = [
            {"start": 0.0, "end": 5.0, "speaker_raw": "A"},
            {"start": 600.0, "end": 610.0, "speaker_raw": "B"},
        ]

        mapping, _ = transcription.map_speakers_from_sample(current, sample)

        self.assertEqual(mapping, {"A": "S1", "B": "S2"})

    def test_mapping_prefixes_ids_with_chunk_index(self) -> None:
        segments = [{"id": "seg_0", "speaker_raw": "A", "start": 0.0, "end": 1.0}]

        mapped = transcription.apply_speaker_mapping(segments, {"A": "S1"}, 3)

        self.assertEqual(mapped[0]["id"], "chunk-003-seg_0")
        self.assertEqual(mapped[0]["speaker"], "S1")
        self.assertEqual(mapped[0]["chunk_index"], 3)


class ChunkMergeTests(unittest.TestCase):
    def test_chunk_segments_move_to_original_timeline(self) -> None:
        raw = {"segments": [{"start": 1, "end": 2.5, "speaker": "A", "text": " Hallo "}]}
        chunk = {"index": 2, "upload_start_seconds": 1190.0}

        segments = transcription.absolute_chunk_segments(raw, chunk)

        self.assertEqual(segments[0]["start"], 1191.0)
        self.assertEqual(segments[0]["end"], 1192.5)
        self.assertEqual(segments[0]["speaker_raw"], "A")
        self.assertEqual(segments[0]["text"], "Hallo")
        self.assertEqual(segments[0]["id"], "segment-1")

    def test_invalid_chunk_segments_stop_the_run(self) -> None:
        chunk = {"index": 2, "upload_start_seconds": 0.0}
        for raw in ({"segments": []}, {"segments": [{"start": "x", "end": 1}]}):
            with self.subTest(raw=raw):
                with self.assertRaises(SystemExit):
                    transcription.absolute_chunk_segments(raw, chunk)

    def test_last_chunk_keeps_segments_after_core_end(self) -> None:
        segments = [{"start": 199.0, "end": 203.0}]
        self.assertEqual(transcription.select_core_segments(segments, 100.0, None), segments)
        self.assertEqual(transcription.select_core_segments(segments, 100.0, 200.0), [])

    def merge_join(self, previous: list[tuple], current: list[tuple]) -> tuple[float, list]:
        def segments(items: list[tuple], chunk: str) -> list[dict]:
            return [
                {"start": start, "end": end, "text": text, "chunk": chunk}
                for start, end, text in items
            ]

        previous_segments = segments(previous, "previous")
        current_segments = segments(current, "current")
        cut = transcription.choose_join_cut(
            previous_segments, current_segments, 1250.0, 1240.0, 1260.0
        )
        merged = transcription.select_core_segments(
            previous_segments, 0.0, cut
        ) + transcription.select_core_segments(current_segments, cut, None)
        return cut, [(segment["chunk"], segment["text"]) for segment in merged]

    def test_join_keeps_speech_that_chunks_segment_differently(self) -> None:
        # Each chunk alone would drop "Grenze": its midpoints fall on opposite
        # sides of the 1250 s boundary.
        cut, merged = self.merge_join(
            [(1200, 1245, "Vorher"), (1245, 1257, "Grenze"), (1257, 1262, "Danach")],
            [(1243, 1255, "Grenze"), (1255, 1260, "Danach"), (1260, 1300, "Ende")],
        )

        self.assertEqual([text for _, text in merged], ["Vorher", "Grenze", "Danach", "Ende"])
        self.assertTrue(1245.0 <= cut <= 1255.0)

    def test_join_does_not_repeat_speech_that_chunks_segment_differently(self) -> None:
        # Each chunk alone would keep "Grenze": its midpoints fall on opposite
        # sides of the 1250 s boundary.
        _, merged = self.merge_join(
            [(1200, 1243, "Vorher"), (1243, 1249, "Grenze"), (1249, 1258, "Danach")],
            [(1242.8, 1251, "Grenze"), (1251, 1258, "Danach"), (1258, 1300, "Ende")],
        )

        self.assertEqual([text for _, text in merged], ["Vorher", "Grenze", "Danach", "Ende"])

    def test_join_moves_to_a_shared_edge_near_the_boundary(self) -> None:
        cut, merged = self.merge_join(
            [(1240, 1248.0, "Eins"), (1248.1, 1254.5, "Zwei")],
            [(1240.2, 1248.1, "Eins"), (1248.1, 1254.2, "Zwei"), (1254.2, 1300, "Drei")],
        )

        self.assertEqual(cut, 1248.1)
        self.assertEqual(
            merged, [("previous", "Eins"), ("current", "Zwei"), ("current", "Drei")]
        )

    def test_join_uses_the_whole_shared_audio(self) -> None:
        # No edge lies within 5 s of the boundary; the clean cuts are 6-7 s away.
        cut, merged = self.merge_join(
            [(1235, 1243, "Vorher"), (1243, 1257, "Grenze"), (1257, 1260, "Danach")],
            [(1242, 1256, "Grenze"), (1256, 1260, "Danach")],
        )

        self.assertEqual([text for _, text in merged], ["Vorher", "Grenze", "Danach"])
        self.assertTrue(1241.0 <= cut <= 1259.0)

    def test_join_ignores_edges_near_the_shared_audio_edges(self) -> None:
        cut, _ = self.merge_join(
            [(1200, 1240.5, "Früh"), (1240.5, 1259.5, "Lang")],
            [(1240, 1259.5, "Lang"), (1259.5, 1300, "Spät")],
        )

        self.assertEqual(cut, 1250.0)


class ChunkCacheTests(unittest.TestCase):
    signature = {
        "audio_sha256": "abc",
        "model": transcription.MODEL,
        "language": transcription.LANGUAGE,
        "response_format": transcription.RESPONSE_FORMAT,
        "chunking_strategy": transcription.CHUNKING_STRATEGY,
        "speaker_references": [],
    }

    def write_cache(self, directory: str, content: str) -> Path:
        path = Path(directory) / "chunk_001_api_raw.json"
        path.write_text(content, encoding="utf-8")
        return path

    def test_matching_cache_is_reused(self) -> None:
        raw = {"segments": [{"start": 0}], "_local_request": self.signature}
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_cache(directory, json.dumps(raw))

            self.assertEqual(transcription.load_cached_chunk(path, self.signature), raw)

    def test_reference_changes_do_not_invalidate_cache(self) -> None:
        raw = {"segments": [{"start": 0}], "_local_request": self.signature}
        signature = {**self.signature, "speaker_references": [{"name": "S1", "sha256": "x"}]}
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_cache(directory, json.dumps(raw))

            self.assertEqual(transcription.load_cached_chunk(path, signature), raw)

    def test_unusable_cache_is_ignored(self) -> None:
        cases = {
            "missing": None,
            "corrupt": "{",
            "no signature": json.dumps({"segments": [{"start": 0}]}),
            "no segments": json.dumps({"segments": [], "_local_request": self.signature}),
            "other audio": json.dumps(
                {
                    "segments": [{"start": 0}],
                    "_local_request": {**self.signature, "audio_sha256": "def"},
                }
            ),
            "other model": json.dumps(
                {
                    "segments": [{"start": 0}],
                    "_local_request": {**self.signature, "model": "other"},
                }
            ),
        }
        for name, content in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "chunk_001_api_raw.json"
                if content is not None:
                    path.write_text(content, encoding="utf-8")

                self.assertIsNone(transcription.load_cached_chunk(path, self.signature))


class ChunkedTranscriptionTests(unittest.TestCase):
    """Run the full chunk loop with the API, ffmpeg and output rendering replaced."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.source = root / "raw" / "Interview 1.mp4"
        self.chunks = [
            {
                "index": 1,
                "path": str(root / "chunk_001.m4a"),
                "sha256": "chunk-1",
                "duration_seconds": 1260.0,
                "upload_start_seconds": 0.0,
                "upload_end_seconds": 1260.0,
                "core_start_seconds": 0.0,
                "core_end_seconds": 1250.0,
            },
            {
                "index": 2,
                "path": str(root / "chunk_002.m4a"),
                "sha256": "chunk-2",
                "duration_seconds": 1260.0,
                "upload_start_seconds": 1240.0,
                "upload_end_seconds": 2500.0,
                "core_start_seconds": 1250.0,
                "core_end_seconds": 2500.0,
            },
        ]
        self.manifest = {
            "source": {
                "path": str(self.source),
                "duration_seconds": 2500.0,
                "sha256": "source",
            },
            "prepared_audio": {},
        }
        # Chunk-relative times; both chunks hear "Grenze" and "Danach" in the overlap.
        self.responses = {
            self.chunks[0]["path"]: [
                (0, 10, "A", "Hallo"),
                (10, 20, "B", "Servus"),
                (1245, 1252, "A", "Grenze"),
                (1253, 1258, "B", "Danach"),
            ],
            self.chunks[1]["path"]: [
                (5, 12, "X", "Grenze"),
                (13, 18, "Y", "Danach"),
                (100, 110, "X", "Ende"),
            ],
        }
        self.rendered: list[dict] = []
        for name, value in (
            ("RESULTS_ROOT", root / "results"),
            ("WORK_ROOT", root / "work"),
        ):
            patcher = mock.patch.object(transcription, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def fake_api(self, audio_path: Path, *args: object, **kwargs: object) -> dict:
        return {
            "segments": [
                {"start": start, "end": end, "speaker": speaker, "text": text}
                for start, end, speaker, text in self.responses[str(audio_path)]
            ],
            "usage": {"chunk": Path(audio_path).name},
        }

    def run_chunked(self, api: mock.Mock) -> dict:
        def render(raw: dict, manifest: dict) -> dict:
            self.rendered.append(raw)
            return {}

        with mock.patch.object(
            transcription, "prepare_chunked_audio", return_value=(self.chunks, self.manifest)
        ), mock.patch.object(transcription, "call_api", api), mock.patch.object(
            transcription, "build_speaker_references", return_value=[]
        ), mock.patch.object(
            transcription, "render_results", side_effect=render
        ), redirect_stdout(
            io.StringIO()
        ):
            transcription.transcribe_chunked(self.source, force=False)
        return self.rendered[-1]

    def test_overlap_is_merged_once_with_continuous_speakers(self) -> None:
        api = mock.Mock(side_effect=self.fake_api)

        merged = self.run_chunked(api)

        self.assertEqual(api.call_count, 2)
        self.assertEqual(
            [(segment["speaker"], segment["text"]) for segment in merged["segments"]],
            [
                ("S1", "Hallo"),
                ("S2", "Servus"),
                ("S1", "Grenze"),
                ("S2", "Danach"),
                ("S1", "Ende"),
            ],
        )
        self.assertEqual(merged["segments"][-1]["start"], 1340.0)
        self.assertEqual(merged["chunks"][1]["speaker_mapping"], {"X": "S1", "Y": "S2"})
        self.assertEqual(merged["chunks"][0]["merge_end_seconds"], 1252.0)
        self.assertNotIn("merge_end_seconds", merged["chunks"][1])
        self.assertEqual(merged["text"], "Hallo Servus Grenze Danach Ende")

    def test_overlap_segmented_differently_is_merged_once(self) -> None:
        # Absolute ranges: chunk 1 hears "Grenze" at 1245-1257, chunk 2 at
        # 1243-1255, so their midpoints fall on opposite sides of the 1250 s join.
        variants = {
            "speech would be lost": ((1245, 1257), (3, 15)),
            "speech would be repeated": ((1243, 1249), (2.8, 11)),
            "chunk 1 starts later": ((1247, 1253), (6.5, 12.5)),
        }
        for name, (first, second) in variants.items():
            with self.subTest(name=name):
                self.responses[self.chunks[0]["path"]][2] = (*first, "A", "Grenze")
                self.responses[self.chunks[1]["path"]][0] = (*second, "X", "Grenze")

                merged = self.run_chunked(mock.Mock(side_effect=self.fake_api))

                self.assertEqual(merged["text"], "Hallo Servus Grenze Danach Ende")
                self.assertEqual(
                    [segment["speaker"] for segment in merged["segments"]],
                    ["S1", "S2", "S1", "S2", "S1"],
                )
                for path in transcription.RESULTS_ROOT.glob("*/full/chunks/*.json"):
                    path.unlink()

    def test_second_run_reuses_saved_chunks(self) -> None:
        self.run_chunked(mock.Mock(side_effect=self.fake_api))
        unavailable_api = mock.Mock(side_effect=AssertionError("API must not be called"))

        merged = self.run_chunked(unavailable_api)

        unavailable_api.assert_not_called()
        self.assertEqual(len(merged["segments"]), 5)

    def test_interrupted_run_resends_only_missing_chunks(self) -> None:
        def fail_on_second_chunk(audio_path: Path, *args: object, **kwargs: object) -> dict:
            if str(audio_path) == self.chunks[1]["path"]:
                raise SystemExit("Error: connection lost")
            return self.fake_api(audio_path)

        with self.assertRaises(SystemExit):
            self.run_chunked(mock.Mock(side_effect=fail_on_second_chunk))
        api = mock.Mock(side_effect=self.fake_api)

        self.run_chunked(api)

        self.assertEqual(
            [str(call.args[0]) for call in api.call_args_list], [self.chunks[1]["path"]]
        )

    def test_validation_sample_sets_first_chunk_labels(self) -> None:
        sample_path = transcription.RESULTS_ROOT / "interview-1" / "sample" / "transcript.json"
        sample_path.parent.mkdir(parents=True)
        sample_path.write_text(
            json.dumps(
                {
                    "source": {"sha256": "source"},
                    "segments": [
                        {"start": 0.0, "end": 10.0, "speaker": "S2", "text": "Hallo"},
                        {"start": 10.0, "end": 20.0, "speaker": "S1", "text": "Servus"},
                    ],
                }
            ),
            encoding="utf-8",
        )

        merged = self.run_chunked(mock.Mock(side_effect=self.fake_api))

        self.assertEqual(
            [segment["speaker"] for segment in merged["segments"]],
            ["S2", "S1", "S2", "S1", "S2"],
        )


class SampleLoadingTests(unittest.TestCase):
    def load(self, sample: object | None) -> list[dict]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            if sample is not None:
                path = root / "interview" / "sample" / "transcript.json"
                path.parent.mkdir(parents=True)
                path.write_text(
                    sample if isinstance(sample, str) else json.dumps(sample),
                    encoding="utf-8",
                )
            manifest = {"source": {"path": "raw/Interview.mp4", "sha256": "source"}}
            with mock.patch.object(transcription, "RESULTS_ROOT", root), redirect_stdout(
                io.StringIO()
            ):
                segments, _ = transcription.load_sample_segments(manifest)
        return segments

    def test_matching_sample_is_loaded(self) -> None:
        segments = self.load(
            {
                "source": {"sha256": "source"},
                "segments": [{"start": "1.5", "end": 2, "speaker": "S1", "text": "Ja"}],
            }
        )

        self.assertEqual(segments, [{"start": 1.5, "end": 2.0, "speaker": "S1", "text": "Ja"}])

    def test_unusable_sample_is_ignored(self) -> None:
        cases = {
            "missing": None,
            "corrupt": "{",
            "other source": {
                "source": {"sha256": "other"},
                "segments": [{"start": 0, "end": 1, "speaker": "S1"}],
            },
            "no segments": {"source": {"sha256": "source"}, "segments": []},
            "invalid segment": {
                "source": {"sha256": "source"},
                "segments": [{"start": 0, "end": 1}],
            },
        }
        for name, sample in cases.items():
            with self.subTest(name=name):
                self.assertEqual(self.load(sample), [])


class FormattingTests(unittest.TestCase):
    def test_timestamps(self) -> None:
        self.assertEqual(transcription.format_timestamp(3725.4567), "01:02:05.457")
        self.assertEqual(transcription.format_timestamp(-1.0), "00:00:00.000")
        self.assertEqual(transcription.compact_timestamp(3725.4567), "010205-457")

    def test_maxqda_timestamps_round_and_truncate(self) -> None:
        self.assertEqual(
            transcription.format_maxqda_timestamp(59.999, precise=True), "(0:01:00.00)"
        )
        self.assertEqual(
            transcription.format_maxqda_timestamp(59.999, precise=False), "[00:00:59]"
        )
        self.assertEqual(
            transcription.format_maxqda_timestamp(36000.0, precise=True), "(10:00:00.00)"
        )

    def test_bitrate_fits_upload_limit(self) -> None:
        self.assertEqual(transcription.choose_bitrate(600.0), 96)
        bitrate = transcription.choose_bitrate(3600.0)
        self.assertLessEqual(bitrate * 1000 / 8 * 3600.0, transcription.TARGET_UPLOAD_BYTES)
        with self.assertRaises(SystemExit):
            transcription.choose_bitrate(10 * 3600.0)

    def test_paragraphs_merge_only_close_turns_of_one_speaker(self) -> None:
        segments = [
            {"start": 0.0, "end": 5.0, "speaker": "S1", "text": "Eins"},
            {"start": 6.0, "end": 10.0, "speaker": "S1", "text": "zwei"},
            {"start": 13.0, "end": 15.0, "speaker": "S1", "text": "Pause"},
            {"start": 15.0, "end": 16.0, "speaker": "S2", "text": ""},
            {"start": 16.0, "end": 100.0, "speaker": "S1", "text": "lang"},
            {"start": 100.5, "end": 110.0, "speaker": "S1", "text": "zu lang"},
        ]

        result = transcription.paragraphs(segments)

        self.assertEqual(
            [(item["start"], item["end"], item["text"]) for item in result],
            [
                (0.0, 10.0, "Eins zwei"),
                (13.0, 100.0, "Pause lang"),
                (100.5, 110.0, "zu lang"),
            ],
        )

    def test_backwards_timestamps_stop_maxqda_export(self) -> None:
        segments = [
            {"start": 10.0, "end": 20.0, "speaker": "S1", "text": "Später"},
            {"start": 5.0, "end": 8.0, "speaker": "S2", "text": "Früher"},
        ]

        with self.assertRaises(SystemExit):
            transcription.render_maxqda_text(segments, precise=False)
        self.assertEqual(
            transcription.render_maxqda_text(segments, precise=None),
            "S1: Später\n\nS2: Früher\n",
        )

    def test_normalization_renames_raw_labels_and_shifts_times(self) -> None:
        raw = {
            "segments": [
                {"start": 0.1234, "end": 1, "speaker": "A", "text": " Hallo "},
                {"start": 1, "end": 2, "speaker": "S1", "text": "Servus"},
                {"start": 2, "end": 3, "text": "Wer?"},
            ]
        }

        segments, mapping = transcription.normalized_segments(raw, 100.0)

        self.assertEqual(mapping, {"A": "S1", "S1": "S2", "unknown": "S3"})
        self.assertEqual(segments[0]["start"], 100.123)
        self.assertEqual(segments[0]["text"], "Hallo")
        with self.assertRaises(SystemExit):
            transcription.normalized_segments({"segments": []}, 0.0)

    def test_export_writes_three_formats_beside_transcript(self) -> None:
        transcript = {
            "segments": [{"start": 0.0, "end": 2.5, "speaker": "S1", "text": "Hallo"}]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcript.json"
            path.write_text(json.dumps(transcript), encoding="utf-8")

            outputs = transcription.export_transcript_formats(path)

            self.assertEqual(
                {label: output.read_text(encoding="utf-8") for label, output in outputs.items()},
                {
                    "maxqda": "S1: Hallo [00:00:02]\n",
                    "maxqda_precise": "S1: Hallo (0:00:02.50)\n",
                    "plain": "S1: Hallo\n",
                },
            )
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), transcript)


class ReviewCandidateTests(unittest.TestCase):
    def reasons(self, segments: list[dict], *args: object) -> list[list[str]]:
        return [item["reasons"] for item in transcription.review_candidates(segments, *args)]

    def test_uncertain_markers_are_flagged(self) -> None:
        for text in ("Das war [unverständlich].", "Ich (lacht) weiß nicht.", "inaudible"):
            with self.subTest(text=text):
                segments = [{"start": 0.0, "end": 5.0, "speaker": "S1", "text": text}]

                self.assertEqual(len(self.reasons(segments, 5.0)), 1)

    def test_fast_speech_and_overlap_are_flagged(self) -> None:
        segments = [
            {
                "start": 0.0,
                "end": 1.0,
                "speaker": "S1",
                "text": "eins zwei drei vier fünf sechs sieben acht",
            },
            {"start": 0.5, "end": 3.0, "speaker": "S2", "text": "Ja genau"},
        ]

        reasons = self.reasons(segments, 3.0)

        self.assertIn("extremely high estimated speech rate", reasons[0])
        self.assertIn("overlapping speaker segments", reasons[1])

    def test_missing_audio_end_and_chunk_joins_are_listed(self) -> None:
        segments = [
            {"start": 0.0, "end": 10.0, "speaker": "S1", "text": "Vorher"},
            {"start": 1198.0, "end": 1203.0, "speaker": "S2", "text": "an der Grenze"},
        ]

        candidates = transcription.review_candidates(segments, 1300.0, [1200.0])

        self.assertEqual(
            [(item["start"], item["end"], item["text"]) for item in candidates],
            [(1195.0, 1205.0, "an der Grenze"), (1203.0, 1300.0, "")],
        )

if __name__ == "__main__":
    unittest.main()
