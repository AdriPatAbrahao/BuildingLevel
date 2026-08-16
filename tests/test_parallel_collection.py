from types import SimpleNamespace

import pytest

import main
from config.settings import ParallelConfig, RunConfig
from main import BuildingOptimizer


class _FakeWorkerPool:
    def __init__(self, **_kwargs):
        self._next_job_id = 1
        self._pending = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def submit(self, _column_polygons, _beam_definitions):
        job_id = self._next_job_id
        self._next_job_id += 1
        self._pending.append(job_id)
        return job_id

    def get_result(self, timeout=None):
        assert timeout is not None
        job_id = self._pending.pop(0)
        return SimpleNamespace(
            job_id=job_id,
            slot_name="TestSlot_01",
            steel=500.0 + job_id,
            concrete=10.0,
            is_valid=True,
            elapsed=0.01,
            success=True,
            error=None,
        )


def test_parallel_collection_refills_after_seed_and_candidate_duplicates(
    monkeypatch,
):
    """A rejected variation must not leave the worker pipeline empty."""
    monkeypatch.setattr(main, "TQSWorkerPool", _FakeWorkerPool)
    monkeypatch.setattr(RunConfig, "CHECKPOINTS_ENABLED", False)
    monkeypatch.setattr(RunConfig, "RESUME_FROM_CHECKPOINT", False)
    monkeypatch.setattr(RunConfig, "MAX_ITERATION_FACTOR", 5)
    monkeypatch.setattr(ParallelConfig, "NUM_WORKERS", 1)
    monkeypatch.setattr(ParallelConfig, "TIMEOUT_SEC", 10)

    optimizer = BuildingOptimizer.__new__(BuildingOptimizer)
    optimizer.num_target_samples = 3
    optimizer.use_vector_input = True
    optimizer._seen_segments_hash = set()
    optimizer._clf_features = []
    optimizer._clf_labels = []
    optimizer.generated_valid_configurations = []
    optimizer._heartbeat_ts = 0.0
    optimizer.segment_plotter = SimpleNamespace(plot_segments=lambda *_a, **_k: None)
    optimizer.length_processor = SimpleNamespace(
        process_segments=lambda segments: ([segments[0]["id"]], [])
    )
    optimizer._get_analysis_results = lambda _segments: (
        500.0,
        10.0,
        ["seed"],
        [],
        True,
    )
    optimizer._extract_feature_vector = lambda columns, _beams: [
        1.0 if columns[0] == "seed" else float(ord(columns[0]))
    ]

    seed = [{"id": "seed"}]
    variations = iter(
        [
            [{"id": "seed"}],  # duplicate of the seed
            [{"id": "a"}],
            [{"id": "a"}],     # duplicate of a submitted candidate
            [{"id": "b"}],
        ]
    )
    calls = []

    def _variation(_base, strategy):
        calls.append(strategy)
        return next(variations)

    optimizer._generate_segment_variation = _variation

    features, outputs = optimizer._collect_training_data_parallel(seed)

    assert len(calls) == 4
    assert features == [[1.0], [97.0], [98.0]]
    assert outputs == [[500.0], [501.0], [502.0]]
    assert optimizer._clf_labels == [1, 1, 1]


@pytest.mark.parametrize(
    ("features", "outputs", "message"),
    [
        ([[1.0] * 22], [[500.0]], "feature shape"),
        ([[1.0] * 23], [[float("nan")]], "NaN or infinite"),
    ],
)
def test_collected_data_validation_rejects_malformed_rows(
    features, outputs, message
):
    optimizer = BuildingOptimizer.__new__(BuildingOptimizer)

    with pytest.raises(RuntimeError, match=message):
        optimizer._validate_collected_data(features, outputs)
