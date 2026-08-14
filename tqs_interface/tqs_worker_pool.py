"""
tqs_interface/tqs_worker_pool.py
================================
Parallel TQS execution pool using isolated building-slot directories.

Each worker is a long-lived subprocess permanently assigned to exactly one
building directory (e.g. ``C:\\TQS\\OptimBuilding_01``).  A round-robin job
dispatcher in the main process ensures that no two workers ever share a
directory — isolation is structural, requiring no file-level locks.

Architecture
------------
::

    Main process                     Worker subprocesses
    ────────────────                 ────────────────────────────────
    TQSWorkerPool.submit()  ──q_01──▶  _worker_main("OptimBuilding_01")
                            ──q_02──▶  _worker_main("OptimBuilding_02")
                            ──q_N ──▶  _worker_main("OptimBuilding_0N")
    TQSWorkerPool.get_result()  ◀── shared result_queue ──────────────

Each worker:
  1. Cleans leftover TQS files in its slot directory (DAT-pattern rules).
  2. Creates the structural model (columns + beams + slabs) via TQSModelManager.
  3. Executes TQS global processing (RunModel) on its dedicated slot.
  4. Reads steel / concrete from RESDES.HTM.
  5. Optionally checks structural validity via the error-reader DLLs.
  6. Returns a WorkerResult to the shared result queue.

Usage
-----
::

    from tqs_interface.tqs_worker_pool import TQSWorkerPool

    with TQSWorkerPool(num_workers=3) as pool:
        # Sliding-window dispatch — submit one job per worker upfront,
        # then replenish as results arrive.
        job_id = pool.submit(column_polygons, beam_definitions)
        result = pool.get_result()   # WorkerResult; blocks until ready

        # — or use the batch helper —
        results = pool.map([(polys, beams), ...])
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading as _threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from shapely.geometry import Polygon

from config.paths import PROJECT_ROOT, TQS_OUTPUT_DIR

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WorkerResult:
    """Outcome of a single TQS analysis job returned by a worker subprocess."""

    job_id:    int
    slot_name: str
    steel:     Optional[float] = None
    concrete:  Optional[float] = None
    is_valid:  bool            = True
    error:     Optional[str]   = None
    elapsed:   float           = 0.0

    @property
    def success(self) -> bool:
        """True when the job produced usable steel and concrete values."""
        return self.error is None and self.steel is not None


# ──────────────────────────────────────────────────────────────────────────────
# TQS execution with hard timeout
# ──────────────────────────────────────────────────────────────────────────────

def _run_model_with_timeout(
    run_model_fn: Callable[[str], None],
    slot_name: str,
    timeout_sec: int,
) -> None:
    """
    Run *run_model_fn(slot_name)* in a daemon thread with a hard timeout.

    ``job.Execute()`` inside ``RunModel`` is a blocking DLL call with no
    built-in timeout.  If TQS shows a modal dialog or crashes silently the
    worker subprocess would hang forever.  This wrapper:

    1. Runs ``RunModel`` in a background thread.
    2. Waits up to *timeout_sec* seconds.
    3. If still running, kills ``NTQSHTM.EXE`` so the DLL call unblocks.
    4. Raises ``TimeoutError`` so the caller can return an error result
       to the queue instead of blocking indefinitely.

    Parameters
    ----------
    run_model_fn : callable
        The ``RunModel`` function imported inside the worker subprocess.
    slot_name : str
        Building slot name passed through to ``RunModel``.
    timeout_sec : int
        Maximum seconds to allow before killing TQS and raising.
    """
    exc_holder: List[Optional[Exception]] = [None]

    def _target() -> None:
        try:
            run_model_fn(slot_name)
        except Exception as e:
            exc_holder[0] = e

    t = _threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        # TQS is hanging — kill the process to unblock the DLL call.
        import subprocess as _sp2
        _sp2.run(
            ["taskkill", "/F", "/IM", "NTQSHTM.EXE", "/T"],
            capture_output=True,
        )
        t.join(timeout=10.0)           # give the thread time to unblock
        raise TimeoutError(
            f"RunModel did not complete within {timeout_sec}s "
            f"for slot '{slot_name}' — TQS process killed."
        )

    if exc_holder[0] is not None:
        raise exc_holder[0]


# ──────────────────────────────────────────────────────────────────────────────
# Worker entry-point  (executes entirely inside a dedicated subprocess)
# ──────────────────────────────────────────────────────────────────────────────

def _worker_main(
    slot_name: str,
    job_q:    "mp.Queue",
    result_q: "mp.Queue",
    dat_dir:  str,      # serialised as str — Path not guaranteed picklable
    tqs_base: str,
    timeout_sec: int,
    validity_check_dll: bool = False,
) -> None:
    """
    Long-lived worker subprocess.

    Reads jobs from *job_q*, runs the full TQS pipeline for its dedicated
    slot, and pushes a :class:`WorkerResult` to *result_q*.

    Terminates when it receives ``None`` (poison-pill) from *job_q*.

    Parameters
    ----------
    slot_name   : Building slot name, e.g. ``"OptimBuilding_01"``.
    job_q       : Per-slot input queue; carries ``(job_id, col_polys, beam_defs)``
                  tuples or ``None`` (shutdown signal).
    result_q    : Shared output queue for :class:`WorkerResult` objects.
    dat_dir     : Path (as str) to directory containing DAT cleanup files.
    tqs_base    : Path (as str) to the root TQS directory (``C:\\TQS``).
    timeout_sec : Maximum seconds to wait for RESDES.HTM after RunModel.
    """
    # ── Ensure T: drive is mapped in this subprocess (subst T: C:\TQSWV26A) ──
    import subprocess as _sp
    _subst = _sp.run(["subst", "T:", r"C:\TQSWV26A"], capture_output=True)
    if _subst.returncode != 0:
        # Drive may already be mapped — query to confirm
        _check = _sp.run(["subst"], capture_output=True, text=True)
        if "T:\\" not in _check.stdout and "T:/" not in _check.stdout:
            print(f"[{slot_name}] WARNING: could not map T: drive — TQS may fail.")

    # ── Deferred imports: TQS DLL is only initialised inside the subprocess ──
    try:
        from tqs_interface.tqs_manager import TQSModelManager
        from tqs_interface.tqs_exec import RunModel
        from tqs_interface.tqs_errors import TQSErrorReader
        from results.resultsext import extract_material_summary
        from utils.file_handler import cleanup_building_files
        from TQS import TQSUtil
    except ImportError as exc:
        # Can't use TQSUtil before it's imported — fall back to print.
        print(
            f"[{slot_name}] FATAL: import failed — {exc}\n"
            f"{traceback.format_exc()}"
        )
        result_q.put(
            WorkerResult(job_id=-1, slot_name=slot_name,
                         error=f"import failed: {exc}")
        )
        return

    _dat_dir  = Path(dat_dir)
    _tqs_base = Path(tqs_base)
    pid       = mp.current_process().pid

    # Each worker owns a persistent TQSModelManager.  The manager reuses the
    # same object across jobs — create_building_model_and_elements() already
    # closes any previously open model at its start.
    manager      = TQSModelManager(building_name=slot_name)
    error_reader = TQSErrorReader()

    TQSUtil.writef(
        f"[{slot_name}] Worker ready "
        f"(PID={pid}, slot_dir={_tqs_base / slot_name})."
    )

    while True:
        job = job_q.get()          # blocks until a job or poison-pill arrives

        if job is None:
            TQSUtil.writef(f"[{slot_name}] Shutdown signal - exiting.")
            break

        job_id, column_polygons, beam_definitions = job
        t0 = time.perf_counter()

        TQSUtil.writef(
            f"[{slot_name}] >> Job #{job_id} - "
            f"{len(column_polygons)} col(s), {len(beam_definitions)} beam(s)."
        )

        try:
            # ── 1. Clean leftover files in this slot's directory ──────────
            cleanup_building_files(
                building_name=slot_name,
                tqs_base_dir=_tqs_base,
                dat_dir=_dat_dir,
            )

            # ── 2. Create the structural model ────────────────────────────
            ok = manager.create_building_model_and_elements(
                column_polygons, beam_definitions
            )
            if not ok:
                raise RuntimeError(
                    "create_building_model_and_elements returned False"
                )

            # ── 3. Execute TQS global processing for THIS slot ────────────
            #    Wrapped with a hard timeout: if TQS hangs (modal dialog,
            #    silent crash), NTQSHTM.EXE is killed and TimeoutError is
            #    raised so the worker always returns a result to the queue.
            _run_model_with_timeout(RunModel, slot_name, timeout_sec)
            TQSUtil.writef(
                f"[{slot_name}] RunModel issued for job #{job_id}."
            )

            # ── 4. Wait for and read the results file ─────────────────────
            results_path = _tqs_base / slot_name / "ESPACIAL" / "RESDES.HTM"
            deadline     = time.time() + timeout_sec
            while not results_path.exists():
                if time.time() > deadline:
                    raise TimeoutError(
                        f"RESDES.HTM not produced within {timeout_sec}s "
                        f"for slot '{slot_name}'"
                    )
                time.sleep(0.5)

            raw = extract_material_summary(results_path)
            if raw is None or raw[0] is None or raw[1] is None:
                raise ValueError(
                    f"Could not parse steel/concrete from RESDES.HTM "
                    f"for '{slot_name}'"
                )

            steel    = float(str(raw[0]).replace(",", "."))
            concrete = float(str(raw[1]).replace(",", "."))

            # ── 5. Optional structural-validity check (DLL-based) ─────────
            is_valid = True
            if validity_check_dll and error_reader._dlls_available():
                try:
                    errors = error_reader.get_critical_errors(
                        building_name=slot_name
                    )
                    if errors:
                        is_valid = False
                        TQSUtil.writef(
                            f"[{slot_name}] Job #{job_id}: "
                            f"{len(errors)} critical error(s) -> invalid."
                        )
                except Exception:
                    pass  # DLL check is non-fatal

            elapsed = time.perf_counter() - t0
            TQSUtil.writef(
                f"[{slot_name}] OK Job #{job_id} in {elapsed:.1f}s - "
                f"aco={steel:.1f} kgf  concreto={concrete:.4f} m3  "
                f"valido={is_valid}"
            )
            result_q.put(
                WorkerResult(
                    job_id=job_id, slot_name=slot_name,
                    steel=steel, concrete=concrete,
                    is_valid=is_valid, elapsed=elapsed,
                )
            )

        except Exception as exc:
            elapsed  = time.perf_counter() - t0
            err_text = f"{exc}\n{traceback.format_exc()}"
            TQSUtil.writef(
                f"[{slot_name}] FAILED Job #{job_id} "
                f"after {elapsed:.1f}s: {err_text}"
            )
            result_q.put(
                WorkerResult(
                    job_id=job_id, slot_name=slot_name,
                    error=str(exc), elapsed=elapsed, is_valid=False,
                )
            )


# ──────────────────────────────────────────────────────────────────────────────
# Pool manager  (runs in the main process)
# ──────────────────────────────────────────────────────────────────────────────

class TQSWorkerPool:
    """
    Manages *num_workers* dedicated TQS worker subprocesses.

    Each worker is permanently assigned to one building slot
    (``{base_name}_01``, ``_02``, …) so that TQS directory access is
    completely isolated — no file-level locks are ever needed.

    Parameters
    ----------
    num_workers : int
        Number of parallel workers / isolated building directories.
        Recommended range: 2–6 (each consumes a TQS licence seat and RAM).
    base_name : str
        Prefix for slot names (default: ``"OptimBuilding"``).
    tqs_base_dir : Path, optional
        Root TQS directory.  Defaults to ``C:\\TQS`` from ``config.paths``.
    dat_dir : Path, optional
        Directory containing the DAT cleanup pattern files
        (``LIMPA ESPACIAL.DAT``, etc.).  Defaults to ``PROJECT_ROOT``.
    timeout_sec : int
        Per-job timeout (seconds) waiting for RESDES.HTM after RunModel.

    Examples
    --------
    ::

        with TQSWorkerPool(num_workers=3) as pool:
            results = pool.map([(col_polys, beam_defs), ...])
    """

    def __init__(
        self,
        num_workers:        int            = 2,
        base_name:          str            = "OptimBuilding",
        tqs_base_dir:       Optional[Path] = None,
        dat_dir:            Optional[Path] = None,
        timeout_sec:        int            = 180,
        validity_check_dll: bool           = False,
    ) -> None:
        self.num_workers        = num_workers
        self.base_name          = base_name
        self.tqs_base_dir       = Path(tqs_base_dir) if tqs_base_dir else TQS_OUTPUT_DIR
        self.dat_dir            = Path(dat_dir)       if dat_dir      else PROJECT_ROOT
        self.timeout_sec        = timeout_sec
        self.validity_check_dll = validity_check_dll

        # Slot names:  OptimBuilding_01, OptimBuilding_02, …
        self.slot_names: List[str] = [
            f"{base_name}_{i + 1:02d}" for i in range(num_workers)
        ]

        # One dedicated input queue per worker — guarantees slot isolation.
        # Two workers can never receive the same slot's job queue.
        self._job_queues: List[mp.Queue] = [
            mp.Queue() for _ in range(num_workers)
        ]
        # Single shared result queue (all workers post here).
        self._result_q: mp.Queue = mp.Queue()

        self._processes:  List[mp.Process] = []
        self._next_slot   = 0          # round-robin counter
        self._job_counter = 0
        # job_id → slot_index; used to track pending work
        self._pending: Dict[int, int] = {}

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> "TQSWorkerPool":
        """
        Spawn all worker subprocesses.  Call once before submitting jobs.

        Returns *self* so the pool can be used as a context manager or
        chained: ``pool = TQSWorkerPool(2).start()``.
        """
        log.info("TQSWorkerPool: starting %d worker(s)…", self.num_workers)
        print(
            f"[TQSWorkerPool] Starting {self.num_workers} worker(s) "
            f"with slots: {', '.join(self.slot_names)}"
        )

        for slot, jq in zip(self.slot_names, self._job_queues):
            p = mp.Process(
                target=_worker_main,
                args=(
                    slot, jq, self._result_q,
                    str(self.dat_dir), str(self.tqs_base_dir),
                    self.timeout_sec, self.validity_check_dll,
                ),
                name=f"TQSWorker-{slot}",
                daemon=True,
            )
            p.start()
            self._processes.append(p)
            log.info("  [%s] started (PID %d).", slot, p.pid)
            print(f"  [{slot}] started (PID {p.pid}).")

        return self

    def stop(self) -> None:
        """Send poison-pills to all workers and wait for clean exit."""
        log.info("TQSWorkerPool: sending shutdown signals…")
        print("[TQSWorkerPool] Sending shutdown signals…")

        for jq in self._job_queues:
            jq.put(None)                       # one poison-pill per queue

        for p in self._processes:
            p.join(timeout=30)
            if p.is_alive():
                log.warning(
                    "Worker %s did not exit within 30 s — terminating.", p.name
                )
                p.terminate()

        self._processes.clear()
        print("[TQSWorkerPool] All workers stopped.")

    def __enter__(self) -> "TQSWorkerPool":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ── job submission ─────────────────────────────────────────────────────────

    def submit(
        self,
        column_polygons:  List[Polygon],
        beam_definitions: List[Dict],
    ) -> int:
        """
        Dispatch a new TQS job to the next available worker (round-robin).

        The caller retains *column_polygons* and *beam_definitions* for
        feature extraction after the result is received.

        Returns
        -------
        int
            A unique job-ID that will appear in the matching
            :class:`WorkerResult`.
        """
        job_id   = self._job_counter
        self._job_counter += 1
        slot_idx = self._next_slot % self.num_workers
        self._next_slot  += 1
        self._pending[job_id] = slot_idx

        self._job_queues[slot_idx].put(
            (job_id, column_polygons, beam_definitions)
        )

        approx_q = self._job_queues[slot_idx].qsize()
        log.debug(
            "Submitted job #%d → [%s] (queue depth ≈ %d).",
            job_id, self.slot_names[slot_idx], approx_q,
        )
        print(
            f"[TQSWorkerPool] Job #{job_id} → [{self.slot_names[slot_idx]}]"
            f"  (queue depth ≈ {approx_q})"
        )
        return job_id

    # ── result retrieval ───────────────────────────────────────────────────────

    def get_result(self, timeout: float = 300.0) -> WorkerResult:
        """
        Block until one job result is available, then return it.

        Parameters
        ----------
        timeout : float
            Maximum wait time in seconds before raising ``queue.Empty``.
        """
        result = self._result_q.get(timeout=timeout)
        self._pending.pop(result.job_id, None)
        log.debug(
            "Result for job #%d from [%s] — success=%s  elapsed=%.1fs",
            result.job_id, result.slot_name, result.success, result.elapsed,
        )
        return result

    def pending_count(self) -> int:
        """Number of submitted jobs not yet retrieved."""
        return len(self._pending)

    # ── batch helper ───────────────────────────────────────────────────────────

    def map(
        self,
        jobs:            List[Tuple[List[Polygon], List[Dict]]],
        timeout_per_job: float = 300.0,
    ) -> List[WorkerResult]:
        """
        Submit all jobs then collect all results, preserving submission order.

        Parameters
        ----------
        jobs : list of ``(column_polygons, beam_definitions)`` tuples.
        timeout_per_job : float
            Passed to :meth:`get_result` for each expected result.

        Returns
        -------
        list of :class:`WorkerResult` in the same order as *jobs*.
        """
        job_ids = [self.submit(polys, beams) for polys, beams in jobs]
        by_id: Dict[int, WorkerResult] = {}
        for _ in job_ids:
            res = self.get_result(timeout=timeout_per_job)
            by_id[res.job_id] = res
        return [by_id[jid] for jid in job_ids]
