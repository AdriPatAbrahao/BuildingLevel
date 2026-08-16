from pathlib import Path
import subprocess
import os
import time

from TQS import TQSUtil, TQSExec
from config.settings import BuildingConfig


class TQSCriticalError(RuntimeError):
    """
    Raised when the TQS DLL API reports critical structural errors.

    Notes
    -----
    This exception is used to propagate critical conditions detected via the
    TQS execution API (DLL) during global processing.
    """
    pass


def _cleanup_report_files(building_name: str = None):
    """
    Remove TQS results file before a new run.

    Removes only the results file (`RESDES.HTM`) to avoid mixing results
    between runs. Error detection no longer relies on `PGLOERR.HTM`.

    Parameters
    ----------
    building_name : str, optional
        Building slot name (e.g. ``"OptimizedBuilding_01"``).  When provided
        the RESDES.HTM path is derived from the TQS base dir + slot name so
        that parallel workers delete their own slot's file rather than the
        default building's file.  Defaults to ``BuildingConfig.TQS_RESULTS_FILE``.
    """
    if building_name:
        from config.paths import TQS_OUTPUT_DIR
        path = Path(TQS_OUTPUT_DIR) / building_name / "ESPACIAL" / "RESDES.HTM"
    else:
        raw_path = getattr(BuildingConfig, "TQS_RESULTS_FILE", None)
        if not raw_path:
            return
        path = Path(raw_path)
    if path.exists():
        try:
            path.unlink()
        except OSError as exc:
            TQSUtil.writef(f"Warning: failed to delete TQS report '{path}': {exc}")


# HTML error report reading removed (obsolete)


# PGLOERR-based structural error check removed (errors are read via DLL)


def RunModel(building_name, terminate_existing_process=True):
    """
    Execute the global processing in TQS for the given building.

    Parameters
    ----------
    building_name : str
        Name of the building folder inside TQS outputs.
    terminate_existing_process : bool, optional
        Preserve the legacy single-run cleanup when ``True``. Concurrent pilot
        workers must pass ``False`` because terminating ``NTQSHTM.EXE`` by its
        global image name would also terminate another building's analysis.

    Raises
    ------
    TQSCriticalError
        If the TQS DLL API reports critical structural errors or execution fails.
    """
    if terminate_existing_process:
        result = subprocess.getoutput('tasklist /FI "IMAGENAME eq NTQSHTM.EXE"')

        if "NTQSHTM.EXE" in result:
            os.system('taskkill /F /IM NTQSHTM.EXE /T >nul 2>&1')
            time.sleep(0.1)

    _cleanup_report_files(building_name)

    job = TQSExec.Job()
    job.EnterTask(TQSExec.TaskFolder(building_name, TQSExec.TaskFolder.FOLDER_FRAMES))
    job.EnterTask(TQSExec.TaskGlobalProc(
        gridSlabsTrnsf=0,
        frameBeamsTrnsf=0,
        slabs=0,
        beams=1,
        columns=2
    ))
    job.EnterTask(TQSExec.TaskStructuralReport())
    try:
        job.Execute()
    except Exception as exc:
        raise TQSCriticalError(f"TQS global processing failed via DLL: {exc}")

    TQSUtil.writef("Global processing completed successfully.")
