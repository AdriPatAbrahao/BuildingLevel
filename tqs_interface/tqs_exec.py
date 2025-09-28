from pathlib import Path
import subprocess
import os
import time

from TQS import TQSUtil, TQSExec
from config.settings import BuildingConfig

_ENC_TRY_ORDER = ("utf-8", "latin-1", "ISO-8859-1")


class TQSCriticalError(RuntimeError):
    """Raised when the TQS structural report flags critical errors."""
    pass


def _cleanup_report_files():
    # Remove both the results file (RESDES.HTM) and the error report (PGLOERR.HTM)
    for raw_path in (
        getattr(BuildingConfig, "TQS_RESULTS_FILE", None),
        getattr(BuildingConfig, "TQS_ERROR_REPORT_FILE", None),
    ):
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.exists():
            try:
                path.unlink()
            except OSError as exc:
                TQSUtil.writef(f"Warning: failed to delete TQS report '{path}': {exc}")


def _read_html_file(file_path: Path):
    for encoding in _ENC_TRY_ORDER:
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return None


def _check_structural_errors() -> bool:
    # Wait a short moment for PGLOERR.HTM to appear and then check for the fatal marker
    raw = getattr(BuildingConfig, "TQS_ERROR_REPORT_FILE", None)
    if not raw:
        print("Warning: BuildingConfig.TQS_ERROR_REPORT_FILE not set; skipping PGLOERR check.")
        return False
    
    error_path = Path(raw)
    timeout = 5.0
    start = time.time()
    print(f"Info: Waiting up to {timeout}s for error report: '{error_path}'")
    while not error_path.exists():
        if time.time() - start > timeout:
           print(
                f"Info: Error report file not found after {timeout}s at '{error_path}'. "
                "Assuming no critical errors."
            )
        return False
        time.sleep(0.2)
    html = _read_html_file(error_path)
    if html is None:
        TQSUtil.writef(f"Warning: unable to read TQS error report '{error_path}'.")
        return False
    marker = getattr(BuildingConfig, "TQS_FATAL_ERROR_MARKER", "").lower()
    if not marker:
        TQSUtil.writef("Warning: TQS_FATAL_ERROR_MARKER not set; cannot detect critical errors.")
        return False
    if marker in html.lower():
        TQSUtil.writef("Critical errors reported by TQS (PGLOERR.HTM).")
        return True
    TQSUtil.writef("Info: No critical errors reported by TQS.")
    return False


def RunModel(building_name):
    """
    Global processing of the building using TQSExec with minimal overhead
    """
    result = subprocess.getoutput('tasklist /FI "IMAGENAME eq NTQSHTM.EXE"')

    if "NTQSHTM.EXE" in result:
        os.system('taskkill /F /IM NTQSHTM.EXE /T >nul 2>&1')
        time.sleep(0.1)

    _cleanup_report_files()

    job = TQSExec.Job()
    job.EnterTask(TQSExec.TaskFolder(building_name, TQSExec.TaskFolder.FOLDER_FRAMES))
    job.EnterTask(TQSExec.TaskGlobalProc(
        gridSlabsTrnsf=0,
        slabs=0,
        beams=3,
        columns=2
    ))
    job.EnterTask(TQSExec.TaskStructuralReport())
    job.Execute()

    if _check_structural_errors():
        raise TQSCriticalError("TQS reported critical structural errors (PGLOERR.HTM)")

    TQSUtil.writef("Global processing completed successfully.")
