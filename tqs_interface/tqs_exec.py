from pathlib import Path
import subprocess
import os
import time

from TQS import TQSUtil, TQSExec
from config.settings import BuildingConfig


def _cleanup_results_file():
    results_path = Path(BuildingConfig.TQS_RESULTS_FILE)
    if results_path.exists():
        try:
            results_path.unlink()
        except OSError as exc:
            TQSUtil.writef(f"Warning: failed to delete TQS results file '{results_path}': {exc}")


def RunModel(building_name):
    """
    Global processing of the building using TQSExec with minimal overhead
    """
    result = subprocess.getoutput('tasklist /FI "IMAGENAME eq NTQSHTM.EXE"')

    if "NTQSHTM.EXE" in result:
        os.system('taskkill /F /IM NTQSHTM.EXE /T >nul 2>&1')
        time.sleep(0.1)

    _cleanup_results_file()

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
    TQSUtil.writef("Global processing completed successfully.")
