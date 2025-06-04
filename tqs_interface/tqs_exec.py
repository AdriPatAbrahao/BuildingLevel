from TQS import TQSUtil, TQSExec
import subprocess
import os
import time

def RunModel(building_name):
    """
    Global processing of the building using TQSExec with minimal overhead
    """
    # Check for NTQSHTM.EXE process
    result = subprocess.getoutput('tasklist /FI "IMAGENAME eq NTQSHTM.EXE"')
    
    # Only close if process is found
    if "NTQSHTM.EXE" in result:
        os.system('taskkill /F /IM NTQSHTM.EXE /T >nul 2>&1')
        time.sleep(0.1)  # Minimal delay
    
    # Run TQS analysis
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

