import subprocess, os, time, shutil
import pandas as pd
from sh import sed
#This is a supervisor program to monitor the `calculate_traj.py` results and re-start the job if
#incomplete and the process fails, be it due to memory issues or what have you

#working directory
WORKDIR = os.getcwd()
os.chdir(WORKDIR)
_PROCESS_ARGS = ['bash', 'run_script_local.sh']
RSR = 'run_script_local_restart.sh'
_RPROCESS_ARGS = ['bash', 'run_script_local_restart_rep.sh']

progfile = 'supervisor.dat'
with open(progfile, 'w') as f:
    f.write('#PID\tSTATUS\tACTION\n')
#start initial process
p = subprocess.Popen(_PROCESS_ARGS)
PREVIOUS_LAST = 99999999
PREVIOUS_COUNT = 0
while True:
    #check process -- returns None if complete
    if p.poll() != None: #process completed
        print('Process completed. Restarting...')
        with open(progfile, 'a') as f:
            f.write(f'{p.pid}\tDONE\tRESTART\n')
        #check progress file for last index
        progress = pd.read_csv('progress', delimiter='\t')
        lastind = progress.iloc[-1]['#idx']
        #copy restart to new file to replace index of
        shutil.copy(RSR, _RPROCESS_ARGS[-1])
        sed(['-i', f's/INSERTINDEXHERE/{lastind+1}/g', _RPROCESS_ARGS[-1]])
        if lastind == PREVIOUS_LAST:
            PREVIOUS_COUNT += 1
        if PREVIOUS_COUNT == 3:
            break
        PREVIOUS_LAST = lastind
        p = subprocess.Popen(_RPROCESS_ARGS)
    else:
        #wait 2 minutes before rechecking
        print('Process still on-going, sleeping...')
        with open(progfile, 'a') as f:
            f.write(f'{p.pid}\tONGOING\tSLEEP\n')
        time.sleep(120)