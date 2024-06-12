import subprocess, os, time, shutil
import pandas as pd
from sh import sed
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--start_script', type=str, action='store', default = 'run_script.sh', help='The starting script that calls calculate_traj.py')
    parser.add_argument('--restart_script', type=str, action='store', default = 'run_script_restart.sh', help='The re-start script that calls calculate_traj.py with a new index')
    parser.add_argument('--replace_str', type=str, action='store', default='INSERTINDEXHERE', help='The string in the restart script to replace with the last calculated index')
    args = parser.parse_args()


    #This is a supervisor program to monitor the `calculate_traj.py` results and re-start the job if
    #incomplete and the process fails, be it due to memory issues or what have you

    #working directory
    WORKDIR = os.getcwd()
    os.chdir(WORKDIR)
    _PROCESS_ARGS = ['bash', args.start_script]
    RSR = 'run_script_local_restart.sh'
    _RPROCESS_ARGS = ['bash', args.restart_script]

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
            sed(['-i', f's/{args.replace_str}/{lastind+1}/g', _RPROCESS_ARGS[-1]])
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