import subprocess, os, time, shutil, psutil
import pandas as pd
from sh import sed
import argparse

def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--start_script', type=str, action='store', default = 'run_script.sh', help='The starting script that calls calculate_traj.py')
    parser.add_argument('--restart_script', type=str, action='store', default = 'run_script_restart.sh', help='The re-start script that calls calculate_traj.py with a new index')
    parser.add_argument('--restart_copy_script', type=str, action='store', default = 'run_script_restart_rep.sh', help='The re-start script that calls calculate_traj.py with a new index')
    parser.add_argument('--replace_str', type=str, action='store', default='CHECKPOINTPATH', help='The string in the restart script to replace with the last calculated index')
    parser.add_argument('--cutoff_epochs', type=int, action='store', default=200, help='The number of unique epochs the supervisor will allow to occur.')
    parser.add_argument('--cutoff_memory', type=float, action='store', default=0.8, help='The fraction of total memory the program will allow to accumulate before forcing a restart')
    args = parser.parse_args()


    #This is a supervisor program to monitor the `calculate_traj.py` results and re-start the job if
    #incomplete and the process fails, be it due to memory issues or what have you
    RESTARTS = 0
    #working directory
    WORKDIR = os.getcwd()
    os.chdir(WORKDIR)
    os.mkdir('run{}'.format(RESTARTS))
    shutil.copy(args.start_script, f'run{RESTARTS}')
    shutil.copy(args.restart_script, f'run{RESTARTS}')
    os.chdir(f'run{RESTARTS}')
    _PROCESS_ARGS = ['bash', args.start_script]
    RSR = args.restart_script
    _RPROCESS_ARGS = ['bash', args.restart_copy_script]

    progfile = 'supervisor.dat'
    with open(progfile, 'w') as f:
        f.write('#PID\tPOLL\tSTATUS\tACTION\tEPOCHS\tMEMUSE\n')
    #start initial process
    p = subprocess.Popen(_PROCESS_ARGS)
    TOTALMEMORY = psutil.virtual_memory().total
    print(f'Total virtual memory available via psutil: {TOTALMEMORY} bytes // {TOTALMEMORY/1e6} MB // {TOTALMEMORY/1e9} GB')
    pp = psutil.Process(p.pid)
    PREVIOUS_LAST = 99999999
    PREVIOUS_COUNT = 0
    EPOCHS_DONE = 0
    REPEATED_SEGMENTS = 0
    while True:
        #check process -- returns None if still running
        if (p.poll() != None) or (psutil.virtual_memory().used / TOTALMEMORY >= args.cutoff_memory) : #process completed
            RESTARTS += 1
            if (psutil.virtual_memory().used / TOTALMEMORY >= args.cutoff_memory):
                print(f'Calculation memory usage exceeds allowed value: {psutil.virtual_memory().used}/{TOTALMEMORY} >= {args.cutoff_memory}')
                print('Killing current process...')
                # p.kill()
                # p.communicate()
                kill(p.pid)
                with open(progfile, 'a') as f:
                    f.write(f'{p.pid}\t{p.poll()}\tDONE\tKILL\t{EPOCHS_DONE}\t{psutil.virtual_memory().used/TOTALMEMORY}\n')

            #Read in the trlog.dat file to get epoch and loss information
            df = pd.read_csv('trlog.dat', delimiter='\t')

            #If EPOCHS_DONE > 0, an epoch has completed, so last_df is created
            if EPOCHS_DONE > 0:
                loss_mean_diff = df['Loss'].mean() - last_df['Loss'].mean()
                if loss_mean_diff == 0:
                    REPEATED_SEGMENTS += 1
            
            #If repeated segments > 2, the training cycle is stuck.
            if REPEATED_SEGMENTS > 2:
                print('Training cycle stuck in a loop. Stopping here.')
                break

            EPOCHS_DONE += len(df)
            if EPOCHS_DONE > args.cutoff_epochs:
                print('Total number of epochs trained has surpassed the allowed limit. Stopping here.')
                print(f'{EPOCHS_DONE} > {args.cutoff_epochs}')
                break
        
            print('Process completed. Restarting...')
            with open(progfile, 'a') as f:
                f.write(f'{p.pid}\t{p.poll()}\tDONE\tRESTART\t{EPOCHS_DONE}\t{psutil.virtual_memory().used/TOTALMEMORY}\n')
            #find number of checkpoints in this directory
            chkpts = sorted([i for i in os.listdir() if 'xc.eqx' in i], key=lambda x: int(x.split('.')[-1]))
            os.mkdir(f'../run{RESTARTS}')
            shutil.copy(args.restart_script, f'../run{RESTARTS}')
            shutil.copy(chkpts[-1], f'../run{RESTARTS}')
            #copy restart to new file to replace index of
            os.chdir(f'../run{RESTARTS}')
            with open(progfile, 'w') as f:
                f.write('#PID\tPOLL\tSTATUS\tACTION\tEPOCHS\tMEMUSE\n')
            shutil.copy(args.restart_script, args.restart_copy_script)
            sed(['-i', f's|{args.replace_str}|./{chkpts[-1]}|g', args.restart_copy_script])
            last_df = df
            p = subprocess.Popen(_RPROCESS_ARGS)
            pp = psutil.Process(p.pid)
            time.sleep(120)
        else:
            #wait 2 minutes before rechecking
            print('Process still on-going, sleeping...')
            print(f'Current process memory usage: {pp.memory_info().rss}/{TOTALMEMORY} = {pp.memory_info().rss/TOTALMEMORY}')
            with open(progfile, 'a') as f:
                f.write(f'{p.pid}\t{p.poll()}\tONGOING\tSLEEP\t{EPOCHS_DONE}\t{psutil.virtual_memory().used/TOTALMEMORY}\n')
            time.sleep(120)