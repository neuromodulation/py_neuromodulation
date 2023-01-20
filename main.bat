call conda activate realtime_decoding
@REM # Without "start" before the script to run your training program,
@REM #     the batch file will wait until the training program finishes
@REM # Adding "start" opens it in a new window, and processes the next line
@REM #     without waiting for the program to finish running

start python "start_decoding.py"
ECHO Running realtime decoding
PAUSE
TIMEOUT /T 5
start python "tmsi_gui.py"
ECHO Running TMSi GUI
TIMEOUT /T 5
start timeflux "timeflux_decoding.yaml"
ECHO Running timeflux
TIMEOUT /T 5
START http://localhost:8000/monitor/

@REM # Adding "PAUSE" makes the script wait for you manually type a key to continue,
@REM #     but it is not required. You can add PAUSE anywhere in the script
PAUSE