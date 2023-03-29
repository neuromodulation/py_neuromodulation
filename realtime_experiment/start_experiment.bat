call conda activate realtime_decoding
@REM # Without "start" before the script to run your training program,
@REM #     the batch file will wait until the training program finishes
@REM # Adding "start" opens it in a new window, and processes the next line
@REM #     without waiting for the program to finish running

start python -m start_decoding
ECHO Running realtime decoding
TIMEOUT /T 10

start timeflux -d timeflux_decoding.yaml
ECHO Running timeflux
TIMEOUT /T 5
START http://localhost:8000/monitor/
TIMEOUT /T 5
call conda activate bsl
start bsl_stream_viewer -s SAGA

PAUSE
