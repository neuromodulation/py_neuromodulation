set ROOT=C:\ProgramData\miniconda3
call %ROOT%\Scripts\activate.bat %ROOT%
call conda activate task_motor_stopping
@REM # If you don't want to see the outputs/print of your training program, 
@REM #     add @ECHO OFF to the start. If you want to see them, remove the @ECHO OFF
@REM @ECHO OFF
@REM # Without "start" before the script to run your training program,
@REM #     the batch file will wait until the training program finishes
@REM python ".\motor_stopping_main.py"
@REM ECHO training program completed

@REM # Adding "start" opens it in a new window, and processes the next line
@REM #     without waiting for the program to finish running
@REM start python ".\motor_stopping_main.py"
@REM start python ".\experiment\motor_stopping.py"
start realtime_decoding
ECHO Running realtime decoding
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