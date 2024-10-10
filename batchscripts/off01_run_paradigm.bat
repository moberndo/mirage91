@echo off

:: generate all subject specific variables
call vars\subjectconfig.bat

echo Starting the paradigm.

cd ..\paradigm

call "%psychopy_path%\pythonw.exe" "%psychopy_path%\Lib\site-packages\psychopy\app\psychopyApp.py" "bci_racingteam.psyexp"
::echo App launched.

exit 0