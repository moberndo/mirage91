@echo off
:: generate all subject specific variables
call vars\subjectconfig.bat

cd ../online_scripts
rem call activate opencv

call python test_connect_to_game.py
pause