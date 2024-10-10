@echo off
:: generate all subject specific variables
call vars\subjectconfig.bat

cd ../offline_scripts
rem call activate opencv

code "%cd%" classifier_ABBR.py
pause