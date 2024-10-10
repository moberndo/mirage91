@echo off
:: generate all subject specific variables
call vars\subjectconfig.bat

cd ../offline_scripts
rem call activate opencv

call python feature_extraction.py
pause