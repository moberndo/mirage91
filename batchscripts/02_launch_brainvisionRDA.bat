@echo off

:: generate all subject specific variables
call vars\subjectconfig.bat

start /D%lsl_apps_brainvisionrda% BrainVisionRDA.exe
echo App launched.

exit 0