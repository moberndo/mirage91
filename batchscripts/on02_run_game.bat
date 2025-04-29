@echo off
:: generate all subject specific variables
call vars\subjectconfig.bat

echo %game_path%

"%game_path%\Cybathlon 2024 BCI.exe" && exit