@echo off

echo We are on %COMPUTERNAME%.
echo Loading config...
call vars/%COMPUTERNAME%.bat
echo Done.

:: querry for new subject code

set /p subjectcode=<vars/subjectcode.txt
set /p "subjectcode=Enter a new subjectcode or press [ENTER] to keep old code [%subjectcode%]: "
echo %subjectcode%

:: save the new code to the subject code file
echo The new subject code is "%subjectcode%"
echo %subjectcode%>vars/subjectcode.txt


:: generate all subject specific variables
call vars/subjectconfig.bat


:: create subject data directory
IF EXIST %subject_data_root_dir% (
	echo ERROR: root directory for given subjectcode exists already!
	pause
	exit 1
) ELSE (
	echo Creating subject recording root directory...
	mkdir %subject_data_root_dir%
	echo Done.
)

exit 0