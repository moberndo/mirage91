:: load conmputer specific config
call vars\%COMPUTERNAME%.bat

:: load the current subject code
set /p subjectcode=<vars\subjectcode.txt

:: define subject based variables
set subject_data_root_dir=%data_root_dir%\%subjectcode%

:: set subject_data_share_dir=%data_share_dir%\%subjectcode%