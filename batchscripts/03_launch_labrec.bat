@echo off



echo We are on %COMPUTERNAME%.
echo Loading config...
:: generate computer specific variables
call vars\%COMPUTERNAME%.bat
:: generate all subject specific variables
call vars\subjectconfig.bat

echo Done.

rem echo StorageLocation=%subject_data_root_dir%\\%subjectcode%-eyeblock%%n.xdf>%subject_data_root_dir%\%labrecorder_eyeblock_cfg%
rem rem echo StudyRoot=%subject_data_root_dir%\\>%subject_data_root_dir%\%labrecorder_eyeblock_cfg%
rem rem echo PathTemplate=%subjectcode%-eyeblock%%n.xdf>>%subject_data_root_dir%\%labrecorder_eyeblock_cfg%
rem rem echo StorageLocation=%subject_data_root_dir%\\%subjectcode%-eyeblock.xdf>%subject_data_root_dir%\%labrecorder_eyeblock_cfg%
rem echo RequiredStreams=%labrec_eyeblock_req_streams%>>%subject_data_root_dir%\%labrecorder_eyeblock_cfg%
rem echo SessionBlocks=[]>>%subject_data_root_dir%\%labrecorder_eyeblock_cfg%
rem echo ExtraChecks={}>>%subject_data_root_dir%\%labrecorder_eyeblock_cfg%
rem echo EnableScriptedActions = False>>%subject_data_root_dir%\%labrecorder_eyeblock_cfg%


start /D%lsl_apps_labrecorder% LabRecorder.exe 
rem --config %subject_data_root_dir%\%labrecorder_eyeblock_cfg%
::echo App launched.