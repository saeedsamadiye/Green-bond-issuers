@echo off
rem Downloads exchange rates, missing prices, firm info, earnings dates and factors,
rem then recomputes data\saved_results.json. Output also goes to update_log.txt.
cd /d "%~dp0"
echo Started %date% %time% > update_log.txt

if not exist venv\Scripts\python.exe (
    echo Creating virtual environment... >> update_log.txt
    py -m venv venv >> update_log.txt 2>&1
)
venv\Scripts\python.exe -m pip install -q -r requirements.txt >> update_log.txt 2>&1

echo ===== update-data ===== >> update_log.txt
venv\Scripts\python.exe -W ignore greenbond_engine.py --update-data >> update_log.txt 2>&1

echo ===== save-snapshot ===== >> update_log.txt
venv\Scripts\python.exe -W ignore greenbond_engine.py --save-snapshot data\saved_results.json >> update_log.txt 2>&1

echo Finished %date% %time% >> update_log.txt
echo Done. See update_log.txt
pause
