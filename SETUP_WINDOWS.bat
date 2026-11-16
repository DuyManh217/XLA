@echo off
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 goto error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error
echo Cai dat xong. Hay mo START_APP.bat
pause
exit /b 0
:error
echo Cai dat that bai. Can Python 3.10+ va ket noi Internet de tai thu vien.
pause
exit /b 1
