@echo off
title Alpha Trading Desk
cd /d "%~dp0"
echo.
echo  Starting Alpha Trading Desk...
echo.
where streamlit >nul 2>&1
if %errorlevel% neq 0 pip install -r requirement.txt
streamlit run app.py --server.port 8501
pause
