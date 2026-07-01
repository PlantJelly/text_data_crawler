@echo off
cd /d "%~dp0"

if not exist venv (
    echo [INFO] Virtual environment not found. Creating venv...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b
    )
    call venv\Scripts\activate
    echo [INFO] Updating pip...
    python -m pip install --upgrade pip
    if exist requirements.txt (
        echo [INFO] Installing dependencies from requirements.txt...
        pip install -r requirements.txt
    )
) else (
    echo [INFO] Activating existing venv...
    call venv\Scripts\activate
)

if not exist app.py (
    echo [ERROR] app.py not found in this directory!
    pause
    exit /b
)

echo [INFO] Running Streamlit app...
streamlit run app.py
pause