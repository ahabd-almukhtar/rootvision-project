@echo off
python -m pip install --upgrade pip
python -m pip install numpy matplotlib pyinstaller
pyinstaller --onefile --windowed --name RootVisionStudio main.py
pause
