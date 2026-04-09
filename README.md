# RootVision Studio

An educational Python GUI project for Engineering Analysis.

## Best idea chosen
Interactive **Numerical Methods Explorer** with visual comparison of:
- Bisection method
- Newton-Raphson method
- Secant method
- Fixed-Point iteration

## Why this is the strongest project idea
- It matches the project guideline exactly: Python, GUI, visual explanation, iterations, parameter control, and dynamic plots.
- It is directly listed in the instructor's suggested topics.
- It is educational because it helps the user understand how convergence changes from one method to another.
- It can be packaged into a standalone Windows `.exe` using PyInstaller.

## Files
- `main.py` → main GUI application
- `requirements.txt` → Python packages
- `user_guide.txt` → short guide required in the guideline
- `build_exe.bat` → one-click Windows build helper

## Run locally
```bash
pip install -r requirements.txt
python main.py
```

## Build EXE on Windows
```bash
pyinstaller --onefile --windowed --name RootVisionStudio main.py
```
