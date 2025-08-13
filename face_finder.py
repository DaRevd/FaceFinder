name: Build Executable

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: windows-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pyinstaller opencv-python-headless pyqt5 pillow numpy
        
    - name: Build executable
      run: |
        cd src
        pyinstaller --onefile --windowed --add-data "models/*;models" --name FaceRecognitionFileManager main.py
        mv dist/FaceRecognitionFileManager.exe ..
    
    - name: Upload artifact
      uses: actions/upload-artifact@v2
      with:
        name: FaceRecognitionFileManager
        path: FaceRecognitionFileManager.exe
