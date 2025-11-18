@echo off
echo Instalando Sistema de Reconhecimento Facial...
echo.

python -m venv venv
call venv\Scripts\activate

python -m pip install --upgrade pip

pip install fastapi uvicorn[standard] websockets python-multipart
pip install mtcnn opencv-python opencv-contrib-python
pip install scikit-learn numpy scipy pillow pydantic python-dotenv aiofiles flask
pip install https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.1-cp311-cp311-win_amd64.whl
pip install face-recognition

echo.
echo Instalacao concluida!
echo Execute: venv\Scripts\activate
pause