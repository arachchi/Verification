copy the source code to a new folder.

Then open a cmd at that location.

pip install virtualenv
python -m virtualenv appenv
appenv\Scripts\activate

pip install pyinstaller
pip install -r requirements.txt
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

pyinstaller -F app.py