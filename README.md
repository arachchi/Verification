copy the source code to a new folder.

Then open a cmd at that location.

pip install virtualenv
python -m virtualenv appenv
appenv\Scripts\activate

pip install pyinstaller
pip install -r requirements.txt
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

pyinstaller -F --add-data "./models/model-inter-11001.pt";"./models/model-inter-11001.pt" --add-data "./models/model-inter-11251.pt";"./models/model-inter-11251.pt" --add-data "./models/model-inter-14501.pt";"./models/model-inter-14501.pt" app.py 
pyinstaller -F --add-data "./models/";"models"  app.py 