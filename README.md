# lstm_compose
Generating music using LSTM. This repository was inspired by "Generative deep learning" book.
## Requirements
Python3.7
numpy
music21
tensorflow
## Dataset
For train gererative neural network I used Bach's chorales from music21 library
## How to generate music
### Train NN
To train neural network you should run command bellow:
python train_nn.py
This command will train NN on Bach's chorales dataset and save best weights and also all weights which improve loss while learning in run/compose/0007_cello/weights.
### Generate 
To generate new melody you should run command bellow:
python predict.py
This command will generate melody and save it in .mid format in run/compose/0007_cello/output
