# lstm_compose
Generating music using LSTM. This repository was inspired by "Generative deep learning" book.
## Requirements
To start generating music you will need:
1. Python3.7
2. numpy
3. music21
4. tensorflow
## Dataset
For train gererative neural network I used Bach's chorales from music21 library
## How to generate music
### Train NN
To train neural network you should run command bellow:
```commandline
python train_nn.py
```
This command will train NN on Bach's chorales dataset and save best weights and also all weights which improve loss while learning in run/compose/0007_cello/weights.
### Generate 
To generate new melody you should run command bellow:
```commandline
python predict.py
```
This command will generate melody and save it in .mid format in run/compose/0007_cello/output
