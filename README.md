# Modeling Spring Reverb with Neural Audio Effects

Working environment for the thesis project.

`config.py` contains the configuration parameters for the project including the paths to the data and the currently loaded model

To make an inference on an audio file execute from terminal:
```
python inference.py --input 'saxophone.wav' 
```

To train the model execute from terminal:
```
python train.py 
```

To evaluate the model execute from terminal:
```
python test.py
```




## Execute these commands from the root folder of the project

To generate the test signals (sweep tone, inverse filter and the reference):
```
python -m utils.generator
```

To compute the transfer function of a pretrained model:
```
python -m utils.transfer_function
```

To measure the RT60 of an audio file execute from terminal:
```
python -m utils.rt60_measure 'input_file'
```

Otherwise the function is automatically called when running the script `inference.py`


## Folder structure:
```
.
├── audio/                  # Audio files for input/output
|  
├── data/
│
├── experiments/            # Experiments results
│
├── imgs/               
|
├── models/                 # Trained and serialized models
|
├── results/
|
├── sandbox/                # For testing purposes
│
├── utils/                  # Utility functions
|
├── config.py               # Configuration file
```

## Sources:
[Baseline Dataset](https://zenodo.org/record/3746119)


[Steerable-Nafx](https://github.com/csteinmetz1/steerable-nafx)
[Micro-tcn](https://github.com/csteinmetz1/micro-tcn.git)

[DeepAFx-ST](https://github.com/adobe-research/DeepAFx-ST#style-evaluation)

[PedalNet](https://github.com/teddykoker/pedalnet)
[PedalNetRT](https://github.com/GuitarML/PedalNetRT)

