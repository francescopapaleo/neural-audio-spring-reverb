# Modeling Spring Reverb with Neural Audio Effects
Working environment for the thesis project.


To train and save a model:
```
python train.py --save MODEL_NAME --iters NUMBEE_OF_ITERATIONS
```

To evaluate the model execute from terminal:
```
python eval.py --load MODEL_NAME 
```

To make an inference on an audio file execute from terminal:
```
python inference.py --input 'saxophone.wav' --load MODEL_NAME
```

To generate/compute the transfer function or measure the RT60:
**Execute these commands from the root folder of the project**
```
python -m utils.generator 
python -m utils.transfer_function --load MODEL_NAME
python -m utils.rt60_measure --input 'saxophone.wav'
```


## Folder structure:
```
.
├── audio/                  # Audio files for input/output
|  
├── data/                   # 
│
├── experiments/            # Experiments results          
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

