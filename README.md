# Modeling Spring Reverb with Neural Audio Effects

Working environment for the thesis project.

To run utility scripts:

```
python -m utils.generator
python -m utils.transfer_function
python -m utils.rt60_measure 'input_file'   # from command line takes an audio file, otherwise np.ndarray 
```



## Folder structure:
```
.
├── audio/                  # Audio files for input/output
|  
├── data/
│
├── experiments/            # Experiment folders
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
[DeepAFx-ST](https://github.com/adobe-research/DeepAFx-ST#style-evaluation)


[PedalNet](https://github.com/teddykoker/pedalnet)
[PedalNetRT](https://github.com/GuitarML/PedalNetRT)

