# Neural Audio Spring Reverb

*This project wants to address the modelling of the spring reverb, using deep neural networks, in particular: CNN (convolutional) and LSTM/GRU (recurrent). The main goal is to investigate the potential of neural networks in modelling such nonlinear time-invariant system, and to compare the results with the state of the art methods.*

## Requirements
The following packages are required to run the code:

```requirements.txt
auraloss==0.4.0
h5py==3.9.0
matplotlib==3.7.2
numpy==1.24.3
PyYAML==6.0.1
scipy==1.11.2
setuptools==65.5.0
torch==2.0.1
torchaudio==2.0.2
torchinfo==1.8.0
neutone_sdk==1.4.2
wandb==0.16.0
```

You can install the dependencies by running the following command:

```terminal
pip install -r requirements.txt
```

Please make sure to use a compatible version of Python, preferably Python 3.11, along with the required packages mentioned above.

## Command Line Arguments

```terminal
-h, --help                      show the help message and exit

POSITIONAL ARGUMENTS:
action     
'download', 'train', 'eval', 'infer', 'config_tools', 'measure_ir', 'measure_rt60', 'print_models' 

OPTIONAL ARGUMENTS:
--data_dir      DATA_DIR      datasets download destination folder
--audio_dir     AUDIO_DIR     audio files storage folder
--log_dir       LOG_DIR       tensorboard logs
--plots_dir     PLOTS_DIR     saved plots
--models_dir    MODELS_DIR    trained models

--dataset       DATASET       'springset', 'egfxset', 'customset'

--sample_rate   SAMPLE_RATE   16000 or 48000
--bit_rate      BIT_RATE      16 or 24


--device        DEVICE        cuda:0 or cpu
--checkpoint    CHECKPOINT    checkpoint to load
--init          CONF          YAML configuration file to load

--input         INPUT         input audio file for inference
--duration      DURATION      duration of the sweep-tone for IR measurement
```
**When you want to pass a checkpoint path or a folder, you can use relative paths.**


## How to Run the Code

The CLI is managed in the [``main.py``](main.py) script. From the project root folder, run the following commands to download one of the datasets or train, test and inference a model.

**To download a dataset:**

```terminal
python main.py download --dataset DATASET_NAME
```

**To train a model:**

You can either start from scratch from a ``YAML`` configuration file or resume the training from a checkpoint (``.pt``). If you resume the training from a checkpoint, this will proceed until the current epoch is equal to MAX_EPOCHS.

```terminal
python main.py train --init YAML_CONF_PATH

python main.py train -c PT_CHECKPOINT_PATH
```

**To test a model:**

```terminal
python main.py eval -c PT_CHECKPOINT_PATH
```
The given checkpoint is loaded and the model is evaluated on the test set. The results are automatically logged by Tensorboard and stored in the  [``logs/``](logs/) folder.

**Use a model for inference:**

```terminal
python main.py infer -i INPUT_FILE_PATH -c PT_CHECKPOINT_PATH
```

## Neutone SDK

This project integrates the [Neutone SDK](), which allows to wrap and export a trained model for using it into a DAW with the [Neutone plugin]() or for using it in a standalone application.

To export a model, run the following command:

```terminal
python main.py wrap -c PT_CHECKPOINT_PATH
```

## Audio Measurement Tools

The folder [``tools``](src/tools/) contains some scripts to measure the impulse response of a spring reverb model or an audio file that contains the impulse response of a physical device. 

### Measure impulse response
This action will call a function that loads the model checkpoint, generate the test signals of the duration specified by the user and perform the IR measurement of the model.

```terminal
python main.py ir --duration DURATION -c PT_CHECKPOINT_PATH
```
A logaritimic sweep tone is generated and is processed by the model inference, the output is then convolved with the inverse filter previously generated. 

- The plot is saved in the [``plots/measured_IR``](docs/plots/measured_IR/) folder.
- The audio file corresponding to the measured IR is saved in the [``audio/measured_IR``](audio/measured_IR/) folder.

### Measure RT60
This action will call a function that loads the input audio and computes the RT60 of the signal. A threshold of -5dB is set by default.

```terminal
python main.py rt60 -i INPUT_FILE_PATH
```

- The plot is saved in the [``plots/``](docs/plots/) folder.


## Utilities

**About Pre-Trained Models**

The training script saves the best model checkpoint if the average validation loss has improved. Saving, loading and initialization are defined in the [checkpoint.py](./src/utils/checkpoints.py) module. The .pt files are saved in the [``models/``](models/) folder and contain a dictionary named: ``checkpoint`` with the following keys:


``['label']``: model label (e.g. 'tcn-2000-cond...'),

``['timestamp']``: timestamp of the latest training,

``['model_state_dict']``: model state dict,

``['optimizer_state_dict']``: optimizer state dict,

``['scheduler_state_dict']``: scheduler state dict,

``['config_state_dict']``: a configuration dict that contains all the parameters used for training and then recalled for evaluation and inference. A list of the keys contained in this dictionary is available in the [``default.yaml``](./models/configs/default.yaml) file.

### Print all models details
To access the ``['config_state_dict']`` of a pretrained model, there is an action that allows to print all the details of the pretrained models, it can be useful for debugging purposes.

This action will print the details of all pretrained models available in the [``models``](models/) folder.

```terminal
python main.py report
```
- results are saved in a txt file in the [``logs/``](logs/) folder.

### Config Tools
This is a more delicate functionality that allows to print the configuration of a pretrained model and asks the user if he wants to modify one of the parameters. If yes it automatically updates the configuration and saves the new model file.

***Please be careful with the modifications***

```terminal
python3 main.py edit -c PT_CHECKPOINT_PATH
```

## Notebooks

Some notebooks are provided to visualize the audio feature extraction process from the datasets, they are located in the [notebooks](./notebooks/) folder. The features are available in the [results](./results/) folder. If you want to proceed to the feature extraction yourself, you will need to install the [Essentia](https://essentia.upf.edu/) library.

## Folder structure

```terminal
.
|-- audio               # audio files
|   |-- eval            # audio examples from the evaluation script
|   |-- measured_IR     # measured impulse responses
|   |-- raw
|   |-- signals         # reference signals for measurements
|   `-- train           # audio examples from the training script
|-- docs
|   |-- plots           # plots
|-- data
|   |-- features
|   `-- raw             # destination folder for the downloaded dataset(s)
|-- logs                # tensorboard logs
|-- models              # trained models
|-- notebooks           # jupyter notebooks
|-- src                 # source code for this project
|   |-- dataload
|   |-- networks
|   |-- tools
|   |-- utils
|   |-- inference.py
|   |-- train.py
|   `-- eval.py
|-- LICENSE
|-- main.py             # main script to access any functionality
|-- README.md           # this file
|-- requirements.txt
`-- setup.py

```

### Citation

This repository is part of my thesis project for the MSc in Sound and Music Computing at the Music Technology Group, Universitat Pompeu Fabra, Barcelona (Spain).

```bibtex
@phdthesis{francesco_papaleo_2023_8380480,
  author       = {Francesco Papaleo},
  title        = {{Neural Audio Effect Modelling Strategies for a 
                   Spring Reverb}},
  school       = {Universitat Pompeu Fabra},
  year         = 2023,
  month        = sep,
  doi          = {10.5281/zenodo.8380480},
  url          = {https://doi.org/10.5281/zenodo.8380480}
}
```  
