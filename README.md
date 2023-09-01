# Neural Audio Effect Strategies for Spring Reverb Modelling

Thesis project for the MSc in Sound and Music Computing at the Music Technology Group, Universitat Pompeu Fabra, Barcelona, Spain.

## :warning: WORK IN PROGRESS :warning:

## Abstract

The spring reverb is an archaic device, simple but rich of nonlinear features, present in most guitar amplifiers, and still quite used in music production in general. Historically, this was the cheapest way to generate a reverberation effect. In this thesis, we will address the problem of creating a spring reverberation model with deep learning.
Different deep learning architectures, based either in time or frequency domain, have been already used for similar tasks. In some cases, certain features can be manipulated by the user to generate results that maintain a similarity to the original emulated effect, providing greater flexibility in the possible outcomes. In this work, we will focus on the use of a Time Convolutional Network (TCN) with Feature-wise Linear Modulation (FiLM) to model the spring reverb.

A basic command line interface is implemented to use the repository. Impulse Response, Transfer Function and RT60 measurements are provided.

### Requirements

To run the code in this repository, you will need the following dependencies installed:

```terminal

# You can install these dependencies by running the following command:

pip install -r requirements.txt
```

Please make sure to use a compatible version of Python, preferably Python 3.11, along with the required packages mentioned above.

### Command Line Arguments

```terminal
-h, --help                      show the help message and exit

--datadir DATA_DIR              Path (rel) to dataset
--audiodir AUDIO_DIR            Path (rel) to audio files
--logdir LOG_DIR                Path (rel) to log directory
--load LOAD                     Path (rel) to checkpoint to load
--input INPUT                   Path (rel) relative to input audio

--sample_rate SAMPLE_RATE       sample rate of the audio
--device DEVICE                 set device to run the model on
--duration DURATION             duration in seconds

--max_epochs N_EPOCHS             the total number of epochs
--batch_size BATCH_SIZE         batch size
--lr LR                         learning rate
--crop CROP                     crop size

--max_length MAX_LENGTH       maximum length of the output audio
--stereo                      flag to indicate if the audio is stereo or mono
--tail                        flag to indicate if tail padding is required
--width WIDTH                 width parameter for the model
--c0 C0                       c0 parameter for the model
--c1 C1                       c1 parameter for the model
--gain_dB GAIN_DB             gain in dB for the model
--mix MIX                     mix parameter for the model
```

### How to Run

From the project root folder, run the following commands to download, train, test and inference:

```terminal

python3 -m data.download_dataset

python3 train.py

python3 test.py --checkpoint_path CHECKPOINT_RELATIVE_PATH

python3 inference.py --input INPUT_RELATIVE_PATH --checkpoint_path CHECKPOINT_RELATIVE_PATH
```

To generate reference signals:
  
```terminal
python3 -m utils.signals
```

### Tensorboard

```terminal
tensorboard dev upload --logdir ./runs/01_train --name "01 training" --description "training with batch size=16, lr=0.001"
```

```terminal
tensorboard dev upload --logdir ./runs/01_test --name "01 testing" --description "testing trained models"
```

## Folder structure

```terminal
.
├── audio
│   ├── generated
│   ├── processed
│   └── raw
├── models
├── notebooks
├── results
│   ├── checkpoints         # saved models
│   ├── plots
│   ├── train               # tensorboard training logs        
│   └── test                # tensorboard testing logs
├── scripts                 # bash scripts
├── src
│   ├── data                # data processing
│   ├── models              # model architectures
│   ├── utils               # utility functions
│   └── visualization       # visualization functions
├── configurations.py       # CLI arguments and hparams
├── inference.py          
├── test.py
├── train.py
├── LICENSE
├── README.md
└── requirements.txt
```

## Main Sources

[Plate-Spring Dataset](https://zenodo.org/record/3746119)

[Steerable-Nafx](https://github.com/csteinmetz1/steerable-nafx)
[Micro-tcn](https://github.com/csteinmetz1/micro-tcn.git)

[DeepAFx-ST](https://github.com/adobe-research/DeepAFx-ST#style-evaluation)

[PedalNet](https://github.com/teddykoker/pedalnet)
[PedalNetRT](https://github.com/GuitarML/PedalNetRT)

### Citation

```bibtex
@misc{papaleo2023,
  title   = {Neural Audio Effects Strategies for Spring Reverb Modelling},
  author  = {Francesco Papaleo and Xavier Lizarraga},
  school  = {Universitat Pompeu Fabra},
  year    = {2023},
  month   = {August},
  doi     = {},
  url     = {}
}
```  
