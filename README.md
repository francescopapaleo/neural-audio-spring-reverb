# Neural Audio Effect Strategies for Spring Reverb Modelling

Thesis project for the MSc in Sound and Music Computing at the Music Technology Group, Universitat Pompeu Fabra, Barcelona, Spain.

## Requirements

The following packages are required to run the code:

```terminal
auraloss==0.4.0
h5py==3.9.0
matplotlib==3.7.2
numpy==1.24.3
scipy==1.11.2
torch==2.0.1
torchaudio==2.0.2
torchinfo==1.8.0
tensorboard==2.14.0

# You can install the dependencies by running the following command:

pip install -r requirements.txt
```

Please make sure to use a compatible version of Python, preferably Python 3.11, along with the required packages mentioned above.

### Command Line Arguments

```terminal
-h, --help                      show the help message and exit

--data_dir DATA_DIR              relative path to dataset
--audio_dir AUDIO_DIR            relative path to audio files
--log_dir LOG_DIR                relative path to log directory
--results_dir RESULTS_DIR        relative path to results directory
--plots_dir PLOTS_DIR            relative path to plots directory
--models_dir MODELS_DIR          relative path to models directory

--sample_rate SAMPLE_RATE       sample rate of the audio
--bit_rate BIT_RATE             bit rate of the audio

--conf CONF                     configuration to select for training
--device DEVICE                 set device to run the model on
--checkpoint CHECKPOINT         relative path to checkpoint

--max_epochs N_EPOCHS           maximum number of epochs for training

--input INPUT                   relative path to input audio
```

### How to Run

From the project root folder, run the following commands to download, train, test and inference:

```terminal

python3 -m data.dataload.download_dataset --dataset DATASET_NAME

python3 train.py --conf CONFIGURATION_NAME --max_epochs N_EPOCHS

python3 test.py --checkpoint_path CHECKPOINT

python3 inference.py --input INPUT_REL_PATH --checkpoint_path CHECKPOINT
```

To generate reference signals:
  
```terminal
python3 -m src.tools.signals
```

### Tensorboard

```terminal
tensorboard --logdir logs/
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
├── plots
├── scripts                 # bash scripts
├── src
│   ├── dataload            # data processing
│   ├── models              # model architectures
│   ├── tools               #
│   └── utils               # 
│
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
