# Neural Audio Effect Strategies for Spring Reverb Modelling

***Thesis project for the MSc in Sound and Music Computing at the Music Technology Group, Universitat Pompeu Fabra, Barcelona, Spain.***

*The need for artificial reverberation first arose in the broadcasting and recording industry, in the 1920s RCA already developed such devices that could respond to their needs. Towards the late 1920s, Hammond started producing the spring reverbs, the very first example of electromechanical reverberation. The spring reverb is as audio effect based on a relatively simple design, the electromechanical functioning of this reverb makes it a highly nonlinear and time-invatiant spatial system. After two decades of improvements and innovations, in the early 1980s AKG introduced the BX 20, a very natural sounding spring reverb that became a standard in the industry, still today this is one of the references in music production.*

*This project wants to address the modelling of the spring reverb, using deep neural networks, in particular: CNN (convolutional) and LSTM (recurrent). The main goal is to investigate the potential of neural networks in modelling such nonlinear time-invariant system, and to compare the results with the state of the art methods.*

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
```
You can install the dependencies by running the following command:
```terminal
pip install -r requirements.txt
```

Please make sure to use a compatible version of Python, preferably Python 3.11, along with the required packages mentioned above.


### Command Line Arguments
```terminal
-h, --help                      show the help message and exit

--data_dir DATA_DIR             relative path to dataset
--audio_dir AUDIO_DIR           relative path to audio files
--log_dir LOG_DIR               relative path to log directory
--results_dir RESULTS_DIR       relative path to results directory
--plots_dir PLOTS_DIR           relative path to plots directory
--models_dir MODELS_DIR         relative path to models directory

--sample_rate SAMPLE_RATE       16000 or 48000
--bit_rate BIT_RATE             16 or 24

--dataset DATASET               'springset' or 'egfxset'
--conf CONF                     one of the configurations in src.train_conf.py
--device DEVICE                 cuda:0 or cpu
--checkpoint CHECKPOINT         relative path to checkpoint

--max_epochs MAX_EPOCHS         maximum number of epochs for training

--input INPUT                   relative path to input audio
```


### How to Run
From the project root folder, run the following commands to download, train, test and inference:

- To download the dataset:
```terminal
python3 -m src.dataload.download_data --dataset DATASET_NAME
```

- To train, test and inference:
```terminal
python3 train.py --conf CONF --max_epochs MAX_EPOCHS

python3 test.py --checkpoint_path CHECKPOINT

python3 inference.py --input INPUT_REL_PATH --checkpoint_path CHECKPOINT
```
The code-structure allows to resume the training from a checkpoint, until the current epoch is equal to MAX_EPOCHS.
To do so, simply add the `--checkpoint` argument to the `train.py` command.


- To generate reference signals:
  
```terminal
python3 -m src.tools.signals
```

- 
### Tensorboard
To visualize the training process ot test results, run the following command from the project root folder:
```terminal
tensorboard --logdir logs/
```

## Folder structure
```terminal
.
├── audio
│   ├── measured_IR
│   ├── raw
│   ├── signals
│   ├── test
│   └── train
├── docs
├── data
│   ├── egfxset
│   └── plate-spring
├── logs
├── models
├── notebooks
├── results
├── scripts                 # bash scripts
├── src
│   ├── dataload            # data pipeline
│   ├── models              # model architectures
│   ├── tools               # audio measurements and signal generation
│   └── utils               # utility functions
│
├── inference.py          
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── test.py
└── train.py
```


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
