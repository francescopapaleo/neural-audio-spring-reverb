# Modeling Spring Reverb with Neural Audio Effects

## Abstract

The spring reverb is an archaic device, simple but rich of nonlinear features, present in most guitar amplifiers, and still quite used in music production in general. Historically, this was the cheapest way to generate a reverberation effect. In this thesis, we will address the problem of creating a spring reverberation model with deep learning.
Different deep learning architectures, based either in time or frequency domain, have been already used for similar tasks. In some cases, certain features can be manipulated by the user to generate results that maintain a similarity to the original emulated effect, providing greater flexibility in the possible outcomes. In this work, we will focus on the use of a Time Convolutional Network (TCN) with Feature-wise Linear Modulation (FiLM) to model the spring reverb.

## How to run

A basic command line interface is provided to train, test and evaluate the model.
Impulse Response, Transfer Function and RT60 measurements are provided.

### Requirements

To run the code in this repository, you will need the following dependencies installed:

```terminal 
auraloss==0.4.0
h5py==3.8.0
matplotlib==3.7.1
numpy==1.23.5
scipy==1.10.1
torch==2.0.1
torchaudio==2.0.2

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

--n_epochs N_EPOCHS             the total number of epochs
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

### Training

From the repository root use the following CLI command to train the model:

```terminal
python train.py
```

### Testing

To test the model on the test set, use the following command:

```terminal
python test.py
```

### Inference

To run inference on a single audio file, use the following command:

```terminal
python inference.py
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
├── audio                                       # audio files
│   ├── generated
│   │   ├── inverse_filter.wav
│   │   ├── reference_ir.wav
│   │   ├── single_impulse.wav
│   │   └── sweep.wav
│   ├── processed
│   └── raw
│       ├── acgtr_clean.wav
│       ├── acgtr_reverb.wav
│       ├── README.md
│       ├── saxophone_input.wav
│       ├── saxophone_output.wav
│       └── vermona_retroverb_ir.wav
├── models                                      # models 
├── notebooks                                   # notebooks
├── results                                     # experiments results
│   ├── checkpoints
│   ├── plots
│   └── runs
├── scripts                                     # bash scripts
├── utils                                       # utility functions
│   ├── generator.py
│   ├── ir.py
│   ├── plotter.py
│   ├── rt60.py
│   └── tf.py
├── data.py
├── inference.py
├── README.md
├── requirements.txt
├── test.py
└── train.py
```

## Main Sources

[Plate-Spring Dataset](https://zenodo.org/record/3746119)

[Steerable-Nafx](https://github.com/csteinmetz1/steerable-nafx)
[Micro-tcn](https://github.com/csteinmetz1/micro-tcn.git)

[DeepAFx-ST](https://github.com/adobe-research/DeepAFx-ST#style-evaluation)

[PedalNet](https://github.com/teddykoker/pedalnet)
[PedalNetRT](https://github.com/GuitarML/PedalNetRT)

## TO DO

### Add metadata to the checkpoint

```python

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'state_epoch': epoch,          
    'name': f'TCN{n_epochs}_{batch_size}_{lr}_{timestamp}',
    'hparams': hparams,
    'criterion': str(criterion)      # Add criterion as a string here
}, save_to)


## Add Loss function metadata to a saved checkpoint

### Load checkpoint
checkpoint = torch.load('path_to_checkpoint.pt')

### Add criterion to the checkpoint dictionary
checkpoint['criterion'] = str(criterion)

### Save checkpoint
torch.save(checkpoint, 'path_to_checkpoint.pt')
```

### Citation

```bibtex
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```  
