# Modeling Spring Reverb with Neural Audio Effects

## TCN model with FiLM

This is the working repository for the thesis project.
A basic command line interface is provided to train, test and evaluate the model.
Impulse Response, Transfer Function and RT60 measurements are provided.

## Train

From the repository root use the following CLI command to train the model:

```terminal
python train.py
```

## Test

To test the model on the test set, use the following command:

```terminal
python test.py
```

## Inference

To run inference on a single audio file, use the following command:

```terminal
python inference.py
```

## Command Line Arguments

```terminal
-h, --help                      show the help message and exit

--data_dir DATA_DIR             Path (rel) to dataset
--audio_dir AUDIO_DIR           Path (rel) to audio files
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

## Tensorboard

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
├── checkpoints
├── plots
├── runs
├── utils
│   ├── decay.py
│   ├── generator.py
│   ├── ir.py
│   ├── plotter.py
│   ├── rt60.py
│   └── tf.py
├── data.py
├── inference.py
├── README.md
├── requirements.txt
├── TCN.py
├── test.py
├── train.py
└── train.sh
```

## Main Sources

[Plate-Spring Dataset](https://zenodo.org/record/3746119)

[Steerable-Nafx](https://github.com/csteinmetz1/steerable-nafx)
[Micro-tcn](https://github.com/csteinmetz1/micro-tcn.git)

[DeepAFx-ST](https://github.com/adobe-research/DeepAFx-ST#style-evaluation)

[PedalNet](https://github.com/teddykoker/pedalnet)
[PedalNetRT](https://github.com/GuitarML/PedalNetRT)


## TODO

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