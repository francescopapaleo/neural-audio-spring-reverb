# Modeling Spring Reverb with Neural Audio Effects

## *Assessment of different models for the task*

This is the working repository for the thesis project.
A basic command line interface is provided to train, test and evaluate the models.
A transfer function and RT60 measurement script is also provided.

## Train

From the repository root use the following CLI command to train the model:

```terminal
python train.py \
--data_dir ./data/plate-spring/spring/ \
--checkpoints ./checkpoints/ \
--batch_size 16 \   
--epochs 50 \
--device cuda:0
--crop 3200
```

## Test

To test the model on the test set, use the following command:

```terminal
python test.py
```

it will automatically load all the checkpoints in `./checkpoints/` and test them on the test set.

## Inference

To run inference on a single audio file, use the following command:

```terminal
python inference.py --load tcn_25_16_0.001_20230605_184451.pt \
--input ./data/processed/inverse_filter.wav \
--max_length 8

```

## Transfer Function and RT60

Execute these commands from the root folder of the project:

```terminal
python -m utils.generator 
python -m utils.transfer_function --load MODEL_NAME --input inverse_filter.wav
python -m utils.rt60_measure --input 'saxophone.wav'
```

## Command Line Arguments

```terminal
options:
options:
  -h, --help                  show this help message and exit
  --data_dir DATA_DIR         dataset
  --n_epochs N_EPOCHS         number of epochs of training
  --batch_size BATCH_SIZE     size of the batches
  --lr LR                     learning rate
  --device DEVICE             device to use
  --crop CROP                 crop size
  --sample_rate SAMPLE_RATE   sample rate
   --load LOAD                checkpoint to load
 ```

## Tensorboard

```terminal
tensorboard dev upload --logdir ./runs --name "01 training" --description "training with batch size=16, lr=0.001"
```

```terminal
tensorboard dev upload --logdir ./results --name "01 testing" --description "testing trained models"
```

## Folder structure

```terminal
.
|
├── checkpoints/            # model checkpoints
|  
├── data/                   # training and test data
│   ├──raw/                 # raw audio files
│   ├──processed/           # processed audio files
│
├── logs/                   # HPC logs
|
├── models/                 # model classes
|
├── results/                # test results
|
├── runs/                   # tensorboard logs
│
├── utils/                  # utility functions
│   ├── generator.py        # generate audio files
│   ├── rt60.py             # measure rt60
│   ├── tf.py               # compute transfer function
|
├── data.py                 # test / evaluation script
├── inference.py            # inference script
├── test.py                 # test / evaluation script
├── train.py                # training script

└── requirements.txt

```

## Sources

[Baseline Dataset](https://zenodo.org/record/3746119)

[Steerable-Nafx](https://github.com/csteinmetz1/steerable-nafx)
[Micro-tcn](https://github.com/csteinmetz1/micro-tcn.git)

[DeepAFx-ST](https://github.com/adobe-research/DeepAFx-ST#style-evaluation)

[PedalNet](https://github.com/teddykoker/pedalnet)
[PedalNetRT](https://github.com/GuitarML/PedalNetRT)
