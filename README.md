# Modeling Spring Reverb with Neural Audio Effects

## Assessment of different models for the task.

This is the working environment for the thesis project.
A basic command line interface is provided to train, evaluate and test the models.

To train:

```bash
python train.py \
--data_dir ./data/plate-spring/spring/ \
--checkpoints ./checkpoints/ \
--batch_size 16 \   
--epochs 50 \
--device cuda:0
--crop 3200
```


## The following scripts are still work in progress::

To test the model on new data execute from terminal:

```bash
python test.py
```


To make an inference on an audio file execute from terminal:

```bash
python inference.py --load MODEL_NAME --input 'saxophone.wav'
python inference.py --load MODEL_NAME --input 'saxophone.wav' --output 'saxophone_out.wav'
```

To generate/compute the transfer function or measure the RT60:
**Execute these commands from the root folder of the project**

```bash
python -m utils.generator 
python -m utils.transfer_function --load MODEL_NAME --input inverse_filter.wav
python -m utils.rt60_measure --input 'saxophone.wav'
```

## Command Line Arguments

```terminal
options:
  -h, --help            show this help message and exit
  --root_dir ROOT_DIR   main folder
  --data_dir DATA_DIR   default dataset folder
  --models_dir MODELS_DIR 
                        folder to store state_dict after training and load for eval or inference
  --results_dir RESULTS_DIR
                        folder to store results/processed files
  --audio_dir AUDIO_DIR
                        folder for raw audio
  --target_dir TARGET_DIR
                        set target folder for a specific function
  --target_file TARGET_FILE
                        set target file for a specific function
  --sr SR               sampling rate frequency, default: 16KHz
  --save SAVE           save weights and biases as
  --load LOAD           load weights and biases from
  --device DEVICE       set the device
  --input INPUT         input file to process
  --split SPLIT         select test/train split of the dataset
  --sample_idx SAMPLE_IDX
                        The index of the sample from a dataset
  --iters ITERS
  --batch_size BATCH_SIZE
  --shuffle SHUFFLE
  --seed SEED
 ```

## Folder structure

```terminal
.
|  
├── data/                   # training and test data
│   ├──raw/                 # raw audio files
│   ├──processed/           # processed audio files
│   ├──plate-spring         # dataset folder
│   │   ├──spring/          # spring reverb subset
│
├── models_trained/         # Trained models state_dict
|
├── results/
│   ├── 00/                 # experiment ## results
│   ├── 01/                 # ...
│
├── utils/                  # utility functions
|
├── config.py               # configuration file
├── eval.py                 # test / evaluation script
├── inference.py            # inference script
├── train.py                # training script
├── README.md
└── requirements.txt

```

## Sources

[Baseline Dataset](https://zenodo.org/record/3746119)

[Steerable-Nafx](https://github.com/csteinmetz1/steerable-nafx)
[Micro-tcn](https://github.com/csteinmetz1/micro-tcn.git)

[DeepAFx-ST](https://github.com/adobe-research/DeepAFx-ST#style-evaluation)

[PedalNet](https://github.com/teddykoker/pedalnet)
[PedalNetRT](https://github.com/GuitarML/PedalNetRT)
