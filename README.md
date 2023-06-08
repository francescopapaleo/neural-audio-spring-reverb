# Modeling Spring Reverb with Neural Audio Effects

## *Assessment of different models for the task*

This is the working repository for the thesis project.
A basic command line interface is provided to train, test and evaluate the models.
A transfer function and RT60 measurement script is also provided.

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
  -h, --help                  show this help message and exit
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
├── README.md
├── TCN.py
├── audio
│   ├── README.md
│   ├── generated
│   │   ├── inverse_filter.wav
│   │   ├── reference_ir.wav
│   │   ├── single_impulse.wav
│   │   └── sweep.wav
│   ├── processed
│   └── raw
│       ├── acgtr_clean.wav
│       ├── acgtr_reverb.wav
│       ├── saxophone_input.wav
│       ├── saxophone_output.wav
│       └── vermona_retroverb_ir.wav
├── checkpoints
│
├── data.py
├── inference.py
├── plots
|
├── requirements.txt
├── results
│ 
├── runs
│ 
├── test.py
├── train.py
├── train.sh
└── utils
    ├── decay.py
    ├── generator.py
    ├── ir.py
    ├── plotter.py
    ├── rt60.py
    └── tf.py


```

## Sources

[Baseline Dataset](https://zenodo.org/record/3746119)

[Steerable-Nafx](https://github.com/csteinmetz1/steerable-nafx)
[Micro-tcn](https://github.com/csteinmetz1/micro-tcn.git)

[DeepAFx-ST](https://github.com/adobe-research/DeepAFx-ST#style-evaluation)

[PedalNet](https://github.com/teddykoker/pedalnet)
[PedalNetRT](https://github.com/GuitarML/PedalNetRT)
