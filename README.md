# Report

I created this repository and cloned it on the HPC, set up a working environment.

## Measured the available samples from Steinmetz et al. :
[measurement](/util/get_audio_leght.py)
Audio file: acgtr_clean.wav, Length: 11.52 seconds, 508032 samples
Audio file: acgtr_reverb.wav, Length: 11.52 seconds, 508032 samples


## Operations on "our" dataset:
[notebook](h5_dataset.ipynb)
- there is no metadata in our dataset
- once measured and checked a random selection has been done for train, test and val


## Current subset:
Audio file: dry_train_subset.h5, Total Length: 34.00 seconds, 544000 samples
Audio file: wet_train_subset.h5, Total Length: 34.00 seconds, 544000 samples
Audio file: wet_val_subset.h5, Total Length: 4.00 seconds, 64000 samples
Audio file: dry_val_subset.h5, Total Length: 4.00 seconds, 64000 samples
Audio file: wet_test_subset.h5, Total Length: 4.00 seconds, 64000 samples
Audio file: dry_test_subset.h5, Total Length: 4.00 seconds, 64000 samples

- lenghts are different because the data is downsampled at 16KHz but the number of samples is the same

## A dataloader is written with Pytorch
[dataloader](dataloader_subset.py)


[tests](dataloader_test.py)

Even with the dataloader the model cannot be trained.
Some modifications have been made, preprocessing, cropping and training loop.


## Training_TCN_mod
[code](training_TCN_mod.py)
Learning rate schedule: 1:1.00e-03 -> 2800:1.00e-04 -> 3325:1.00e-05
Loss: 7.914e-01 | : 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3500/3500 [09:58<00:00,  5.84it/s]



## Training_TCN
[code](training_TCN.py)


[parameters](config.py)


After many trials and errors, with concatenation a way to train the original model without modifications in the training loop is achieved

Learning rate schedule: 1:1.00e-03 -> 2800:1.00e-04 -> 3325:1.00e-05
 Loss: 9.323e-01 | : 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3500/3500 [1:01:01<00:00,  1.05s/it]


## Some Questions
- Since there's no metadata I couldn't separate bass and guitar notes from the dataset
- I have some doubts about the evaluation part in the training script, is it correct or I should change something?
- Shouldn't we use the validation set at some point?

- How should I proceed afterwards? Use the metrics to measure the performance.

I don't know if that's clear enough, you get an idea of where I am and then I can explain you in detail with more infromations.

