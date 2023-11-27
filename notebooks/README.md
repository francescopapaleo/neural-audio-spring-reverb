# Neural Audio Spring Reverb Notebooks

This directory contains the notebooks used for the preparatory work and to generate the plots and audio examples. If you want to re-run these notebooks you need to install some additional packages:

```
pip install essentia soundfile pyloudnorm
```

## Table of Contents

1. [](01_SpringSetAnalysis.ipynb)

1. [](02_EgfxSetAnalysis.ipynb)

1. [](03_AudioFeaturesEssentiaStreaming.ipynb)
    Extract the audio features from the two dataset, save in multiple csv files available [here](../data/features/).    

- [](03_FeatureVisualization.ipynb)
    Visualize the extracted features on both datasets.

- [Plot_Audio_Samples.ipynb](PlotAudioSamples.ipynb)
    A "service notebook" to plot the audio samples saved during training or evaluation runs and located in the [audio](../audio/) directory.

# Steerable discovery of neural audio effects
The preparatory work on the TCN model for this project is based on the paper [Steerable discovery of neural audio effects](https://arxiv.org/abs/2110.00484) by Christian J. Steinmetz and Joshua D. Reiss. My fork of the original repository contains the Jupyter notebooks related to the spring reverb:  [Steerable discovery of neural audio effects](https://github.com/francescopapaleo/steerable-nafx).