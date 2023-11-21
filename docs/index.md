---
layout: base
title: Home
permalink: /
---
<h2>Abstract</h2>
<p>Virtual analog modeling emulates the processing characteristics of a given physical device. This has been an active
    field of research and commercial innovation in which two main perspectives have been historically adopted. The first
    one: <strong>white-box</strong>, seeks to reproduce the exact behavior through algorithmic simulation of circuits or
    physical phenomena. The second one: <strong>black-box</strong>, aims to learn the approximation function from
    examples recorded at the input and output stages of the target device. In this second approach, deep learning has
    emerged as a valuable strategy for linear systems, such as filters, as well as nonlinear time-dependent ones like
    distortion circuits or compressors.</p>

<p>The spring reverb is an audio effect with a very long and rooted history in music production and performance. Based
    on a relatively simple design, this device is an effective tool for artificial reverberation. The electromechanical
    functioning of this reverb makes it a nonlinear time-invariant spatial system that is difficult to fully emulate in
    the digital domain with white-box modeling techniques.</p>

<p>This thesis aims to address the modeling of spring reverb, leveraging end-to-end neural audio effect architectures
    through supervised learning. Recurrent, convolutional, and hybrid models have successfully been used for similar
    tasks, especially compressor and distortion circuit emulations. Using two available datasets of guitar recordings,
    we evaluate with quantitative metrics, acoustical analysis, and signal processing measurements the efficiency and
    the results of four neural network architectures to model this effect. We present the results and outline different
    strategies for this modeling task, providing a reproducible experimental environment with code.</p>

<h2>Audio Examples</h2>

Some dry and processed audio examples are available below. The audio files are in `48kHz 24bit wav` format.
All the models have been trained with the same dataset, and the same training procedure with a single `NVIDIA GeForce GTX 1080 Ti GPU`. 

**Drum loop (dry / raw)**

<audio src="assets/audio/raw/drums-48k24b.wav" controls preload></audio>

- Drum loop processed with the GCN model

<audio src="assets/audio/processed/drums-48k24b*gcn-3250.wav" controls preload></audio>

- Drum loop processed with the TCN model

<audio src="assets/audio/processed/drums-48k24b*tcn-3900-updated.wav" controls preload></audio>

- Drum loop processed with the WaveNet model

<audio src="assets/audio/processed/drums-48k24b*wavenet-900.wav" controls preload></audio>


- **Pluck synth sample (dry / raw)**

<audio src="assets/audio/raw/pluck-48k24b.wav" controls preload></audio>

- Pluck synth sample processed with the GCN model

<audio src="assets/audio/processed/pluck-48k24b*gcn-3250.wav" controls preload></audio>

- Pluck synth sample processed with the TCN model

<audio src="assets/audio/processed/pluck-48k24b*tcn-3900-updated.wav" controls preload></audio>

- Pluck synth sample processed with the WaveNet model

<audio src="assets/audio/processed/pluck-48k24b*wavenet-900.wav" controls preload></audio>


<h2>Bibtex Citation</h2>

This is not a PhD thesis, but a Master's thesis, Zenodo does not allow to specify this in the metadata.  

```bibtex
@phdthesis{francesco_papaleo_2023_8380480,
  author       = {Francesco Papaleo},
  title        = {Neural Audio Effect Modelling Strategies for a 
                   Spring Reverb},
  school       = {Universitat Pompeu Fabra},
  year         = 2023,
  month        = sep,
  doi          = {10.5281/zenodo.8380480},
  url          = {https://doi.org/10.5281/zenodo.8380480}
}
```