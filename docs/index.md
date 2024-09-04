<h1>Neural Audio Spring Reverb</h1>
Reverberation is a key element in spatial audio perception, histor- ically achieved with the use of analogue devices, such as plate and spring reverb, and in the last decades with digital signal process- ing techniques that have allowed different approaches for Virtual Analogue Modelling (VAM). The electromechanical functioning of the spring reverb makes it a nonlinear system that is difficult to fully emulate in the digital domain with white-box modelling techniques. In this study, we compare five different neural net- work architectures, including convolutional and recurrent models, to assess their effectiveness in replicating the characteristics of this audio effect. The evaluation is conducted on two datasets at sam- pling rates of 16 kHz and 48 kHz. This paper specifically focuses on neural audio architectures that offer parametric control, aiming to advance the boundaries of current black-box modelling tech- niques in the domain of spring reverberation.


<h2>Audio Examples</h2>

Some dry and processed audio examples are available below. The audio files are in `48kHz 24bit wav` format.
All the models have been trained with the same dataset, and the same training procedure with a single `NVIDIA GeForce GTX 1080 Ti GPU`. 

<h3>Drum loop</h3>

- Dry / raw 

<audio src="assets/audio/raw/drums-48k24b.wav" controls preload></audio>

- Processed with the GCN model

<audio src="assets/audio/processed/drums-48k24b*gcn-3250.wav" controls preload></audio>

- Processed with the TCN model

<audio src="assets/audio/processed/drums-48k24b*tcn-3900-updated.wav" controls preload></audio>

- Processed with the WaveNet model

<audio src="assets/audio/processed/drums-48k24b*wavenet-900.wav" controls preload></audio>


<h3>Pluck synth sample</h3>

- Dry / raw

<audio src="assets/audio/raw/pluck-48k24b.wav" controls preload></audio>

- Processed with the GCN model

<audio src="assets/audio/processed/pluck-48k24b*gcn-3250.wav" controls preload></audio>

- Processed with the TCN model

<audio src="assets/audio/processed/pluck-48k24b*tcn-3900-updated.wav" controls preload></audio>

- Processed with the WaveNet model

<audio src="assets/audio/processed/pluck-48k24b*wavenet-900.wav" controls preload></audio>


<h2>Bibtex Citation</h2>

```bibtex
@inproceedings{DAFx24_paper_77,
    author = "Papaleo, Francesco and Lizarraga-Seijas, Xavier and Font, Frederic",
    title = "{Evaluating Neural Networks Architectures for Spring Reverb Modelling}",
    booktitle = "Proceedings of the 27-th Int. Conf. on Digital Audio Effects (DAFx24)",
    editor = "De Sena, E. and Mannall, J.",
    location = "Guildford, Surrey, UK",
    eventdate = "2024-09-03/2024-09-07",
    year = "2024",
    month = "Sept.",
    publisher = "",
    issn = "2413-6689",
    doi = "",
    pages = ""
}
```