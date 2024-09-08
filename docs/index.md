---
title: Evaluating Neural Networks Architectures for Spring Reverb Modelling
layout: home
---

## Abstract

Reverberation is a key element in spatial audio perception, histor- ically achieved with the use of analogue devices, such as plate and spring reverb, and in the last decades with digital signal process- ing techniques that have allowed different approaches for Virtual Analogue Modelling (VAM). The electromechanical functioning of the spring reverb makes it a nonlinear system that is difficult to fully emulate in the digital domain with white-box modelling techniques. In this study, we compare five different neural net- work architectures, including convolutional and recurrent models, to assess their effectiveness in replicating the characteristics of this audio effect. The evaluation is conducted on two datasets at sam- pling rates of 16 kHz and 48 kHz. This paper specifically focuses on neural audio architectures that offer parametric control, aiming to advance the boundaries of current black-box modelling tech- niques in the domain of spring reverberation.

## Audio Examples collected during the evaluation

| Model | Target | Prediction | 
|-------|--------|------------|
| GCN | <audio src="assets/audio/eval/target-GCN-3-stft-48k.wav" controls preload style="width: 100px;"></audio> | <audio src="assets/audio/eval/pred-GCN-3-stft-48k.wav" controls preload style="width: 100px;"></audio> |
| TCN |<audio src="assets/audio/eval/target-TCN-99-stft-48k.wav" controls preload style="width: 100px;"></audio> | <audio src="assets/audio/eval/pred-TCN-99-stft-48k.wav" controls preload style="width: 100px;"></audio> |
| WaveNet | <audio src="assets/audio/eval/target-WaveNet-99-stft-48k.wav" controls preload style="width: 100px;"></audio> | <audio src="assets/audio/eval/pred-WaveNet-99-stft-48k.wav" controls preload style="width: 100px;"></audio> |
| GRU | <audio src="assets/audio/eval/target-GRU-99-stft-48k.wav" controls preload style="width: 100px;"></audio> | <audio src="assets/audio/eval/pred-GRU-99-stft-48k.wav" controls preload style="width: 100px;"></audio> |
| LSTM | <audio src="assets/audio/eval/target-LSTM-99-stft-48k.wav" controls preload style="width: 100px;"></audio> | <audio src="assets/audio/eval/pred-LSTM-99-stft-48k.wav" controls preload style="width: 100px;"></audio> |


## Audio Examples with other kind of sounds obtained at inference

| Sound | Dry | GCN | TCN | WaveNet | GRU | LSTM |
|-------|-----|-----|-----|---------|-----|------|
| Drums | <audio src="assets/audio/raw/drums-48k24b.wav" controls preload style="width: 100px;"></audio> | <audio src="assets/audio/processed/drums-48k24b*gcn-3250.wav" controls preload style="width: 100px;"></audio> | <audio src="assets/audio/processed/drums-48k24b*tcn-3900-updated.wav" controls preload style="width: 100px;"></audio> | <audio src="assets/audio/processed/drums-48k24b*wavenet-900.wav" controls preload style="width: 100px;"></audio> | <audio src="assets/audio/processed/drums-48k24b*gru-4layer-updated.wav" controls preload style="width: 100px;"></audio> | <audio src="assets/audio/processed/drums-48k24b*lstm-4layer-updated.wav" controls preload style="width: 100px;"></audio> |
| Synth | <audio src="assets/audio/raw/pluck-48k24b.wav" controls preload style="width: 100px;"></audio> | <audio src="assets/audio/processed/pluck-48k24b*gcn-3250.wav" controls preload style="width: 100px;"></audio> | <audio src="assets/audio/processed/pluck-48k24b*tcn-3900-updated.wav" controls preload style="width: 100px;"></audio> | <audio src="assets/audio/processed/pluck-48k24b*wavenet-900.wav" controls preload style="width: 100px;"></audio> | <audio src="assets/audio/processed/pluck-48k24b*gru-4layer-updated.wav" controls preload style="width: 100px;"></audio> | <audio src="assets/audio/processed/pluck-48k24b*lstm-4layer-updated.wav" controls preload style="width: 100px;"></audio> |


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