---
title: Evaluating Neural Networks Architectures for Spring Reverb Modelling
layout: home
---

<center> <h3>WORK IN PROGRESS</h3> </center>


## Audio Examples compared with target data

| Target | GCN | TCN | WaveNet | GRU | LSTM |
|--------|-----|-----|---------|-----|------|
|        |     |     |         |     |      |


## Audio Examples with other kind of sounds

| Sound | Dry | GCN | TCN | WaveNet | GRU | LSTM |
|-------|-----|-----|-----|---------|-----|------|
| Drums | <audio src="assets/audio/raw/drums-48k24b.wav" controls preload style="width: 100px;"></audio> | <audio src="assets/audio/processed/drums-48k24b*gcn-3250.wav" controls preload style="width: 100px;"></audio> | <audio src="assets/audio/processed/drums-48k24b*tcn-3900-updated.wav" controls preload style="width: 100px;"></audio> | <audio src="assets/audio/processed/drums-48k24b*wavenet-900.wav" controls preload style="width: 100px;"></audio> |  |  |
| Synth | <audio src="assets/audio/raw/pluck-48k24b.wav" controls preload style="width: 100px;"></audio> | <audio src="assets/audio/processed/pluck-48k24b*gcn-3250.wav" controls preload style="width: 100px;"></audio> | <audio src="assets/audio/processed/pluck-48k24b*tcn-3900-updated.wav" controls preload style="width: 100px;"></audio> | <audio src="assets/audio/processed/pluck-48k24b*wavenet-900.wav" controls preload style="width: 100px;"></audio> |  |  |


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