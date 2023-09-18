# Neural Audio Effects Modelling Strategies for a Spring Reverb

Virtual analog modelling emulates the processing characteristics of a given physical device. This has been an active field of research and commercial innovation in which two main perspectives have been historically adopted.  The first one: **knowledge based**, or white-box, seeks to reproduce the exact behaviour through algorithmic simulation of circuits or physical phenomena. The second one: **data-driven**, or black-box, aims to learn the approximation function from examples recorded at the input and output stages of the target device. In this second approach, deep learning has emerged as a valuable strategy for linear systems, such as filters, as well as nonlinear time-dependent ones like distortion circuits or compressors. 

The spring reverb is as audio effect with a very long and rooted history in music production and performance, based on a relatively simple design, this device is an effective tool for artificial reverberation. The electromechanical functioning of this reverb makes it a nonlinear time-invariant spatial system that is difficult to fully emulate in the digital domain with white-box modelling techniques.     

This thesis wants to address the modelling of spring reverb, leveraging end-to-end neural audio effect architectures through supervised learning. Recurrent, convolutional and hybrid models have successfully been used for similar tasks, especially  compressors and distortion circuits emulations. Using two available datasets of guitar recordings, we evaluate with quantitative metrics, acoustical analysis and signal processing measurements the efficiency and the results of four neural network architectures to model this effect. We present the results and outline different strategies for this modelling task, providing a reproducible experimental environment with code.

### Audio Examples

<audio src="../audio/eval/pred-gcn-2500-kt-c2-mrstft-48kHz.wav" controls preload></audio>

<audio src="../audio/eval/pred-gcn-2500-kt-c2-mrstft-48kHz.wav" controls preload></audio>

<audio src="../audio/eval/pred-gcn-2500-kt-c2-mrstft-48kHz.wav" controls preload></audio>

<audio src="../audio/eval/pred-gcn-2500-kt-c2-mrstft-48kHz.wav" controls preload></audio>
