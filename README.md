Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

# Fast Random Approximation of Room Impulse Response (FRA-RIR)

This is the repository for a Pytorch-based implementation of FRA-RIR method for data augmentation purpurse in simulating reverberant signals. FRA-RIR is a Image-source method (ISM) based RIR simulation method that replaces the explicit calculation of the delay and reflections of the virtual sound sources by random approximations. With a CPU-only simulation pipeline, FRA-RIR can be significantly faster than other RIR simulation tools, enabling fully on-the-fly data simulation during training, but also improves the model performance when evaluated on realistic RIR data.

Please refer to our paper on [Arxiv](https://arxiv.org/abs/2208.04101) for details.

![](https://github.com/yluo42/FRA-RIR/blob/main/FRA-RIR-result.png)
