# Fast Random Approximation of Room Impulse Response (FRA-RIR)

This is the repository for a Pytorch-based implementation of FRA-RIR method for data augmentation purpurse in simulating reverberant signals. FRA-RIR is a Image-source method (ISM) based RIR simulation method that replaces the explicit calculation of the delay and reflections of the virtual sound sources by random approximations. With a CPU-only simulation pipeline, FRA-RIR can be significantly faster than other RIR simulation tools, enabling fully on-the-fly data simulation during training, but also improves the model performance when evaluated on realistic RIR data.

Please refer to our paper on [Arxiv](https://arxiv.org/abs/2208.04101) for details.

![](https://github.com/yluo42/FRA-RIR/blob/main/FRA-RIR-result.png)

# The multi-channel version (FRAM-RIR)

Paper: [Arxiv](https://arxiv.org/abs/2304.08052)

Code: [github](https://github.com/tencent-ailab/FRA-RIR/tree/fram_rir)


## Disclaimer
This is not an officially supported Tencent product.
