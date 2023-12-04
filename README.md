# FRAM-RIR
Python implementation for FRAM-RIR, a fast and plug-and-use multi-channel room impulse response simulation tool without the need of specific hardward acceleration platforms (e.g., GPUs). 

[Long paper](https://arxiv.org/abs/2304.08052)

[Interspeech'23 short paper](https://www.isca-speech.org/archive/pdfs/interspeech_2023/luo23b_interspeech.pdf)

## Update 2023/12/04

* Add support for customizing microphone and source orientations
* Fix issues in image sampling which may cause suboptimal performance

## Dependencies 
* numpy
* torch (tested for v1.13 & v2.0.0)
* torchaudio (tested for v0.11)

## Usage
```python
rir, early_rir = FRAM_RIR(mic_pos, sr, T60, room_dim, src_pos, num_src=1, direct_range=(-6, 50), n_image=(512, 2049), src_pattern='omni', src_orientation_rad=None, mic_pattern='omni', mic_orientation_rad=None)
```
### Parameters:
* ``mic_pos``: The microphone(s) position with respect to the room coordinates, with shape [num_mic, 3] (in meters). Room coordinate system must be defined in advance, with the constraint that the origin of the coordinate is on the floor (so positive z axis points up).
* ``sr``: RIR sampling rate (Hz).
* ``rt60``: RT60 (second).
* ``room_dim``: Room size with shape [3] (meters).
* ``src_pos``: The source(s) position with respect to the room coordinate system, with shape [num_src, 3] (meters).
* ``num_src``: Number of sources. Default: 1.
* ``direct_range``: 2-element tuple, range of early reflection time (milliseconds, defined as the context around the direct path signal) of RIRs. Default: (-6, 50).
* ``n_image``: 2-element tuple, minimum and maximum number of images to sample from. Default: (512, 2049).
* ``src_pattern``: Polar pattern for all of the sources. {_"omni", "half_omni", "cardioid", "hyper_cardioid", "sub_cardioid", "bidirectional"_}. Default: *omni*. See *test_samples.py* for examples.
* ``src_orientation_rad``: Array-like with shape [num_src, 2]. Orientation (rad) of all the sources, where the first column indicate azimuth and the second column indicate elevation, all calculated with respect to the room coordinate system. None (default) is only valid for omnidirectional patterns. For other patterns with *src_orientation_rad=None*, apply random source orientation.
* ``mic_pattern``: Polar pattern for all of the receivers. {_"omni", "half_omni", "cardioid", "hyper_cardioid", "sub_cardioid", "bidirectional"_}. Default: *omni*. See *test_samples.py* for examples.
* ``mic_orientation_rad``: Array-like with shape [num_mic, 2]. Orientation (rad) of all the microphones, where the first column indicate azimuth and the second column indicate elevation, all calculated with respect to the room coordinate system. None (default) is only valid for omnidirectional patterns. For other patterns with *mic_orientation_rad=None*, assume all microphone pointing up (positive z axis) to mimic the scenario where all microphones are put on a table.
### Outputs:
* ``rir``: RIR filters for all mic-source pairs, with shape [num_mic, num_src, rir_length].
* ``early_rir``: Early reflection (direct path) RIR filters for all mic-source pairs, with shape [num_mic, num_src, rir_length].

## Reference

If you use FRAM-RIR in your project, please consider citing the following papers.

> @article{luo2023fast,  
> title={Fast Random Approximation of Multi-channel Room Impulse Response},   
> author={Luo, Yi and Gu, Rongzhi},   
> year={2023},  
> eprint={2304.08052},  
> archivePrefix={arXiv},  
> primaryClass={cs.SD}   
> }

> @inproceedings{luo2023fra,  
> title={{FRA}-{RIR}: Fast Random Approximation of the Image-source Method},  
> author= {Luo, Yi and Yu, Jianwei},  
> year=2023,  
> booktitle={Proc. Interspeech},  
> pages={3884--3888}  
> }

## Disclaimer

This is not an officially supported Tencent product.