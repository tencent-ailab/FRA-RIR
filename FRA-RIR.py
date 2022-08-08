import numpy as np
import torch
from torchaudio.transforms import Resample
from torchaudio.functional import highpass_biquad

def FRA_RIR(nsource=1, sr=16000, direct_range=[-6, 50], max_T60=0.8,
            alpha=0.25, a=-2.0, b=2.0, tau=0.2):
    """
    The fast random approximation of room impulse response (FRA-RIR) method.
    args:
        nsource: number of sources (RIR filters) to simulate. Default: 1.
        sr: target sample rate. Default: 16000.
        direct_range: the context range (at milliseconds) at the first peak of the RIR filter to define the direct-path RIR. Default: [-6, 50] ms.
        max_T60: the maximum range of T60 to sample from. Default: 0.8.
        alpha: controlling the probability distribution to sample the distance of the virtual sound sources from. Default: 0.25.
        a, b: controlling the random pertubation added to each virtual sound source. Default: -2, 2.
        tau: controlling the relationship between the distance and the number of reflections of each virtual sound source. Default: 0.25.
    output:
        rir_filter: simulated RIR filter for all sources, shape: (nsource, nsample)
        direct_rir_filter: simulated direct-path RIR filter for all sources, shape: (nsource, nsample)
    """
    
    eps = np.finfo(np.float16).eps
    
    # sample distance between the sound sources and the receiver (d_0)
    direct_dist = torch.FloatTensor(nsource).uniform_(0.2, 12)

    # sample T60 of the room
    T60 = torch.FloatTensor(1).uniform_(0.1, max_T60)[0].data
    
    # sample room-related statistics for calculating the reflection coefficient R
    R = torch.FloatTensor(1).uniform_(0.1, 1.2)[0].data
    
    # number of virtual sound sources
    image = sr*2
    
    # the sample rate at which the original RIR filter is generated
    ratio = 64
    sample_sr = sr*ratio
    
    # sound velocity
    velocity = 340.
    
    # indices of direct-path signals based on the sampled d_0
    direct_idx = torch.ceil(direct_dist * sample_sr / velocity).long()
    
    # length of the RIR filter based on the sampled T60
    rir_length = int(np.ceil(sample_sr * T60))
    
    # two resampling operations
    resample1 = Resample(sample_sr, sample_sr//int(np.sqrt(ratio)))
    resample2 = Resample(sample_sr//int(np.sqrt(ratio)), sr)
    
    # calculate the reflection coefficient based on the Eyring's empirical equation
    reflect_coef = (1 - (1 - torch.exp(-0.16*R/T60)).pow(2)).sqrt()
    
    # randomly sample the propagation distance for all the virtual sound sources
    dist_range = [torch.linspace(1., velocity*T60/direct_dist[i]-1, image) for i in range(nsource)]
    # a simple quadratic function
    dist_prob = torch.linspace(alpha, 1., image).pow(2)
    dist_prob = dist_prob / dist_prob.sum()
    dist_select_idx = dist_prob.multinomial(num_samples=image*nsource, replacement=True).view(nsource, image)
    # the distance is sampled as a ratio between d_0 and each virtual sound sources
    dist_ratio = torch.stack([dist_range[i][dist_select_idx[i]] for i in range(nsource)], 0)
    dist = direct_dist.view(-1, 1) * dist_ratio
    
    # sample the number of reflections (can be nonintegers)
    # calculate the maximum number of reflections
    reflect_max = (torch.log10(velocity*T60) - torch.log10(direct_dist) - 3) / torch.log10(reflect_coef + eps)
    # calculate the number of reflections based on the assumption that 
    # virtual sound sources which have longer propagation distances may reflect more frequently
    reflect_ratio = (dist / (velocity*T60)).pow(2) * (reflect_max.view(nsource, -1) - 1) + 1
    # add a random pertubation based on the assumption that
    # virtual sound sources which have similar propagation distances can have different routes and reflection patterns
    reflect_pertub = torch.FloatTensor(nsource, image).uniform_(a, b) * dist_ratio.pow(tau)
    # all virtual sound sources should reflect for at least once
    reflect_ratio = torch.maximum(reflect_ratio + reflect_pertub, torch.ones(1))
    
    # calculate the rescaled dirac comb as RIR filter
    dist = torch.cat([direct_dist.reshape(-1, 1), dist], 1)
    reflect_ratio = torch.cat([torch.zeros(nsource, 1), reflect_ratio], 1)
    rir = torch.zeros(nsource, rir_length)
    delta_idx = torch.minimum(torch.ceil(dist * sample_sr / velocity), torch.ones(1)*rir_length-1).long()
    delta_decay = reflect_coef.pow(reflect_ratio) / dist
    for i in range(nsource):
        rir[i][delta_idx[i]] += delta_decay[i]
    
    # a binary mask for direct-path RIR
    direct_mask = torch.zeros(nsource, rir_length).float()
    for i in range(nsource):
        direct_mask[i,max(direct_idx[i]+sample_sr*direct_range[0]//1000, 0):min(direct_idx[i]+sample_sr*direct_range[1]//1000, rir_length)] = 1.
    rir_direct = rir * direct_mask
    
    # downsample
    all_rir = torch.stack([rir, rir_direct], 1).view(nsource*2, -1)
    rir_downsample = resample1(all_rir)
    
    # apply high-pass filter
    rir_hp = highpass_biquad(rir_downsample, sample_sr//int(np.sqrt(ratio)), 80.)
    
    # downsample again
    rir = resample2(rir_hp).float().view(nsource, 2, -1)
    
    # RIR filter and direct-path RIR filter at target sample rate
    rir_filter = rir[:,0]  # nsource, nsample
    direct_rir_filter = rir[:,1]  # nsource, nsample

    return rir_filter, direct_rir_filter


if __name__ == '__main__':
    rir, direct_rir = FRA_RIR(nsource=2)
    print(rir.shape, direct_rir.shape)
