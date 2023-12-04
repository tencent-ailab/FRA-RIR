"""!
Author: Rongzhi Gu, Yi Luo
Copyright: Tencent AI Lab
"""

import os
from FRAM_RIR import FRAM_RIR
import numpy as np
import torch

def sample_mic_arch(n_mic, mic_spacing=None, bounding_box=None):
    if mic_spacing is None:
        mic_spacing = [0.02, 0.10]
    if bounding_box is None:
        bounding_box = [0.08, 0.12, 0]

    sample_n_mic = np.random.randint(n_mic[0], n_mic[1] + 1)
    if sample_n_mic == 1:
        mic_arch = np.array([[0, 0, 0]])
    else:
        mic_arch = []
        while len(mic_arch) < sample_n_mic:
            this_mic_pos = np.random.uniform(
                np.array([0, 0, 0]), np.array(bounding_box))

            if len(mic_arch) != 0:
                ok = True
                for other_mic_pos in mic_arch:
                    this_mic_spacing = np.linalg.norm(this_mic_pos - other_mic_pos)
                    if this_mic_spacing < mic_spacing[0] or this_mic_spacing > mic_spacing[1]:
                        ok = False
                        break
                if ok:
                    mic_arch.append(this_mic_pos)
            else:
                mic_arch.append(this_mic_pos)
        mic_arch = np.stack(mic_arch, 0)  # [nmic, 3]
    return mic_arch


def sample_src_pos(room_dim, num_src, array_pos,
                   min_mic_dis=0.5, max_mic_dis=5, min_dis_wall=None):
    if min_dis_wall is None:
        min_dis_wall = [0.5, 0.5, 0.5]

    # random sample the source positon
    src_pos = []
    while len(src_pos) < num_src:
        pos = np.random.uniform(np.array(min_dis_wall), np.array(
            room_dim) - np.array(min_dis_wall))
        dis = np.linalg.norm(pos - np.array(array_pos))
        
        if dis >= min_mic_dis and dis <= max_mic_dis:
            src_pos.append(pos)

    return np.stack(src_pos, 0)


def sample_mic_array_pos(mic_arch, room_dim, min_dis_wall=None):
    """
    Generate the microphone array position according to the given microphone architecture (geometry)
    :param mic_arch: np.array with shape [n_mic, 3]
                    the relative 3D coordinate to the array_pos in (m)
                    e.g., 2-mic linear array [[-0.1, 0, 0], [0.1, 0, 0]];
                    e.g., 4-mic circular array [[0, 0.035, 0], [0.035, 0, 0], [0, -0.035, 0], [-0.035, 0, 0]]
    :param min_dis_wall: minimum distance from the wall in (m)
    :return
        mic_pos: microphone array position in (m) with shape [n_mic, 3]
        array_pos: array CENTER / REFERENCE position in (m) with shape [1, 3]
    """
    def rotate(angle, valuex, valuey):
        rotate_x = valuex * np.cos(angle) + valuey * np.sin(angle)  # [nmic]
        rotate_y = valuey * np.cos(angle) - valuex * np.sin(angle)
        return np.stack([rotate_x, rotate_y, np.zeros_like(rotate_x)], -1)  # [nmic, 3]
    
    if min_dis_wall is None:
        min_dis_wall = [0.5, 0.5, 0.5]

    if isinstance(mic_arch, dict):  # ADHOC ARRAY
        n_mic, mic_spacing, bounding_box = mic_arch["n_mic"], mic_arch["spacing"], mic_arch["bounding_box"]
        sample_n_mic = np.random.randint(n_mic[0], n_mic[1] + 1)

        if sample_n_mic == 1:
            mic_arch = np.array([[0, 0, 0]])
        else:
            mic_arch = [np.random.uniform(np.array([0, 0, 0]), np.array(bounding_box))]
            while len(mic_arch) < sample_n_mic:
                this_mic_pos = np.random.uniform(np.array([0, 0, 0]), np.array(bounding_box))
                ok = True
                for other_mic_pos in mic_arch:
                    this_mic_spacing = np.linalg.norm(this_mic_pos - other_mic_pos)
                    if this_mic_spacing < mic_spacing[0] or this_mic_spacing > mic_spacing[1]:
                        ok = False
                        break
                if ok:
                    mic_arch.append(this_mic_pos)
            mic_arch = np.stack(mic_arch, 0)  # [nmic, 3]
    else:
        mic_arch = np.array(mic_arch)

    mic_array_center = np.mean(mic_arch, 0, keepdims=True)  # [1, 3]
    max_radius = max(np.linalg.norm(mic_arch - mic_array_center, axis=-1))
    array_pos = np.random.uniform(np.array(min_dis_wall) + max_radius,
                                  np.array(room_dim) - np.array(min_dis_wall) - max_radius).reshape(1, 3)
    mic_pos = array_pos + mic_arch
    # assume the array is always horizontal
    rotate_azm = np.random.uniform(-np.pi, np.pi)
    mic_pos = array_pos + rotate(rotate_azm, mic_arch[:, 0], mic_arch[:, 1])  # [n_mic, 3]

    return mic_pos, array_pos

def sample_a_config(simu_config):
    room_config = simu_config["min_max_room"]
    rt60_config = simu_config["rt60"]
    mic_dist_config = simu_config["mic_dist"]
    num_src = simu_config["num_src"]
    room_dim = np.random.uniform(np.array(room_config[0]), np.array(room_config[1]))
    rt60 = np.random.uniform(rt60_config[0], rt60_config[1])
    sr = simu_config["sr"]

    if "array_pos" not in simu_config.keys():   # mic_arch must be given in this case
        mic_arch = simu_config["mic_arch"]
        mic_pos, array_pos = sample_mic_array_pos(mic_arch, room_dim)
    else:
        array_pos = simu_config["array_pos"]

    if "src_pos" not in simu_config.keys():
        src_pos = sample_src_pos(room_dim, num_src, array_pos, min_mic_dis=mic_dist_config[0], max_mic_dis=mic_dist_config[1])
    else:
        src_pos = np.array(simu_config["src_pos"])

    return mic_pos, sr, rt60, room_dim, src_pos, array_pos


# === single-channel FRA-RIR ===
def single_channel(simu_config):
    mic_arch = {
        'n_mic': [1, 1],
        'spacing': None,
        'bounding_box': None
    }
    simu_config["mic_arch"] = mic_arch
    mic_pos, sr, rt60, room_dim, src_pos, array_pos = sample_a_config(simu_config)

    rir, rir_direct = FRAM_RIR(mic_pos, sr, rt60, room_dim, src_pos, array_pos)
    # with shape [1, n_src, rir_len]
    print(rir.shape, rir_direct.shape)


# === multi-channel (fixed) ===
def multi_channel_array(simu_config):    
    mic_arch = [[-0.05, 0, 0], [0.05, 0, 0]]
    
    simu_config["mic_arch"] = mic_arch
    mic_pos, sr, rt60, room_dim, src_pos, array_pos = sample_a_config(simu_config)

    rir, rir_direct = FRAM_RIR(mic_pos, sr, rt60, room_dim, src_pos)
    # with shape [n_mic, n_src, rir_len]
    print(rir.shape, rir_direct.shape)


# === multi-channel (adhoc) ===
def multi_channel_adhoc(simu_config):
    mic_arch = {
        'n_mic': [1, 3],
        'spacing': [0.02, 0.05],
        'bounding_box': [0.5, 1.0, 0],  # x, y, z
    }
    simu_config["mic_arch"] = mic_arch
    mic_pos, sr, rt60, room_dim, src_pos, array_pos = sample_a_config(simu_config)

    rir, rir_direct = FRAM_RIR(mic_pos, sr, rt60, room_dim, src_pos)
    # with shape [sample_n_mic, n_src, rir_len]
    print(rir.shape, rir_direct.shape)


def multi_channel_src_orientation():
    """
    ========================= → y axis
    |                       |
    |    *1          *2     |      
    |                       |
    |          ↑            |      
    |                       |
    |    *3          *4     |
    |                       |
    =========================
    ↓
    x axis
    """
    sr = 16000
    rt60 = 0.6
    room_dim = [8, 8, 3]
    src_pos = np.array([[4, 4, 1.5]])   # middle of the room
    mic_pos = np.array([[2, 2, 1.5], [2, 6, 1.5],   # mic 1, 2
                        [6, 2, 1.5], [6, 6, 1.5]])  # mic 3, 4
    src_pattern = "sub_cardioid"
    src_orientation_rad = np.array([180, 90]) / 180. * np.pi    # facing *front* (negative x axis) 

    rir, rir_direct = FRAM_RIR(mic_pos, sr, rt60, room_dim=room_dim, 
                               src_pos=src_pos, src_pattern=src_pattern, src_orientation_rad=src_orientation_rad)
    
    print(rir.shape, rir_direct.shape)
    

def multi_channel_mic_orientation():
    """    
    ========================= → y axis
    |                       |
    |    ↑1          ↓2     |      
    |                       |
    |          o            |      
    |                       |
    |    ↑3          ↓4     |
    |                       |
    =========================
    ↓
    x axis
    """

    sr = 16000
    rt60 = 0.6
    room_dim = [8, 8, 3]
    src_pos = np.array([[4, 4, 1.5]])   # middle of the room
    mic_pos = np.array([[2, 2, 1.5], [2, 6, 1.5],   # mic 1, 2
                        [6, 2, 1.5], [6, 6, 1.5]])  # mic 3, 4
    mic_pattern = "sub_cardioid"
    mic_orientation_rad = np.array([[180, 90], [0, 90],    # mic 1 (negative x axis), 2 (positive x axis)
                                    [180, 90], [0, 90]]) / 180. * np.pi    # mic 3 (negative x axis), 4 (positive x axis)

    rir, rir_direct = FRAM_RIR(mic_pos, sr, rt60, room_dim=room_dim, 
                               src_pos=src_pos, mic_pattern=mic_pattern, mic_orientation_rad=mic_orientation_rad)
    
    print(rir.shape, rir_direct.shape)
    
if __name__ == "__main__":

    seed = 20231
    np.random.seed(seed)
    torch.manual_seed(seed)

    simu_config = {
        "min_max_room": [[3, 3, 2.5], [10, 6, 4]],
        "rt60": [0.1, 0.7],
        "sr": 16000,
        "mic_dist": [0.2, 5.0],
        "num_src": 1,
    }
    single_channel(simu_config)
    multi_channel_array(simu_config)
    multi_channel_adhoc(simu_config)
    multi_channel_src_orientation()
    multi_channel_mic_orientation()