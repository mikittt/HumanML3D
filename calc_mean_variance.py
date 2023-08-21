import numpy as np
import argparse
from tqdm import tqdm
import sys
import os
from os.path import join as pjoin


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, help='NTU RGB+D root path'
    )
    return parser.parse_args()


def mean_variance(data_dir, save_dir, joints_num):
    file_list = os.listdir(data_dir)
    data_list = []
    init_pos_list = []

    for file in tqdm(file_list):
        data = np.load(pjoin(data_dir, file))
        if np.isnan(data).any():
            print(file)
            continue
        data_list.append(data[0,:-1])
        data_list.append(data[1,:-1])
        init_pos_list.append(data[0,-1,:4][None,:])
        init_pos_list.append(data[1,-1,:4][None,:])

    data = np.concatenate(data_list, axis=0)
    init_pos = np.concatenate(init_pos_list, axis=0)
    
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    init_Mean = init_pos.mean(axis=0)
    init_Std = init_pos.std(axis=0)
    Std[0:1] = Std[0:1].mean() / 1.0 # rotation velocity along y-axis
    Std[1:3] = Std[1:3].mean() / 1.0 # linear velovity on xz plane
    Std[3:4] = Std[3:4].mean() / 1.0 # root height
    Std[4: 4+(joints_num - 1) * 3] = Std[4: 4+(joints_num - 1) * 3].mean() / 1.0
    Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9] = Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9].mean() / 1.0
    Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3] = Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3].mean() / 1.0
    #Std[4 + (joints_num - 1) * 9 + joints_num * 3: ] = Std[4 + (joints_num - 1) * 9 + joints_num * 3: ].mean() / 1.0
    Std[4 + (joints_num - 1) * 9 + joints_num * 3: 4 + (joints_num - 1) * 9 + joints_num * 3 + 4] = Std[4 + (joints_num - 1) * 9 + joints_num * 3: 4 + (joints_num - 1) * 9 + joints_num * 3 + 4].mean() / 1.0
    #Std[4 + (joints_num - 1) * 9 + joints_num * 3 + 4:] = Std[4 + (joints_num - 1) * 9 + joints_num * 3 + 4:].mean() / 1.0
    #print(Std[4 + (joints_num - 1) * 9 + joints_num * 3: 4 + (joints_num - 1) * 9 + joints_num * 3 + 4].shape)
    #print(Std[4 + (joints_num - 1) * 9 + joints_num * 3 + 4:].shape)
    #print(Std[4 + (joints_num - 1) * 9 + joints_num * 3: 4 + (joints_num - 1) * 9 + joints_num * 3 + 4])
    #print(Std[4 + (joints_num - 1) * 9 + joints_num * 3: 4 + (joints_num - 1) * 9 + joints_num * 3 + 4].mean() / 1.0)
    #print(Std[4 + (joints_num - 1) * 9 + joints_num * 3 + 4:].mean() / 1.0)
    Mean = np.concatenate([Mean, init_Mean])
    Std = np.concatenate([Std, init_Std])
    print(Std, Mean)
    np.save(pjoin(save_dir, 'Mean.npy'), Mean)
    np.save(pjoin(save_dir, 'Std.npy'), Std)

    return Mean, Std

if __name__ == '__main__':
    args = get_args()
    data_dir = os.path.join(args.dataset_path, 'preprocessed', 'NTURGBD_multi/new_joint_vecs/')
    save_dir = os.path.join(args.dataset_path, 'preprocessed', 'NTURGBD_multi/')
    mean, std = mean_variance(data_dir, save_dir, 22)