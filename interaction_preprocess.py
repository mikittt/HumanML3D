import pickle
from os.path import join as pjoin
import shutil
import argparse
import numpy as np
import os
from glob import glob

import torch
from tqdm import tqdm
import random
from common.skeleton import Skeleton
from common.quaternion import *
from paramUtil import *

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import ArtistAnimation

#filter for smoothing
from scipy.signal import savgol_filter

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_dir", type=str, help='None'
    )
    parser.add_argument(
        '--vis', action='store_true'
    )
    return parser.parse_args()

args = get_args()
with open(os.path.join(args.video_dir, 'merged.pkl'), 'rb') as f:
    data = pickle.load(f)
    

def draw_original(one_data, save_path, fps=20):
    chains = [[0, 1, 4, 7, 10], 
            [0, 2, 5, 8, 11], 
            [0, 3, 6, 9, 12, 15], 
            [9, 13, 16, 18, 20, 22], 
            [9, 14, 17, 19, 21, 23]]
    skels = np.copy(one_data['joints'])
    skel0 = skels[0]
    skel1 = skels[1]
    skel0[:,:,1] = -skel0[:,:,1]
    skel0[:,:,2] = -skel0[:,:,2]
    skel1[:,:,1] = -skel1[:,:,1]
    skel1[:,:,2] = -skel1[:,:,2]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(20, 50)

    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([-0.8, 0.8])
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_facecolor('none')

    frames = []  
    for frame_idx in range(len(skel0)):
        x0 = skel0[frame_idx, :, 0]
        z0 = skel0[frame_idx, :, 1]
        y0 = skel0[frame_idx, :, 2]
        x1 = skel1[frame_idx, :, 0]
        z1 = skel1[frame_idx, :, 1]
        y1 = skel1[frame_idx, :, 2]
        
        artist = []
        for part in chains:
            x_plot = x0[part]
            y_plot = y0[part]
            z_plot = z0[part]
            artist += ax.plot(x_plot, y_plot, z_plot, color='b', marker='o', markerfacecolor='r')
            x_plot = x1[part]
            y_plot = y1[part]
            z_plot = z1[part]
            artist += ax.plot(x_plot, y_plot, z_plot, color='y', marker='o', markerfacecolor='r')
        
        frames.append(artist)
    
    ani = ArtistAnimation(fig, frames)
    ani.save(save_path, fps=fps)


def plot_3d_motion(save_path, kinematic_tree, joints1, joints2, title, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')
    title_sp = title.split(' ')
    if len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])
    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data1 = joints1.copy().reshape(len(joints1), -1, 3)
    data2 = joints2.copy().reshape(len(joints2), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    MINS = np.vstack([data1.min(axis=0).min(axis=0), data2.min(axis=0).min(axis=0)]).min(axis=0)
    MAXS = np.vstack([data1.max(axis=0).max(axis=0), data2.max(axis=0).max(axis=0)]).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',  
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
             'darkred', 'darkred','darkred','darkred','darkred']
    frame_number = data1.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data1[:, :, 1] -= height_offset
    data2[:, :, 1] -= height_offset
    trajec1 = data1[:, 0, [0, 2]]
    trajec2 = data2[:, 0, [0, 2]]
    
    #data1[..., 0] -= data1[:, 0:1, 0]
    #data1[..., 2] -= data1[:, 0:1, 2]
    #data2[..., 0] -= data2[:, 0:1, 0]
    #data2[..., 2] -= data2[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0], MAXS[0], 0, MINS[2], MAXS[2])
#         ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)
        if index > 1:
            #ax.plot3D(trajec1[:index, 0]-trajec1[index, 0], np.zeros_like(trajec1[:index, 0]), trajec1[:index, 1]-trajec1[index, 1], linewidth=1.0,
            #          color='blue')
            #ax.plot3D(trajec2[:index, 0]-trajec2[index, 0], np.zeros_like(trajec2[:index, 0]), trajec2[:index, 1]-trajec2[index, 1], linewidth=1.0,
            #          color='darkred')
            ax.plot3D(trajec1[:index, 0], np.zeros_like(trajec1[:index, 0]), trajec1[:index, 1], linewidth=1.0,
                      color='blue')
            ax.plot3D(trajec2[:index, 0], np.zeros_like(trajec2[:index, 0]), trajec2[:index, 1], linewidth=1.0,
                      color='darkred')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])
        
        
        for i, (chain, color, inv_color) in enumerate(zip(kinematic_tree, colors, colors[4::-1])):
#             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data1[index, chain, 0], data1[index, chain, 1], data1[index, chain, 2], linewidth=linewidth, color=color)
            ax.plot3D(data2[index, chain, 0], data2[index, chain, 1], data2[index, chain, 2], linewidth=linewidth, color=inv_color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)

    ani.save(save_path, fps=fps)
    print(save_path, 'saved')
    plt.close()


def smooth_root_move(vecs):
    # vecs: (seq_num, pose_num, pose_dim)
 
    #use Savitzky-Golay filter along temporal direction
    smoothed = savgol_filter(np.array(vecs[:,0]), window_length=21, polyorder=3, axis=0)
    diff = np.array(vecs[:,0]) - smoothed
    vecs = vecs - np.expand_dims(diff, axis=1)

    weight = np.array([0, 0.98, 0])
    last_root=vecs[0,0]
    smoothed=[]
    for vec in vecs:
        smoothed_root_vec=last_root*weight+(1-weight)*vec[0]
        diff = vec[0]-smoothed_root_vec
        smoothed.append(vec-diff)
        last_root=smoothed_root_vec
    
    return np.array(smoothed)


def uniform_skeleton(positions, target_offset):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    # print(src_offset)
    # print(tgt_offset)
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # print(quat_params.shape)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


def calc_min_distance(joints1, joints2, mode='none'):
    if mode=='arm':
        all_min_dis = []
        for seq in range(len(joints1)):
            hand1_l = ((joints1[seq, 0, [0,2]]-joints1[seq, 20, [0,2]])**2).sum()
            hand1_r = ((joints1[seq, 0, [0,2]]-joints1[seq, 21, [0,2]])**2).sum()
            hand1 = [20,21][np.argmax([hand1_l, hand1_r])]
            hand2_l = ((joints2[seq, 0, [0,2]]-joints2[seq, 20, [0,2]])**2).sum()
            hand2_r = ((joints2[seq, 0, [0,2]]-joints2[seq, 21, [0,2]])**2).sum()
            hand2 = [20,21][np.argmax([hand2_l, hand2_r])]
            dist = ((joints1[seq, hand1]-joints2[seq, hand2])**2).sum()
            all_min_dis.append(dist)
        min_dis = min(all_min_dis)
    else:
        all_min_dis = []
        for seq in range(len(joints1)):
            min_dis = []
            for pos in range(len(joints1[0])):
                min_dis.append(np.linalg.norm(joints1[seq][pos]-joints2[seq], axis=-1).min())
            all_min_dis.append(min(min_dis))
        min_dis = min(all_min_dis)
    return min_dis


def check_y_move(positions1, positions2, y_move_thresh=0.5):
    flag = True

    ## jump category does not exist this time
    float_threshold = 0.15
    float_num_threshold = 7
    floating_num1 = len(np.where(
        (positions1[:,[8, 11],1].min(axis=-1)>float_threshold) * (positions1[:,[7, 10],1].min(axis=-1)>float_threshold)
        )[0]) 
    floating_num2 = len(np.where(
        (positions2[:,[8, 11],1].min(axis=-1)>float_threshold) * (positions2[:,[7, 10],1].min(axis=-1)>float_threshold)
        )[0])
    if floating_num1>float_num_threshold or floating_num2>float_num_threshold:
        flag=False

    ## check y axis move (drop floating samples)
    height1 = max(positions1[:,[7, 8, 10, 11],1].min(axis=1))
    height2 = max(positions2[:,[7, 8, 10, 11],1].min(axis=1))
    #print(height1, height2)
    if height1>y_move_thresh or height2>y_move_thresh:
        #print(height1, height2)
        flag = False
    return flag

def detect_fast_move(positions1, positions2, threshold=0.1, hand_move_threshold=0.4):
    ## detect too fast move
    flag = True
    move1 = np.linalg.norm(positions1[1:]-positions1[:-1], axis=-1)
    move2 = np.linalg.norm(positions2[1:]-positions2[:-1], axis=-1)
    acc1 = np.abs(move1[1:]-move1[:-1])
    acc2 = np.abs(move2[1:]-move2[:-1])
    #print(acc1[:,0].max(), acc2[:,0].max(), acc1[:,[20,21]].max(), acc2[:,[20,21]].max())
    ## root
    if acc1[:,0].max()>threshold or acc2[:,0].max()>threshold:
        flag=False
    ## hand
    if acc1[:,[20,21]].max()>hand_move_threshold or acc2[:,[20,21]].max()>hand_move_threshold:
        flag=False
    return flag

def process_file(positions1, positions2, class_name, feet_thre, vis=False):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''
    positions1 = uniform_skeleton(positions1, tgt_offsets)
    positions2 = uniform_skeleton(positions2, tgt_offsets)
    #positions1+=trans[0][:,None]
    #positions2+=trans[1][:,None]
    
    #     print(floor_height)

    '''XZ at origin'''
    root_pos_init = (positions1[0]+positions2[0])/2
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions1 = positions1 - root_pose_init_xz
    positions2 = positions2 - root_pose_init_xz


    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    '''All initially face Z+'''
    
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions1.shape[:-1] + (4,)) * root_quat_init

    positions1 = qrot_np(root_quat_init, positions1)
    positions2 = qrot_np(root_quat_init, positions2)

    move1 = positions1[0,0]
    move2 = positions2[0,0]
    positions1 = positions1-move1[None, None, :]
    positions2 = positions2-move2[None, None, :]
    forward_init_1 = (positions1[0,9]-positions1[0,0])[None, :]
    forward_init_1 = forward_init_1 / np.sqrt((forward_init_1 ** 2).sum(axis=-1))[..., np.newaxis]
    forward_init_2 = (positions2[0,9]-positions2[0,0])[None, :]
    forward_init_2 = forward_init_1 / np.sqrt((forward_init_2 ** 2).sum(axis=-1))[..., np.newaxis]
    if forward_init_1[0,1]<0 or forward_init_2[0,1]<0:
        return None
    target = np.array([[0, 1, 0]])
    root_quat_init1 = np.ones(positions1.shape[:-1] + (4,)) * qbetween_np(forward_init_1, target)
    root_quat_init2 = np.ones(positions1.shape[:-1] + (4,)) * qbetween_np(forward_init_2, target)

    positions1 = qrot_np(root_quat_init1, positions1)+move1[None, None, :]
    positions2 = qrot_np(root_quat_init2, positions2)+move2[None, None, :]


    '''smooth root pos move and Put on Floor'''
    positions1 = smooth_root_move(positions1)
    positions2 = smooth_root_move(positions2)
    floor_height1 = positions1.min(axis=0).min(axis=0)[1]
    positions1[:, :, 1] -= floor_height1
    floor_height2 = positions2.min(axis=0).min(axis=0)[1]
    positions2[:, :, 1] -= floor_height2
    
    flag = detect_fast_move(positions1, positions2)
    if not flag:
        print('fast move detected')
        return None

    flag = check_y_move(positions1, positions2)
    if not flag:
        return None

    move1 = positions1[0]
    move1 = move1[0] * np.array([1, 0, 1])
    move2 = positions2[0]
    move2 = move2[0] * np.array([1, 0, 1])
    #move1 = positions1[0,0]
    #move2 = positions2[0,0]
    positions1 = positions1-move1[None, None, :]
    positions2 = positions2-move2[None, None, :]

    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    root_pos_init = positions1[0]
    across = root_pos_init[r_hip] - root_pos_init[l_hip] + root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init1 = np.ones(positions1.shape[:-1] + (4,)) * root_quat_init
    positions1 = qrot_np(root_quat_init1, positions1)

    root_pos_init = positions2[0]
    across = root_pos_init[r_hip] - root_pos_init[l_hip] + root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init2 = np.ones(positions2.shape[:-1] + (4,)) * root_quat_init
    positions2 = qrot_np(root_quat_init2, positions2)
    
    '''New ground truth positions'''
    global_positions1 = positions1.copy()
    global_positions2 = positions2.copy()
    

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l1, feet_r1 = foot_detect(positions1, feet_thre)
    feet_l2, feet_r2 = foot_detect(positions2, feet_thre)

    '''Quaternion and Cartesian representation'''
    r_rot1 = None
    r_rot2 = None

    def get_rifke(positions, r_rot):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params) 
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy() 
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity) 
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params1, r_velocity1, velocity1, r_rot1 = get_cont6d_params(positions1)
    positions1 = get_rifke(positions1, r_rot1)
    cont_6d_params2, r_velocity2, velocity2, r_rot2 = get_cont6d_params(positions2)
    positions2 = get_rifke(positions2, r_rot2)

    #     trejec = np.cumsum(np.concatenate([np.array([[0, 0, 0]]), velocity], axis=0), axis=0)
    #     r_rotations, r_pos = recover_ric_glo_np(r_velocity, velocity[:, [0, 2]])

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(ground_positions[:, 0, 0], ground_positions[:, 0, 2], marker='o', color='r')
    # plt.plot(trejec[:, 0], trejec[:, 2], marker='^', color='g')
    # plt.plot(r_pos[:, 0], r_pos[:, 2], marker='s', color='y')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    '''Root height'''
    root_y1 = positions1[:, 0, 1:2]
    root_y2 = positions2[:, 0, 1:2]
    
    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity1 = np.arcsin(r_velocity1[:, 2:3])
    l_velocity1 = velocity1[:, [0, 2]]
    r_velocity2 = np.arcsin(r_velocity2[:, 2:3])
    l_velocity2 = velocity2[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)

    root_data1 = np.concatenate([r_velocity1, l_velocity1, root_y1[:-1]], axis=-1)
    root_data2 = np.concatenate([r_velocity2, l_velocity2, root_y2[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data1 = cont_6d_params1[:, 1:].reshape(len(cont_6d_params1), -1)
    rot_data2 = cont_6d_params2[:, 1:].reshape(len(cont_6d_params2), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data1 = positions1[:, 1:].reshape(len(positions1), -1)
    ric_data2 = positions2[:, 1:].reshape(len(positions2), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel1 = qrot_np(np.repeat(r_rot1[:-1, None], global_positions1.shape[1], axis=1),
                        global_positions1[1:] - global_positions1[:-1])
    local_vel1 = local_vel1.reshape(len(local_vel1), -1)
    local_vel2 = qrot_np(np.repeat(r_rot2[:-1, None], global_positions2.shape[1], axis=1),
                        global_positions2[1:] - global_positions2[:-1])
    local_vel2 = local_vel2.reshape(len(local_vel2), -1)

    data1 = root_data1 # (seq_len, 4)(4: rotation v, linear v, root height)
    data1 = np.concatenate([data1, ric_data1[:-1]], axis=-1) # (seq_len, 4+63) (63: Joint Rotation Invariant Position Representio (joints_num-1)*3)
    data1 = np.concatenate([data1, rot_data1[:-1]], axis=-1) # (seq_len, 4+63+126) (126: Joint Rotation Representation (joints_num-1)*6(quaternion))
    data1 = np.concatenate([data1, local_vel1], axis=-1) # (seq_len, 4+63+126+66) (66: Joint Velocity joints_num*3)
    data1 = np.concatenate([data1, feet_l1, feet_r1], axis=-1) # (seq_len, 4+63+126+66+4) (4: foot_detect)

    init_pos1 = np.zeros((1, data1.shape[1]))
    init_pos1[0][0] = move1[0]
    init_pos1[0][1] = move1[2]
    init_pos1[0][2] = root_quat_init1[0,0,0]
    init_pos1[0][3] = -root_quat_init1[0,0,2]
    data1 = np.concatenate([data1, init_pos1], axis=0)

    data2 = root_data2
    data2 = np.concatenate([data2, ric_data2[:-1]], axis=-1)
    data2 = np.concatenate([data2, rot_data2[:-1]], axis=-1)
    data2 = np.concatenate([data2, local_vel2], axis=-1)
    data2 = np.concatenate([data2, feet_l2, feet_r2], axis=-1)

    init_pos2 = np.zeros((1, data2.shape[1]))
    init_pos2[0][0] = move2[0]
    init_pos2[0][1] = move2[2]
    init_pos2[0][2] = root_quat_init2[0,0,0]
    init_pos2[0][3] = -root_quat_init2[0,0,2]
    data2 = np.concatenate([data2, init_pos2], axis=0)

    return data1, global_positions1, positions1, l_velocity1, data2, global_positions2, positions2, l_velocity2

# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)
    
    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device) # (1, seq_len, 3)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3] # dataのlinear_vをr_posに入れる。
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data1, data2, joints_num):
    data1, init_state1, data2, init_state2 = data1[:,:-1], data1[:,-1], data2[:,:-1], data2[:,-1]
    init_pos1, init_pos2 = init_state1[:,:2], init_state2[:,:2]
    
    root_quat_init1, root_quat_init2 = torch.zeros((init_state1.shape[0],4)), torch.zeros((init_state1.shape[0],4))
    root_quat_init1[:,0] = init_state1[:,2]
    root_quat_init1[:,2] = init_state1[:,3]
    root_quat_init2[:,0] = init_state2[:,2]
    root_quat_init2[:,2] = init_state2[:,3]

    r_rot_quat1, r_pos1 = recover_root_rot_pos(data1)
    r_rot_quat2, r_pos2 = recover_root_rot_pos(data2)
    positions1 = data1[..., 4:(joints_num - 1) * 3 + 4] ## root, face Z+
    positions1 = positions1.view(positions1.shape[:-1] + (-1, 3))
    positions2 = data2[..., 4:(joints_num - 1) * 3 + 4] ## root, face Z+
    positions2 = positions2.view(positions2.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions1 = qrot(qinv(r_rot_quat1[..., None, :]).expand(positions1.shape[:-1] + (4,)), positions1)
    positions2 = qrot(qinv(r_rot_quat2[..., None, :]).expand(positions2.shape[:-1] + (4,)), positions2)

    '''Add root XZ to joints'''
    positions1[..., 0] += r_pos1[..., 0:1]
    positions1[..., 2] += r_pos1[..., 2:3]
    positions2[..., 0] += r_pos2[..., 0:1]
    positions2[..., 2] += r_pos2[..., 2:3]

    '''Concate root and joints'''
    positions1 = torch.cat([r_pos1.unsqueeze(-2), positions1], dim=-2)
    positions2 = torch.cat([r_pos2.unsqueeze(-2), positions2], dim=-2)

    positions1 = qrot(torch.ones(positions1.shape[:-1] + (4,)) * root_quat_init1, positions1)
    positions2 = qrot(torch.ones(positions2.shape[:-1] + (4,)) * root_quat_init2, positions2)
    
    positions1[...,0]+=init_pos1[:, None, 0]
    positions1[...,2]+=init_pos1[:, None, 1]
    positions2[...,0]+=init_pos2[:, None, 0]
    positions2[...,2]+=init_pos2[:, None, 1]

    return positions1, positions2


def down_sample(data, original_fps=30, target_fps=20):
    down_sample = int(original_fp / target_fps)


if __name__=='__main__':
    #trans_matrix = np.array([[1.0, 0.0, 0.0],
    #                        [0.0, 0.0, 1.0],
    #                        [0.0, 1.0, 0.0]])
    example_data = data[0]['joints'][0][0]
    #example_data[...,0]*=-1
    example_data[...,1]*=-1
    example_data[...,2]*=-1
    example_data = torch.from_numpy(example_data)
    # Lower legs
    l_idx1, l_idx2 = 5, 8
    # Right/Left foot
    fid_r, fid_l = [8, 11], [7, 10]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [2, 1, 17, 16]
    # l_hip, r_hip
    r_hip, l_hip = 2, 1
    joints_num = 22
    # ds_num = 8
    data_dir = './joints/'
    save_dir1 = './NTURGBD_multi/new_joints/'
    save_dir2 = './NTURGBD_multi/new_joint_vecs/'
    
    #n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    #kinematic_chain = t2m_kinematic_chain

    #tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    #tgt_offsets = tgt_skel.get_offsets_joints(example_data)

    example_id = "000021"
    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain

    # Get offsets of target skeleton
    example_data = np.load(os.path.join(data_dir, example_id + '.npy'))
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    # (joints_num, 3)
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
    
    visualization_mode = args.vis

    if not visualization_mode:
        if os.path.exists(save_dir1):
            shutil.rmtree(save_dir1)
        if os.path.exists(save_dir2):
            shutil.rmtree(save_dir2)
        os.makedirs(save_dir1, exist_ok=True)
        os.makedirs(save_dir2, exist_ok=True)
        for one_data in tqdm(data):
            try:
                source_file = one_data['file_name']+'.npy'
                class_name = source_file.split('A')[1][:3]
                source_data = one_data['joints'][:, :, :joints_num]
                source_data[...,1]*=-1
                source_data[...,2]*=-1
                data1, _, _, _, data2, _, _, _ = process_file(
                    source_data[0], source_data[1], class_name, 0.002)
                rec_ric_data1, rec_ric_data2 = recover_from_ric(
                    torch.from_numpy(data1).unsqueeze(0).float(), 
                    torch.from_numpy(data2).unsqueeze(0).float(),
                    joints_num)
                rec_ric_data1, rec_ric_data2 = rec_ric_data1.numpy(), rec_ric_data2.numpy()
                rec_ric_data = np.concatenate([rec_ric_data1, rec_ric_data2], axis=0)
                data = np.concatenate([data1[None,:], data2[None,:]], axis=0)
                #min_distance = calc_min_distance(rec_ric_data[0], rec_ric_data[1])
                #if class_name in ['058']:
                np.save(pjoin(save_dir1, source_file), rec_ric_data)
                np.save(pjoin(save_dir2, source_file), data)
            except:
                print(source_file)
                pass
    else:
        for one_path in glob('positions_*.gif'):
            os.remove(one_path)

        random_target_num = random.randint(0, len(data)-1)
        one_data = data[random_target_num]
        #draw_original(one_data, './original.gif')
        source_file = one_data['file_name']+'.npy'
        class_name = source_file.split('A')[1][:3]
        source_data = one_data['joints'][:, :, :joints_num]
        source_data[...,1]*=-1
        source_data[...,2]*=-1
        data1, _, _, _, data2, _, _, _ = process_file(
            source_data[0], source_data[1], class_name, 0.002, vis=False)
        rec_ric_data1, rec_ric_data2 = recover_from_ric(
            torch.from_numpy(data1).unsqueeze(0).float(), 
            torch.from_numpy(data2).unsqueeze(0).float(),
            joints_num)
        rec_ric_data1, rec_ric_data2 = rec_ric_data1.squeeze(0).numpy(), rec_ric_data2.squeeze(0).numpy()
        calc_min_distance(rec_ric_data1, rec_ric_data2)
        
        plot_3d_motion("./positions.gif", kinematic_chain, rec_ric_data1, rec_ric_data2, 'title', fps=20)