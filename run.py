#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### Import functions
import functions as fs
import numpy as np
import h5py
import os
import tensorflow as tf


"""run primme on mode filter"""
# gpus=tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=50000)])
#         logical_gpus=tf.config.list_logical_devices('GPU')
#     except RuntimeError as e:
#         print(e)

# trainset_location='/home/zhihui.tian/blue_gator_group/zhihui.tian/PRIMME_newest/image_own_textured_data_changed_cov20_0_exp_full_steps_cut0_random_from200sims.h5'
# model_location = fs.train_primme(trainset_location, num_eps=100, obs_dim=17, act_dim=17, lr=5e-5, reg=0, pad_mode="circular", if_plot=False)

# """run previous primme"""
# ic = np.load("/blue/joel.harley/zhihui.tian/PRIMME_newest/ic_compare.npy")
# ea = np.load("/blue/joel.harley/zhihui.tian/PRIMME_newest/ea_compare.npy")
# ims_id, fp_save = fs.run_primme(ic, ea, nsteps=10, modelname='/home/zhihui.tian/blue_gator_group/zhihui.tian/PRIMME_summary/data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(0)_ep(100)_ktom200sims.h5', pad_mode='circular')

# # """plot"""
# fp_save='/home/zhihui.tian/blue_gator_group/zhihui.tian/PRIMME_summary/data/primme_sz(1024x1024)_ng(4096)_nsteps(10)_freq(1)_kt_dim(2)_sz(17_17)_lr(5e-05)_reg(0)_ep(100)_ktom200sims.h5'
# fs.compute_grain_stats(fp_save)
# fs.make_videos(fp_save)
# fs.make_time_plots(fp_save)   




"""run primme on inclination dependent MCP"""
gpus=tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=15000)])
        logical_gpus=tf.config.list_logical_devices('GPU')
    except RuntimeError as e:
        print(e)

# trainset_location = "/blue/joel.harley/zhihui.tian/PRIMME_newest/only_inclination_rand_new.h5"
# model_location = fs.train_primme(trainset_location, num_eps=10, obs_dim=13, act_dim=13, lr=5e-5, reg=1, pad_mode="circular", if_plot=False)

# model_location = '/home/zhihui.tian/blue_gator_group/zhihui.tian/PRIMME_summary/data/model_dim(2)_sz(13_13)_lr(5e-05)_reg(1)_ep(10)_ktand_new.h5'
# with open("/blue/joel.harley/zhihui.tian/MF_new/MF-main/Case4.init", 'r') as file:
#     content = file.readlines()
# all=[]
# for i in range(5760000):
#     all.append(int(content[i+3].split()[1]))
# k=np.reshape(all,(2400,2400))
# ic=np.flipud(k).copy()
# ic=ic-1
# ea=fs.init2euler("/blue/joel.harley/zhihui.tian/MF_new/MF-main/Case4.init")[0,:,:]
# ims_id, fp_primme = fs.run_primme(ic,ea, nsteps=600, modelname=model_location, pad_mode='circular', if_plot=False)
# fs.compute_grain_stats(fp_primme)
# fs.make_videos(fp_primme)
# fs.make_time_plots(fp_primme)   


"""run primme on phase field data"""
# import h5py
# import numpy as np

# # Open the existing HDF5 file
# with h5py.File('/home/zhihui.tian/blue_gator_group/zhihui.tian/PRIMME_summary/training_dataset/trainset_PF_256_dt6_start1_end100_Case1.h5', 'r') as input_file:
#     dataset1 = input_file['ims_id'][:]
#     modified_dataset1 =dataset1[:,:2,:,:,:]

#     dataset2 = input_file['miso_array'][:]
#     with h5py.File('/home/zhihui.tian/blue_gator_group/zhihui.tian/PRIMME_summary/training_dataset/trainset_PF_256_dt6_start1_end100_Case1_bs2.h5', 'w') as output_file:
#         output_file.create_dataset('ims_id', data=modified_dataset1)
#         output_file.create_dataset('miso_array', data=dataset2)


gpus=tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=15000)])
        logical_gpus=tf.config.list_logical_devices('GPU')
    except RuntimeError as e:
        print(e)
trainset_location = '/home/zhihui.tian/blue_gator_group/zhihui.tian/PRIMME_summary/training_dataset/trainset_PF_256_dt6_start1_end100_Case1_bs2.h5'
model_location = fs.train_primme(trainset_location, num_eps=10, obs_dim=17, act_dim=17, lr=5e-5, reg=1, pad_mode="circular", if_plot=False)

# f = h5py.File('/home/zhihui.tian/blue_gator_group/zhihui.tian/PRIMME_summary/training_dataset/trainset_PF_256_dt6_start1_end100_Case1_bs2.h5','r')
# ic = f['ims_id'][0,0,0]
# miso_array = f['miso_array'][0]
# # ic, ea, _ = fs.voronoi2image(size=[512, 512], ngrain=512) #nsteps=500, pad_mode='circular'
ims_id, fp_primme = fs.run_primme(ic,ea=None,nsteps=2, modelname='/home/zhihui.tian/blue_gator_group/zhihui.tian/PRIMME_summary/data/phase_field/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(10)_ktt1_end100_Case1.h5',miso_array = miso_array, pad_mode='circular', if_plot=False)

# fp_primme = '/home/zhihui.tian/blue_gator_group/zhihui.tian/PRIMME_summary/data/primme_sz(256x256)_ng(118)_nsteps(2)_freq(1)_kt_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(10)_ktt1_end100_Case1.h5'

# fs.compute_grain_stats(fp_primme)
# fs.make_videos(fp_primme)
# fs.make_time_plots(fp_primme)   


