#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IF THIS CODE IS USED FOR A RESEARCH PUBLICATION, please cite:
    Yan, W., Melville, J., Yadav, V., Everett, K., Yang, L., Kesler, M. S., ... & Harley, J. B. (2022). A novel physics-regularized interpretable machine learning model for grain growth. Materials & Design, 222, 111032.
"""

# IMPORT LIBRARIES
import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, Input, Flatten, BatchNormalization, Dropout
import keras.backend as K
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam
import functions as fs
import torch
import h5py
import matplotlib.pyplot as plt


# Setup gpu access
import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(physical_devices[4], 'GPU')
# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.set_visible_devices(physical_devices[1], 'GPU') #0:5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PRIMME:
    def __init__(self, obs_dim=9, act_dim=9, pad_mode="circular", learning_rate=0.00005, reg=1, num_dims=2, cfg='./cfg/dqn_setup.json'):
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pad_mode = pad_mode
        self.learning_rate = learning_rate
        self.reg = reg
        self.num_dims = num_dims
        self.model = self._build_model()#.to(self.device)
        self.training_loss = []
        self.validation_loss = []
        self.training_acc = []
        self.validation_acc = []

    
    def _build_model(self):
        state_input = Input(shape=(self.obs_dim,)*self.num_dims)
        h0 = state_input
        h1 = Flatten()(h0)
        h2 = BatchNormalization()(h1)
        h3 = Dense(21*21*4, activation='relu')(h2)
        h4 = Dropout(0.25)(h3)
        h5 = BatchNormalization()(h4)
        h6 = Dense(21*21*2, activation='relu')(h5)
        h7 = Dropout(0.25)(h6)
        h9 = BatchNormalization()(h7)
        h8 = Dense(21*21, activation='relu')(h7)
        h9 = BatchNormalization()(h8)
        output = Dense(self.act_dim**self.num_dims,  activation='sigmoid')(h9)
        model = Model(inputs=state_input, outputs=output)
        adam = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=adam, loss='mse')
        return model

    
    def sample_data(self, h5_path='spparks_data_size257x257_ngrain256-256_nsets200_future4_max100_offset1_kt0.h5', batch_size=1):
        with h5py.File(h5_path, 'r') as f:
            i_max = f['ims_id'].shape[0]
            i_batch = np.sort(np.random.randint(low=0, high=i_max, size=(batch_size,)))
            batch = f['ims_id'][i_batch,]
            miso_array = f['miso_array'][i_batch,] 
        self.im_seq = torch.from_numpy(batch[0,].astype(float)).to(self.device)
        miso_array = torch.from_numpy(miso_array.astype(float)).to(self.device)

        

        # self.miso_matrix0 = fs.miso_array_to_matrix(miso_array[:,:,0])
        # self.miso_matrix1 = fs.miso_array_to_matrix(miso_array[:,:,1])
        # self.miso_matrix2 = fs.miso_array_to_matrix(miso_array[:,:,2])
        # self.miso_matrix3 = fs.miso_array_to_matrix(miso_array[:,:,3])
        # self.miso_matrix_4d=torch.cat((self.miso_matrix0, self.miso_matrix1, self.miso_matrix2, self.miso_matrix3), dim=0)
        self.miso_matrix = fs.miso_array_to_matrix(miso_array)

        del miso_array
        torch.cuda.empty_cache()        ### unnecessary since the variable is removed after the function is finished
        
        #Compute features and labels
        # self.features = fs.compute_features(self.im_seq[0:1,], obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        self.labels = fs.compute_labels(self.im_seq, obs_dim=self.obs_dim, act_dim=self.act_dim, reg=self.reg, pad_mode=self.pad_mode) # labels is calculated by whole sequence
        
        #Use miso functions
        self.features = fs.compute_features_miso(self.im_seq[0:1,], self.miso_matrix, obs_dim=self.obs_dim, pad_mode=self.pad_mode)  # feature is calculate by the first time point
        # self.labels = fs.compute_labels_miso(self.im_seq, self.miso_matrix, obs_dim=self.obs_dim, act_dim=self.act_dim, reg=self.reg, pad_mode=self.pad_mode)
        
    # def sample_data(self, h5_path='spparks_data_size257x257_ngrain256-256_nsets200_future4_max100_offset1_kt0.h5', batch_size=1):
    #     with h5py.File(h5_path, 'r') as f:
    #         i_max = f['ims_id'].shape[0]
    #         i_batch = np.sort(np.random.randint(low=0, high=i_max, size=(batch_size,)))
    #         batch = f['ims_id'][i_batch,]
    #         miso_array_all=f['miso_array'][()]
    #         new_array = miso_array_all[np.newaxis, :, :]
    #         miso_array = np.broadcast_to(new_array, (595, 199990000, 4))
    #         miso_array = miso_array[i_batch,]
    #     self.im_seq = torch.from_numpy(batch[0,].astype(float)).to(self.device)
    #     self.miso_array = torch.from_numpy(miso_array.astype(float)).to(self.device)

    #     self.miso_matrix0 = fs.miso_array_to_matrix(self.miso_array[:,:,0])
    #     self.miso_matrix1 = fs.miso_array_to_matrix(self.miso_array[:,:,1])
    #     self.miso_matrix2 = fs.miso_array_to_matrix(self.miso_array[:,:,2])
    #     self.miso_matrix3 = fs.miso_array_to_matrix(self.miso_array[:,:,3])
    #     self.miso_matrix=torch.cat((self.miso_matrix0, self.miso_matrix1, self.miso_matrix2, self.miso_matrix3), dim=0)

    #     # self.miso_matrix = fs.miso_array_to_matrix(miso_array)
        
    #     #Compute features and labels
    #     # self.features = fs.compute_features(self.im_seq[0:1,], obs_dim=self.obs_dim, pad_mode=self.pad_mode)
    #     del self.miso_array
    #     del self.miso_matrix0
    #     del self.miso_matrix1
    #     del self.miso_matrix2
    #     del self.miso_matrix3
    #     torch.cuda.empty_cache() 
    #     self.labels = fs.compute_labels(self.im_seq, obs_dim=self.obs_dim, act_dim=self.act_dim, reg=self.reg, pad_mode=self.pad_mode) # labels is calculated by whole sequence
        
    #     #Use miso functions
    #     self.features = fs.compute_features_miso(self.im_seq[0:1,], self.miso_matrix, obs_dim=self.obs_dim, pad_mode=self.pad_mode)  # feature is calculate by the first time point
    #     # self.labels = fs.compute_labels_miso(self.im_seq, self.miso_matrix, obs_dim=self.obs_dim, act_dim=self.act_dim, reg=self.reg, pad_mode=self.pad_mode)
        
    def step(self, im, miso_matrix, evaluate=True):
        # features = fs.compute_features(im, obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        features = fs.compute_features_miso(im, miso_matrix, obs_dim=self.obs_dim, pad_mode=self.pad_mode) #use miso functions
        mid_ix = (np.array(features.shape[1:])/2).astype(int)
        ind = tuple([slice(None)]) + tuple(mid_ix)
        indx_use = torch.nonzero(features[ind])[:,0]
        features = features[indx_use,]
        
        action_features = fs.my_unfoldNd(im, kernel_size=self.act_dim, pad_mode=self.pad_mode)[0,] 
        action_features = action_features[...,indx_use]
        
        batch_size = 5000
        features_split = torch.split(features, batch_size)
        predictions_split = []
        action_values_split = []
        total_list=[]
        for e in features_split:
            
            predictions = torch.Tensor(self.model.predict_on_batch(e.cpu().numpy())).to(self.device)
            # predictions = self.model(e)

            # for i in range(len(e)):
            #     total_list.append(np.array(predictions[i]).reshape(17,17))
            tmp=torch.rand(predictions.shape).to(self.device)/1e2
            predictions=predictions+tmp
            action_values = torch.argmax(predictions, dim=1)


            if evaluate==True: 
                predictions_split.append(predictions)
            action_values_split.append(action_values)
        ###
        # arg_array_all=[]
        # for i in range(len(total_list)):
        #
        #     arg_array_1=np.zeros((17,17))
        #     [a, b] = np.where(total_list[i] == np.max(total_list[i]))
        #     arg_array_1[a,b]=1
        #     arg_array_all.append(arg_array_1)
        # array_final_argmax = np.array(arg_array_all).mean(axis=0)
        # np.save('argmax_average.npy', array_final_argmax)
        if evaluate==True: self.predictions = torch.cat(predictions_split, dim=0)
        action_values = torch.hstack(action_values_split)
        
        # self.im_next = torch.gather(action_features, dim=0, index=action_values.unsqueeze(0)).reshape(im.shape)
        upated_values = torch.gather(action_features, dim=0, index=action_values.unsqueeze(0))[0,]
        self.im_next = im.flatten().float()
        self.im_next[indx_use] = upated_values
        self.im_next = self.im_next.reshape(im.shape)
        self.indx_use = indx_use
        
        return self.im_next


    def step_new(self, im, miso_matrix, n,evaluate=True):
        # features = fs.compute_features(im, obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        features = fs.compute_features_miso(im, miso_matrix, obs_dim=self.obs_dim, pad_mode=self.pad_mode) #use miso functions
        mid_ix = (np.array(features.shape[1:])/2).astype(int)
        ind = tuple([slice(None)]) + tuple(mid_ix)
        indx_use = torch.nonzero(features[ind])[:,0]
        features = features[indx_use,]
        
        action_features = fs.my_unfoldNd(im, kernel_size=self.act_dim, pad_mode=self.pad_mode)[0,] 
        action_features = action_features[...,indx_use]
        
        batch_size = 5000
        features_split = torch.split(features, batch_size)
        predictions_split = []
        action_values_split = []
        total_list=[]
        for e in features_split:
            
            predictions = torch.Tensor(self.model.predict_on_batch(e.cpu().numpy())).to(self.device)
            # predictions = self.model(e)

            for j in range(len(e)):
                total_list.append(np.array(predictions[j].cpu()).reshape(17,17))
            tmp=torch.rand(predictions.shape).to(self.device)/1e2
            predictions=predictions+tmp
            action_values = torch.argmax(predictions, dim=1)


            if evaluate==True: 
                predictions_split.append(predictions)
            action_values_split.append(action_values)
        ###
        arg_array_all=[]
        for i in range(len(total_list)):
        
            arg_array_1=np.zeros((17,17))
            [a, b] = np.where(total_list[i] == np.max(total_list[i]))
            arg_array_1[a,b]=1
            arg_array_all.append(arg_array_1)
        array_final_argmax = np.array(arg_array_all).mean(axis=0)
        # print(np.array(arg_array_all).shape)
        np.save('./all_npy/argmax_average'+str(n)+'.npy', array_final_argmax)

        if evaluate==True: self.predictions = torch.cat(predictions_split, dim=0)
        action_values = torch.hstack(action_values_split)
        
        # self.im_next = torch.gather(action_features, dim=0, index=action_values.unsqueeze(0)).reshape(im.shape)
        upated_values = torch.gather(action_features, dim=0, index=action_values.unsqueeze(0))[0,]
        self.im_next = im.flatten().float()
        self.im_next[indx_use] = upated_values
        self.im_next = self.im_next.reshape(im.shape)
        self.indx_use = indx_use
        
        return self.im_next



    def step_new(self, im, miso_matrix, n,evaluate=True):
        # features = fs.compute_features(im, obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        features = fs.compute_features_miso(im, miso_matrix, obs_dim=self.obs_dim, pad_mode=self.pad_mode) #use miso functions
        mid_ix = (np.array(features.shape[1:])/2).astype(int)
        ind = tuple([slice(None)]) + tuple(mid_ix)
        indx_use = torch.nonzero(features[ind])[:,0]
        features = features[indx_use,]
        
        action_features = fs.my_unfoldNd(im, kernel_size=self.act_dim, pad_mode=self.pad_mode)[0,] 
        action_features = action_features[...,indx_use]
        
        batch_size = 5000
        features_split = torch.split(features, batch_size)
        predictions_split = []
        action_values_split = []
        total_list=[]
        for e in features_split:
            
            predictions = torch.Tensor(self.model.predict_on_batch(e.cpu().numpy())).to(self.device)
            # predictions = self.model(e)

            for j in range(len(e)):
                total_list.append(np.array(predictions[j].cpu()).reshape(17,17))
            tmp=torch.rand(predictions.shape).to(self.device)/1e2
            predictions=predictions+tmp
            action_values = torch.argmax(predictions, dim=1)


            if evaluate==True: 
                predictions_split.append(predictions)
            action_values_split.append(action_values)
        ###
        arg_array_all=[]
        for i in range(len(total_list)):
        
            arg_array_1=np.zeros((17,17))
            [a, b] = np.where(total_list[i] == np.max(total_list[i]))
            arg_array_1[a,b]=1
            arg_array_all.append(arg_array_1)
        array_final_argmax = np.array(arg_array_all).mean(axis=0)
        # print(np.array(arg_array_all).shape)
        np.save('./all_npy/argmax_average'+str(n)+'.npy', array_final_argmax)

        if evaluate==True: self.predictions = torch.cat(predictions_split, dim=0)
        action_values = torch.hstack(action_values_split)
        
        # self.im_next = torch.gather(action_features, dim=0, index=action_values.unsqueeze(0)).reshape(im.shape)
        upated_values = torch.gather(action_features, dim=0, index=action_values.unsqueeze(0))[0,]
        self.im_next = im.flatten().float()
        self.im_next[indx_use] = upated_values
        self.im_next = self.im_next.reshape(im.shape)
        self.indx_use = indx_use
        
        return self.im_next

    def step_new_new(self, im, miso_matrix, n,evaluate=True):
        # features = fs.compute_features(im, obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        features,features_window = fs.compute_features_miso(im, miso_matrix, obs_dim=self.obs_dim, pad_mode=self.pad_mode) #use miso functions
        mid_ix = (np.array(features.shape[1:])/2).astype(int)
        ind = tuple([slice(None)]) + tuple(mid_ix)
        indx_use = torch.nonzero(features[ind])[:,0]
        features = features[indx_use,]
        
        action_features = fs.my_unfoldNd(im, kernel_size=self.act_dim, pad_mode=self.pad_mode)[0,] 
        action_features = action_features[...,indx_use]
        
        batch_size = 5000
        features_split = torch.split(features, batch_size)
        predictions_split = []
        action_values_split = []
        total_list=[]
        for e in features_split:
            
            predictions = torch.Tensor(self.model.predict_on_batch(e.cpu().numpy())).to(self.device)

            # np.save('all_features.npy',e.cpu().numpy())
            # np.save('all_predictions.npy',predictions.cpu().numpy())

            # predictions = self.model(e)

            # for j in range(len(e)):
            #     total_list.append(np.array(predictions[j].cpu()).reshape(17,17))
            
            total_array=np.array(predictions.cpu()).reshape(len(e),17,17)
            tmp=torch.rand(predictions.shape).to(self.device)/1e2
            predictions=predictions+tmp
            action_values = torch.argmax(predictions, dim=1)


            if evaluate==True: 
                predictions_split.append(predictions)
            action_values_split.append(action_values)
        ###
        # arg_array_all=[]
        # for i in range(len(total_list)):
        
        #     arg_array_1=np.zeros((17,17))
        #     [a, b] = np.where(total_list[i] == np.max(total_list[i]))
        #     arg_array_1[a,b]=1
        #     arg_array_all.append(arg_array_1)
        # array_final_argmax = np.array(arg_array_all).mean(axis=0)

        ###
        
        # max_indices = np.argmax(total_array.reshape(len(total_list), -1), axis=1)
        max_indices = np.argmax(total_array.reshape(len(e), -1), axis=1)
        arg_array_all = np.zeros_like(total_array)
        for idx, max_idx in enumerate(max_indices):
            arg_array_all[idx].flat[max_idx] = 1
        array_final_argmax = arg_array_all.mean(axis=0)

        array_only_ave=np.mean(total_array,axis=0)

        if evaluate==True: self.predictions = torch.cat(predictions_split, dim=0)
        action_values = torch.hstack(action_values_split)
        
        # self.im_next = torch.gather(action_features, dim=0, index=action_values.unsqueeze(0)).reshape(im.shape)
        upated_values = torch.gather(action_features, dim=0, index=action_values.unsqueeze(0))[0,]    #### action_values is action position, upated_values is the predicted grain id
        self.im_next = im.flatten().float()
        self.im_next[indx_use] = upated_values
        self.im_next = self.im_next.reshape(im.shape)
        self.indx_use = indx_use
        
        return self.im_next,array_final_argmax,array_only_ave,0





    def step_sum_max(self, im, miso_matrix, n,evaluate=True):
        # features = fs.compute_features(im, obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        features = fs.compute_features_miso(im, miso_matrix, obs_dim=self.obs_dim, pad_mode=self.pad_mode) #use miso functions
        mid_ix = (np.array(features.shape[1:])/2).astype(int)
        ind = tuple([slice(None)]) + tuple(mid_ix)
        indx_use = torch.nonzero(features[ind])[:,0]
        features = features[indx_use,]
        
        action_features = fs.my_unfoldNd(im, kernel_size=self.act_dim, pad_mode=self.pad_mode)[0,] 
        action_features = action_features[...,indx_use]
        
        batch_size = 5000
        features_split = torch.split(features, batch_size)
        predictions_split = []
        action_values_split = []
        total_list=[]
        
        for e in features_split:
            
            predictions = torch.Tensor(self.model.predict_on_batch(e.cpu().numpy())).to(self.device)
            # np.save('all_features.npy',e.cpu().numpy())
            # np.save('all_predictions.npy',predictions.cpu().numpy())
            # predictions = self.model(e)
            # for j in range(len(e)):
            #     total_list.append(np.array(predictions[j].cpu()).reshape(17,17))
            total_array=np.array(predictions.cpu()).reshape(len(e),17,17)
            tmp=torch.rand(predictions.shape).to(self.device)/1e2
            predictions=predictions+tmp
            
            all_feature=e.cpu().numpy()
            all_predictions=predictions.cpu().numpy()

            

            action_value_summax=[]
            for i in range(all_feature.shape[0]):
                sum_max_value = fs.sum_max(all_feature[i], all_predictions[i])
                action_value_summax.append(sum_max_value)
            action_values2=np.array(action_value_summax).reshape(all_feature.shape[0],)
            action_values=torch.from_numpy(action_values2).to(self.device)

            # action_values = torch.argmax(predictions, dim=1)


            if evaluate==True: 
                predictions_split.append(predictions)
            action_values_split.append(action_values)

        max_indices = action_values.cpu().numpy()
        arg_array_all = np.zeros_like(total_array)
        for idx, max_idx in enumerate(max_indices):
            arg_array_all[idx].flat[max_idx] = 1
        array_final_argmax = arg_array_all.mean(axis=0)

        array_only_ave=np.mean(total_array,axis=0)

        if evaluate==True: self.predictions = torch.cat(predictions_split, dim=0)
        action_values = torch.hstack(action_values_split)
        
        # self.im_next = torch.gather(action_features, dim=0, index=action_values.unsqueeze(0)).reshape(im.shape)
        upated_values = torch.gather(action_features, dim=0, index=action_values.unsqueeze(0))[0,]    #### action_values is action position
        self.im_next = im.flatten().float()
        self.im_next[indx_use] = upated_values
        self.im_next = self.im_next.reshape(im.shape)
        self.indx_use = indx_use
        
        return self.im_next,array_final_argmax,array_only_ave



    def step_sum_max_new(self, im, miso_matrix, n,evaluate=True):
        # features = fs.compute_features(im, obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        features = fs.compute_features_miso(im, miso_matrix, obs_dim=self.obs_dim, pad_mode=self.pad_mode) #use miso functions
        mid_ix = (np.array(features.shape[1:])/2).astype(int)
        ind = tuple([slice(None)]) + tuple(mid_ix)
        indx_use = torch.nonzero(features[ind])[:,0]
        features = features[indx_use,]
        
        action_features = fs.my_unfoldNd(im, kernel_size=self.act_dim, pad_mode=self.pad_mode)[0,] 
        action_features = action_features[...,indx_use]
        
        batch_size = 5000
        features_split = torch.split(features, batch_size)
        predictions_split = []
        action_values_split = []
        total_list=[]
        for e in features_split:
            
            predictions = torch.Tensor(self.model.predict_on_batch(e.cpu().numpy())).to(self.device)
            # np.save('all_features.npy',e.cpu().numpy())
            # np.save('all_predictions.npy',predictions.cpu().numpy())
            # predictions = self.model(e)
            # for j in range(len(e)):
            #     total_list.append(np.array(predictions[j].cpu()).reshape(17,17))
            total_array=np.array(predictions.cpu()).reshape(len(e),17,17)
            tmp=torch.rand(predictions.shape).to(self.device)/1e2
            predictions=predictions+tmp
            
            all_feature=e.cpu().numpy()
            all_predictions=predictions.cpu().numpy()
            action_value_summax=[]
            indx = indx_use.cpu().numpy()
            im_input=im.clone().cpu().numpy()
            for i in range(all_feature.shape[0]):
                sum_max_value = fs.sum_max_new(im_input,indx[i],all_predictions[i])
                action_value_summax.append(sum_max_value)
            action_values2=np.array(action_value_summax).reshape(all_feature.shape[0],)
            action_values=torch.from_numpy(action_values2).to(self.device)

            # action_values = torch.argmax(predictions, dim=1)


            if evaluate==True: 
                predictions_split.append(predictions)
            action_values_split.append(action_values)

        max_indices = action_values.cpu().numpy()
        arg_array_all = np.zeros_like(total_array)
        for idx, max_idx in enumerate(max_indices):
            arg_array_all[idx].flat[max_idx] = 1
        array_final_argmax = arg_array_all.mean(axis=0)

        array_only_ave=np.mean(total_array,axis=0)

        if evaluate==True: self.predictions = torch.cat(predictions_split, dim=0)
        action_values = torch.hstack(action_values_split)
        
        # self.im_next = torch.gather(action_features, dim=0, index=action_values.unsqueeze(0)).reshape(im.shape)
        upated_values = torch.gather(action_features, dim=0, index=action_values.unsqueeze(0))[0,]    #### action_values is action position
        self.im_next = im.flatten().float()
        self.im_next[indx_use] = upated_values
        self.im_next = self.im_next.reshape(im.shape)
        self.indx_use = indx_use
        
        return self.im_next,array_final_argmax,array_only_ave





    def step_sum_max_poly(self, im, miso_matrix, n,evaluate=True):
        # features = fs.compute_features(im, obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        features, features_window = fs.compute_features_miso(im, miso_matrix, obs_dim=self.obs_dim, pad_mode=self.pad_mode) #use miso functions
        mid_ix = (np.array(features.shape[1:])/2).astype(int)
        ind = tuple([slice(None)]) + tuple(mid_ix)
        indx_use = torch.nonzero(features[ind])[:,0]
        features = features[indx_use,]
        features_window_use=features_window[indx_use,]
        
        action_features = fs.my_unfoldNd(im, kernel_size=self.act_dim, pad_mode=self.pad_mode)[0,] 
        action_features = action_features[...,indx_use]
        
        batch_size = 5000

        features_split = torch.split(features, batch_size)
        features_window_split = torch.split(features_window_use, batch_size)

        predictions_split = []
        action_values_split = []
        total_list=[]

        stepi_all_predictions=[]
        results_0_1_all = []
        for e,f in zip(features_split,features_window_split):
            
            predictions = torch.Tensor(self.model.predict_on_batch(e.cpu().numpy())).to(self.device)
            # np.save('all_features.npy',e.cpu().numpy())
            # np.save('all_predictions.npy',predictions.cpu().numpy())
            # predictions = self.model(e)
            # for j in range(len(e)):
            #     total_list.append(np.array(predictions[j].cpu()).reshape(17,17))
            total_array=np.array(predictions.cpu()).reshape(len(e),17,17)
            tmp=torch.rand(predictions.shape).to(self.device)/1e2
            predictions=predictions+tmp
            
            all_feature=e.cpu().numpy()
            feature_window=f.cpu().numpy()
            all_predictions=predictions.cpu().numpy()
            action_value_summax=[]
            indx = indx_use.cpu().numpy()

            stepi_all_predictions.append(all_predictions)

            for i in range(feature_window.shape[0]):
                sum_max_value,result_0_1 = fs.sum_max_poly(feature_window[i],all_predictions[i])   ### return the position in 17*17
                results_0_1_all.append(result_0_1)
                action_value_summax.append(sum_max_value)
            action_values2=np.array(action_value_summax).reshape(feature_window.shape[0],)
            action_values=torch.from_numpy(action_values2).to(self.device)

            # action_values = torch.argmax(predictions, dim=1)


            if evaluate==True: 
                predictions_split.append(predictions)
            action_values_split.append(action_values)

        # if n == 24:
        #     np.save('step'+str(n)+'all_prediction.npy',np.concatenate(stepi_all_predictions, axis=0))
        max_indices = action_values.cpu().numpy()
        arg_array_all = np.zeros_like(total_array)
        for idx, max_idx in enumerate(max_indices):
            arg_array_all[idx].flat[max_idx] = 1
        array_final_argmax = arg_array_all.mean(axis=0)
        
        array_only_ave=np.mean(total_array,axis=0)

        result_0_1_ave = np.mean(results_0_1_all,axis=0)

        if evaluate==True: self.predictions = torch.cat(predictions_split, dim=0)
        action_values = torch.hstack(action_values_split)
        
        # self.im_next = torch.gather(action_features, dim=0, index=action_values.unsqueeze(0)).reshape(im.shape)
        upated_values = torch.gather(action_features, dim=0, index=action_values.unsqueeze(0))[0,]    #### action_values is action position
        self.im_next = im.flatten().float()
        self.im_next[indx_use] = upated_values
        self.im_next = self.im_next.reshape(im.shape)
        self.indx_use = indx_use
        
        return self.im_next,array_final_argmax,array_only_ave,result_0_1_ave



    def step_sum_max_poly_new(self, im, miso_matrix, n,evaluate=True):  ### change the size of probability map, from last batch to all!
        

        # features, features_window = fs.compute_features_miso(im, miso_matrix, obs_dim=self.obs_dim, pad_mode=self.pad_mode) #use miso functions
        features, features_window = fs.run_compute_features_miso(im, miso_matrix, obs_dim=17, pad_mode=self.pad_mode) #use miso functions
        mid_ix = (np.array(features.shape[1:])/2).astype(int)
        ind = tuple([slice(None)]) + tuple(mid_ix)
        indx_use = torch.nonzero(features[ind])[:,0]
        features = features[indx_use,]
        features_window_use=features_window[indx_use,]
        
        action_features = fs.my_unfoldNd(im, kernel_size=self.act_dim, pad_mode=self.pad_mode)[0,] 
        # action_features = fs.circular_window(im, kernel_size=5, pad_mode=self.pad_mode)[0,] 

        action_features = action_features[...,indx_use]
        
        batch_size = 5000

        features_split = torch.split(features, batch_size)
        features_window_split = torch.split(features_window_use, batch_size)

        predictions_split = []
        action_values_split = []
        total_list=[]

        stepi_all_predictions=[]
        results_0_1_all = []

        argmax_all = []
        directly_all = []
        sum_max_all = []
        grainid_all = []

        r_all = []
        c_all = []
        # np.save('all_features_cov20_step1.npy',features_window_use.cpu().numpy())  ### save all features of one step
        for e,f in zip(features_split,features_window_split):
            
            predictions = torch.Tensor(self.model.predict_on_batch(e.cpu().numpy())).to(self.device)
            # np.save('all_features.npy',e.cpu().numpy())
            # np.save('all_predictions.npy',predictions.cpu().numpy())
            # predictions = self.model(e)
            # for j in range(len(e)):
            #     total_list.append(np.array(predictions[j].cpu()).reshape(17,17))

            total_array=np.array(predictions.cpu()).reshape(len(e),17,17)
            # total_array=np.array(predictions.cpu()).reshape(len(e),5,5)
            # total_array=np.array(predictions.cpu()).reshape(len(e),197)
            tmp=torch.rand(predictions.shape).to(self.device)/1e2
            predictions=predictions+tmp
            
            all_feature=e.cpu().numpy()
            feature_window=f.cpu().numpy()
            all_predictions=predictions.cpu().numpy()
            action_value_summax=[]
            indx = indx_use.cpu().numpy()

            stepi_all_predictions.append(all_predictions)

            for i in range(feature_window.shape[0]):
                # sum_max_value, result_0_1,r,c = fs.sum_max_poly(feature_window[i],all_predictions[i])   ### return the position in 17*17
                grain_ID = fs.sum_max_poly(feature_window[i],all_predictions[i])   ### return the position in 17*17
                grainid_all.append(grain_ID)

        # np.save('all_predictions_cov20_step1.npy',np.concatenate(stepi_all_predictions, axis=0)) ### save all predictions of one step
        upated_values = torch.Tensor(np.array(grainid_all)).to(device)
        self.im_next = im.flatten().float()
        self.im_next[indx_use] = upated_values
        self.im_next = self.im_next.reshape(im.shape)
        self.indx_use = indx_use
        
        # return self.im_next,argmax_all_average,directly_all_average,sum_max_all_average
        return self.im_next




    def step_sum_max_poly_bayes(self, im, miso_matrix, n,evaluate=True):  ### change the size of probability map, from last batch to all!
        # features = fs.compute_features(im, obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        features, features_window = fs.compute_features_miso(im, miso_matrix, obs_dim=self.obs_dim, pad_mode=self.pad_mode) #use miso functions
        mid_ix = (np.array(features.shape[1:])/2).astype(int)
        ind = tuple([slice(None)]) + tuple(mid_ix)
        indx_use = torch.nonzero(features[ind])[:,0]
        features = features[indx_use,]
        features_window_use=features_window[indx_use,]
        num_window = features_window_use.shape[0]
        
        action_features = fs.my_unfoldNd(im, kernel_size=self.act_dim, pad_mode=self.pad_mode)[0,] 
        action_features = action_features[...,indx_use]
        
        batch_size = 5000

        features_split = torch.split(features, batch_size)
        features_window_split = torch.split(features_window_use, batch_size)

        predictions_split = []
        action_values_split = []
        total_list=[]

        stepi_all_predictions=[]
        results_0_1_all = []

        argmax_all = []
        directly_all = []
        sum_max_all = []
        grainid_all = []
        result_bayes_all = np.zeros((17,17))
        result_bayes_nochange_all = np.zeros((17,17))
        for e,f in zip(features_split,features_window_split):
            
            predictions = torch.Tensor(self.model.predict_on_batch(e.cpu().numpy())).to(self.device)
            total_array=np.array(predictions.cpu()).reshape(len(e),17,17)
            tmp=torch.rand(predictions.shape).to(self.device)/1e2
            predictions=predictions+tmp
            
            all_feature=e.cpu().numpy()
            feature_window=f.cpu().numpy()
            all_predictions=predictions.cpu().numpy()
            action_value_summax=[]
            indx = indx_use.cpu().numpy()

            stepi_all_predictions.append(all_predictions)

            for i in range(feature_window.shape[0]):
                result_bayes_nochange = fs.sum_max_bayes_nochange(feature_window[i],all_predictions[i],num_window) 
                result_bayes_nochange_all += result_bayes_nochange

                grainid, result_bayes = fs.sum_max_bayes(feature_window[i],all_predictions[i],num_window)   ### return the position in 17*17
                grainid_all.append(grainid)
                result_bayes_all += result_bayes

        if evaluate==True: self.predictions = torch.cat(predictions_split, dim=0)

        upated_values = torch.Tensor(np.array(grainid_all)).to(device)   #### action_values is action position
        self.im_next = im.flatten().float()
        self.im_next[indx_use] = upated_values
        self.im_next = self.im_next.reshape(im.shape)
        self.indx_use = indx_use
        
        return self.im_next,result_bayes_all,result_bayes_nochange_all


    def step_sum_max_poly_bayes_new(self, im, miso_matrix, n,evaluate=True):  ### change the size of probability map, from last batch to all!
        # features = fs.compute_features(im, obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        features, features_window = fs.compute_features_miso(im, miso_matrix, obs_dim=self.obs_dim, pad_mode=self.pad_mode) #use miso functions
        mid_ix = (np.array(features.shape[1:])/2).astype(int)
        ind = tuple([slice(None)]) + tuple(mid_ix)
        indx_use = torch.nonzero(features[ind])[:,0]
        features = features[indx_use,]
        features_window_use=features_window[indx_use,]
        num_window = features_window_use.shape[0]
        
        action_features = fs.my_unfoldNd(im, kernel_size=self.act_dim, pad_mode=self.pad_mode)[0,] 
        action_features = action_features[...,indx_use]
        
        batch_size = 5000

        features_split = torch.split(features, batch_size)
        features_window_split = torch.split(features_window_use, batch_size)

        predictions_split = []
        action_values_split = []
        total_list=[]

        stepi_all_predictions=[]
        results_0_1_all = []

        argmax_all = []
        directly_all = []
        sum_max_all = []
        grainid_all = []
        ct_all = []
        area_bayes_all = []
        area_nochange_all = []
        result_bayes_all = np.zeros((17,17))
        result_bayes_nochange_all = np.zeros((17,17))
        feature_window_all = []
        array01_all = []
        for e,f in zip(features_split,features_window_split):
            
            predictions = torch.Tensor(self.model.predict_on_batch(e.cpu().numpy())).to(self.device)
            total_array=np.array(predictions.cpu()).reshape(len(e),17,17)
            tmp=torch.rand(predictions.shape).to(self.device)/1e2
            predictions=predictions+tmp
            
            all_feature=e.cpu().numpy()
            feature_window=f.cpu().numpy()
            all_predictions=predictions.cpu().numpy()
            action_value_summax=[]
            indx = indx_use.cpu().numpy()

            stepi_all_predictions.append(all_predictions)


            for i in range(feature_window.shape[0]):

                array01 = fs.sum_max_bayes_pos(feature_window[i],all_predictions[i],num_window)   ### return the position in 17*17
                array01_all.append(array01)

            for i in range(feature_window.shape[0]):

                grainid, area_bayes = fs.sum_max_bayes_new(feature_window[i],all_predictions[i],num_window)   ### return the position in 17*17
                ct, area_bayes_nochange = fs.sum_max_bayes_nochange_new(feature_window[i],all_predictions[i],num_window)

                grainid_all.append(int(grainid))
                area_bayes_all.append(area_bayes) 

                ct_all.append(ct)
                area_nochange_all.append(area_bayes_nochange)

                feature_window_all.append(feature_window[i])

        # arrays = [np.array(lst) for lst in feature_window_all]
        feature_window_all = np.array(feature_window_all)
        """for regular result"""
        for j in np.unique(grainid_all):
            grainj_pos = np.where(np.array(grainid_all) == j)[0]
            grainj_area = 0
            for k in grainj_pos:
                grainj_area += area_bayes_all[k]
            bayes_value = grainj_area/(num_window*17*17)
            resultj_bayes_all = np.zeros((17,17))
            for k2 in grainj_pos:
                resultj_bayes = np.where(feature_window_all[k2] == j,bayes_value,0)
                resultj_bayes_all += resultj_bayes
            resultj_bayes_ave = resultj_bayes_all/len(grainj_pos)
            # resultj_bayes_ave = resultj_bayes_all
            result_bayes_all += resultj_bayes_ave
        """for result without change"""
        # for j in np.unique(ct_all):
        #     grainj_pos = np.where(np.array(ct_all) == j)[0]
        #     grainj_area_nochange = 0
        #     for k in grainj_pos:
        #         grainj_area_nochange += area_nochange_all[k]
        #     bayes_value = grainj_area_nochange/(num_window*17*17)
        #     resultj_bayes_nochange_all = np.zeros((17,17))
        #     for k2 in grainj_pos:
        #         resultj_bayes = np.where(feature_window_all[k2] == j,bayes_value,0)
        #         resultj_bayes_nochange_all += resultj_bayes
        #     resultj_bayes_nochange_ave = resultj_bayes_nochange_all/len(grainj_pos)
        #     result_bayes_nochange_all += resultj_bayes_nochange_ave

        if evaluate==True: self.predictions = torch.cat(predictions_split, dim=0)

        upated_values = torch.Tensor(np.array(grainid_all)).to(device)   #### action_values is action position
        self.im_next = im.flatten().float()
        self.im_next[indx_use] = upated_values
        self.im_next = self.im_next.reshape(im.shape)
        self.indx_use = indx_use
        
        return self.im_next,result_bayes_all
    #         # 分子
    # def each_pos_1(new_pro):
    #     array_zeros = np.zeros((17,17))
    #     for i in range(17):
    #         for j in range(17):
    #             pro_ij = 0
    #             for n in range(new_pro.shape[0]):
    #                 new_pro_pro = new_pro[n]
    #                 pro_ij += new_pro_pro[i,j]*np.sum(new_pro_pro)
    #             array_zeros[i][j] = pro_ij
    #     return array_zeros

    # #分母
    # def each_pos_2(new_pro):
    #     array_zeros2 = np.zeros((17,17))
    #     for i in range(17):
    #         for j in range(17):
    #             pro_ij = 0
    #             k=0
    #             for n in range(new_pro.shape[0]):
    #                 new_pro_pro = new_pro[n]
    #                 if new_pro_pro[i,j] == 1:
    #                     pro_ij += 1/np.sum(new_pro_pro)
    #                     k+=1
    #             array_zeros2[i][j] = pro_ij/k
    #     return array_zeros2


    def step_sum_max_poly_bayes_pos(self, im, miso_matrix, n,evaluate=True):  ### change the size of probability map, from last batch to all!
        # features = fs.compute_features(im, obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        features, features_window = fs.compute_features_miso(im, miso_matrix, obs_dim=self.obs_dim, pad_mode=self.pad_mode) #use miso functions
        mid_ix = (np.array(features.shape[1:])/2).astype(int)
        ind = tuple([slice(None)]) + tuple(mid_ix)
        indx_use = torch.nonzero(features[ind])[:,0]
        features = features[indx_use,]
        features_window_use=features_window[indx_use,]
        num_window = features_window_use.shape[0]
        
        action_features = fs.my_unfoldNd(im, kernel_size=self.act_dim, pad_mode=self.pad_mode)[0,] 
        action_features = action_features[...,indx_use]
        
        batch_size = 5000

        features_split = torch.split(features, batch_size)
        features_window_split = torch.split(features_window_use, batch_size)

        predictions_split = []
        action_values_split = []
        total_list=[]

        stepi_all_predictions=[]
        results_0_1_all = []

        argmax_all = []
        directly_all = []
        sum_max_all = []
        grainid_all = []
        ct_all = []
        area_bayes_all = []
        area_nochange_all = []
        result_bayes_all = np.zeros((17,17))
        result_bayes_nochange_all = np.zeros((17,17))
        feature_window_all = []
        array01_all = []
        for e,f in zip(features_split,features_window_split):
            
            predictions = torch.Tensor(self.model.predict_on_batch(e.cpu().numpy())).to(self.device)
            total_array=np.array(predictions.cpu()).reshape(len(e),17,17)
            tmp=torch.rand(predictions.shape).to(self.device)/1e2
            predictions=predictions+tmp
            
            all_feature=e.cpu().numpy()
            feature_window=f.cpu().numpy()
            all_predictions=predictions.cpu().numpy()
            action_value_summax=[]
            indx = indx_use.cpu().numpy()

            stepi_all_predictions.append(all_predictions)


            for i in range(feature_window.shape[0]):

                array01 = fs.sum_max_bayes_pos(feature_window[i],all_predictions[i],num_window)   ### return the position in 17*17
                array01_all.append(array01)
            

            for i in range(feature_window.shape[0]):

                grainid, area_bayes = fs.sum_max_bayes_new(feature_window[i],all_predictions[i],num_window)   ### return the position in 17*17


                grainid_all.append(int(grainid))
                area_bayes_all.append(area_bayes) 

                feature_window_all.append(feature_window[i])

        new_pro = np.array(array01_all).copy()
        array01_all = np.mean(np.array(array01_all),axis=0)
        
        # # 分子
        # def each_pos_pro(new_pro):
        #     array_zeros = np.zeros((17,17))
        #     for i in range(17):
        #         for j in range(17):
        #             pro_ij = 0
        #             for n in range(new_pro.shape[0]):
        #                 new_pro_pro = new_pro[n]
        #                 pro_ij += new_pro_pro[i,j]*np.sum(new_pro_pro)
        #             array_zeros[i][j] = pro_ij
        #     return array_zeros

        ans1 = fs.each_pos_1(new_pro)  ##调用函数计算分子
        ans1 = ans1/new_pro.shape[0]

        # #分母
        # def each_pos_2(new_pro):
        #     array_zeros2 = np.zeros((17,17))
        #     for i in range(17):
        #         for j in range(17):
        #             pro_ij = 0
        #             k=0
        #             for n in range(new_pro.shape[0]):
        #                 new_pro_pro = new_pro[n]
        #                 if new_pro_pro[i,j] == 1:
        #                     pro_ij += 1/np.sum(new_pro_pro)
        #                     k+=1
        #             array_zeros2[i][j] = pro_ij/k
        #     return array_zeros2


        ans2 = fs.each_pos_2(new_pro) #调用函数计算分母


        #分子
        # pro_00 = 0
        # for i in range(new_pro.shape[0]):
        #     new_pro_pro = new_pro[i]
        #     pro_00 += new_pro_pro[0,0]*np.sum(new_pro_pro)
        #分母
        # pro_00 = 0
        # k = 0
        # for i in range(new_pro.shape[0]):
        #     new_pro_pro = new_pro[i]
        #     if new_pro_pro[0,0] == 1:
        #         pro_00 += 1/np.sum(new_pro_pro)
        #         k+=1

        result = array01_all*ans1/ans2 #计算返回bayes value

        if evaluate==True: self.predictions = torch.cat(predictions_split, dim=0)

        upated_values = torch.Tensor(np.array(grainid_all)).to(device)   #### action_values is action position
        self.im_next = im.flatten().float()
        self.im_next[indx_use] = upated_values
        self.im_next = self.im_next.reshape(im.shape)
        self.indx_use = indx_use
        
        return self.im_next,result
    



    
    def compute_metrics(self):
        im_next_predicted = self.step(self.im_seq[0:1,], self.miso_matrix)
        # im_next_predicted = self.im_next
        im_next_actual = self.im_seq[1:2,]
        accuracy = torch.mean((im_next_predicted==im_next_actual).float())
        loss = np.mean(tf.keras.losses.mse(self.predictions.cpu().numpy(), np.reshape(self.labels[self.indx_use,].cpu(),(-1,self.act_dim**self.num_dims))))
        # _, loss = self.model.evaluate(self.predictions, self.labels[self.indx_use,].reshape(-1,self.act_dim**self.num_dims))
        return loss, accuracy
        
    
    def train(self, evaluate=True):
        
        if evaluate: 
            loss, accuracy = self.compute_metrics()
            self.validation_loss.append(loss)
            self.validation_acc.append(accuracy)
        
        # features, labels = fs.unison_shuffled_copies(self.features.cpu().numpy(), self.labels.cpu().numpy()) #random shuffle 
        features, labels = fs.unison_shuffled_copies(self.features, self.labels) #random shuffle 
        # ss = int(self.obs_dim/2)
        # indx_use = np.nonzero(features[:,ss,ss,ss])[0]
        
        mid_ix = (np.array(features.shape[1:])/2).astype(int)
        ind = tuple([slice(None)]) + tuple(mid_ix)
        indx_use = torch.nonzero(features[ind])[:,0]   ### use random few features(indx_use) to train the model
        

        # features=features[indx_use[:100]].cpu().numpy()
        # labels = labels[indx_use[:100]].cpu().numpy()


        # torch.manual_seed(42)
        # indices = torch.randperm(indx_use.size(0))
        # random_indices = indices[:100]
        # features=features[indx_use[random_indices]].cpu().numpy()
        # labels = labels[indx_use[random_indices]].cpu().numpy()

        # features=features[indx_use[-100:]].cpu().numpy()
        # labels = labels[indx_use[-100:]].cpu().numpy()
        features = features[indx_use,].cpu().numpy()
        labels = labels[indx_use,].cpu().numpy()

        _ = self.model.fit(features, np.reshape(labels,(-1,self.act_dim**self.num_dims)), epochs=1, verbose=0)
        # _ = self.model.fit(features, labels, epochs=1, verbose=0)


        # self.training_loss.append(history.history['loss'][0])
        # features, labels = fs.unison_shuffled_copies(self.features, self.labels) #random shuffle 
        # ss = int(self.obs_dim/2)
        # indx_use = torch.nonzero(features[:,ss,ss,ss])[:,0]
        # features = features[indx_use,]
        # labels = labels[indx_use,]
        # _ = self.model.fit(features, labels.reshape(-1,self.act_dim**self.num_dims))
        del self.features
        torch.cuda.empty_cache()

        if evaluate: 
            loss, accuracy = self.compute_metrics()
            self.training_loss.append(loss)
            self.training_acc.append(accuracy)


        # del self.labels
        # del self.im_seq
        # del self.miso_matrix
        # torch.cuda.empty_cache()
        
    
    def plot(self, fp_results='./plots'):
        
        if self.num_dims==2:
            #Plot the next images, predicted and true, together
            fig, axs = plt.subplots(1,3)
            axs[0].matshow(self.im_seq[0,0,].cpu().numpy())
            axs[0].set_title('Current')
            axs[0].axis('off')
            axs[1].matshow(self.im_next[0,0,].cpu().numpy()) 
            axs[1].set_title('Predicted Next')
            axs[1].axis('off')
            axs[2].matshow(self.im_seq[1,0,].cpu().numpy()) 
            axs[2].set_title('True Next')
            axs[2].axis('off')
            plt.savefig('%s/sim_vs_true.png'%fp_results)
            plt.show()
            
            #Plot the action distributions, predicted and true, together
            ctr = int((self.act_dim-1)/2)
            pred = self.predictions.reshape(-1, self.act_dim, self.act_dim).detach().cpu().numpy()
            fig, axs = plt.subplots(1,2)
            p1 = axs[0].matshow(np.mean(pred, axis=0), vmin=0, vmax=1)
            fig.colorbar(p1, ax=axs[0])
            axs[0].plot(ctr,ctr,marker='x')
            axs[0].set_title('Predicted')
            axs[0].axis('off')
            p2 = axs[1].matshow(np.mean(self.labels.cpu().numpy(), axis=0), vmin=0, vmax=1) 
            
            # p2 = axs[1].matshow(np.mean(self.labels.cpu().numpy(), axis=0)) 
            # p2 = axs[1].matshow(self.labels.cpu().numpy()[0]) 
            
            fig.colorbar(p2, ax=axs[1])
            axs[1].plot(ctr,ctr,marker='x')
            axs[1].set_title('True')
            axs[1].axis('off')
            plt.savefig('%s/action_likelihood.png'%fp_results)
            plt.show()
            
            #Plot loss and accuracy
            fig, axs = plt.subplots(1,2)
            axs[0].plot(self.validation_loss, '-*', label='Validation')
            axs[0].plot(self.training_loss, '--*', label='Training')
            axs[0].set_title('Loss')
            axs[0].legend()
            axs[1].plot(self.validation_acc, '-*', label='Validation')
            axs[1].plot(self.training_acc, '--*', label='Training')
            axs[1].set_title('Accuracy')
            axs[1].legend()
            plt.savefig('%s/train_val_loss_accuracy.png'%fp_results)
            plt.show()
            
            plt.close('all')
        
        if self.num_dims==3:
            bi = int(self.im_seq.shape[-1]/2)
            
            #Plot the next images, predicted and true, together
            fig, axs = plt.subplots(1,3)
            axs[0].matshow(self.im_seq[0,0,...,bi].cpu().numpy())
            axs[0].set_title('Current')
            axs[0].axis('off')
            axs[1].matshow(self.im_next[0,0,...,bi].cpu().numpy()) 
            axs[1].set_title('Predicted Next')
            axs[1].axis('off')
            axs[2].matshow(self.im_seq[1,0,...,bi].cpu().numpy()) 
            axs[2].set_title('True Next')
            axs[2].axis('off')
            plt.savefig('%s/sim_vs_true.png'%fp_results)
            plt.show()
            
            #Plot the action distributions, predicted and true, together
            ctr = int((self.act_dim-1)/2)
            pred = self.predictions.reshape(-1, self.act_dim, self.act_dim, self.act_dim).detach().cpu().numpy()
            fig, axs = plt.subplots(1,2)
            p1 = axs[0].matshow(np.mean(pred, axis=0)[...,ctr], vmin=0, vmax=1)
            fig.colorbar(p1, ax=axs[0])
            axs[0].plot(ctr,ctr,marker='x')
            axs[0].set_title('Predicted')
            axs[0].axis('off')
            p2 = axs[1].matshow(np.mean(self.labels.cpu().numpy(), axis=0)[...,ctr], vmin=0, vmax=1) 
            fig.colorbar(p2, ax=axs[1])
            axs[1].plot(ctr,ctr,marker='x')
            axs[1].set_title('True')
            axs[1].axis('off')
            plt.savefig('%s/action_likelihood.png'%fp_results)
            plt.show()
            
            #Plot loss and accuracy
            fig, axs = plt.subplots(1,2)
            axs[0].plot(self.validation_loss, '-*', label='Validation')
            axs[0].plot(self.training_loss, '--*', label='Training')
            axs[0].set_title('Loss')
            axs[0].legend()
            axs[1].plot(self.validation_acc, '-*', label='Validation')
            axs[1].plot(self.training_acc, '--*', label='Training')
            axs[1].set_title('Accuracy')
            axs[1].legend()
            plt.savefig('%s/train_val_loss_accuracy.png'%fp_results)
            plt.show()
            
            plt.close('all')
        
    
    def load(self, name):
        self.model = load_model(name)
        self.num_dims = len(self.model.layers[0].get_output_at(0).get_shape().as_list()) - 1
        self.obs_dim = self.model.layers[0].get_output_at(0).get_shape().as_list()[1]
        # self.obs_dim = 5

        model_out_dim = self.model.layers[-1].get_output_at(0).get_shape().as_list()[1]

        self.act_dim = int(np.rint(model_out_dim**(1/self.num_dims)))
        # self.act_dim = 5
        self.learning_rate = K.eval(self.model.optimizer.lr)


    def save(self, name):
        self.model.save(name)