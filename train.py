from __future__ import print_function, division
import sys
sys.path.append('model')
sys.path.append('util')
from util import *

from TorchLossComputer import TorchLossComputer
from dataloader import get_loader

import matplotlib.pyplot as plt
import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb
import scipy.io as sio
import gc
from tqdm import tqdm
import glob
import re
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle, islice
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt

import clip

from landmark_model import DualLRNet
from rppg_model import ViT_ST_ST_Compact3_TDC_gra_sharp
from model_rppg_landmark_text import PAD_Classifier

def HR_F0_to_120(HR_rate):
    if HR_rate<60:
        HR_rate = 60
    elif HR_rate>=120:
        HR_rate = 120
    else:
        pass
    return HR_rate

def butter_bandpass(sig, lowcut, highcut, fs, order=2):
    # butterworth bandpass filter
    
    sig = np.reshape(sig, -1)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    
    y = filtfilt(b, a, sig)
    return y

def hr_fft(sig, fs, harmonics_removal=True):

    sig = sig.reshape(-1)
    sig = sig * signal.windows.hann(sig.shape[0])
    sig_f = np.abs(fft(sig))
    low_idx = np.round(0.6 / fs * sig.shape[0]).astype('int')
    high_idx = np.round(4 / fs * sig.shape[0]).astype('int')
    sig_f_original = sig_f.copy()
    
    sig_f[:low_idx] = 0
    sig_f[high_idx:] = 0

    peak_idx, _ = signal.find_peaks(sig_f)
    sort_idx = np.argsort(sig_f[peak_idx])
    sort_idx = sort_idx[::-1]

    peak_idx1 = peak_idx[sort_idx[0]]
    peak_idx2 = peak_idx[sort_idx[1]]

    f_hr1 = peak_idx1 / sig_f.shape[0] * fs
    hr1 = f_hr1 * 60

    f_hr2 = peak_idx2 / sig_f.shape[0] * fs
    hr2 = f_hr2 * 60
    if harmonics_removal:
        if np.abs(hr1-2*hr2)<10:
            hr = hr2
        else:
            hr = hr1
    else:
        hr = hr1

    x_hr = np.arange(len(sig_f))/len(sig_f)*fs*60
    return hr, sig_f_original, x_hr

def normalize(x):
    return (x-x.mean())/x.std()

def load_net(device, rPPG_path=None, LRNet_comp=None, model_text=None):
    
    if LRNet_comp is None or LRNet_comp == "":
        LRNet_load = False
        LRNet_comp = ""
    else:
        LRNet_load = True

    net_pad = DualLRNet(load_pretrained=LRNet_load,
                        pretrained_comp=LRNet_comp,
                        device=device)
    
    def load_model(model,path):
        pretrained_state = torch.load(path, map_location=device)
        
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_state.items() if k in model.state_dict()}
        model_dict = model.state_dict()

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict) # model_dict or pretrained_dict
        
        
        return model

    net_downstream = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(300,128,128),
                                                      patches=(4,4,4),
                                                        dim=96,
                                                        ff_dim=144,
                                                        num_heads=4,
                                                        num_layers=12,
                                                        dropout_rate=0.1,
                                                        theta=0.7)
    if rPPG_path is not None:
        net_downstream = load_model(net_downstream, rPPG_path="model\rppg_weight\Physformer_VIPL_fold1.pkl")
        print("pretrained rPPG model is loaded")

    else:
        print("No pretrained rPPG model is loaded")

    net = PAD_Classifier(net_pad,net_downstream,model_text)
    net.to(device)
    
    return net

class Pearson(nn.Module):    
    def __init__(self):
        super(Pearson,self).__init__()
        return
    def forward(self, preds, labels):    
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])               
            sum_y = torch.sum(labels[i])              
            sum_xy = torch.sum(preds[i]*labels[i])       
            sum_x2 = torch.sum(torch.pow(preds[i],2)) 
            sum_y2 = torch.sum(torch.pow(labels[i],2)) 
            N = preds.shape[1]
            
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
            loss +=  pearson
            
        loss = loss/preds.shape[0]
        return loss

torch.autograd.set_detect_anomaly(True)

# loader

train_loader_raw_real = get_loader(train=True, seq_length=160, batch_size=1, if_fg=True, shuffle=False,
                         real_or_fake="real", real="youtube", fake=DF, comp=raw)
train_loader_raw_fake = get_loader(train=True, seq_length=160, batch_size=1, if_fg=True, shuffle=False,
                         real_or_fake="fake", real="youtube", fake=DF, comp=raw)



# text model
model_text, _ = clip.load("ViT-B/16", 'cuda:0')

# model
model = load_net(device=device, rPPG_path="Physformer_VIPL_fold1.pkl", LRNet_comp=raw, model_text=model_text)

# optim and  BCE_loss 
opt_fg = optim.AdamW(model.parameters(), lr=0.00001)
BCE_loss = nn.CrossEntropyLoss()  

similar = torch.tensor([1], dtype=torch.float).to(device)
dissimilar = torch.tensor([-1], dtype=torch.float).to(device)


for epoch in range(epoch_number,epoch_number+1):
    for step, (data_raw_r,data_raw_f) in enumerate(zip(_train_loader_raw_real, _train_loader_raw_fake)):
        
        # Multi-modal Prompt-guided Learning
        face_frames_r, landmarks_r, landmarks_diff_r, label_r, subjects_r = data_raw_r
        face_frames_f, landmarks_f, landmarks_diff_f, label_f, subjects_f = data_raw_f

        face_frames_r = face_frames_r.to(device)
        face_frames_f = face_frames_f.to(device)
        landmarks_r = landmarks_r.to(device)
        landmarks_f = landmarks_f.to(device)
        landmarks_diff_r = landmarks_diff_r.to(device)
        landmarks_diff_f = landmarks_diff_f.to(device)
        label_r = label_r.to(device) 
        label_f = label_f.to(device)
    

        out_Real, out_land_Real, out_rPPG_Real, rppg_Real, cosine_similarity_feature_text_Real, cosine_similarity_feature_text_other_Real \
        ,out_rppg_cross_landmark_cat_prompt_Real = model(face_frames_Real, landmarks_Real, landmarks_diff_Real, similar,size=128)
       
        out_Fake, out_land_Fake, out_rPPG_Fake, rppg_Fake, cosine_similarity_feature_text_Fake, cosine_similarity_feature_text_other_Fake \
        ,out_rppg_cross_landmark_cat_prompt_Fake = model(face_frames_Fake, landmarks_Fake, landmarks_diff_Fake, dissimilar,size=128)
        
        loss_BCE = BCE_loss(out_rppg_cross_landmark_cat_prompt_R ,label_r[:, 0].long())
        loss_BCE = BCE_loss(out_rppg_cross_landmark_cat_prompt_F ,label_f[:, 0].long())
        total_loss.backward()
        
        # Multi-modal Prompt-guided Contrastive Learning
        out_R, out_land_R, out_rPPG_R, rppg_R, cosine_similarity_feature_text_R, cosine_similarity_feature_text_other_R \
        ,out_rppg_cross_landmark_cat_prompt_R   = model(face_frames_R, landmarks_R, landmarks_diff_R, similar,size=64)
       
        out_F, out_land_F, out_rPPG_F, rppg_F, cosine_similarity_feature_text_F, cosine_similarity_feature_text_other_F \
        ,out_rppg_cross_landmark_cat_prompt_F    = model(face_frames_F, landmarks_F, landmarks_diff_F, dissimilar,size=64)

        out_r, out_land_r, out_rPPG_r, rppg_r, cosine_similarity_feature_text_r, cosine_similarity_feature_text_other_r \
        ,out_rppg_cross_landmark_cat_prompt_r   = model(face_frames_R, landmarks_R, landmarks_diff_R, similar,size=32)
       
        out_f, out_land_f, out_rPPG_f, rppg_f, cosine_similarity_feature_text_f, cosine_similarity_feature_text_other_f \
        ,out_rppg_cross_landmark_cat_prompt_f    = model(face_frames_F, landmarks_F, landmarks_diff_F, dissimilar,size=32)

        # Cross-Quality Similarity Learning
        PearsoLoss = Pearson()
        loss_rppg_between_RF = abs(PearsoLoss(rppg_R, rppg_F)) 
        loss_rppg_between_Rr = 1 - PearsoLoss(rppg_R, rppg_r)
        loss_rppg_between_rf = abs(PearsoLoss(rppg_r, rppg_f)) 
        loss_rppg_between_Ff = 1 - PearsoLoss(rppg_F, rppg_f)
        loss_rppg_between_Rf = abs(PearsoLoss(rppg_R, rppg_f))
        loss_rppg_between_rF = abs(PearsoLoss(rppg_r, rppg_F))

        loss_rPPG_pull = loss_rppg_between_Rr + loss_rppg_between_Ff
        loss_rPPG_push = loss_rppg_between_RF + loss_rppg_between_rf + loss_rppg_between_Rf + loss_rppg_between_rF
        
        rppg_R_butter = butter_bandpass(rppg_R.detach().cpu().numpy(), lowcut=0.6, highcut=4, fs=30)
        rppg_F_butter = butter_bandpass(rppg_F.detach().cpu().numpy(), lowcut=0.6, highcut=4, fs=30)
        rppg_r_butter = butter_bandpass(rppg_r.detach().cpu().numpy(), lowcut=0.6, highcut=4, fs=30)
        rppg_f_butter = butter_bandpass(rppg_f.detach().cpu().numpy(), lowcut=0.6, highcut=4, fs=30)

        hr_R, psd_y_R, psd_x_R = hr_fft(rppg_R_butter, fs=30)
        hr_F, psd_y_F, psd_x_F = hr_fft(rppg_F_butter, fs=30)
        hr_r, psd_y_r, psd_x_r = hr_fft(rppg_r_butter, fs=30)
        hr_f, psd_y_f, psd_x_f = hr_fft(rppg_f_butter, fs=30)

        hr_R = HR_F0_to_120(hr_R)
        hr_F = HR_F0_to_120(hr_F)
        hr_r = HR_F0_to_120(hr_r)
        hr_f = HR_F0_to_120(hr_f)

        loss_norm_Rr = abs(hr_1-hr_3)/60
        loss_norm_Ff = abs(hr_2-hr_4)/60 

        loss_hr = loss_norm_Rr + loss_norm_Ff

        physiological_loss = loss_rPPG_pull + loss_rPPG_push + loss_hr
        
        # Cross-Modality Consistency Learning
        MSELoss = nn.MSELoss()
        loss_mse_between_Rr = MSELoss(cosine_similarity_feature_text_R, cosine_similarity_feature_text_r) 
        loss_mse_between_Ff = MSELoss(cosine_similarity_feature_text_F, cosine_similarity_feature_text_f) 
        
        loss_mse = loss_mse_between_Rr + loss_mse_between_Ff
        
        loss_cosine_similarity_R = 1 - cosine_similarity_feature_text_R
        loss_cosine_similarity_F = 1 - cosine_similarity_feature_text_F
        loss_cosine_similarity_r = 1 - cosine_similarity_feature_text_r
        loss_cosine_similarity_f = 1 - cosine_similarity_feature_text_f

        loss_cosine_similarity_other_R = cosine_similarity_feature_text_other_R
        loss_cosine_similarity_other_F = cosine_similarity_feature_text_other_F
        loss_cosine_similarity_other_r = cosine_similarity_feature_text_other_r
        loss_cosine_similarity_other_f = cosine_similarity_feature_text_other_f

        loss_text_pull = loss_cosine_similarity_R + loss_cosine_similarity_F+ loss_cosine_similarity_r + loss_cosine_similarity_f
        loss_text_push = loss_cosine_similarity_other_R + loss_cosine_similarity_other_F + loss_cosine_similarity_other_r +loss_cosine_similarity_other_f
        
        cross_modal_consistency_loss = loss_text_pull + loss_text_push + loss_mse

        
        total_loss = 0.2*physiological_loss + 0.25*cross_modal_consistency_loss

        total_loss.backward()
        opt_fg.step()
        opt_fg.zero_grad()

    torch.save(model.state_dict(), result_dir + '/weight/fg_epoch%d.pt' % epoch) 
    gc.collect()
    torch.cuda.empty_cache()


        
    