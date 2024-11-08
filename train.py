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

def HR_60_to_120(HR_rate):
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
    # get heart rate by FFT
    # return both heart rate and PSD

    sig = sig.reshape(-1)
    sig = sig * signal.windows.hann(sig.shape[0])
    sig_f = np.abs(fft(sig))
    low_idx = np.round(0.6 / fs * sig.shape[0]).astype('int')
    high_idx = np.round(4 / fs * sig.shape[0]).astype('int')
    sig_f_original = sig_f.copy()
    # sig_f_original = sig_f.detach()
    
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


def load_net(args, device, rPPG_path=None, LRNet_comp=None, model_text=None):
    
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
        
        if args.fix_weight:
            print("Fix the weights of the rPPG estimation model")
            model.requires_grad_(False)

        else:
            print("Finetune all weights")
        
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

    net = PAD_Classifier(net_pad,net_downstream,model_text)# numClasses = 2, net_adapter
    net.to(device)
    
    return net

class Pearson(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Pearson,self).__init__()
        return
    def forward(self, preds, labels):       # all variable operation
        loss = 0
        # print(f"{preds.shape[1] = }") # 160 frames
        # print(f"{preds.shape[0] = }") # batch size 
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])                # x
            sum_y = torch.sum(labels[i])               # y
            sum_xy = torch.sum(preds[i]*labels[i])        # xy
            sum_x2 = torch.sum(torch.pow(preds[i],2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i],2)) # y^2
            N = preds.shape[1]
            
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
            loss +=  pearson
            
        loss = loss/preds.shape[0]
        # print(preds, labels)
        return loss

torch.autograd.set_detect_anomaly(True)


# loader
seq_len = args.seq_len

train_loader_raw_real = get_loader(train=True, seq_length=seq_len, batch_size=args.bs, if_fg=True, shuffle=False,
                         real_or_fake="real", real="youtube", fake=args.subset, comp=args.comp_3_loader)
train_loader_raw_fake = get_loader(train=True, seq_length=seq_len, batch_size=args.bs, if_fg=True, shuffle=False,
                         real_or_fake="fake", real="youtube", fake=args.subset, comp=args.comp_3_loader)



# text model
model_text, _ = clip.load("ViT-B/16", 'cuda:0')

# model
model = load_net(args, device=device, rPPG_path="Physformer_VIPL_fold1.pkl", LRNet_comp=args.comp_3_loader, model_text=model_text)

# optim and  BCE_loss 
opt_fg = optim.AdamW(model.parameters(), lr=args.lr)
BCE_loss = nn.CrossEntropyLoss()  


# loss 
similar = torch.tensor([1], dtype=torch.float).to(device)
dissimilar = torch.tensor([-1], dtype=torch.float).to(device)


for epoch in range(epoch_number,epoch_number+1):
    print(f"epoch_train: {epoch_number} / {epoch_number+1}:")
    for step, (data_raw_r,data_raw_f) in enumerate(zip(_train_loader_raw_real, _train_loader_raw_fake)):

        face_frames_r, landmarks_r, landmarks_diff_r, label_r, subjects_r = data_raw_r 
        face_frames_f, landmarks_f, landmarks_diff_f, label_f, subjects_f = data_raw_f 
        if face_frames_r.shape[0] != args.bs or face_frames_f.shape[0] != args.bs:
            print(f"{face_frames_R.shape[0]=}, continue")
            print(f"{face_frames_F.shape[0]=}, continue")
            continue

        # load raw video
        face_frames_r = face_frames_r.to(device)
        face_frames_f = face_frames_f.to(device)
        landmarks_r = landmarks_r.to(device)
        landmarks_f = landmarks_f.to(device)
        landmarks_diff_r = landmarks_diff_r.to(device)
        landmarks_diff_f = landmarks_diff_f.to(device)
        label_r = label_r.to(device) 
        label_f = label_f.to(device)
        
        # torch.cuda.reset_peak_memory_stats()    
        
        # TODO model input raw video feature ,rppg ,landmark feature
        # Classifier_output= model(INPUT)
        out, out_land, out_rPPG, rppg, cosine_similarity_feature_text, cosine_similarity_feature_text_other \
        ,out_rppg_cross_landmark_cat_prompt   = model(face_frames, landmarks, landmarks_diff, similar,size=128)
   
        # paper loss(4) BCE loss
        loss_BCE = BCE_loss(out_rppg_cross_landmark_cat_prompt ,label[:, 0].long())
        
        # backward
        total_loss.backward()
        

        # Multi-modal Prompt-guided Contrastive Learning
        # downsample views 
        # OUTPUT= model(INPUT)
 
        # paper loss(sim),loss(dis)
        # nn.CosineSimilarity(rppg_cat_landmark,text_embedding_real/fake_prompt) 

        """
        text embedding from FLIP method
        """
        # # self.text_encode = model_text
        # class_embeddings = self.text_encode.encode_text(texts) 
        # class_embeddings = class_embeddings.mean(dim=0) 
        # # normalized features
        # class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        # cosine_similarity_feature_text = self.cosine_similarity(rppg_cat_landmark,class_embeddings)

        # loss_cosine_similarity = 1 - cosine_similarity_feature_text
        # # loss_cosine_similarity_other = cosine_similarity_feature_text_other
        
        # paper loss(12),loss(13)
        total_cosine_similarity_loss = loss_cosine_similarity + loss_cosine_similarity_other
        total_cosine_similarity_other_loss = loss_cosine_similarity + loss_cosine_similarity_other


        # paper loss(5),loss(7)
        PearsoLoss = Pearson()
        loss_rppg_between_RF = abs(PearsoLoss(rppg_R, rppg_F)) 
        loss_rppg_between_Rr = 1 - PearsoLoss(rppg_R, rppg_r)
        loss_rppg_between_rf = abs(PearsoLoss(rppg_r, rppg_f)) 
        loss_rppg_between_Ff = 1 - PearsoLoss(rppg_F, rppg_f)
        loss_rppg_between_Rf = abs(PearsoLoss(rppg_R, rppg_f))
        loss_rppg_between_rF = abs(PearsoLoss(rppg_r, rppg_F)) 
        rPPG_pull_loss = loss_rppg_between_Rr + loss_rppg_between_Ff
        rPPG_push_loss = loss_rppg_between_RF + loss_rppg_between_rf + loss_rppg_between_Rf + loss_rppg_between_rF
        
        # paper loss(9) HR
        # HR 
        rppg_R_butter = butter_bandpass(rppg_R.detach().cpu().numpy(), lowcut=0.6, highcut=4, fs=30)
        rppg_F_butter = butter_bandpass(rppg_F.detach().cpu().numpy(), lowcut=0.6, highcut=4, fs=30)
        rppg_r_butter = butter_bandpass(rppg_r.detach().cpu().numpy(), lowcut=0.6, highcut=4, fs=30)
        rppg_f_butter = butter_bandpass(rppg_f.detach().cpu().numpy(), lowcut=0.6, highcut=4, fs=30)

        hr_R, psd_y_R, psd_x_R = hr_fft(rppg_R_butter, fs=30)
        hr_F, psd_y_F, psd_x_F = hr_fft(rppg_F_butter, fs=30)
        hr_r, psd_y_r, psd_x_r = hr_fft(rppg_r_butter, fs=30)
        hr_f, psd_y_f, psd_x_f = hr_fft(rppg_f_butter, fs=30)

        hr_R = HR_60_to_120(hr_R)
        hr_F = HR_60_to_120(hr_F)
        hr_r = HR_60_to_120(hr_r)
        hr_f = HR_60_to_120(hr_f)
        # print(hr_1,hr_2,hr_3,hr_4)

        # HR_diff_1 = abs(hr_1-hr_2)/60 # c23 real - c23 fake dissimilar (120-60)
        HR_diff_2 = abs(hr_1-hr_3)/60 # c23 real - c40 real similar
        # HR_diff_3 = abs(hr_3-hr_4)/60 # c40 real - c40 fake dissimilar
        HR_diff_4 = abs(hr_2-hr_4)/60 # c23 fake - c40 fake similar
        # HR_diff_5 = abs(hr_1-hr_4)/60 # c23 real - c40 fake dissimilar
        # HR_diff_6 = abs(hr_3-hr_2)/60 # c40 real - c23 fake dissimilar
        # print(HR_diff_1,HR_diff_2,HR_diff_3,HR_diff_4,HR_diff_5,HR_diff_6)    
        HR_diff_pos = ( HR_diff_2 + HR_diff_4) / 2
        # HR_diff_neg = ((1-HR_diff_1) + (1-HR_diff_3) + (1-HR_diff_5) + (1-HR_diff_6)) / 4

        # paper loss(10)
        physiological_loss = rPPG_pull_loss + rPPG_push_loss +  HR_diff_pos
        
        # paper loss(11)
        MSELoss = nn.MSELoss()
        loss_mse_between_Rr = MSELoss(cosine_similarity_feature_text_R, cosine_similarity_feature_text_r) 
        loss_mse_between_Ff = MSELoss(cosine_similarity_feature_text_F, cosine_similarity_feature_text_f) 
        total_mse_loss = loss_mse_between_Rr + loss_mse_between_Ff

        # paper loss(14)
        crossmodal_consistency_loss = total_cosine_similarity_loss + total_cosine_similarity_other_loss + total_mse_loss

        # paper loss(15)
        total_loss = 0.2*physiological_loss + 0.25*crossmodal_consistency_loss


        total_loss.backward()
        opt_fg.step()
        opt_fg.zero_grad()

    torch.save(model.state_dict(), result_dir + '/weight/fg_epoch%d.pt' % epoch) 
    gc.collect()
    torch.cuda.empty_cache()


        
    