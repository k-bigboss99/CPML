from __future__ import print_function, division
import os
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import glob
import random
import numpy as np



def get_frame(paths, transform_face):
    face_frames = []
    for path in paths:
        frame =  Image.open(path)
        face_frame = transform_face(frame)
        face_frames.append(face_frame)
    return face_frames




def get_landmarks(path, start_idx, block):

    vectors = np.loadtxt(path, dtype=np.float32)
    
    if vectors.shape[0] < block:
        vectors = np.pad(vectors, ((0, block - vectors.shape[0]), (0, 0)), "edge")
    
    vec = vectors[start_idx:start_idx + block, :]
    vec_next = vectors[start_idx + 1:start_idx + block, :]
    vec_next = np.pad(vec_next, ((0, 1), (0, 0)), 'constant', constant_values=(0, 0))
    vec_diff = (vec_next - vec)[:block - 1, :]
        
    return vec, vec_diff




def getSubjects(configPath, real, fake):
    
    f = open(configPath, "r")
    
    all_live, all_spoof = [], []
    ls_dict = {}
    
    while(True):
        line = f.readline()
        if not line:
            break
        line = line.strip()
        
        subset, subj, classNum = line.split(",")
        
        if classNum == "+1" and real != "all" and subset != real:
            continue
        if classNum == "-1" and fake != "all" and subset != fake:
            continue
        
        if(classNum == "+1"):
            all_live.append(line)
            ls_dict[line] = 1

        else:
            all_spoof.append(line)
            ls_dict[line] = 0

    
    print(f"{configPath=}")
    print(f"{len(all_live)=}, {len(all_spoof)=}")
    
    return all_live, all_spoof, ls_dict



class FFv2_Dataset(Dataset):
    def __init__(self, train, seq_length=300, real_or_fake="both", real="all", fake="all", comp="c23", 
                 if_fg=True, if_land=True, if_bg=False,validate=False):
        
        real_subsets = ['actors', 'youtube', 'all']
        fake_subsets = ['DF', "F2F", "FS", "NT", 'all']
        FF_fullname = {
            'DF':  'Deepfakes',
            'F2F': 'Face2Face',
            'FS':  'FaceSwap',
            'NT':  'NeuralTextures',
            'all': 'all',
        }
        
        assert real in real_subsets
        print(f"real = {real}")
        print(f"fake = {fake}")
        assert fake in fake_subsets
        assert real_or_fake in ["real", "fake", "both"]
        fake = FF_fullname[fake]
        
        self.train = train
        if self.train:
            self.config = "./FF++_train.txt" 


        self.real_root_dir = f"/work/u1657859/DeepFake_dataset/FF++_landmark/original_sequences"
        self.fake_root_dir = f"/work/u1657859/DeepFake_dataset/FF++_landmark/manipulated_sequences"        

        self.real = real
        self.fake = fake
        
        self.if_fg = if_fg
        self.if_land = if_land
        self.if_bg = if_bg
        self.seq_length = seq_length

        self.subject_images = {}
        self.landmark_files = {}

        face_folder = "crop_MTCNN"
        landmark_folder = "video_landmark_new"

        all_real, all_fake, label_dict = getSubjects(self.config, real, fake)
        
        self.label_dict = label_dict

        if real_or_fake == "real":
            self.all_subjects = all_real
        elif real_or_fake == "fake":
            self.all_subjects = all_fake
        elif real_or_fake == "both":
            self.all_subjects = all_real + all_fake
        
        enough_frame_subjects = []
        
        for line in self.all_subjects:
            
            subset, subj, classNum = line.split(",")
            
            root = self.real_root_dir if classNum == "+1" else self.fake_root_dir
            
            image_paths = sorted(glob.glob(os.path.join(root, subset, comp, face_folder, subj, "face0", "*.jpg")))
            
            N = len(image_paths)
            if(N < self.seq_length-30):
                continue
                
                
            landmark_paths = os.path.join(root, subset, comp, landmark_folder, f"{subj}.txt")
            
            self.subject_images[line] = image_paths
            self.landmark_files[line] = landmark_paths
            
            enough_frame_subjects.append(line)
        
        print(f"{len(self.all_subjects)-len(enough_frame_subjects)} subjects have less than {self.seq_length-30} frames")
        self.all_subjects = enough_frame_subjects

        self.transform_face = transforms.Compose([
            transforms.Resize((128, 128)), # TODO
            transforms.ToTensor()
        ])
            

    def __getitem__(self, idx):

        subject = self.all_subjects[idx]
        _image_paths = self.subject_images[subject]
        _landmark_path = self.landmark_files[subject]
        label = torch.tensor(self.label_dict[subject]).unsqueeze(0)


        _face_frame, _landmarks, _landmarks_diff  = [], [], []# , [], _bg_frame
        
        start = 0

        if self.if_fg:
            _image_paths = _image_paths[start:start+self.seq_length]
            _face_frame = torch.stack(get_frame(_image_paths, self.transform_face)).transpose(0, 1)
            if _face_frame.shape[1] < self.seq_length:
                _face_frame = F.pad(_face_frame, (0, 0, 0, 0, 0, self.seq_length-_face_frame.shape[1]))
            

        if self.if_land:
            _landmarks, _landmarks_diff = get_landmarks(_landmark_path, start_idx=start, block=self.seq_length)
        
        return _face_frame, _landmarks, _landmarks_diff, label, subject # , _bg_frame


    def __len__(self):
        return len(self.all_subjects)


def get_loader(train=1, seq_length=300, batch_size=1, if_fg=True, if_land=True, if_bg=False,shuffle=True,
               real_or_fake="both", real="all", fake="all", comp="c23",validate=False):
    
    _dataset = FFv2_Dataset(train=train, seq_length=seq_length, if_fg=if_fg, if_land=if_land, if_bg=if_bg, 
                            real_or_fake=real_or_fake, real=real, fake=fake, comp=comp,validate=validate)
    
    return DataLoader(_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)


def collate_batch(batch):
       
    face_frames_list, image_paths_list, bg_frames_list, bg_paths_list = [], [], [], []
    
    for (face_frames, image_paths, bg_frames, bg_paths) in batch:
        
        face_frames_list.append(face_frames)
        image_paths_list.append(image_paths)
        bg_frames_list.append(bg_frames)
        bg_paths_list.append(bg_paths)

    return face_frames_list, image_paths_list, bg_frames_list, bg_paths_list




if __name__ == "__main__":
    

    train_loader = get_loader(train=False, seq_length=160, batch_size=1, if_fg=True, if_land=True, if_bg=False, shuffle=False,
                              real_or_fake="both", real="youtube", fake="all", comp="c23",validate=True)

    for i, (face_frames, landmarks, _landmarks_diff, bg_frames, ls, subjects) in enumerate(train_loader):
        print(face_frames.shape)
        print(landmarks.shape) 
        print(_landmarks_diff.shape) 
        print(ls)
        print(ls.shape)
        print(subjects)

        break
    
    