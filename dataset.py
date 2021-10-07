import torch
from torch.utils.data import Dataset
import numpy as np
import decord
decord.bridge.set_bridge('torch')
import h5py

"""
The input txt format: "RGB_video_path"*"Depth_video_path"*"Label"*
Example: "/data/Final_P1/P0218/H/0/color_1.avi*/data/Final_P1/P0218/H/0/depth_1.h5*0*"

"""
 
class RGBDDataset(Dataset):
    def __init__(self, args,txt_file):
        self.rgb=list()
        self.depth=list()
        self.labels=list()
        self.args=args
        f=open(txt_file)
        lines=f.readlines()
        count=0
        while count< len(lines):
            rgb,depth,label,_ = lines[count].split('\n')[0].split('*')
            self.rgb.append(rgb)
            self.depth.append(depth)
            self.labels.append(label)
            count+=1      
            
        f.close()
        
    def extract_clips_equal(self, rgb_reader, depth_reader):
        clip_length=self.args.clip_length
        resize_ratio=self.args.resize_ratio
        indexes = np.linspace(0,len(rgb_reader)-1,clip_length,dtype=int)
        rgb_clip = rgb_reader.get_batch(indexes)  
        rgb_clip = rgb_clip.permute(3, 0, 1, 2) / 255.0 
        
        depth_clip = depth_reader['depth'][indexes]
        depth_clip = (depth_clip-np.mean(depth_clip))/np.std(depth_clip)
        depth_clip = depth_clip[np.newaxis, :]
        depth_clip = torch.from_numpy(depth_clip)
        
        target_size = (int(rgb_clip.shape[-2]/resize_ratio), int(rgb_clip.shape[-1]/resize_ratio))        
        rgb_clip = torch.nn.functional.interpolate(rgb_clip, size=target_size, mode="bilinear")
        depth_clip = torch.nn.functional.interpolate(depth_clip, size=target_size, mode="bilinear")
        
        return rgb_clip.float(),depth_clip.float()
            
    def __getitem__(self, index):
        rgb=decord.VideoReader(self.rgb[index])
        depth = h5py.File(self.depth[index],'r')
        label=int(self.labels[index])
        rgb_clip,depth_clip = self.extract_clips_equal(rgb, depth)
        return (rgb_clip, depth_clip, torch.tensor(label))
 
    def __len__(self):
        return len(self.rgb)