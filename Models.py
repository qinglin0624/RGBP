import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def _make_layer(inchannel,outchannel,block_num,stride = 1):
    
    shortcut = nn.Sequential(
        nn.Conv3d(inchannel,outchannel,1,stride,bias=False),
        nn.BatchNorm3d(outchannel)
    )
    layers = []
    layers.append(ResidualBlock(inchannel,outchannel,stride,shortcut))
    
    for i in range(1,block_num):
        layers.append(ResidualBlock(outchannel,outchannel))

    return nn.Sequential(*layers)
# ---------------------------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride = 1,shortcut = None):

        super().__init__()
        self.left = nn.Sequential(
            nn.Conv3d(inchannel,outchannel,3,stride,1,bias=False),
            nn.BatchNorm3d(outchannel),
            nn.ReLU(),
            nn.Conv3d(outchannel,outchannel,3,1,1,bias=False), 
            nn.BatchNorm3d(outchannel)
         )
        self.right = shortcut

    def forward(self, input):
        out = self.left(input)
        residual = input if self.right is None else self.right(input)
        out+=residual
        return F.relu(out)

    
class ResNet_small(nn.Module):

    def __init__(self,num_class=5, in_ch = 3):
        super().__init__()
    
        self.pre = nn.Sequential(
            nn.Conv3d(in_ch,16,7,2,3,bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(3,2,1)
        )

        self.layer1 = _make_layer(16,32,3)
        self.layer2 = _make_layer(32,64,4,stride=2) 
        self.layer3 = _make_layer(64,128,6,stride=2)
        self.layer4 = _make_layer(128,128,3,stride=2)
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Linear(128,num_class)   

    def forward(self, input):
        x = self.pre(input)

        x = self.layer1(x)
        one = x
        x = self.layer2(x)
        two = x
        x = self.layer3(x)
        three = x
        x = self.layer4(x)
        four = x
        x = self.maxpool(x) 
                                
        x = x.view(x.size(0),-1)
        
        return self.fc(x), one, two, three, four
# ---------------------------------------------------------------------------------------------    

class RGBD_late(nn.Module):
    
    def __init__(self, rgb_weight=None, d_weight=None, num_class=5):
        super().__init__()
        self.rgb_stream = ResNet_small(in_ch=3,num_class=num_class)
        self.depth_stream = ResNet_small(in_ch=1,num_class=num_class)
        
        if rgb_weight:
            self.rgb_stream.load_state_dict(torch.load(rgb_weight))
        if d_weight:
            self.depth_stream.load_state_dict(torch.load(d_weight))
        
        self.rgb_stream = torch.nn.Sequential(*(list(self.rgb_stream.children())[:-1]))
        self.depth_stream = torch.nn.Sequential(*(list(self.depth_stream.children())[:-1]))
        
        self.fc = nn.Linear(256,num_class)

    def forward(self, rgb_input, depth_input):
        rgb_o  = self.rgb_stream(rgb_input)
        rgb_o = rgb_o.view(rgb_o.size(0),-1)
        depth_o  = self.depth_stream(depth_input)
        depth_o = depth_o.view(depth_o.size(0),-1)
        
        final_fc = torch.cat((rgb_o,depth_o),dim=1)
        final_fc = self.fc(final_fc)
        
        return final_fc
# ---------------------------------------------------------------------------------------------
    
class RGBD_center(nn.Module):

    def __init__(self, rgb_weight=None, d_weight=None, num_class=5):
        super().__init__()
        self.rgb_stream = ResNet_small(in_ch=3,num_class=num_class)
        self.depth_stream = ResNet_small(in_ch=1,num_class=num_class)
        if rgb_weight:
            self.rgb_stream.load_state_dict(torch.load(rgb_weight))
        if d_weight:
            self.depth_stream.load_state_dict(torch.load(d_weight))
        
        self.layer1 = _make_layer(64,128,4,stride=2) 
        self.layer2 = _make_layer(256,512,6,stride=2)
        self.layer3 = _make_layer(768,1024,3,stride=2)
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Linear(1280,num_class) 

    def forward(self, rgb_input, depth_input):
        _, rgb_one, rgb_two, rgb_three, rgb_four = self.rgb_stream(rgb_input)
        _, d_one, d_two, d_three, d_four = self.depth_stream(depth_input)
        x = self.layer1(torch.cat((rgb_one,d_one),dim=1))
        x = self.layer2(torch.cat((rgb_two,x,d_two),dim=1))
        x = self.layer3(torch.cat((rgb_three,x,d_three),dim=1))
        x = self.maxpool(torch.cat((rgb_four,x,d_four),dim=1))
        x = x.view(x.size(0),-1)
        return self.fc(x)
# ---------------------------------------------------------------------------------------------

class RGBD_left(nn.Module):

    def __init__(self,rgb_weight=None,d_weight=None,num_class=5):
        super().__init__()
        
        self.depth_stream = ResNet_small(in_ch=1,num_class=num_class)
        if d_weight:
            self.depth_stream.load_state_dict(torch.load(d_weight))
            
        self.pre = nn.Sequential(
            nn.Conv3d(3,16,7,2,3,bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(3,2,1)
        )
        self.layer1 = _make_layer(16,32,3)
        self.layer2 = _make_layer(32,64,4,stride=2)
        
        if rgb_weight:
            rgb_model=ResNet_small(5,in_ch=3)
            rgb_model.load_state_dict(torch.load(rgb_weight))
            self.pre=torch.nn.Sequential(*(list(rgb_model.children())[0]))
            self.layer1=torch.nn.Sequential(*(list(rgb_model.children())[1]))
            self.layer2=torch.nn.Sequential(*(list(rgb_model.children())[2]))

        self.layer3 = _make_layer(128,128,6,stride=2)
        self.layer4 = _make_layer(256,256,3,stride=2)
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Linear(256+128,num_class)
        
    def forward(self, rgb_input, d_input):
        
        _, one, two, three, four = self.depth_stream(d_input)
        
        x = self.pre(rgb_input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(torch.cat((x,two),dim=1))
        x = self.layer4(torch.cat((x,three),dim=1))
        x = torch.cat((x,four),dim=1) 
        x = self.maxpool(x)                      
        x = x.view(x.size(0),-1)
        
        return self.fc(x)
# ---------------------------------------------------------------------------------------------

class RGBP_late(nn.Module):
    def __init__(self, rgbd_weight=None, pd_weight=None, num_class=5):
        super().__init__()
        rgbd = RGBD_late(num_class=num_class)
        pdepth_stream = ResNet_small(in_ch=3,num_class=num_class)
        
        if rgbd_weight:
            rgbd.load_state_dict(torch.load(rgbd_weight))
        if pd_weight:
            pdepth_stream.load_state_dict(torch.load(pd_weight))
            
        pdepth_stream = torch.nn.Sequential(*(list(pdepth_stream.children())[:-1]))        
        rgbd.depth_stream = pdepth_stream
        self.rgbp=rgbd

    def forward(self, rgb_input): 
        return self.rgbp(rgb_input,rgb_input)
# ---------------------------------------------------------------------------------------------

class RGBP_center(nn.Module):
    def __init__(self, rgbd_weight=None, pd_weight=None, num_class=5):
        super().__init__()
        rgbd = RGBD_center(num_class=num_class)
        pdepth_stream = ResNet_small(in_ch=3,num_class=num_class)
        
        if rgbd_weight:
            rgbd.load_state_dict(torch.load(rgbd_weight))
        if pd_weight:
            pdepth_stream.load_state_dict(torch.load(pd_weight))
            
        rgbd.depth_stream = pdepth_stream
        self.rgbp=rgbd

    def forward(self, rgb_input):
        return self.rgbp(rgb_input,rgb_input)
# ---------------------------------------------------------------------------------------------

class RGBP_left(nn.Module):
    def __init__(self, rgbd_weight=None, pd_weight=None, num_class=5):
        super().__init__()
        rgbd = RGBD_left(num_class=num_class)
        pdepth_stream = ResNet_small(in_ch=3,num_class=num_class)
        
        if rgbd_weight:
            rgbd.load_state_dict(torch.load(rgbd_weight))
        if pd_weight:
            pdepth_stream.load_state_dict(torch.load(pd_weight))
            
        rgbd.depth_stream = pdepth_stream
        self.rgbp=rgbd

    def forward(self, rgb_input):
        return self.rgbp(rgb_input,rgb_input)

      