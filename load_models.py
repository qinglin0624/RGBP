from Models import *

def load_model(model_name,weights,num_class=5,FREEZE=False):
    """
    model_name: the model name to load
    weights: {} the dictionary of corresponding weights path to load
   
    """
    models={}
    if model_name == 'RGB':
        models['RGB'] = ResNet_small(num_class,3) 
            
    elif model_name == 'Depth':
        models['Depth'] = ResNet_small(num_class,1)
                
    elif model_name == 'Pdepth':
        models['Depth'] = ResNet_small(num_class,1)
        models['Pdepth'] = ResNet_small(num_class,3)
        if weights['Depth']:
            models['Depth'].load_state_dict(torch.load(weights['Depth'])) 
        models['Pdepth'].fc = models['Depth'].fc
        models['Depth'].cuda()
        
    elif model_name.startswith('RGBD'):
        models['RGBD'] = globals()[model_name](weights['RGB'],weights['Depth'],num_class)
        
        if FREEZE:
            if model_name=='RGBD_late' or model_name=='RGBD_center':
                for param in models['RGBD'].rgb_stream.parameters():
                    param.requires_grad = False
                for param in models['RGBD'].depth_stream.parameters():
                    param.requires_grad = False
            elif model_name=='RGBD_left':
                for param in models['RGBD'].depth_stream.parameters():
                    param.requires_grad=False
                for param in models['RGBD'].pre.parameters():
                    param.requires_grad=False
                for param in models['RGBD'].layer1.parameters():
                    param.requires_grad=False
                for param in models['RGBD'].layer2.parameters():
                    param.requires_grad=False
        
    elif model_name.startswith('RGBP'):
        models['RGBP'] = globals()[model_name](weights[model_name.replace('RGBP','RGBD')],weights['Pdepth'],num_class)
        if FREEZE:
            for param in models['RGBP'].parameters():
                param.requires_grad=False
            for param in models['RGBP'].rgbp.fc.parameters():
                param.requires_grad = True
    
    
    if model_name.startswith('RGBD') or model_name.startswith('RGBP'):
        if weights[model_name]:
            models[model_name[:4]].load_state_dict(torch.load(weights[model_name])) 
        models[model_name[:4]].cuda()
    else:
        if weights[model_name]:
            models[model_name].load_state_dict(torch.load(weights[model_name])) 
        models[model_name].cuda()
    
        
    return models