import torch

def corr(feature_map):
    # reshape (B,C,T,W,H) into d*C
    feature_map = torch.transpose(feature_map, 1, 4)
    
    B,H,T,W,C = feature_map.shape
    ft_map = torch.reshape(feature_map, (B,W*H*T,C))
    
    # calculate F^
    variance, sample_mean = torch.var_mean(ft_map, dim=2, keepdim=True)
    sub_map = torch.sub(ft_map, sample_mean)
    div_map = torch.div(sub_map, torch.sqrt(variance))
    norm = torch.norm(div_map, p=2, dim=2, keepdim=True)
    _ft_map = torch.div(div_map, norm)    
    
    result = _ft_map.matmul(torch.transpose(_ft_map, 1, 2))
    return result


def calculate_loss(criterion,lambdas,depth_fms,pseudo_fms,depth_out,pseudo_out,y):
    pseudo_CE =0
    
    # consistency loss on layer 2,3,4
    consis_loss = 0
    for i in [1,2,3]:
        B,C,T,W,H = (depth_fms[i]).shape
        size = B*C*T*W*H
        consis_loss += lambdas[i] * torch.square(torch.norm(depth_fms[i] - pseudo_fms[i])) / size
    
    corr_loss = 0
    d_corr = corr(depth_fms[0])
    pd_corr = corr(pseudo_fms[0])
    a,b,c = (d_corr).shape
    size = a*b*c
    corr_loss = lambdas[0] * torch.square(torch.norm(d_corr - pd_corr)) / size
    total_loss = consis_loss + corr_loss

    pseudo_CE = criterion(pseudo_out,y)
    return total_loss, pseudo_CE.item(), consis_loss.item(), corr_loss.item()