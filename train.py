import time
import json
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.utils.data as data
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from FSLoss import *
from utils import *
from dataset import *
from load_models import *

def calculate_step(set_size, dot_per_epoch):
    num_iter = set_size / args.BS
    step = num_iter / dot_per_epoch
    return int(step)

# def train_model(TRIAL, MODE, models, optimizer, train_loader, eval_loader, criterion, EPOCH, dot_per_epoch=10):
def train_model(args, models,train_loader, eval_loader):
    TRIAL = args.TRIAL
    if args.model_name.startswith('RGBD') or args.model_name.startswith('RGBP'):
        MODE = args.model_name[:4]
    else:
        MODE = args.model_name
    EPOCH = args.EPOCH
    dot_per_epoch = args.dot_per_epoch
    lambdas = args.lambdas
    
    stats = {'train':([],[]), 'eval':([],[])}
    train_begin = time.time()
    
    train_size=len(train_loader.dataset)
    eval_size=len(eval_loader.dataset)  
    
    STEP = calculate_step(set_size=train_size, dot_per_epoch=args.dot_per_epoch)    
    
    optimizer = optim.Adadelta(models[MODE].parameters(), rho=args.rho, eps=args.eps, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    
    for epoch in range(EPOCH):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, EPOCH - 1))
        print('-' * 10) 
        
        models[MODE].train()
        train_loss, train_correct = 0,0 
        
        if MODE == 'Pdepth':
            models['Depth'].eval()
            FS_train_loss = 0.0             
        
        ptime = time.time()
        for itera, (rgb, depth, y) in enumerate(train_loader):
            itime = time.time()
            y = y.cuda(non_blocking=True)
            
            if MODE == 'Pdepth':               
                rgb, depth = rgb.cuda(non_blocking=True), depth.cuda(non_blocking=True)
                depth_out,done, dtwo, dthree, dfour = models['Depth'](depth)
                preds,pone, ptwo, pthree, pfour = models[MODE](rgb)
                loss,CELoss, consis, corr = calculate_loss(criterion,lambdas,[done, dtwo, dthree, dfour],[pone, ptwo, pthree, pfour],depth_out,preds,y)                
            
            elif MODE == 'RGBD': 
                rgb, depth = rgb.cuda(non_blocking=True), depth.cuda(non_blocking=True)
                preds = models[MODE](rgb, depth)
                loss = criterion(preds,y)
                CELoss=loss.item()
            
            elif MODE == 'Depth':
                depth = depth.cuda(non_blocking=True)
                preds, _, _, _, _ = models[MODE](depth)
                loss = criterion(preds,y)
                CELoss=loss.item()
                
            elif MODE == 'RGB':
                rgb = rgb.cuda(non_blocking=True)                
                preds, _, _, _, _ = models[MODE](rgb)
                loss = criterion(preds,y)  
                CELoss=loss.item()
                
            else:
                rgb = rgb.cuda(non_blocking=True)                
                preds = models[MODE](rgb)
                loss = criterion(preds,y)  
                CELoss=loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += CELoss * y.size(0)            
            preds_o = torch.argsort(-preds, axis=1)
            c = (preds_o[:,0]==y).sum().item()
            train_correct += c
            
            if MODE == 'Pdepth':
                FS_train_loss += loss.item() * y.size(0)
                print('Iter {}: FS Loss: {:.4f}  CE loss: {:.4f}  model time: {:.3f}s  iter time: {:.3f}s'.format(itera, loss.item(), CELoss, time.time()-itime, time.time()-ptime))
                
            else:
                print('Iter {}:  loss: {:.4f}  model time: {:.3f}s  iter time: {:.3f}s'.format(itera, CELoss, time.time()-itime, time.time()-ptime))
            ptime =time.time()
            
            if itera%STEP==0 and itera//STEP>0:
                
                with torch.no_grad():
                    models[MODE].eval()
                    eval_loss, eval_correct = 0, 0  
                     
                    if MODE == 'Pdepth':
                        FS_eval_loss = 0.0
                        
                    
                    for batch_idx, (rgb, depth, y) in enumerate(eval_loader):
                        y = y.cuda(non_blocking=True)
                        
                        if MODE == 'Pdepth':
                            rgb, depth = rgb.cuda(non_blocking=True), depth.cuda(non_blocking=True)
                            depth_out,done, dtwo, dthree, dfour = models['Depth'](depth)
                            preds,pone, ptwo, pthree, pfour = models[MODE](rgb)
                            
                            loss,CELoss, consis, corr = calculate_loss(criterion,lambdas,[done, dtwo, dthree, dfour],[pone, ptwo, pthree, pfour],depth_out,preds,y)    
                            FS_eval_loss += loss.item() * y.size(0)

                        elif MODE == 'RGBD': 
                            rgb, depth = rgb.cuda(non_blocking=True), depth.cuda(non_blocking=True)
                            preds = models[MODE](rgb, depth)
                            loss = criterion(preds,y)
                            CELoss=loss.item()
                            

                        elif MODE == 'Depth':
                            depth = depth.cuda(non_blocking=True)
                            preds, _, _, _, _ = models[MODE](depth)
                            loss = criterion(preds,y)
                            CELoss=loss.item()

                        elif MODE == 'RGB':
                            rgb = rgb.cuda(non_blocking=True)                
                            preds, _, _, _, _ = models[MODE](rgb)
                            loss = criterion(preds,y)  
                            CELoss=loss.item()
                            
                        else:
                            rgb = rgb.cuda(non_blocking=True)                
                            preds = models[MODE](rgb)
                            loss = criterion(preds,y)  
                            CELoss=loss.item()


                        eval_loss += CELoss* y.size(0)
                        
                        preds_o = torch.argsort(-preds, axis=1)
                        eval_correct += (preds_o[:,0]==y).sum().item()
                    
                   
                    eval_loss = eval_loss / eval_size
                    eval_acc = eval_correct / eval_size
                    stats['eval'][1].append(eval_acc)
                    
                    if MODE == 'Pdepth':
                        FS_eval_loss = FS_eval_loss / eval_size                        
                        print('FS eval loss: {:.4f}   CE eval loss: {:.4f}   acc: {:.4f}'.format(FS_eval_loss, eval_loss, eval_acc))
                    
                        stats['eval'][0].append([FS_eval_loss, eval_loss])
                        
                    else:
                        stats['eval'][0].append(eval_loss)
                        
                        print ('eval loss: {:.4f}   acc: {:.4f}'.format(eval_loss, eval_acc))
                        
                models[MODE].train()
            
        epoch_loss = train_loss / train_size
        epoch_acc = train_correct / train_size
        stats['train'][1].append(epoch_acc)
        
        if MODE == 'Pdepth':
            FS_train_loss = FS_train_loss / eval_size
            stats['train'][0].append([FS_train_loss, epoch_loss])
            print ('train FS loss: {:.4f}   train CE loss: {:.4f}   acc: {:.4f}'.format(FS_train_loss, epoch_loss, epoch_acc))
        else:
            stats['train'][0].append(epoch_loss)
            print ('train loss: {:.4f}   acc: {:.4f}'.format(epoch_loss, epoch_acc))
        
        json.dump(stats, open('Outputs/'+TRIAL + '_stats.json','w'))
        torch.save(models[MODE].state_dict(), 'Outputs/'+TRIAL+ '_'+ str(epoch)+'e.pth')
        
        epoch_length = time.time() - epoch_start
        print('epoch time {:.0f}m {:.0f}s'.format(epoch_length // 60, epoch_length % 60))
        print()
        
    print('Training Complete.')
    
    
if __name__ == '__main__':
    parser = get_train_parser()
    args = parser.parse_args()
    weights={'RGB':args.RGB_weight,'Depth':args.Depth_weight,'Pdepth':args.Pdepth_weight,'RGBD_late':args.RGBD_late_weight,
         'RGBD_center':args.RGBD_center_weight,'RGBD_left':args.RGBD_left_weight,'RGBP_late':args.RGBP_late_weight,
        'RGBP_center':args.RGBP_center_weight,'RGBP_left':args.RGBP_left_weight}
    models = load_model(args.model_name,weights,num_class=args.num_class,FREEZE=args.FREEZE)
    train_set = RGBDDataset(args,args.train_txt)
    eval_set = RGBDDataset(args,args.eval_txt)
    train_loader = data.DataLoader(train_set, shuffle=True, num_workers=args.num_workers, pin_memory=True, batch_size=args.BS)
    eval_loader = data.DataLoader(eval_set, num_workers=args.num_workers, pin_memory=True, batch_size=args.BS)
    
    train_model(args, models, train_loader, eval_loader)