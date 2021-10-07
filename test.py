import time
import json
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.utils.data as data
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from thop import profile
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from FSLoss import *
from utils import *
from dataset import *
from load_models import *

def test_model(args, models, test_loader):
    # Calculate FLOPs and params
    num_class=args.num_class
    TRIAL=args.TRIAL
    criterion = torch.nn.CrossEntropyLoss().cuda()
    if args.model_name.startswith('RGBD') or args.model_name.startswith('RGBP'):
        MODE = args.model_name[:4]
    else:
        MODE = args.model_name
    lambdas = args.lambdas
    
    rgb=torch.randn(1, 3, 16, 135, 240).cuda()
    depth=torch.randn(1, 1, 16, 135, 240).cuda()
    if MODE == 'Depth':
        inputs=(depth,)
    elif MODE == 'RGBD':
        inputs=(rgb,depth,)
    else:
        inputs=(rgb,)
        
    flops, params = profile(models[MODE], inputs=inputs)
    print("Begin Testing {} ...".format(MODE))
    print("FLOPs=", str(flops/1e9) +'{} per sample'.format("G"))
    print("params=", str(params/1e6)+'{}'.format("M"))
    
    test_size=len(test_loader.dataset) 
    
    
    y_pre = np.array([])
    y_tru = np.array([])
    t_total=0.0
        
    with torch.no_grad():
        models[MODE].eval()
        test_loss, test_correct = 0, 0  

        if MODE == 'Pdepth':
            FS_test_loss = 0.0

        
        for batch_idx, (rgb, depth, y) in enumerate(test_loader):
            y = y.cuda(non_blocking=True)

            if MODE == 'Pdepth':
                rgb, depth = rgb.cuda(non_blocking=True), depth.cuda(non_blocking=True)
                t0=time.perf_counter()
                depth_out,done, dtwo, dthree, dfour = models['Depth'](depth)
                preds,pone, ptwo, pthree, pfour = models[MODE](rgb)
                t1=time.perf_counter()
                
                loss,CELoss, consis, corr = calculate_loss(criterion,lambdas,[done, dtwo, dthree, dfour],[pone, ptwo, pthree, pfour],depth_out,preds,y)    
                FS_test_loss += loss.item() * y.size(0)

            elif MODE == 'RGBD': 
                rgb, depth = rgb.cuda(non_blocking=True), depth.cuda(non_blocking=True)
                t0=time.perf_counter()
                preds = models[MODE](rgb, depth)
                t1=time.perf_counter()
                
                loss = criterion(preds,y)
                CELoss=loss.item()


            elif MODE == 'Depth':
                depth = depth.cuda(non_blocking=True)
                t0=time.perf_counter()
                preds,_,_,_,_ = models[MODE](depth)
                t1=time.perf_counter()
                
                loss = criterion(preds,y)
                CELoss=loss.item()

            elif MODE=='RGB':
                rgb = rgb.cuda(non_blocking=True)
                t0=time.perf_counter()
                preds,_,_,_,_ = models[MODE](rgb)
                t1=time.perf_counter()
                
                loss = criterion(preds,y)  
                CELoss=loss.item()
                
            elif MODE=='RGBP':
                rgb = rgb.cuda(non_blocking=True)
                t0=time.perf_counter()
                preds= models[MODE](rgb)
                t1=time.perf_counter()
                
                loss = criterion(preds,y)  
                CELoss=loss.item()
            
            t_total+=t1-t0
                
            preds_o = torch.argsort(-preds, axis=1)
            test_correct += (preds_o[:,0]==y).sum().item()
            test_loss += CELoss* y.size(0)        
            y_pre = np.concatenate((y_pre,preds_o[:,0].detach().cpu().numpy()))
            y_tru = np.concatenate((y_tru,y.detach().cpu().numpy()))
            


        test_loss = test_loss / test_size
        test_acc = test_correct / test_size
        F1_score = f1_score(y_tru,y_pre,average='weighted')
        
        if MODE == 'Pdepth':
            FS_test_loss = FS_test_loss / test_size                        
            print('Test FS loss: {:.4f}   CE test loss: {:.4f}   acc: {:.4f}   F1 score: {:.4f}'.format(FS_test_loss, test_loss, test_acc, F1_score))

        else:

            print ('Test CE loss: {:.4f}   acc: {:.4f}   F1 score: {:.4f}'.format(test_loss, test_acc, F1_score))
            
        cm = confusion_matrix(y_tru, y_pre)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Confusion Matrix:')
        print (cm_normalized)
            
        print('\n testing time {:.0f}m {:.0f}s'.format(t_total // 60, t_total % 60))  


if __name__ == '__main__':
    parser = get_test_parser()
    args = parser.parse_args()
    weights={'RGB':args.RGB_weight,'Depth':args.Depth_weight,'Pdepth':args.Pdepth_weight,'RGBD_late':args.RGBD_late_weight,
         'RGBD_center':args.RGBD_center_weight,'RGBD_left':args.RGBD_left_weight,'RGBP_late':args.RGBP_late_weight,
        'RGBP_center':args.RGBP_center_weight,'RGBP_left':args.RGBP_left_weight}
    models = load_model(args.model_name,weights,num_class=args.num_class,FREEZE=False)
    test_set = RGBDDataset(args,args.test_txt)
    test_loader = data.DataLoader(test_set, shuffle=False, num_workers=args.num_workers, pin_memory=True, batch_size=args.BS)
    
    test_model(args, models, test_loader)
    