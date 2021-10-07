import argparse

def get_train_parser():
    parser = argparse.ArgumentParser(description='Train parser')
    parser.add_argument('--TRIAL',default='Train_RGBD_late')
    parser.add_argument('--num_class',type=int,default=5)
    parser.add_argument('--model_name',default='RGBD_late')
    parser.add_argument('--train_txt',default='input_template/train_txt_file.txt')
    parser.add_argument('--eval_txt',default='input_template/eval_txt_file.txt')    
    parser.add_argument('--RGB_weight',default=None)
    parser.add_argument('--Depth_weight',default=None)
    parser.add_argument('--Pdepth_weight',default=None)
    parser.add_argument('--RGBD_late_weight',default=None)
    parser.add_argument('--RGBD_center_weight',default=None)
    parser.add_argument('--RGBD_left_weight',default=None)
    parser.add_argument('--RGBP_late_weight',default=None)
    parser.add_argument('--RGBP_center_weight',default=None)
    parser.add_argument('--RGBP_left_weight',default=None)
    parser.add_argument('--lambdas',type=list,default=[1,2,3,5])
    parser.add_argument('--EPOCH',type=int,default=10)
    parser.add_argument('--BS',type=int,default=16)
    parser.add_argument('--num_workers',type=int,default=8)
    parser.add_argument('--rho',type=float,default=0.9)
    parser.add_argument('--eps',type=float,default=1e-6)
    parser.add_argument('--FREEZE',default=False)
    parser.add_argument('--weight_decay',type=float,default=1e-3)
    parser.add_argument('--dot_per_epoch',type=int,default=10)
    parser.add_argument('--clip_length',type=int,default=16)
    parser.add_argument('--resize_ratio',type=int,default=8)
    return parser

def get_test_parser():
    parser = argparse.ArgumentParser(description='Test parser')
    parser.add_argument('--num_class',type=int,default=5)    
    parser.add_argument('--TRIAL',default='Test_RGBP_late')
    parser.add_argument('--model_name',default='RGBP_late')
    parser.add_argument('--test_txt',default='input_template/test_txt_file.txt')    
    parser.add_argument('--RGB_weight',default=None)
    parser.add_argument('--Depth_weight',default=None)
    parser.add_argument('--Pdepth_weight',default=None)
    parser.add_argument('--RGBD_late_weight',default=None)
    parser.add_argument('--RGBD_center_weight',default=None)
    parser.add_argument('--RGBD_left_weight',default=None)
    parser.add_argument('--RGBP_late_weight',default=None)
    parser.add_argument('--RGBP_center_weight',default=None)
    parser.add_argument('--RGBP_left_weight',default=None)
    parser.add_argument('--lambdas',type=list,default=[1,2,3,5])
    parser.add_argument('--BS',type=int,default=1)
    parser.add_argument('--num_workers',type=int,default=4)    
    parser.add_argument('--clip_length',type=int,default=16)
    parser.add_argument('--resize_ratio',type=int,default=8)
    return parser

    
    
    
    
    
    
    
    