# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:48:05 2020

@author: Administrator
"""


import torch
import torch.nn as nn

#from tensorboardX import SummaryWriter
import numpy as np
import cv2
import time
import os
from model import *
import utils_train


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_checkpoint(checkpoint_dir,IsGPU):
    
	if IsGPU == 1:
		model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
		net = Main()
		device_ids = [0]
		model = nn.DataParallel(net, device_ids=device_ids).cuda()
		model.load_state_dict(model_info['state_dict'])
		optimizer = torch.optim.Adam(model.parameters())
		optimizer.load_state_dict(model_info['optimizer'])
		cur_epoch = model_info['epoch']
	else:

		model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar',map_location=torch.device('cpu'))        
		net = Main()
		device_ids = [0]
		model = nn.DataParallel(net, device_ids=device_ids)
		model.load_state_dict(model_info['state_dict'])
		optimizer = torch.optim.Adam(model.parameters())
		optimizer.load_state_dict(model_info['optimizer'])
		cur_epoch = model_info['epoch']

	return model, optimizer,cur_epoch

def adjust_learning_rate(optimizer, epoch, lr_update_freq):
	if not epoch % lr_update_freq and epoch:
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] * 0.1
			print( param_group['lr'])
	return optimizer

def train_psnr(train_in,train_out):
	
	psnr = utils_train.batch_psnr(train_in,train_out,1.)
	return psnr


def FSobel_XY(data):
    data = cv2.cvtColor(chw_to_hwc(data),cv2.COLOR_BGR2GRAY)

    
    x = cv2.Sobel(np.clip(data*255,0,255).astype('uint8'),cv2.CV_8U,0,1,ksize =3)
    absX = cv2.convertScaleAbs(x)
    y = cv2.Sobel(np.clip(data*255,0,255).astype('uint8'),cv2.CV_8U,1,0,ksize =3)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
    
    return dst/255.0

def FLap(data):
    data = cv2.cvtColor(chw_to_hwc(data),cv2.COLOR_BGR2GRAY)
    x = cv2.GaussianBlur(data, (3,3),0)
    x = cv2.Laplacian(np.clip(x*255,0,255).astype('uint8'),cv2.CV_8U,ksize =3)
    Lap = cv2.convertScaleAbs(x)
    return Lap/255.0

def GFSobel_XY(data):

    x = cv2.Sobel(np.clip(data*255,0,255).astype('uint8'),cv2.CV_8U,0,1,ksize =3)
    absX = cv2.convertScaleAbs(x)
    y = cv2.Sobel(np.clip(data*255,0,255).astype('uint8'),cv2.CV_8U,1,0,ksize =3)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
    
    return dst/255.0

def GFLap(data):
    x = cv2.GaussianBlur(data, (3,3),0)
    x = cv2.Laplacian(np.clip(x*255,0,255).astype('uint8'),cv2.CV_8U,ksize =3)
    Lap = cv2.convertScaleAbs(x)
    return Lap/255.0	


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])
	

if __name__ == '__main__': 	
	checkpoint_dir = './checkpoint/'
	test_dir = './dataset/Test_2k'
	result_dir = './Test_result'    
	testfiles = os.listdir(test_dir)
    
	IsGPU = 1    #GPU is 1, CPU is 0

	print('> Loading dataset ...')

	lr_update_freq = 30
	model,optimizer,cur_epoch = load_checkpoint(checkpoint_dir,IsGPU)

	if IsGPU == 1:
		for f in range(len(testfiles)):
			model.eval()
			with torch.no_grad():
				img = cv2.imread(test_dir + '/' + testfiles[f])
				img_g = cv2.imread(test_dir + '/' + testfiles[f],0)

				img_g = img_g / 255.0 
				h,w,c = img.shape
				img_ccc = img / 255.0
				img_h = hwc_to_chw(img_ccc)
				input_var = torch.from_numpy(img_h.copy()).type(torch.FloatTensor).unsqueeze(0).cuda()
                
				input_svar = torch.from_numpy(GFSobel_XY(img_g.copy())).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).cuda()
				input_lapvar = torch.from_numpy(GFLap(img_g.copy())).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).cuda()
                
				s = time.time()
				lapout,lbpout,e_out = model(input_var,input_svar,input_lapvar)                
				e = time.time()   
				print(input_var.shape)       
				print(e-s)    
	             

				e_out = e_out.squeeze().cpu().detach().numpy()			               
				e_out = chw_to_hwc(e_out) 
				e_out = cv2.resize(e_out,(w,h))

				newlap = np.zeros((e_out.shape))		              
				newlbp = np.zeros((e_out.shape))	
                
				lap_out = lapout.squeeze().squeeze().cpu().detach().numpy()			               
				lap_out = cv2.resize(lap_out,(w,h))
                
				for clap in range(3):
					newlap[:,:,clap] = lap_out
                

				lbp_out = lbpout.squeeze().squeeze().cpu().detach().numpy()			               
				lbp_out = cv2.resize(lbp_out,(w,h))	

                
				for clbp in range(3):
					newlbp[:,:,clbp] = lbp_out
                    
				temp = e_out#np.concatenate((img/255,e_out,newlap, newlbp), axis=1)
                
                
				cv2.imwrite(result_dir + '/' + testfiles[f][:-4] +'_Grad4'+'.png',np.clip(temp*255,0.0,255.0))
           
	  
				
			
			

