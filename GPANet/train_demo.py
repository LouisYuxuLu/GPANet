# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:48:05 2020

@author: Administrator
"""

import os, time, scipy.io, shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
import numpy as np
import cv2
import scipy.misc
from model import *
from makedataset import Dataset
import utils_train



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_checkpoint(checkpoint_dir):
	if os.path.exists(checkpoint_dir + 'checkpoint.pth.tar'):
		model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
		print('loading existing model ......', checkpoint_dir + 'checkpoint.pth.tar')
		net = Main()
		device_ids = [0]
		model = nn.DataParallel(net, device_ids=device_ids).cuda()
		model.load_state_dict(model_info['state_dict'])
		optimizer = torch.optim.Adam(model.parameters())
		optimizer.load_state_dict(model_info['optimizer'])
		cur_epoch = model_info['epoch']
		
	else:
		net = Main()
		device_ids = [0]
		model = nn.DataParallel(net, device_ids=device_ids).cuda()
		optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
		cur_epoch = 0
		
	return model, optimizer,cur_epoch


def save_checkpoint(state, is_best, PSNR,SSIM,filename='checkpoint.pth.tar'):
	torch.save(state, checkpoint_dir + 'PSNR_%.4f_SSIM_%.4f_'%(PSNR,SSIM) + 'checkpoint.pth.tar')
	if is_best:
		shutil.copyfile(checkpoint_dir + 'checkpoint.pth.tar',checkpoint_dir + 'model_best.pth.tar')


        
def adjust_learning_rate(optimizer, epoch, lr_update_freq):
	if not epoch % lr_update_freq and epoch:
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] * 0.1
			print( param_group['lr'])
	return optimizer

def train_psnr(train_in,train_out):
	
	psnr = utils_train.batch_psnr(train_in,train_out,1.)
	return psnr


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])

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
    x = data#cv2.GaussianBlur(data, (3,3),0)
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
    x = data#cv2.GaussianBlur(data, (3,3),0)
    x = cv2.Laplacian(np.clip(x*255,0,255).astype('uint8'),cv2.CV_8U,ksize =3)
    Lap = cv2.convertScaleAbs(x)
    return Lap/255.0	

def thresholded(center, pixels):
    out = []
    for a in pixels:
        if a >= center:
            out.append(1)
        else:
            out.append(0)
    return out

def get_pixel_else_0(l, idx, idy, default=0):
    try:
        return l[idx,idy]
    except IndexError:
        return default

def LowLightType01(img):
   
    g = np.random.uniform(0.1,0.5)
    img_l = img*g
    
    return img_l


if __name__ == '__main__': 	
	checkpoint_dir = './checkpoint/'
	test_dir1 = './dataset/Test'

	result_dir = './result'
    
	testfiles1 = os.listdir(test_dir1)

    
	maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    
	print('> Loading dataset ...')
	dataset = Dataset(trainrgb=True, trainsyn=True, shuffle=True)
	loader_dataset = DataLoader(dataset=dataset, num_workers=0, batch_size=16, shuffle=True)
	count = len(loader_dataset)
	
	lr_update_freq = 20
	model,optimizer,cur_epoch = load_checkpoint(checkpoint_dir)
    
	L1_loss = torch.nn.L1Loss(reduce=True, size_average=True).cuda()
	L2_loss = torch.nn.MSELoss(reduce=True, size_average=True).cuda()

	
	for epoch in range(cur_epoch,80):
		optimizer = adjust_learning_rate(optimizer, epoch, lr_update_freq)
		learnrate = optimizer.param_groups[-1]['lr']
		model.train()

        
		aaa = 0
		for i,data in enumerate(loader_dataset,0):

			img_c       = torch.zeros(data[:,0:3,:,:].size())		
			img_l       = torch.zeros(data[:,0:3,:,:].size())

			img_c_sobel = torch.zeros(data[:,0:1,:,:].size())
			img_l_sobel = torch.zeros(data[:,0:1,:,:].size())

			img_c_lap   = torch.zeros(data[:,0:1,:,:].size())
			img_l_lap   = torch.zeros(data[:,0:1,:,:].size())
            

			for nx in range(data.shape[0]):             
				img_c[nx,:,:,:]       = data[nx,0:3,:,:]
				img_c_sobel[nx,:,:,:] =  torch.from_numpy(FSobel_XY(data[nx,0:3,:,:].numpy()))
				img_c_lap[nx,:,:,:]   = torch.from_numpy(FLap(data[nx,0:3,:,:].numpy()))
                
	
			for nxx in range(data.shape[0]):

				sor = np.random.uniform(0,1)                    
				if sor <= 1:
					img_l[nxx] = data[nxx,3:6,:,:]
				else:
					img_l[nxx] = LowLightType01(data[nxx,0:3,:,:])

				img_l_sobel[nxx,:,:,:] = torch.from_numpy(FSobel_XY(img_l[nxx,0:3,:,:].numpy()))
				img_l_lap[nxx,:,:,:]   = torch.from_numpy(FLap(img_l[nxx,0:3,:,:].numpy()))

                
                    
									
			input_var = Variable(img_l.cuda(), volatile=True)        
			target_final = Variable(img_c.cuda(), volatile=True)

			input_var_s = Variable(img_l_sobel.cuda(), volatile=True)        
			target_final_s = Variable(img_c_sobel.cuda(), volatile=True)
            
			input_var_l = Variable(img_l_lap.cuda(), volatile=True)        
			target_final_l = Variable(img_c_lap.cuda(), volatile=True)
            
			lapout,sobelout,eout = model(input_var,input_var_s,input_var_l)

			enloss = 0.8*(0.5*L2_loss(eout,target_final)+0.5*L1_loss(eout,target_final)) +\
                   0.1*(0.5*L2_loss(lapout,target_final_l)+0.5*L1_loss(lapout,target_final_l)) +\
                   0.1*(0.5*L2_loss(sobelout,target_final_s)+0.5*L1_loss(sobelout,target_final_s)) +\            

			optimizer.zero_grad()
            
			enloss.backward()
            
			optimizer.step()
            
			SN1_psnr = train_psnr(target_final,eout)		           
			print("[Epoch %d][Type--Enhancement--][%d/%d] lr :%f loss: %.4f PSNR_train: %.4f" %(epoch+1, i+1, count, learnrate, enloss.item(), SN1_psnr))
			
		for f in range(len(testfiles1)):
			model.eval()
			with torch.no_grad():
				img = cv2.imread(test_dir1 + '/' + testfiles1[f])
				img_g = cv2.imread(test_dir1 + '/' + testfiles1[f],0)
				img_g = img_g /255.0 #cv2.resize(img_g,(512,512)) / 255.0 
				h,w,c = img.shape
				img_ccc = img /255.0#cv2.resize(img,(512,512)) / 255.0
				img_h = hwc_to_chw(img_ccc)
				input_var = torch.from_numpy(img_h.copy()).type(torch.FloatTensor).unsqueeze(0).cuda()
                
				input_svar = torch.from_numpy(GFSobel_XY(img_g.copy())).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).cuda()
				#input_svar = torch.from_numpy(GFLbp(img_g.copy())).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).cuda()
				input_lapvar = torch.from_numpy(GFLap(img_g.copy())).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).cuda()
                
				s = time.time()
				lapout,sobelout,e_out = model(input_var,input_svar,input_lapvar)                
				e = time.time()   
				print(input_var.shape)       
				print(e-s)    
	             

				e_out = e_out.squeeze().cpu().detach().numpy()			               
				e_out = chw_to_hwc(e_out) 


				newlap = np.zeros((e_out.shape))		              
				newlbp = np.zeros((e_out.shape))	
                
				lap_out = lapout.squeeze().squeeze().cpu().detach().numpy()			               

                
				for clap in range(3):
					newlap[:,:,clap] = lap_out
                
				sobel_out = sobelout.squeeze().squeeze().cpu().detach().numpy()			               
                
				for clbp in range(3):
					newsobel[:,:,clbp] = sobel_out
				temp = np.concatenate((e_out,newlap, newsobel), axis=1)
                
                
				cv2.imwrite(result_dir + '/' + testfiles1[f][:-4] +'_%d'%(epoch)+'.png',np.clip(temp*255,0.0,255.0))
				cv2.imwrite('./GPANet/' + testfiles1[f][:-4] +'_GPANet.png',np.clip(e_out*255,0.0,255.0))



		print('PSNR_%.4f_SSIM_%.4f'%(ps,ss))
		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict()}, is_best=0,PSNR=ps,SSIM=ss)
			
			

