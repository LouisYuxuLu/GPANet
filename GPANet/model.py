# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 16:14:37 2021

@author: Administrator
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F



class Main(nn.Module):
	def __init__(self,channel = 8):
		super(Main,self).__init__()


		self.Low_E  = Encoder(channel)
              
		self.Share  = ShareNet(channel)
        
		self.Lap_D = Decoder1(channel)
		self.Lbp_D  = Decoder1(channel)
        
		self.Fusion_D = Decoder(channel)


		self.Low_in = nn.Conv2d(5,channel,kernel_size=1,stride=1,padding=0,bias=False)        
        
		self.Fusion_out = nn.Conv2d(channel,3,kernel_size=1,stride=1,padding=0,bias=False)          

        
		self.Lap_out = nn.Conv2d(channel,1,kernel_size=1,stride=1,padding=0,bias=False)  
		self.Lbp_out = nn.Conv2d(channel,1,kernel_size=1,stride=1,padding=0,bias=False)  
        
	def forward(self,xin,xs,xl):
        
        
		x = torch.cat((xin,xs,xl),1)
        
    
		x_l_in   = self.Low_in(x)             
		L,M,S,SS = self.Low_E(x_l_in)
    
		Lap,Lbp,Share = self.Share(SS)
    
		lap0,lap1,lap_out0,lap_out = self.Lap_D(Lap,SS,S,M,L)
		lbp0,lbp1,lbp_out0,lbp_out = self.Lbp_D(Lbp,SS,S,M,L)
		_,_,_,x_out = self.Fusion_D(Share,SS+lbp0+lap0,S+lbp1+lap1,M+lap_out0+lbp_out0,L+lap_out+lbp_out)


		lap_o = self.Lap_out(lap_out) #+ xs
		lbp_o = self.Lbp_out(lbp_out) #+ xl

		x_o   = self.Fusion_out(x_out) #+ xin
            
		return lap_o, lbp_o, x_o

class Encoder(nn.Module):
	def __init__(self,channel):
		super(Encoder,self).__init__()    

		self.el = ResidualBlock(channel)
		self.em = ResidualBlock(channel*2)
		self.es = ResidualBlock(channel*4)
		self.ess = ResidualBlock(channel*8)
        
		self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
		self.conv_eltem = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_emtes = nn.Conv2d(2*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_estess = nn.Conv2d(4*channel,8*channel,kernel_size=1,stride=1,padding=0,bias=False)  
        
	def forward(self,x):
        
		elout = self.el(x)
		x_emin = self.conv_eltem(self.maxpool(elout))
		emout = self.em(x_emin)
		x_esin = self.conv_emtes(self.maxpool(emout))        
		esout = self.es(x_esin)
		x_essin = self.conv_estess(self.maxpool(esout))        
		essout = self.ess(x_essin)

        
		return elout,emout,esout,essout

class ShareNet(nn.Module):
	def __init__(self,channel):
		super(ShareNet,self).__init__()    

		self.s1 = MTRB(channel*8)

        
	def forward(self,x):

		s1_1,s1_2,s1_3 = self.s1(x)

		return s1_1,s1_2,s1_3



class Decoder(nn.Module):
	def __init__(self,channel):
		super(Decoder,self).__init__()    


		self.dss = ResidualBlock(channel*8)
		self.ds = ResidualBlock(channel*4)
		self.dm = ResidualBlock(channel*2)
		self.dl = ResidualBlock(channel)

          
		self.conv_dsstds = nn.Conv2d(8*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)           
		self.conv_dstdm = nn.Conv2d(4*channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_dmtdl = nn.Conv2d(2*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)
        
	def _upsample(self,x,y):
		_,_,H,W = y.size()
		return F.upsample(x,size=(H,W),mode='bilinear')
    
	def forward(self,x,ss,s,m,l):

		dssout = self.dss(x+ss)
		x_dsin = self.conv_dsstds(self._upsample(dssout,s))        
		dsout = self.ds(x_dsin+s)
		x_dmin = self.conv_dstdm(self._upsample(dsout,m))
		dmout = self.dm(x_dmin+m)
		x_dlin = self.conv_dmtdl(self._upsample(dmout,l))
		dlout = self.dl(x_dlin+l)
        
		return dssout, dsout, dmout, dlout

class Decoder1(nn.Module):
	def __init__(self,channel):
		super(Decoder1,self).__init__()    


		self.dss = ResidualBlock(channel*8)
		self.ds = ResidualBlock(channel*4)
		self.dm = ResidualBlock(channel*2)
		self.dl = ResidualBlock(channel)
        
		self.out1 = Convolutional(channel*8,channel*8)
		self.out2 = Convolutional(channel*4,channel*4)
		self.out3 = Convolutional(channel*2,channel*2)
		self.out4 = Convolutional(channel,channel)
          
		self.conv_dsstds = nn.Conv2d(8*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)           
		self.conv_dstdm = nn.Conv2d(4*channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_dmtdl = nn.Conv2d(2*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)
        
	def _upsample(self,x,y):
		_,_,H,W = y.size()
		return F.upsample(x,size=(H,W),mode='bilinear')
    
	def forward(self,x,ss,s,m,l):

		dssout = self.dss(x+ss)
		x_dsin = self.conv_dsstds(self._upsample(dssout,s))        
		dsout = self.ds(x_dsin+s)
		x_dmin = self.conv_dstdm(self._upsample(dsout,m))
		dmout = self.dm(x_dmin+m)
		x_dlin = self.conv_dmtdl(self._upsample(dmout,l))
		dlout = self.dl(x_dlin+l)
        
		return self.out1(dssout), self.out2(dsout), self.out3(dmout), self.out4(dlout)


    
class MTRB(nn.Module):# Edge-oriented Residual Convolution Block
	def __init__(self,channel,norm=False):                                
		super(MTRB,self).__init__()

		self.conv_1_1 = nn.Conv2d(channel,  channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.conv_2_1 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_2_2 = nn.Conv2d(channel,  channel,kernel_size=3,stride=1,padding=1,bias=False)

        
		self.conv_3_1 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		#self.conv_3_2_1 = ResidualBlock(channel*4)   
		self.conv_3_2 = nn.Conv2d(channel*5,channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.conv_3_3 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)

 
		self.conv_4_1 = nn.Conv2d(channel*2,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_4_2 = nn.Conv2d(channel*2,channel,kernel_size=3,stride=1,padding=1,bias=False)


		self.conv_5_1 = nn.Conv2d(channel*3,channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.conv_add_out = nn.Conv2d(channel*1,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_cat_out = nn.Conv2d(channel*4,channel,kernel_size=3,stride=1,padding=1,bias=False)
     
		self.act = nn.PReLU(channel)

		self.norm = nn.GroupNorm(num_channels=channel,num_groups=1)# nn.InstanceNorm2d(channel)#

	def forward(self,x):
        
		x_1_1 = self.act(self.norm(self.conv_1_1(x)))

		x_2_1 = self.act(self.norm(self.conv_2_1(x_1_1)))
		x_2_2 = self.act(self.norm(self.conv_2_2(x_1_1)))

        
		x_3_1 = self.act(self.norm(self.conv_3_1(x_2_1)))
		x_3_3 = self.act(self.norm(self.conv_3_3(x_2_2)))
		x_3_2 = self.act(self.norm(self.conv_3_2(torch.cat((x_1_1, x_2_1 , x_2_2, x_3_1, x_3_3),1))))

        
		x_4_1 = self.act(self.norm(self.conv_4_1(torch.cat((x_3_1 , x_3_2),1))))
		x_4_2 = self.act(self.norm(self.conv_4_2(torch.cat((x_3_2 , x_3_3),1))))

        
		x_5_1 = self.act(self.norm(self.conv_5_1(torch.cat((x_3_2,x_4_1 , x_4_2),1))))
        
		x_add = x_3_1 + x_3_3 + x_5_1
		x_add_out = self.act(self.norm(self.conv_add_out(x_add)))

		x_cat = torch.cat((x_5_1,x_add_out,x_3_1, x_3_3),1)
		x_cat_out = self.act(self.norm(self.conv_cat_out(x_cat)))        
 
		return x_3_1, x_3_3, x_cat_out

class ResidualBlock(nn.Module):# Edge-oriented Residual Convolution Block
	def __init__(self,channel,norm=False):                                
		super(ResidualBlock,self).__init__()

		self.conv_1_1 = nn.Conv2d(channel,  channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_2_1 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.conv_out = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.act = nn.PReLU(channel)

		self.norm =nn.GroupNorm(num_channels=channel,num_groups=1)
   


	def forward(self,x):
        
		x_1 = self.act(self.norm(self.conv_1_1(x)))
		x_2 = self.act(self.norm(self.conv_2_1(x_1))+x)
		#x_out = self.act(self.norm(self.conv_out(x_2)) + x)


		return	x_2        
    
class Convolutional(nn.Module):# Edge-oriented Residual Convolution Block
	def __init__(self,inchannel,outchannel,norm=False):                                
		super(Convolutional,self).__init__()

		self.conv = nn.Conv2d(inchannel, outchannel,kernel_size=3,stride=1,padding=1,bias=False)
		self.act = nn.PReLU(outchannel)
		self.norm =nn.GroupNorm(num_channels=outchannel,num_groups=1)
   

	def forward(self,x):

		x_out = self.act(self.norm(self.conv(x)))

		return	x_out     
        