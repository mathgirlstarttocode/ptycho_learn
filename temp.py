# -*- coding: utf-8 -*-
import numpy as np
import scipy 
import math
import cv2
import matplotlib.pyplot as plt



nx=16 # frame size
Dx=5 # Step size
nnx=8 # number of frames in x direction
ny=nx # y dimensions are the same
nny=nnx;
Dy=Dx;
nframes=nnx*nny; #total number of frames

#load image problem
#img0=cv2.imread('gold_balls.png')
#img0=cv2.normalize(img0.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

from scipy.io import loadmat
img0 = loadmat('gold.mat')['img0']


def cropmat(img,size):
    left0=math.floor((np.size(img,0)-size[0])/2)
    right0=(size[0]+math.floor((np.size(img,0)-size[0])/2))
    left1=math.floor((np.size(img,1)-size[1])/2)
    right1=(size[1]+math.floor((np.size(img,1)-size[1])/2))
    crop_img= img[left0:right0,left1:right1]
    return crop_img

#ground truth
truth=cropmat(img0,[math.floor(nnx*Dx),math.floor(nny*Dy)])
Nx,Ny=np.size(truth,0),np.size(truth,1)

def make_probe(nx,ny):
    xx,yy=np.meshgrid(np.arange(1,nx+1)-nx/2,np.transpose(np.arange(1,ny+1)-ny/2))
    rr=np.sqrt(xx**2 + yy**2) #calculate distance
    r1= 0.025*nx*3 #define zone plate circles
    r2= 0.085*nx*3
    Fprobe=np.fft.fftshift((rr>=r1) & (rr<=r2))
    probe=np.fft.fftshift(np.fft.ifft2(Fprobe))
    probe=probe/max(abs(probe).flatten())
    return probe
    
illumination=make_probe(nx,ny)
    
def make_translations(Dx,Dy,nnx,nny,Nx,Ny):
    ix,iy=np.meshgrid(np.arange(0,Dx*nnx,Dx)+Nx/2-Dx*nnx/2,
                      np.arange(0,Dy*nny,Dy)+Ny/2-Dy*nny/2)
    xshift=math.floor(Dx/2)*np.mod(np.arange(1,np.size(ix,1)+1),2)
    ix=np.transpose(np.add(np.transpose(ix),xshift))
    ix=np.add(ix,1)
    iy=np.add(iy,1)
    return ix,iy
    
translations_x,translations_y=make_translations(Dx,Dy,nnx,nny,Nx,Ny)

def map_frames(translations_x,translations_y,nx,ny,Nx,Ny):
    xframeidx,yframeidx=np.meshgrid(np.arange(nx),np.arange(ny))
    spv_x=np.add(xframeidx,np.reshape(np.transpose(translations_x),(np.size(translations_x),1,1))) 
    spv_y=np.add(yframeidx,np.reshape(np.transpose(translations_y),(np.size(translations_y),1,1))) 
    mapidx=np.mod(spv_x,Nx)
    mapidy=np.mod(spv_y,Ny)
    mapid=np.add(mapidx*Nx,mapidy)
    return mapidx,mapidy,mapid
    
mapidx,mapidy,mapid=map_frames(translations_x,translations_y,nx,ny,Nx,Ny)  
    
def Split(img,mapidx,mapidy):
    row=mapidx.astype(int)
    col=mapidy.astype(int)
    Split=img[col,row]         
    return Split

# im1=Split(truth,mapidx,mapidy) check OK

def Overlap(frames): #check
    idx_list=np.squeeze(np.reshape(mapid,(1,np.size(mapid))).astype(int))
    weig=np.squeeze(np.reshape(frames,(1,np.size(frames))))
    accumr=np.bincount(idx_list,weights=weig.real)
    accumi=np.bincount(idx_list,weights=weig.imag)
    accum=np.reshape((accumr+1j* accumi), [Nx,Ny])
    return accum

def Illuminate_frames(frames,Illumination):
    Illuminated=np.multiply(frames,Illumination)
    return Illuminated

    
def Replicate_frame(frame,nframes):
    Replicated= np.repeat(frame[np.newaxis,:, :], nframes, axis=0)
    return Replicated

def Sum_frames(frames):
    Sumed=np.add(frames,axis=0)
    return Sumed

frames=Illuminate_frames(Split(truth,mapidx,mapidy),illumination) #check

normalization=np.transpose(Overlap(Replicate_frame(np.abs(illumination)**2,nframes))) #check

img=np.divide(np.transpose(Overlap(Illuminate_frames(frames,np.conj(illumination)))),normalization)
 #checked 
   