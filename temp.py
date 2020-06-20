# -*- coding: utf-8 -*-
import numpy as np
#import scipy 
#import math
#import cv2
import matplotlib.pyplot as plt
from Operators import cropmat, make_probe, make_translations, map_frames, Overlapc,  Illuminate_frames, Replicate_frame, Splitc



# define x dimensions (frames, step, image)
nx=16 # frame size
Dx=5 # Step size
nnx=8 # number of frames in x direction
Nx = Dx*nnx

# same thing for y
ny=nx 
nny=nnx;
Dy=Dx;
nframes=nnx*nny; #total number of frames
Ny = Dy*nny

#############################3

#load image problem
img0 = np.sum(plt.imread('gold_balls.png'),2)+1j

truth=cropmat(img0,[Nx,Ny])
illumination=make_probe(nx,ny)
translations_x,translations_y=make_translations(Dx,Dy,nnx,nny,Nx,Ny)

mapidx,mapidy,mapid=map_frames(translations_x,translations_y,nx,ny,nnx,nny,Nx,Ny)  

Overlap = lambda frames: Overlapc(frames,Nx,Ny,mapid)
Split = lambda img: Splitc(img,mapid)


#frames=Illuminate_frames(Split(truth,mapidx,mapidy),illumination) #check
frames=Illuminate_frames(Split(truth),illumination) #check

normalization=Overlap(Replicate_frame(np.abs(illumination)**2,nframes)) #check

img=Overlap(Illuminate_frames(frames,np.conj(illumination)))/normalization
   
fig, axs = plt.subplots(1,2)
axs[0].imshow(np.abs(truth))
axs[0].set_title('truth')

axs[1].imshow(np.abs(truth))
axs[1].set_title('recovered')

