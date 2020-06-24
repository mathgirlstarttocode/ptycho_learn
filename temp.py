# -*- coding: utf-8 -*-
import numpy as np
#import scipy as sp
#import math
#from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from Operators import cropmat, make_probe, make_translations, map_frames
from Operators import Overlapc, Illuminate_frames,  Replicate_frame, Splitc #, frames_overlap, Stack_frames
#from Operators import Gramiam, Eigensolver, Precondition
from Operators import Gramiam_plan
from Operators import synchronize_frames_c

#from Operators import Project_data

from Operators import Alternating_projections
from Operators import mse_calc





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

#load image
from scipy.io import loadmat
img0 = loadmat('gold.mat')['img0']

############################

#def main():
if True:
    #generate truth
    truth=cropmat(img0,[Nx,Ny])
    
    #generate benchmark
    illumination=make_probe(nx,ny)
    translations_x,translations_y=make_translations(Dx,Dy,nnx,nny,Nx,Ny)
    
    #print('translations shape',np.shape(translations_x))
    # make plan for Overlap and Split    
    mapid=map_frames(translations_x,translations_y,nx,ny,Nx,Ny)  
    
    Overlap = lambda frames: Overlapc(frames,Nx,Ny,mapid)
    Split = lambda img: Splitc(img,mapid)
    Gramiam = Gramiam_plan(translations_x,translations_y,nframes,nx,ny,Nx,Ny)
    
    # generate normalization
    normalization=Overlap(Replicate_frame(np.abs(illumination)**2,nframes)) #check
    # generate frames
    frames=Illuminate_frames(Split(truth),illumination) #check

    # recover the image 
    img=Overlap(Illuminate_frames(frames,np.conj(illumination)))/normalization
   
    #randomize framewise phases
    phases=np.exp(1j*np.random.random((nframes,1,1))*2*np.pi)
    frames_rand=frames*phases
    img1=Overlap(Illuminate_frames(frames_rand,np.conj(illumination)))/normalization
   
    #Phase synchronization
    #frames=frames_rand
    #omega = synch_frames(frames, illumination, normalization)

    inormalization_split = Split(1/normalization)
    
    #omega=synchronize_frames_c(frames, illumination, inormalization_split,translations_x,translations_y,nframes,nx,ny,Nx,Ny)
    omega=synchronize_frames_c(frames_rand, illumination, inormalization_split, Gramiam)
    
    #synchronize frames
    frames_sync=frames_rand*omega

    img2=Overlap(Illuminate_frames(frames_sync,np.conj(illumination)))/normalization
    
    # simulate data
    
    frames_data = np.abs(np.fft.fft2(frames))**2
    # initial guess of all ones
    img3 = np.ones(np.shape(img))
    #img3 = truth #np.ones(np.shape(img))
    
    img3,frames = Alternating_projections(img3, frames_data, illumination, normalization, Overlap, Split, maxiter=100)
    
    # frames= Illuminate_frames(Split(img3), illumination)
        
    # maxiter=10
    # for ii in np.arange(maxiter):
    #     frames = Project_data(frames,frames_data)
    #     img3= Overlap(Illuminate_frames(frames,np.conj(illumination)))/normalization
    #     frames = Illuminate_frames(Split(img3),illumination)

    #for ii=np.arange(maxiter):
        
    
#    return truth,img,img1,img2


#call main()
# truth,img,img1,img2=main()


# def mse_calc(img0,img1):
#     # calculate the MSE after global phase correction
#     nnz=np.size(img0)
#     # compute the best phase
#     phase=np.dot(np.reshape(np.conj(img1),(1,nnz)),np.reshape(img0,(nnz,1)))[0,0]
#     phase=phase/np.abs(phase)
#     # compute mse
#     mse=np.linalg.norm(img0-img1*phase)
#     return mse


nrm0=np.linalg.norm(truth)
#nmse0=np.linalg.norm(truth-img)/nrm0
nmse0=mse_calc(truth,img)/nrm0
nmse1=mse_calc(truth,img1)/nrm0
nmse2=mse_calc(truth,img2)/nrm0
nmse3=mse_calc(truth,img3)/nrm0



#phase2=np.dot(np.reshape(img2,(1,np.size(img1))),np.reshape(truth,(np.size(img1),1)))[0,0]

#print("nmse",nmse0,nmse1,nmse2)


fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True,figsize=(10,10))

axs[0,0].set_title('Truth')
axs[0,0].imshow(abs(truth))

axs[0,1].set_title('True frames, nmse:%2.2g' % (nmse0))
axs[0,1].imshow(abs(img))

axs[1,0].set_title('Random phase, No Sync:%2.2g' %( nmse1))
axs[1,0].imshow(abs(img1))

axs[1,1].set_title('Random phase with Sync:%2.2g' %( nmse2))
axs[1,1].imshow(abs(img2))

axs[0,2].set_title('Alternating Projections:%2.2g' %( nmse3))
axs[0,2].imshow(abs(img3))

axs[1,2].set_title('empty:%2.2g' %( np.infty))
axs[1,2].imshow(abs(truth*0))

plt.show()

