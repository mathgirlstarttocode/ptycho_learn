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

#from Operators import Alternating_projections_c
from Operators import mse_calc

from Solvers import Alternating_projections_c





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
    #Gramiam = Gramiam_plan(translations_x,translations_y,nframes,nx,ny,Nx,Ny)
    bw=3
    Gplan = Gramiam_plan(translations_x,translations_y,nframes,nx,ny,Nx,Ny,bw)
    # Gramiam = lambda framesl,framesr: Gramiam_calc(framesl,framesr,plan)
    
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
    omega=synchronize_frames_c(frames_rand, illumination, inormalization_split, Gplan)
    
    #synchronize frames
    frames_sync=frames_rand*omega

    img2=Overlap(Illuminate_frames(frames_sync,np.conj(illumination)))/normalization
    
    # simulate data
    
    frames_data = np.abs(np.fft.fft2(frames))**2 #squared magnitude from the truth
    # initial guess of all ones
    img_initial = np.ones(np.shape(img))
    
    #img3 calculated using AP without phase sync
    Alternating_projections=lambda opt,img_initial,maxiter: Alternating_projections_c(opt,img_initial,Gplan,frames_data, illumination, normalization, Overlap, Split, maxiter, img_truth=truth)
    img3,frames, residuals_nosync = Alternating_projections(False,img_initial,maxiter=100)
    
    #img4 calculated using AP with phase sync
    img4,frames, residuals_wsync = Alternating_projections(True,img_initial,maxiter=100)


#calculate mse
nrm0=np.linalg.norm(truth)
#nmse0=np.linalg.norm(truth-img)/nrm0
nmse0=mse_calc(truth,img)/nrm0
nmse1=mse_calc(truth,img1)/nrm0
nmse2=mse_calc(truth,img2)/nrm0
nmse3=mse_calc(truth,img3)/nrm0
nmse4=mse_calc(truth,img4)/nrm0



#phase2=np.dot(np.reshape(img2,(1,np.size(img1))),np.reshape(truth,(np.size(img1),1)))[0,0]

#print("nmse",nmse0,nmse1,nmse2)


fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True,figsize=(10,10))

axs[0,0].set_title('Truth',fontsize=10)
axs[0,0].imshow(abs(truth))

axs[0,1].set_title('True frames, nmse:%2.2g' % (nmse0),fontsize=10)
axs[0,1].imshow(abs(img))

axs[1,0].set_title('Random phase, No Sync:%2.2g' %( nmse1),fontsize=10)
axs[1,0].imshow(abs(img1))

axs[1,1].set_title('Random phase with Sync:%2.2g' %( nmse2),fontsize=10)
axs[1,1].imshow(abs(img2))

axs[0,2].set_title('Alternating Projections:%2.2g' %( nmse3),fontsize=10)
axs[0,2].imshow(abs(img3))

axs[1,2].set_title('Alternating Projections with Sync:%2.2g' %( nmse4),fontsize=10)
axs[1,2].imshow(abs(img4))

plt.show()

##
# make a new figure with residuals
fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True,figsize=(10,10))
# fig=plt.figure()
axs[0].semilogy(residuals_nosync[:,0])
axs[0].semilogy(residuals_wsync[:,0])
axs[0].legend(['nosync', 'sync'])
axs[0].set_title('img-truth')
axs[1].semilogy(residuals_nosync[:,1])
axs[1].semilogy(residuals_wsync[:,1])
axs[1].legend(['nosync', 'sync'])
#axs[1].title('frames-data')
axs[1].set_title('frames-data')

axs[2].semilogy(residuals_nosync[:,2])
axs[2].semilogy(residuals_wsync[:,2])
axs[2].legend(['nosync', 'sync'])
axs[2].set_title('frames overlapped')

