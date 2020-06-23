# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
from scipy.sparse.linalg import eigsh
import math
import matplotlib.pyplot as plt
from Operators import cropmat, make_probe, make_translations, map_frames, Overlapc,Illuminate_frames, frames_overlap,Replicate_frame, Splitc, Stack_frames,Gramiam



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

def main():
    #generate truth
    truth=cropmat(img0,[Nx,Ny])
    
    #generate benchmark
    illumination=make_probe(nx,ny)
    translations_x,translations_y=make_translations(Dx,Dy,nnx,nny,Nx,Ny)

    mapidx,mapidy,mapid=map_frames(translations_x,translations_y,nx,ny,Nx,Ny)  

    Overlap = lambda frames: Overlapc(frames,Nx,Ny,mapid)
    Split = lambda img: Splitc(img,mapid)

    frames=Illuminate_frames(Split(truth),illumination) #check
    normalization=Overlap(Replicate_frame(np.abs(illumination)**2,nframes)) #check
    
    img=Overlap(Illuminate_frames(frames,np.conj(illumination)))/normalization
   
    #randomize framewise phases
    phases=np.exp(1j*np.random.random((nframes,1))*2*math.pi)
    stv_rand=Stack_frames(frames,phases)
    img1=Overlap(Illuminate_frames(stv_rand,np.conj(illumination)))/normalization
   
    #Phase synchronization
    framesl=Illuminate_frames(frames,np.conj(illumination))
    framesr=np.divide(framesl,normalization[mapidy.astype(int),mapidx.astype(int)])
    
    col,row,dx,dy=frames_overlap(translations_x,translations_y,nframes,nx,ny,Nx,Ny)
    H=Gramiam(nframes,framesl,framesr,col,row,nx,ny,dx,dy)

    #preconditioner 
    frames_norm=np.linalg.norm(frames,axis=(1,2))
    D=sp.sparse.diags(1/frames_norm)
    H1=D @ H @ D
    H1=(H1+np.transpose(H1))/2

    #compute the largest eigenvalue of H
    v0=sp.ones((nframes,1))
    eigenvalues, eigenvectors = eigsh(H1, k=2,which='LM',v0=v0)
    #if dont specify starting point v0, converges to another eigenvector
    omega=eigenvectors[:,0]
    omega=omega/np.abs(omega)
    
    #synchronize frames
    stv_sync=Stack_frames(frames,omega)
    img2=Overlap(Illuminate_frames(stv_sync,np.conj(illumination)))/normalization
    
    return truth,img,img1,img2

#call main()
    
truth,img,img1,img2=main()

#plot

fig, axs = plt.subplots(nrows=4, sharex=True,figsize=(20,20))

axs[0].set_title('Truth')
axs[0].imshow(abs(truth))

axs[1].set_title('Recovered')
axs[1].imshow(abs(img))

axs[2].set_title('Random phase, No Sync')
axs[2].imshow(abs(img1))

axs[3].set_title('Random phase with Sync')
axs[3].imshow(abs(img2))

plt.show()

