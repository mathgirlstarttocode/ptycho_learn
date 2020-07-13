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
from Operators import make_tiles, map_tiles, group_frames,overlap_tiles_c,split_tiles_c,flatten,Tiles_plan,Sync_tiles_c
from Operators import get_times, reset_times
reset_times()


#from Operators import Project_data

#from Operators import Alternating_projections_c
from Operators import mse_calc

from Solvers import Alternating_projections_c,Alternating_projections_tiles_c





# define x dimensions (frames, step, image)
nx=16 # frame size
Dx=5 # Step size
nnx=16 # number of frames in x direction
Nx = Dx*nnx
#Nx=256
bw1=3 # cropping border width 
bw2=0 #cropping border width for sync tile-wise
NTx=nnx//7 #number of tiles in x direction

# same thing for y
ny=nx 
nny=nnx
Dy=Dx
nframes=nnx*nny #total number of frames
Ny = Dy*nny
#Ny=256
NTy=NTx
Ntiles=NTx*NTy

maxiter=100
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
    
    Gplan = Gramiam_plan(translations_x,translations_y,nframes,nx,ny,Nx,Ny,bw1)
    # Gramiam = lambda framesl,framesr: Gramiam_calc(framesl,framesr,plan)
    
    # generate frames
    frames=Illuminate_frames(Split(truth),illumination) #check
    
    # generate normalization
    normalization=Overlap(Replicate_frame(np.abs(illumination)**2,nframes)) #check
    
    # recover the image 
    img=Overlap(Illuminate_frames(frames,np.conj(illumination)))/normalization
   
    #randomize framewise phases
    phases=np.exp(1j*np.random.random((nframes,1,1))*2*np.pi)
    frames_rand=frames*phases
    img1=Overlap(Illuminate_frames(frames_rand,np.conj(illumination)))/normalization
   
    #phase sync for rand frame
    inormalization_split = Split(1/normalization)
    frames_norm=np.linalg.norm(frames,axis=(1,2))

    import scipy as sp
    Gplan['Preconditioner']=sp.sparse.diags(1/frames_norm)
    omega=synchronize_frames_c(frames_rand, illumination, inormalization_split, Gplan)
    frames_sync=frames_rand*omega
    img2=Overlap(Illuminate_frames(frames_sync,np.conj(illumination)))/normalization
    
    
    #phase sync for rand frames by tiles
    
    #make tiles
    shift_Tx, shift_Ty=make_tiles(max(translations_x.ravel())+1,max(translations_y.ravel())+1,NTx,NTy) #calculate divide point of image
    #coordinates of each tile
    translations_tx,translations_ty=np.meshgrid(shift_Tx[0:-1],shift_Ty[0:-1]) 
    
    #generate mapid for tiles
    tiles_idx=map_tiles(shift_Tx,shift_Ty,NTx,NTy,Nx,Ny,nx,ny,nnx,nny,Dx,Dy)
    
    #sort frames into tiles
    groupie=group_frames(translations_x,translations_y,shift_Tx,shift_Ty)
    
    if groupie.any() != None:
        #seperate frames into groups, according to tiles
        grouped=[np.where(groupie==i) for i in range(Ntiles)]
        
        #initialization for tile sync
        frames_sync_tiles=[[] for i in range(Ntiles)]
        img_tiles=[[] for i in range(Ntiles)]
        sync_tiles=[[] for i in range(Ntiles)]
        tiles_size=np.zeros((Ntiles,2),dtype=int)
        #loop over all tiles
        for j in range(len(grouped)):

            #find the tile size, including the halo
            Nxi=shift_Tx[j%NTx+1]-shift_Tx[j%NTx]+nx
            Nyi=shift_Ty[j//NTx+1]-shift_Ty[j//NTx]+ny
            tiles_size[j,:]=[Nxi,Nyi]
            #Nxi=min(shift_Tx[j%NTx+1]-shift_Tx[j%NTx]+nx,Nx) #get the image size within each tile
            #Nyi=min(shift_Ty[j//NTx+1]-shift_Ty[j//NTx]+ny,Ny)
            
            #get the shift of tile
            dxi=shift_Tx[j%NTx]
            dyi=shift_Ty[j//NTx]
            
            #find all frames that are in the tile
            idxi=np.in1d(Gplan["col"], grouped[j]) 
            idyi=np.in1d(Gplan["row"],grouped[j]) 
            idxi=idxi & idyi
            nframesi=np.size(grouped[j])
            frames_rand_i=np.array([frames_rand[i] for i in grouped[j]])[0,:,:,:]
            
            #extract information from Gplan
            Gplani={'col':Gplan['col'][idxi],'row':Gplan['row'][idxi],'dd':Gplan['dd'][idxi], 'val':Gplan['val'][idxi],'bw':Gplan['bw']//NTx,'nx':Gplan['nx'],'ny':Gplan['ny']}
            
            #find the mapid that corresponds to the frames in the tile
            #mapidi=np.array([mapid[i] for i in grouped[j]])[0,:,:,:]
            translations_xi=np.array([translations_x[i] for i in grouped[j]])[0,:,:,:]
            translations_yi=np.array([translations_y[i] for i in grouped[j]])[0,:,:,:]
            
            #Overlap and Split for frames in the tile
            Overlapi=lambda frames: overlap_tiles_c(dxi,dyi,Nxi,Nyi,translations_xi,translations_yi,frames)
            Spliti = lambda image: split_tiles_c (dxi,dyi,nx,ny,translations_xi,translations_yi,image)
            
            
            #get normalization
            reg=1e-8 #may have zero values in normalization
            normalizationi=Overlapi(Replicate_frame(np.abs(illumination)**2,nframesi))
            inormalization_split_i=Spliti(1/(normalizationi+reg)) 
            
            #frames_norm_i=np.linalg.norm(framesi,axis=(1,2))
            
            #sychronize tiles
            omega_i=synchronize_frames_c(frames_rand_i, illumination, inormalization_split_i, Gplani) #ok
            
            sync_tiles[j]=frames_rand_i*omega_i
            
            img_tiles[j]=Overlapi(Illuminate_frames(sync_tiles[j],np.conj(illumination)))/(normalizationi+reg)
            
            
            
        ####sync tiles,tiles may have different sizes
        
        #get the sizes of tiles
        tiles_sizes = np.array([tiles_idx[i].shape for i in range(Ntiles)])
        
        #Overlap and Split for tiles sync
        Overlap_tiles=lambda img_tiles:Overlapc(flatten(img_tiles),Nx,Ny, flatten(tiles_idx))           
        Split_tiles=lambda img:Splitc(img,tiles_idx)
        
        #calculate normalization, illumination=1
        average_normalization=Overlap_tiles(np.ones((flatten(img_tiles).shape)))
            
        #img without tile sync
        img6=Overlap_tiles(img_tiles)/average_normalization
        
        #tile phase sync
        inormalization_split_tiles = Split_tiles(1/average_normalization) #now list
           
        Gplan_tiles=Gramiam_plan(translations_tx.T,translations_ty.T,Ntiles,tiles_sizes[:,0].reshape(Ntiles,1),tiles_sizes[:,1].reshape(Ntiles,1),Nx,Ny,bw=0)
              
        omega_tiles=synchronize_frames_c(np.array(img_tiles), 1+0j, inormalization_split_tiles, Gplan_tiles)
        
        tiles_sync=[img_tiles[i]*omega_tiles[i] for i in range(Ntiles)]
        
        img7=Overlap_tiles(tiles_sync)/average_normalization
    
    ##AP 
    # simulate data
    
    frames_data = np.abs(np.fft.fft2(frames))**2 #squared magnitude from the truth

    # precompute the preconditioner
    #frames_norm=np.sqrt(np.sum(frames_data,axis=(1,2)))/np.sqrt(nx*ny) # norm (fft-rescaled)
    #Gplan['Preconditioner']=sp.sparse.diags(1/frames_norm)


    # initial guess of all ones
    img_initial = np.ones(np.shape(img))
    
   
    Alternating_projections=lambda opt,img_initial,maxiter: Alternating_projections_c(opt,img_initial,Gplan,frames_data, illumination, normalization, Overlap, Split, maxiter, img_truth=truth)
   
    #img3 calculated using AP without phase sync
    img3,frames, residuals_nosync,time_sync1 = Alternating_projections(False,img_initial,maxiter)
    reset_times()
    #img4 calculated using AP with phase sync
    img4,frames, residuals_wsync,time_sync2 = Alternating_projections(True,img_initial,maxiter)
    timers=get_times()
    #print('timer no tilewise',timers)
    reset_times()
    #img10 calculated using AP with phase sync tile-wise
    Tiles_plan=Tiles_plan(translations_x,translations_y,NTx,NTy,Nx,Ny,nx,ny,nnx,nny,Dx,Dy)
    Sync_tiles_plan=Sync_tiles_c(frames_data,frames,illumination,Tiles_plan,Gplan,translations_x,translations_y,NTx,NTy,nx,ny)
    Alternating_projections_tiles=lambda opt,img_initial,maxiter:Alternating_projections_tiles_c(opt,img_initial,frames_data, illumination,Sync_tiles_plan,Tiles_plan,Overlapc,Split,Splitc,flatten,Gramiam_plan,maxiter, Nx,Ny,img_truth = truth)
    img10,residuals_tiles_wsync,time_sync_total1=Alternating_projections_tiles(True,img_initial,maxiter)
    #print('timer w tilewise',timers)
    reset_times()
    img11,residuals_tiles_nosync,time_sync_total2=Alternating_projections_tiles(False,img_initial,maxiter)
    
#calculate mse
nrm0=np.linalg.norm(truth)
#nmse0=np.linalg.norm(truth-img)/nrm0
nmse0=mse_calc(truth,img)/nrm0
nmse1=mse_calc(truth,img1)/nrm0
nmse2=mse_calc(truth,img2)/nrm0
nmse3=mse_calc(truth,img3)/nrm0
nmse4=mse_calc(truth,img4)/nrm0
nmse6=mse_calc(truth,img6)/nrm0
nmse7=mse_calc(truth,img7)/nrm0
nmse10=mse_calc(truth,img10)/nrm0
nmse11=mse_calc(truth,img11)/nrm0

#phase2=np.dot(np.reshape(img2,(1,np.size(img1))),np.reshape(truth,(np.size(img1),1)))[0,0]

#print("nmse",nmse0,nmse1,nmse2)


fig, axs = plt.subplots(nrows=5, ncols=2, sharex=True,figsize=(10,10))

axs[0,0].set_title('Truth',fontsize=10)
axs[0,0].imshow(abs(truth))

axs[0,1].set_title('True frames, nmse:%2.2g' % (nmse0),fontsize=10)
axs[0,1].imshow(abs(img))

axs[1,0].set_title('Random phase, No Sync:%2.2g' %( nmse1),fontsize=10)
axs[1,0].imshow(abs(img1))

axs[1,1].set_title('Random phase with Sync:%2.2g' %( nmse2),fontsize=10)
axs[1,1].imshow(abs(img2))

axs[2,0].set_title('Random phase, Sync tile-wise:%2.2g' %( nmse6),fontsize=10)
axs[2,0].imshow(abs(img6))

axs[2,1].set_title('Random phase, Sync Sync tile-wise:%2.2g' %( nmse7),fontsize=10)
axs[2,1].imshow(abs(img7))

axs[3,0].set_title('Alternating Projections:%2.2g' %( nmse3),fontsize=10)
axs[3,0].imshow(abs(img3))

axs[3,1].set_title('Alternating Projections with Sync:%2.2g' %( nmse4),fontsize=10)
axs[3,1].imshow(abs(img4))

axs[4,0].set_title('Alternating Projections with Sync tile-wise:%2.2g' %( nmse10),fontsize=10)
axs[4,0].imshow(np.minimum(abs(img10),max(abs(truth).ravel())))#for scaling purpose. The recovered image at the slit are big

axs[4,1].set_title('Alternating Projections tile-wise:%2.2g' %( nmse11),fontsize=10)
axs[4,1].imshow(np.minimum(abs(img11),max(abs(truth).ravel())))

plt.show()

##
# make a new figure with residuals
fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True,figsize=(10,10))
# fig=plt.figure()
axs[0].semilogy(residuals_nosync[:,0])
axs[0].semilogy(residuals_wsync[:,0])
axs[0].semilogy(residuals_tiles_wsync[:,0])
axs[0].semilogy(residuals_tiles_nosync[:,0])
axs[0].legend(['nosync', 'sync','sync_tilewise','tile-wise'])
axs[0].set_title('img-truth')

axs[1].semilogy(residuals_nosync[:,1])
axs[1].semilogy(residuals_wsync[:,1])
axs[1].semilogy(residuals_tiles_wsync[:,1])
axs[1].semilogy(residuals_tiles_nosync[:,1])
axs[1].legend(['nosync', 'sync','sync_tilewise','tile-wise'])
axs[1].set_title('frames-data')

axs[2].semilogy(residuals_nosync[:,2])
axs[2].semilogy(residuals_wsync[:,2])
axs[2].semilogy(residuals_tiles_wsync[:,2])
axs[2].semilogy(residuals_tiles_nosync[:,2])
axs[2].legend(['nosync', 'sync','sync_tilewise','tile-wise'])
axs[2].set_title('frames overlapped')

