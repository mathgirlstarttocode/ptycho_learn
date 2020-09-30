import numpy as np
from timeit import default_timer as timer
from Operators import Illuminate_frames, Project_data, synchronize_frames_c, mse_calc
#from Operators import shared_array


def Alternating_projections_c(opt,img,Gramiam,frames_data, illumination, normalization, Overlap, Split, maxiter,  img_truth = None):
    reg=1e-8
    # we need the frames norm to normalize
    frames_norm = np.linalg.norm(np.sqrt(frames_data))
    # renormalize the norm for the ifft2 space
    frames_norm_r= frames_norm/np.sqrt(np.prod(frames_data.shape[-2:]))
    
    
    # get the frames from the inital image
    frames = Illuminate_frames(Split(img),illumination)
    inormalization_split = Split(1/(normalization+reg))
    time_sync=np.zeros((maxiter,1),dtype=float)

    
    residuals = np.zeros((maxiter,3))
    if type(img_truth) != type(None):
        nrm_truth = np.linalg.norm(img_truth)
    
   
    for ii in np.arange(maxiter):
        print('ii',ii)
        # data projection
        frames, mse_data = Project_data(frames,frames_data)
        residuals[ii,1] = mse_data/frames_norm
        
        frames_old =frames+0. # make a copy
        ####################
        # here goes the synchronization
        if opt==True:
            time0 = timer()
            H,omega=synchronize_frames_c(frames, illumination, inormalization_split, Gramiam)
            frames=frames*omega
            time_sync[ii] += timer()-time0

        ##################
        # overlap projection
        img= Overlap(Illuminate_frames(frames,np.conj(illumination)))/(normalization+reg)
        
        frames = Illuminate_frames(Split(img),illumination)

        residuals[ii,2] = np.linalg.norm(frames-frames_old)/frames_norm_r
        

        if type(img_truth) != type(None):
                nmse0=mse_calc(img_truth,img)/nrm_truth
                residuals[ii,0] = nmse0
    
        if opt ==False:
            omega=None
            H=None
    return img, frames, residuals,time_sync,omega,H


def Alternating_projections_tiles_c(opt,img,frames_data, illumination,Sync_tiles_plan,Tiles_plan,normalization,Overlapc,Split,Splitc,flatten,Gramiam_plan,maxiter, Nx,Ny,img_truth = None): 
    reg=1e-8
    
    Ntiles=Tiles_plan['Ntiles']
    tiles_idx=Tiles_plan['tiles_idx']
    grouped=Tiles_plan['grouped']
    tiles_size=Tiles_plan['tiles_size']
    
    
    #Overlap and Split or tiles sync
    Overlap_tiles=lambda image_tiles:Overlapc(flatten(image_tiles),Nx,Ny, flatten(tiles_idx))           
    Split_tiles=lambda image:Splitc(image,tiles_idx) 
    
    #initializatin for sync between tiles
    #tiles_sizes = np.array([tiles_idx[i].shape for i in range(Ntiles)])   
    average_normalization=Overlap_tiles(np.ones((np.sum(tiles_size[:,0]*tiles_size[:,1]),1)))
    inormalization_split_tiles = Split_tiles(1/(average_normalization+reg)) #now list
    Gplan_tiles=Gramiam_plan(Tiles_plan['translations_tx'].T,Tiles_plan['translations_ty'].T,Ntiles,tiles_size[:,0].reshape(Ntiles,1),tiles_size[:,1].reshape(Ntiles,1),Nx,Ny,bw=0)                 

    #initialization
    img_tiles=[[] for i in range(Ntiles)]
    residuals = np.zeros((maxiter,3))
    time_sync=np.zeros((maxiter,1),dtype=float)
    
    # we need the frames norm to normalize
    frames_norm = np.linalg.norm(np.sqrt(frames_data))
    # renormalize the norm for the ifft2 space
    frames_norm_r= frames_norm/np.sqrt(np.prod(frames_data.shape[-2:]))
    
    
    # get the frames from the inital image
    frames = Illuminate_frames(Split(img),illumination)
    
    if type(img_truth) != type(None):
        nrm_truth = np.linalg.norm(img_truth)
    
   
    for ii in range(maxiter):
        print(ii)
        #frames, mse_data = Project_data(frames,frames_data)
        #residuals[ii,1] = mse_data/frames_norm
        frames_old =frames+0.
        
        #sync within each tile
        for jj in range(Ntiles):
            #Overlap and Split for within each tile
            Overlapi=Sync_tiles_plan[jj]['Overlapi']
            #Spliti = Sync_tiles_plan[jj]['Spliti']
            
            #data projection
            frames_i=np.array([frames[i] for i in grouped[jj]])[0,:,:,:]
            frames_data_i=Sync_tiles_plan[jj]['frames_datai']
                
            frames_i, mse_data_i = Project_data(frames_i,frames_data_i)
            residuals[ii,1] += mse_data_i
            
            #sync within each tile
            if opt==True:
                time0 = timer()
                omega_i=synchronize_frames_c(frames_i, illumination, Sync_tiles_plan[jj]['inormalization_split_i'], Sync_tiles_plan[jj]['Gplani'])[1] #ok
                frames_i=frames_i*omega_i
                time_sync[ii] += timer()-time0
                
            #img_tiles[jj]=Overlapi(Illuminate_frames(frames_i,np.conj(illumination)))/(Sync_tiles_plan[jj]['normalizationi']+reg)
            img_tiles[jj]=Overlapi(Illuminate_frames(frames_i,np.conj(illumination)))
        
        ####sync tiles,tiles may have different sizes
        if opt==True and Ntiles !=1:
            time0=timer()
            omega_tiles=synchronize_frames_c(np.array(img_tiles), 1+0j, inormalization_split_tiles, Gplan_tiles)[1]      
            img_tiles=[img_tiles[i]*omega_tiles[i] for i in range(Ntiles)]
            time_sync[ii] += timer()-time0
            
        #img_out=Overlap_tiles(img_tiles)/average_normalization
        img_out=Overlap_tiles(img_tiles)/normalization
        frames = Illuminate_frames(Split(img_out),illumination)
            
        residuals[ii,1] = residuals[ii,1]/frames_norm
        residuals[ii,2] = np.linalg.norm(frames-frames_old)/frames_norm_r
    
        if type(img_truth) != type(None):
            nmse0=mse_calc(img_truth,img_out)/nrm_truth
            residuals[ii,0] = nmse0
            
    #print('time sync:',time_sync)

    return img_out,img_tiles, residuals,time_sync
