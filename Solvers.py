import numpy as np
from timeit import default_timer as timer
from Operators import Illuminate_frames, Project_data, synchronize_frames_c, mse_calc
#from Operators import shared_array


def Alternating_projections_c(opt, img,Gramiam,frames_data, illumination, normalization, Overlap, Split, maxiter,  img_truth = None):
 
    # we need the frames norm to normalize
    frames_norm = np.linalg.norm(np.sqrt(frames_data))
    # renormalize the norm for the ifft2 space
    frames_norm_r= frames_norm/np.sqrt(np.prod(frames_data.shape[-2:]))
    
    
    # get the frames from the inital image
    frames = Illuminate_frames(Split(img),illumination)
    inormalization_split = Split(1/(normalization+1e-8))
    time_sync=np.zeros((maxiter,1),dtype=float)

    
    residuals = np.zeros((maxiter,3))
    if type(img_truth) != type(None):
        nrm_truth = np.linalg.norm(img_truth)
        
    for ii in np.arange(maxiter):
        print(ii)
        # data projection
        frames, mse_data = Project_data(frames,frames_data)
        residuals[ii,1] = mse_data/frames_norm
        
        frames_old =frames+0. # make a copy
        ####################
        # here goes the synchronization
        if opt==True:
            time0 = timer()
            omega=synchronize_frames_c(frames, illumination, inormalization_split, Gramiam)
            frames=frames*omega
            time_sync[ii] += timer()-time0

        ##################
        # overlap projection
        img= Overlap(Illuminate_frames(frames,np.conj(illumination)))/(normalization+1e-8)
        
        frames = Illuminate_frames(Split(img),illumination)

        residuals[ii,2] = np.linalg.norm(frames-frames_old)/frames_norm_r
        

        if type(img_truth) != type(None):
                nmse0=mse_calc(img_truth,img)/nrm_truth
                residuals[ii,0] = nmse0

    return img, frames, residuals,time_sync


def Alternating_projections_tiles_c(opt,img,frames_data, illumination,Sync_tiles_plan,Tiles_plan,Overlapc,Split,Splitc,flatten,Gramiam_plan,maxiter, Nx,Ny,img_truth = None): 
   
    Ntiles=Tiles_plan['Ntiles']
    tiles_idx=Tiles_plan['tiles_idx']
    grouped=Tiles_plan['grouped']
    
    #Overlap and Split for tiles sync
    Overlap_tiles=lambda img_tiles:Overlapc(flatten(img_tiles),Nx,Ny, flatten(tiles_idx))           
    Split_tiles=lambda img:Splitc(img,tiles_idx) 
    
    #initialization
    img_tiles=[[] for i in range(Ntiles)]
    total_residuals=np.zeros((Ntiles,maxiter,3),dtype=float)
    time_sync_total=np.zeros((maxiter,Ntiles),dtype=float)
   
    #sync within each tile
    for ii in range(Ntiles):
        #initialization
        normalizationi=Sync_tiles_plan[ii]['normalizationi']
        Gplani=Sync_tiles_plan[ii]['Gplani']
        Overlapi=Sync_tiles_plan[ii]['Overlapi']
        Spliti=Sync_tiles_plan[ii]['Spliti']
        #data for sync within tiles
        frames_data_i=np.array([frames_data[i] for i in grouped[ii]])[0,:,:,:]
        img_i=Split_tiles(img)[ii]
        truth_i=Split_tiles(img_truth)[ii]
        img_tiles[ii], frames_i, residuals,time_sync=Alternating_projections_c(opt,img_i,Gplani,frames_data_i, illumination, normalizationi, Overlapi, Spliti, maxiter, img_truth=truth_i)
        time_sync_total[:,ii]+=time_sync.ravel()
        total_residuals[ii]=residuals
    
    total_residuals=np.sqrt(np.sum(total_residuals**2,axis=0))
    ####sync tiles,tiles may have different sizes
    time0=timer()
    tiles_sizes = np.array([tiles_idx[i].shape for i in range(Ntiles)])   
    
    #calculate normalization, illumination=1
    average_normalization=Overlap_tiles(np.ones((np.sum(tiles_sizes[:,0]*tiles_sizes[:,1]),1)))
    inormalization_split_tiles = Split_tiles(1/average_normalization) #now list
    Gplan_tiles=Gramiam_plan(Tiles_plan['translations_tx'].T,Tiles_plan['translations_ty'].T,Ntiles,tiles_sizes[:,0].reshape(Ntiles,1),tiles_sizes[:,1].reshape(Ntiles,1),Nx,Ny,bw=0)                 
    omega_tiles=synchronize_frames_c(np.array(img_tiles), 1+0j, inormalization_split_tiles, Gplan_tiles)      
    tiles_sync=[img_tiles[i]*omega_tiles[i] for i in range(Ntiles)]
    img=Overlap_tiles(tiles_sync)/average_normalization
    
    time_sync_total=np.sum(time_sync_total,axis=1)
    time_sync_total+=timer()-time0
        
    #residuals[j,2] = np.linalg.norm(frames-frames_old)/frames_norm_r
        
    #if type(img_truth) != type(None):
           # nmse0=mse_calc(img_truth,img)/nrm_truth
            #residuals[j,0] = nmse0
            
        #print('time sync:',time_sync)
    return img, total_residuals,time_sync_total