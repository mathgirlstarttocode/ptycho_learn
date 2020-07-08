import numpy as np
from timeit import default_timer as timer
from Operators import Illuminate_frames, Project_data, synchronize_frames_c, mse_calc,group_frames


def Alternating_projections_c(opt, img,Gramiam,frames_data, illumination, normalization, Overlap, Split, maxiter,  img_truth = None):
    
    # we need the frames norm to normalize
    frames_norm = np.linalg.norm(np.sqrt(frames_data))
    # renormalize the norm for the ifft2 space
    frames_norm_r= frames_norm/np.sqrt(np.prod(frames_data.shape[-2:]))
    
    
    # get the frames from the inital image
    frames = Illuminate_frames(Split(img),illumination)
    inormalization_split = Split(1/normalization)
    time_sync = 0 

    
    residuals = np.zeros((maxiter,3))
    if type(img_truth) != type(None):
        nrm_truth = np.linalg.norm(img_truth)
        
    for ii in np.arange(maxiter):
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
            time_sync += timer()-time0

        ##################
        # overlap projection
        img= Overlap(Illuminate_frames(frames,np.conj(illumination)))/normalization
        
        frames = Illuminate_frames(Split(img),illumination)

        residuals[ii,2] = np.linalg.norm(frames-frames_old)/frames_norm_r
        

        if type(img_truth) != type(None):
            nmse0=mse_calc(img_truth,img)/nrm_truth
            residuals[ii,0] = nmse0
        
    print('time sync:',time_sync)
    return img, frames, residuals

def Alternating_projections_tiles_c():
    
    #sort frames into tiles
    groupie=group_frames(translations_x,translations_y,shift_Tx,shift_Ty)
    
    if groupie.any() != None:
        grouped=[np.where(groupie==i) for i in range(NTx*NTy)]
        frames_sync_tiles=[[],[],[],[]]
        img_tiles=[[],[],[],[]]
        for j in range(len(grouped)):
            #group frames and mapid, sync within each tile
            Nxi=min(shift_Tx[j%NTx+1]-shift_Tx[j%NTx]+nx,Nx) #get the image size within each tile
            Nyi=min(shift_Ty[j//NTx+1]-shift_Ty[j//NTx]+ny,Ny)
            idxi=np.in1d(Gplan["col"], grouped[j]) #check if the frames is in tiles
            idyi=np.in1d(Gplan["row"],grouped[j]) 
            idxi=idxi & idyi
            nframesi=np.size(grouped[j])
            
            #framesi=np.array([frames_rand[i] for i in grouped[j]])[0,:,:,:]
            mapidi=np.array([mapid[i] for i in grouped[j]])[0,:,:,:]
            
            Overlapi = lambda frames: Overlapc(frames,Nxi,Nyi,mapidi)
            Spliti = lambda img: Splitc(img,mapidi)
            
            normalizationi=Overlapi(Replicate_frame(np.abs(illumination)**2,nframesi))
            
            inormalization_split_i=Spliti(1/normalizationi) 
            
            #frames_norm_i=np.linalg.norm(framesi,axis=(1,2))
            
            frames_rand_i=np.array([frames_rand[i] for i in grouped[j]])[0,:,:,:]
            
            Gplani={'col':Gplan['col'][idxi],'row':Gplan['row'][idxi],'dd':Gplan['dd'][idxi], 'val':Gplan['val'][idxi],'bw':Gplan['bw']}
            
            omega_i=synchronize_frames_c(frames_rand_i, illumination, inormalization_split_i, Gplani) #ok
            
            frames_sync_tiles[j]=frames_rand_i*omega_i
            
            img_tiles[j]=Overlapi(Illuminate_frames(frames_sync_tiles[j],np.conj(illumination)))/normalizationi
            
            #------simply add the img_tiles gives good results/each img_tiles is a good recovery
            
        #sync tiles
        #Overlap_test=lambda frames:Overlapc(frames,Nx,Ny,tiles_idx) #not right
            
        #Split_test=lambda img:Splitc(img,tiles_idx)
            
        #average_illumination=np.ones((np.shape(img_tiles[j])))
            
        #average_normalization=Overlap_test(Replicate_frame(np.abs(average_illumination)**2,NTx*NTy))
            
        #img6=Overlap_test(np.array(img_tiles))/average_normalization
        
        #inormalization_split_test = Split_test(1/average_normalization)
            
        #frames_test=img_tiles
           
        #Gplan_tiles=Gramiam_plan(translations_tx,translations_ty,Ntiles,Nx,Ny,Nx,Ny,bw=0)
            
        #omega_tiles=synchronize_frames_c(np.array(frames_test), average_illumination, inormalization_split_test, Gplan_tiles)
        
        #tiles_sync=img_tiles*omega_tiles
        
        #img5=Overlap_test(Illuminate_frames(tiles_sync,np.conj(average_illumination)))/average_normalization