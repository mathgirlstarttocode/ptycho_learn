"""
Ptycho operators

"""
import numpy as np
import scipy as sp
import math
import numpy_groupies

from timeit import default_timer as timer
timers={'Gramiam':0, 'Eigensolver':0}

def get_times():
    return timers

def crop_center(img, cropx, cropy):
    # crop an image
       y, x = img.shape
       startx = x // 2 - (cropx // 2)
       starty = y // 2 - (cropy // 2)
       return img[starty:starty + cropy, startx:startx + cropx]


def cropmat(img,size):
    # crop an image to a given size
    left0=math.floor((np.size(img,0)-size[0])/2)
    right0=(size[0]+math.floor((np.size(img,0)-size[0])/2))
    left1=math.floor((np.size(img,1)-size[1])/2)
    right1=(size[1]+math.floor((np.size(img,1)-size[1])/2))
    crop_img= img[left0:right0,left1:right1]
    return crop_img


def make_probe(nx,ny):
    # make an illumination (probe)
    xi=np.reshape(np.arange(1,nx+1)-nx/2,(nx,1))
    rr=np.sqrt(xi**2+(xi.T)**2)
    r1= 0.025*nx*3 #define zone plate circles
    r2= 0.085*nx*3
    Fprobe=np.fft.fftshift((rr>=r1) & (rr<=r2))
    probe=np.fft.fftshift(np.fft.ifft2(Fprobe))
    probe=probe/max(abs(probe).flatten())
    return probe
    
    
def make_translations(Dx,Dy,nnx,nny,Nx,Ny):
    # make scan position
    ix,iy=np.meshgrid(np.arange(0,Dx*nnx,Dx)+Nx/2-Dx*nnx/2+1,
                      np.arange(0,Dy*nny,Dy)+Ny/2-Dy*nny/2+1)
    xshift=math.floor(Dx/2)*np.mod(np.arange(1,np.size(ix,1)+1),2)
    ix=np.transpose(np.add(np.transpose(ix),xshift))
    
    ix=np.reshape(ix,(nnx*nny,1,1))
    iy=np.reshape(iy,(nnx*nny,1,1))
    
    #+1 to account for counting
    return ix,iy
    

def map_frames(translations_x,translations_y,nx,ny,Nx,Ny):
    # map frames to image indices 
    translations_x=np.reshape(np.transpose(translations_x),(np.size(translations_x),1,1))
    translations_y=np.reshape(np.transpose(translations_y),(np.size(translations_y),1,1))

    
    xframeidx,yframeidx=np.meshgrid(np.arange(nx),np.arange(ny))
    print('translations shapes:',np.shape(translations_x),'frameidx',np.shape(xframeidx))
    
    spv_x=np.add(xframeidx,translations_x) 
    spv_y=np.add(yframeidx,translations_y) 
    
    mapidx=np.mod(spv_x,Nx)
    mapidy=np.mod(spv_y,Ny)
    mapid=np.add(mapidx,mapidy*Nx) 
    mapid=mapid.astype(int)
    return mapid
    
    
#def Split(img,col,row):
#    Split=img[row,col]         
#    return Split

def Splitc(img,mapid):
    # Split an image into frames given mapping
    return (img.ravel())[mapid]

def Overlapc(frames,Nx,Ny, mapid): #check
    # overlap frames onto an image
    accum = np.reshape(numpy_groupies.aggregate(mapid.ravel(),frames.ravel()),(Nx,Ny))
    return accum

def Illuminate_frames(frames,Illumination):
    Illuminated=frames*np.reshape(Illumination,(1,np.shape(Illumination)[0],np.shape(Illumination)[1]))
    return Illuminated
  
def Replicate_frame(frame,nframes):
    # replicate a frame along the first dimension
    Replicated= np.repeat(frame[np.newaxis,:, :], nframes, axis=0)
    return Replicated

def Sum_frames(frames):
    Summed=np.add(frames,axis=0)
    return Summed

def Stack_frames(frames,omega):
    # multiply frames by a vector in the first dimension
    omega=omega.reshape([len(omega),1,1])
    #stv=np.multiply(frames,omega)
    stv=frames*omega
    return stv


def ket(ystackr,dx,dy,bw=0): 
    #extracts the portion of the left frame that overlaps
    #dxi=dx[ii,jj].astype(int)
    #dyi=dy[ii,jj].astype(int)
    nx,ny = ystackr.shape
    dxi=dx.astype(int)
    dyi=dy.astype(int)
    ket=ystackr[max([0,dyi])+bw:min([nx,nx+dyi])-bw,
                max([0,dxi])+bw:min([nx,nx+dxi])-bw]
    return ket

def bra(ystackl,dx,dy,bw=0):
    #calculates the portion of the right frame that overlaps
    bra=ket(ystackl,dx,dy,bw)
    return bra

def braket(ystackl,ystackr,dd,bw):
    #calculates inner products between the overlapping portion
#    dxi=dx[ii,jj]
#    dyi=dy[ii,jj]
    dxi=dd.real
    dyi=dd.imag
    
    #bracket=np.sum(np.multiply(bra(ystackl[jj],nx,ny,-dxi,-dyi),ket(ystackr[ii],nx,ny,dxi,dyi)))
    #bracket=np.vdot(bra(ystackl[jj],nx,ny,-dxi,-dyi),ket(ystackr[ii],nx,ny,dxi,dyi))
    bket=np.vdot(bra(ystackl,-dxi,-dyi,bw),ket(ystackr,dxi,dyi,bw))
    
    
    return bket


#def Gramiam_calc(framesl,framesr,col,row,dd, bw=0):
def Gramiam_calc0(framesl,framesr,plan):
    # computes all the inner products between overlaping frames
    col=plan['col']
    row=plan['row']
    dd=plan['dd']
    bw=plan['bw']

    nframes=framesl.shape[0]
    nnz=len(col)
    val=np.zeros((nnz,1),dtype='complex128')
        
    for ii in range(nnz):         
        val[ii]=braket(framesl[col[ii]],framesr[row[ii]],dd[ii],bw)

    H=sp.sparse.csr_matrix((val.ravel(), (col, row)), shape=(nframes,nframes))
    
    H=H+(sp.sparse.triu(H,1)).getH()
    
    return H

#from multiprocessing import Process
import multiprocessing as mp

def Gramiam_calc(framesl,framesr,plan):
    # computes all the inner products between overlaping frames
    col=plan['col']
    row=plan['row']
    dd=plan['dd']
    bw=plan['bw']
    val=plan['val']
    
    nframes=framesl.shape[0]
    nnz=len(col)
    #val=np.empty((nnz,1),dtype=framesl.dtype)
    #val = shared_array(shape=(nnz),dtype=np.complex128)
 
    def braket_i(ii):
        val[ii] = braket(framesl[col[ii]],framesr[row[ii]],dd[ii],bw)
    
    #val=np.array(list(map(proc1, range(nnz))))
    for ii in range(nnz):
        braket_i(ii)
    
    
    H=sp.sparse.csr_matrix((val.ravel(), (col, row)), shape=(nframes,nframes))
    
    H=H+(sp.sparse.triu(H,1)).getH()
    
    return H


def Gramiam_plan(translations_x,translations_y,nframes,nx,ny,Nx,Ny,bw =0):
    # embed all geometric parameters into the gramiam function
    #calculates the difference of the coordinates between all frames
    dx=translations_x.ravel(order='F').reshape(nframes,1)
    dy=translations_y.ravel(order='F').reshape(nframes,1)
    dx=np.subtract(dx,np.transpose(dx))
    dy=np.subtract(dy,np.transpose(dy))
    
    #calculates the wrapping effect for a period boundary
    dx=-(dx+Nx*((dx < (-Nx/2)).astype(float)-(dx > (Nx/2)).astype(float)))
    dy=-(dy+Ny*((dy < (-Ny/2)).astype(float)-(dy > (Ny/2)).astype(float)))    
 
    #find the frames idex that overlaps
    #col,row=np.where(np.tril(np.logical_and(abs(dy)< nx,abs(dx) < ny)).T)
    col,row=np.where(np.tril((abs(dy)< nx-2*bw)*(abs(dx) < ny-2*bw)).T)
    # complex displacement (x, 1j y)
    dd = dx[row,col]+1j*dy[row,col]

    #col,row,dd=frames_overlap(translations_x,translations_y,nframes,nx,ny,Nx,Ny, bw)
 
    nnz=col.size
#    val=np.empty((nnz,1),dtype=np.complex128)
    val = shared_array(shape=(nnz,1),dtype=np.complex128)
    
    plan={'col':col,'row':row,'dd':dd, 'val':val,'bw':bw}
    #Gramiam = lambda framesl,framesr: Gramiam_calc(framesl,framesr,plan)
    return  plan
    
    

#    lambda Gramiam1
#    H=Gramiam(nframes,framesl,framesr,col,row,nx,ny,dx,dy)


def Precondition(H,frames, bw = 0):
    fw,fh=frames.shape[1:]
    frames_norm=np.linalg.norm(frames[:,bw:fw-bw ,bw:fh-bw],axis=(1,2))
    D=sp.sparse.diags(1/frames_norm)
    H1=D @ H @ D
    H1=(H1+H1.getH())/2
    return H1, D

    
from scipy.sparse.linalg import eigsh
def Eigensolver(H):
    nframes=np.shape(H)[0]
    #print('nframes',nframes)
    v0=np.ones((nframes,1))
    eigenvalues, eigenvectors = eigsh(H, k=2,which='LM',v0=v0)
    #if dont specify starting point v0, converges to another eigenvector
    omega=eigenvectors[:,0]
    omega=omega/np.abs(omega)
    
    # subtract the average phase
    so=np.conj(np.sum(omega))
    so/=abs(so)    
    omega*=so
    ########
    
    omega=np.reshape(omega,(nframes,1,1))
    return omega


#def synchronize_frames_c(frames, illumination, normalization,translations_x,translations_y,nframes,nx,ny,Nx,Ny):
def synchronize_frames_c(frames, illumination, normalization,plan, bw=0):
    #col,row,dx,dy=frames_overlap(translations_x,translations_y,nframes,nx,ny,Nx,Ny)
    # Gramiam = Gramiam_plan(translations_x,translations_y,nframes,nx,ny,Nx,Ny)

        
    framesl=Illuminate_frames(frames,np.conj(illumination))
    framesr=framesl*normalization    
    H=Gramiam_calc(framesl,framesr,plan)

    #preconditioner 
    H1, D = Precondition(H,frames,bw)

    
    #compute the largest eigenvalue of H1    
    omega=Eigensolver(H1)
    return omega

#def synchronize_frames_plan(inormalization_split,Gramiam):
#    omega=lambda frames synchronize_frames_c(frames, illumination, inormalization_split, Gramiam)
#    Gramiam = lambda framesl,framesr: Gramiam_calc(framesl,framesr,nframes,col,row,nx,ny,dx,dy)
#    return Gramiam

def mse_calc(img0,img1):
    # calculate the MSE between two images after global phase correction
    nnz=np.size(img0)
    # compute the best phase
    phase=np.dot(np.reshape(np.conj(img1),(1,nnz)),np.reshape(img0,(nnz,1)))[0,0]
    phase=phase/np.abs(phase)
    # compute norm after correcting the phase
    mse=np.linalg.norm(img0-img1*phase)
    return mse

def Propagate(frames):
    # simple propagation
    return np.fft.fft2(frames)

def IPropagate(frames):
    # simple inverse propagation
    return np.fft.ifft2(frames)

eps = 1e-8
def Project_data(frames,frames_data):
    # apply Fourier magnitude projections
    frames = Propagate(frames)
    mse = np.linalg.norm(np.abs(frames)-np.sqrt(frames_data))

    frames *= np.sqrt((frames_data+eps)/(np.abs(frames)**2+eps))
    frames = IPropagate(frames)
    return frames, mse
    


def Alternating_projections_c(opt,img,Gramiam,frames_data, illumination, normalization, Overlap, Split, maxiter, img_truth = None):
    
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


import ctypes
from multiprocessing import sharedctypes
def shared_array(shape=(1,), dtype=np.float32):  
    np_type_to_ctype = {np.float32: ctypes.c_float,
                        np.float64: ctypes.c_double,
                        np.bool: ctypes.c_bool,
                        np.uint8: ctypes.c_ubyte,
                        np.uint64: ctypes.c_ulonglong,
                        np.complex128: ctypes.c_double,
                        np.complex64: ctypes.c_float}

    numel = np.int(np.prod(shape))
    iscomplex=(dtype == np.complex128 or dtype == np.complex64)
    #numel *= 
    arr_ctypes = sharedctypes.RawArray(np_type_to_ctype[dtype], numel*(1+iscomplex))
    np_arr = np.frombuffer(arr_ctypes, dtype=dtype, count=numel)
    np_arr.shape = shape

    return np_arr     
        