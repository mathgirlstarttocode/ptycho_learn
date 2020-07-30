"""
Ptycho operators

"""
import numpy as np
import scipy as sp
import math
import numpy_groupies
import bisect

#import multiprocessing as mp

# timers
from timeit import default_timer as timer
timers={'Gramiam':0, 'Gramiam_completion':0, 'Precondition':0,
        'Eigensolver':0, 'Sync_setup':0, 'Overlap':0, 'Project_data':0,'Sync':0}

def get_times():
    return timers
def reset_times():
    for keys in timers: timers[keys]=0


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

def make_tiles(Nx,Ny,NTx,NTy,translations_x,translations_y):    
    #gives coord for tiles, evenly distrubited (NTx,NTy)tiles over image of size(Nx,Ny)
   # shift_Tx=np.floor(np.linspace(0,Nx,NTx+1)).astype(int)
    #shift_Ty=np.floor(np.linspace(0,Ny,NTy+1)).astype(int)
    #to make the upper left corner of the tiles coincide with a frame. 
    #therefore every pixel within the tile is covered
    ix=np.unique(translations_x).shape[0]
    iy=np.unique(translations_y).shape[0]
    shift_Tx=np.unique(translations_x)[::(ix//NTx)][0:NTx]
    #shift_Tx=np.append(shift_Tx,translations_x[-1]+1)
    shift_Tx=np.append(shift_Tx,Nx)
    shift_Ty=np.unique(translations_y)[::(iy//NTy)][0:NTy]
    shift_Ty=np.append(shift_Ty,Ny)
    #shift_Ty=np.append(shift_Ty,translations_y[-1]+1)
    return shift_Tx.astype(int), shift_Ty.astype(int)

def make_translations(Dx,Dy,nnx,nny,Nx,Ny):
    # make scan position
    #ix,iy=np.meshgrid(np.arange(0,Dx*nnx,Dx)+Nx/2-Dx*nnx/2+1,
    #                  np.arange(0,Dy*nny,Dy)+Ny/2-Dy*nny/2+1)
    ix,iy=np.meshgrid(np.arange(0,Dx*nnx,Dx)+Nx/2-Dx*nnx/2,
                       np.arange(0,Dy*nny,Dy)+Ny/2-Dy*nny/2)
    #xshift=math.floor(Dx/2)*np.mod(np.arange(1,np.size(ix,1)+1),2)
    #ix=np.transpose(np.add(np.transpose(ix),xshift))
    
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
    #mapid=np.add(mapidx*Nx,mapidy) 
    mapid=mapid.astype(int)
    return mapid


#def map_tiles(shift_Tx,shift_Ty,NTx,NTy,Nx,Ny,nx,ny,nnx,nny,Dx,Dy): 
def map_tiles(shift_Tx,shift_Ty,tiles_size,NTx,NTy,Nx,Ny,nx,ny,Dx,Dy):     
    #map tiles to image indices 
    #xframeidx,yframeidx=np.meshgrid(np.arange(max(Nx,max(shift_Tx)+nx))%Nx,np.arange(max(Ny,max(shift_Ty)+ny))%Ny)
    xframeidx,yframeidx=np.meshgrid(np.arange(Nx-Dx+nx)%Nx,np.arange(Ny-Dy+ny)%Ny)
    
    idxx=xframeidx+yframeidx*Nx
    tiles_idx = [[] for i in range(NTx*NTy)]
    
    for m in range(NTx):
        for n in range(NTy):
            #tiles_idx[m+n*NTx]= idxx[shift_Ty[n]: (shift_Ty[n+1]+nx),shift_Tx[m]: (shift_Tx[m+1]+ny)]
            tiles_idx[m+n*NTx]= idxx[shift_Ty[n]:shift_Ty[n]+tiles_size[m+n*NTx,0] ,shift_Tx[m]: shift_Tx[m]+tiles_size[m+n*NTx,1]]
    tiles_idx=np.array(tiles_idx)     
    # test=np.hstack((tiles_idx[i] for i in range(4)))                  
    return tiles_idx

def size_tiles(Ntiles,NTx,shift_Tx,shift_Ty,Dx,Dy,nx,ny):
    tiles_size=np.zeros((Ntiles,2),dtype=int)
        #loop over all tiles
    for j in range(Ntiles):

            #find the tile size, including the halo
            #Nxi=shift_Tx[j%NTx+1]-shift_Tx[j%NTx]+nx           
        Nxi=shift_Tx[j%NTx+1]-shift_Tx[j%NTx]+nx-Dx
            #Nyi=shift_Ty[j//NTx+1]-shift_Ty[j//NTx]+ny
        Nyi=shift_Ty[j//NTx+1]-shift_Ty[j//NTx]+ny-Dy
            
            #tiles_size[j,:]=[Nxi,Nyi]
        tiles_size[j,:]=[Nyi,Nxi]
    return tiles_size
#def Split(img,col,row):
#    Split=img[row,col]         
#    return Split

def Splitc(img,mapid):
    try:
        return (img.ravel())[mapid]
    except IndexError:
        splited=[[] for i in range(mapid.size)]
        for i in range(len(mapid)):
            splited[i]=(img.ravel())[mapid[i]]
        return splited

def split_tiles_c (dxi,dyi,nx,ny,translations_xi,translations_yi,image):

    framesi=np.zeros((translations_xi.shape[0],nx,ny),dtype='complex128')
    for i in range(translations_xi.shape[0]):
        framesi[i] = image[int(translations_yi[i]-dyi): int(translations_yi[i]-dyi + ny), \
                           int(translations_xi[i]-dxi): int(translations_xi[i]-dxi + nx)] 
      
    return framesi

    
    
def Overlapc(frames,Nx,Ny, mapid): #check
    # overlap frames onto an image
    time0=timer()
    accum = np.reshape(numpy_groupies.aggregate(mapid.ravel(),frames.ravel()),(Ny,Nx))
    timers['Overlap']+=timer()-time0
    return accum

def overlap_tiles_c (dxi,dyi,Nxi,Nyi,translations_xi,translations_yi,frames):
    imgi=np.zeros((Nyi,Nxi),dtype='complex128')
    
    for i in range(translations_xi.shape[0]):
        #print(i)
        imgi[int(translations_yi[i]-dyi): int(translations_yi[i] + frames.shape[-2]-dyi),\
             int(translations_xi[i]-dxi): int(translations_xi[i] + frames.shape[-1]-dxi)] \
             += frames[i % frames.shape[0]]
    
    return imgi

def flatten(objects):
    a = []
    for l in objects:
        a.extend(l.ravel())
    return np.array(a)    
    
def Illuminate_frames(frames,Illumination):
    try:
        Illuminated=frames*np.reshape(Illumination,(1,np.shape(Illumination)[0],np.shape(Illumination)[1]))
        return Illuminated
    except IndexError:
        Illuminated=frames
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

def group_frames(translations_x,translations_y,shift_Tx,shift_Ty):
    #find the interval for which the frames lie in
    
    find_x=lambda x: bisect.bisect_right(shift_Tx, x)
    find_y=lambda y: bisect.bisect_right(shift_Ty, y)
    
    grouped_x=np.array([find_x(i) for i in translations_x])
    grouped_y=np.array([find_y(i) for i in translations_y])
    grouped=(grouped_x-1)+(grouped_y-1)*(np.shape(shift_Tx)[0]-1) 
    return grouped

    
    
def ket(ystackr,dx,dy,x,y,nx,ny,bw=0):  
    #extracts the portion of the left frame that overlaps
    #dxi=dx[ii,jj].astype(int)
    #dyi=dy[ii,jj].astype(int)
    #nx,ny = ystackr.shape # to account for tiles of different sizes
    dxi=dx.astype(int)
    dyi=dy.astype(int)
    
    #plus the frame size, 16 is the frame size
    #ket=ystackr[max([0,dyi])+bw:min(min([Tx,Tx+dyi]),max([0,dyi])+8)-bw,
    #            max([0,dxi])+bw:min(min([Ty,Ty+dxi]),max([0,dxi])+8)-bw] 
    ket=ystackr[max([0,dyi])+bw:min(x,x+dyi)-bw,
                max([0,dxi])+bw:min(y,y+dxi)-bw] 
    #ket=ystackr[max([0,dyi])+bw:min([Tx,Tx+dyi])-bw,
    #            max([0,dxi])+bw:min([Ty,Ty+dxi])-bw] 
        

    return ket

def bra(ystackl,dx,dy,x,y,nx,ny,bw=0):
    #calculates the portion of the right frame that overlaps
    bra=ket(ystackl,dx,dy,x,y,nx,ny,bw)
    return bra

def braket(ystackl,ystackr,dd,nx,ny,bw=0):
    #calculates inner products between the overlapping portion
#    dxi=dx[ii,jj]
#    dyi=dy[ii,jj]
    dxi=dd.real
    dyi=dd.imag
    x1,y1=ystackl.shape
    x2,y2=ystackr.shape
    #consider tiles with different sizes
    if dxi>0:
        y=y2
    else:y=y1
    if dyi>0:
        x=x2
    else:x=x1
    #bracket=np.sum(np.multiply(bra(ystackl[jj],nx,ny,-dxi,-dyi),ket(ystackr[ii],nx,ny,dxi,dyi)))
    bket=np.vdot(bra(ystackl,-dxi,-dyi,x,y,nx,ny,bw),ket(ystackr,dxi,dyi,x,y,nx,ny,bw))
    
    return bket
    
    


#from multiprocessing import Process

def Gramiam_calc(framesl,framesr,plan):
    # computes all the inner products between overlaping frames
    col=plan['col']
    row=plan['row']
    dd=plan['dd']
    bw=plan['bw']
    val=plan['val']
    nx=plan['nx']
    ny=plan['ny']
    
    #col_map=lambda x:np.argwhere(np.unique(col)==x).flatten()#to account for indexing of frames when divide into tiles
    #row_map=lambda x:np.argwhere(np.unique(row)==x).flatten()
    col=np.array([np.argwhere(col[i]==np.unique(col)) for i in range(np.size(col))]).ravel()
    row=np.array([np.argwhere(row[i]==np.unique(row)) for i in range(np.size(row))]).ravel()
    
    nframes=framesl.shape[0]
    nnz=len(col)
    #val=np.empty((nnz,1),dtype=framesl.dtype)
    #val = shared_array(shape=(nnz),dtype=np.complex128)
 
    def braket_i(ii):
        #val[ii] = braket(framesl[col_map(col[ii])][0,:,:],framesr[row_map(row[ii])][0,:,:],dd[ii],bw)
        val[ii] = braket(framesl[col[ii]],framesr[row[ii]],dd[ii],nx,ny,bw)
    #def proc1(ii):
    #    return braket(framesl[col[ii]],framesr[row[ii]],dd[ii],bw)
    
    time0=timer()        
    for ii in range(nnz):
        braket_i(ii)
    timers['Gramiam']+=timer()-time0
    time0=timer()
    
    time0=timer()
    
    #H=sp.sparse.csr_matrix((val.ravel(), (col, row)), shape=(nframes,nframes))
    H=sp.sparse.coo_matrix((val.ravel(), (col,row)), shape=(nframes,nframes))    
    H=H+(sp.sparse.triu(H,1)).getH()
    H=H.tocsr()
    timers['Gramiam_completion']+=timer()-time0
    
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
    # why are col-row swapped?
    dd = dx[row,col]+1j*dy[row,col]

    #col,row,dd=frames_overlap(translations_x,translations_y,nframes,nx,ny,Nx,Ny, bw)
 
    nnz=col.size
#   val=np.empty((nnz,1),dtype=np.complex128)
    val = shared_array(shape=(nnz,1),dtype=np.complex128)
    
    plan={'col':col,'row':row,'dd':dd, 'val':val,'bw':bw,'nx':nx,'ny':ny}
    #Gramiam = lambda framesl,framesr: Gramiam_calc(framesl,framesr,plan)
    return  plan
    
    

#    lambda Gramiam1
#    H=Gramiam(nframes,framesl,framesr,col,row,nx,ny,dx,dy)


def Precondition(H,frames, bw = 0):
    time0=timer()
    try:
        fw,fh=frames.shape[1:]
        frames_norm=np.linalg.norm(frames[:,bw:fw-bw ,bw:fh-bw],axis=(1,2))
        D=sp.sparse.diags(1/frames_norm)
        H1=D @ H @ D
        timers['Precondition']+=timer()-time0  
        return H1,D
    
    except ValueError:
        frames_norm=np.ones((len(frames)))
        for i in range(len(frames)):
            fw,fh=frames[i].shape
            frames_norm[i]=np.linalg.norm(frames[i][bw:fw-bw,bw:fh-bw])
     
        D=sp.sparse.diags(1/frames_norm)
        H1=D @ H @ D
        #H1=(H1+H1.getH())/2
        timers['Precondition']+=timer()-time0       
            
        return H1, D

    
from scipy.sparse.linalg import eigsh
def Eigensolver(H):
    time0=timer()
    
    nframes=np.shape(H)[0]
    #print('nframes',nframes)
    v0=np.ones((nframes,1))
    eigenvalues, eigenvectors = eigsh(H, k=1,which='LM',v0=v0, tol=1e-9)
    #eigenvalues, eigenvectors = eigsh(H, k=2,which='LM',v0=v0)
    #if dont specify starting point v0, converges to another eigenvector
    omega=eigenvectors[:,0]
    timers['Eigensolver']+=timer()-time0

    omega=omega/np.abs(omega)
    
    # subtract the average phase
    so=np.conj(np.sum(omega))
    so/=abs(so)    
    omega*=so
    ########
    
    omega=np.reshape(omega,(nframes,1,1))
    return omega



def synchronize_frames_c(frames, illumination, normalization,Gplan):
    #col,row,dx,dy=frames_overlap(translations_x,translations_y,nframes,nx,ny,Nx,Ny)
    # Gramiam = Gramiam_plan(translations_x,translations_y,nframes,nx,ny,Nx,Ny)

    time0=timer()
    bw=Gplan['bw']
    framesl=Illuminate_frames(frames,np.conj(illumination))
    framesr=framesl*normalization    
    timers['Sync_setup']+=timer()-time0

    H=Gramiam_calc(framesl,framesr,Gplan)
    
    
    if 'Preconditioner' in Gplan: 
        time0=timer()
        #print('hello')
        D = Gplan['Preconditioner']
        H1 = D @ H @ D
        timers['Precondition']=timer()-time0
    else:
        H1, D = Precondition(H,frames,bw)

    #compute the largest eigenvalue of H1    
    omega=Eigensolver(H1)
    return omega

def Tiles_plan_c(shift_Tx,shift_Ty,translations_x,translations_y,NTx,NTy,Nx,Ny,nx,ny,nnx,nny,Dx,Dy):
    #number of tiles
    Ntiles=NTx*NTy
    
    #find the size of tiles
    tiles_size=np.zeros((Ntiles,2),dtype=int)
    #loop over all tiles
    for j in range(Ntiles):
        #find the tile size, including the halo
        #Nxi=shift_Tx[j%NTx+1]-shift_Tx[j%NTx]+nx           
        Nxi=shift_Tx[j%NTx+1]-shift_Tx[j%NTx]+nx-Dx
        #Nyi=shift_Ty[j//NTx+1]-shift_Ty[j//NTx]+ny
        Nyi=shift_Ty[j//NTx+1]-shift_Ty[j//NTx]+ny-Dy
            
        #tiles_size[j,:]=[Nxi,Nyi]
        tiles_size[j,:]=[Nyi,Nxi]
        
    #make tiles
    #shift_Tx, shift_Ty=make_tiles(max(translations_x.ravel())+1,max(translations_y.ravel())+1,NTx,NTy,translations_x,translations_y) #calculate divide point of image
    shift_Tx, shift_Ty=make_tiles(Nx,Ny,NTx,NTy,translations_x,translations_y)
    
    #coordinates of each tile
    translations_tx,translations_ty=np.meshgrid(shift_Tx[0:-1],shift_Ty[0:-1]) 
    
    #generate mapid for tiles
    #tiles_idx=map_tiles(shift_Tx,shift_Ty,NTx,NTy,Nx,Ny,nx,ny,nnx,nny,Dx,Dy)
    tiles_idx=map_tiles(shift_Tx,shift_Ty,tiles_size,NTx,NTy,Nx,Ny,nx,ny,Dx,Dy)
    
    #sort frames into tiles
    groupie=group_frames(translations_x,translations_y,shift_Tx,shift_Ty)
        
    #seperate frames into groups, according to tiles
    grouped=[np.where(groupie==i) for i in range(NTx*NTy)]
    
    Tiles_plan={'Ntiles':Ntiles,'shift_Tx':shift_Tx,'shift_Ty':shift_Ty,'translations_tx':translations_tx,\
                'translations_ty':translations_ty,'tiles_idx':tiles_idx,'groupie':groupie,\
                'grouped':grouped,'tiles_size':tiles_size}
    
    return Tiles_plan


#def Sync_tiles_c(frames_data,frames,illumination,Tiles_plan,Gplan,translations_x,translations_y,NTx,NTy,nx,ny):  
def Sync_tiles_c(frames_data,illumination,Tiles_plan,Gplan,translations_x,translations_y,NTx,NTy,nx,ny,Dx,Dy):     
    Ntiles=NTx*NTy
    
    shift_Tx=Tiles_plan['shift_Tx']
    shift_Ty=Tiles_plan['shift_Ty']
    grouped =Tiles_plan['grouped']
    
    #initialization 
    Sync_tiles_plan=[{} for i in range(Ntiles)]
        
    #loop over all tiles
    for j in range(len(grouped)):
        
        #find the tile size, including the halo
        #Nxi=shift_Tx[j%NTx+1]-shift_Tx[j%NTx]+nx
        Nxi=shift_Tx[j%NTx+1]-shift_Tx[j%NTx]+nx-Dx
        #Nyi=shift_Ty[j//NTx+1]-shift_Ty[j//NTx]+ny
        Nyi=shift_Ty[j//NTx+1]-shift_Ty[j//NTx]+ny-Dy
        
        #get the shift of tile
        dxi=shift_Tx[j%NTx]
        dyi=shift_Ty[j//NTx]
            
        #find all frames that are in the tile
        idxi=np.in1d(Gplan["col"], grouped[j]) 
        idyi=np.in1d(Gplan["row"],grouped[j]) 
        idxi=idxi & idyi
        
        nframesi=np.size(grouped[j])
        frames_datai=np.array([frames_data[i] for i in grouped[j]])[0,:,:,:]
        
        #extract information from Gplan
        Gplani={'col':Gplan['col'][idxi],'row':Gplan['row'][idxi],'dd':Gplan['dd'][idxi], 'val':Gplan['val'][idxi],'bw':Gplan['bw']//NTx,'nx':Gplan['nx'],'ny':Gplan['ny']}
        
        #find the mapid that corresponds to the frames in the tile
        #mapidi=np.array([mapid[i] for i in grouped[j]])[0,:,:,:]
        translations_xi=np.array([translations_x[i] for i in grouped[j]])[0,:,:,:]
        translations_yi=np.array([translations_y[i] for i in grouped[j]])[0,:,:,:]
    
        #Overlap and Split for frames in the tile
        Overlapi=lambda frames,dx=dxi,dy=dyi,Nx=Nxi,Ny=Nyi,translations_x=translations_xi,translations_y=translations_yi \
            : overlap_tiles_c(dx,dy,Nx,Ny,translations_x,translations_y,frames)
        Spliti = lambda image,dx=dxi,dy=dyi,translations_x=translations_xi,translations_y=translations_yi\
            : split_tiles_c (dx,dy,nx,ny,translations_x,translations_y,image)
        
        #get normalization
        reg=1e-8 #may have zero values in normalization
        normalizationi=Overlapi(Replicate_frame(np.abs(illumination)**2,nframesi))           
        inormalization_split_i=Spliti(1/(normalizationi+reg)) 
        
        Sync_tiles_plan[j]={'Gplani':Gplani,'frames_datai':frames_datai,\
                            'dxi':dxi,'dyi':dyi,'Nxi':Nxi,'Nyi':Nyi,'translations_xi':translations_xi,\
                            'translations_yi':translations_yi,'Overlapi':Overlapi,'Spliti':Spliti,\
                             'normalizationi':normalizationi,'inormalization_split_i':inormalization_split_i }
    
    return Sync_tiles_plan
    
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
    time0=timer()
    # apply Fourier magnitude projections
    frames = Propagate(frames)
    mse = np.linalg.norm(np.abs(frames)-np.sqrt(frames_data))

    frames *= np.sqrt((frames_data+eps)/(np.abs(frames)**2+eps))
    frames = IPropagate(frames)
    timers['Project_data']+=timer()-time0
    return frames, mse
    
##illum refinement#################################
def refine_illum(illum,frames, Overlap,Split):

    illum_epsilon = 1e-02
    
    illum=np.sum(frames*Split(Overlap(Illuminate_frames(np.conj(frames),np.conj(illum)))),axis=0)
    
    illum_normalization = 1/(np.sum(Split(Overlap(np.abs(frames)**2)), axis = 0)+illum_epsilon)

    illum=illum*illum_normalization
    
    illum=illum/np.linalg.norm(illum)
    
    return illum

def refine_frames(illum,frames,Overlap,Split):
    
    frames_epsilon=1e-02
    
    frames1=Illuminate_frames(Split(Overlap(Illuminate_frames(frames,np.conj(illum)))),illum)
    
    frames=frames1/(Split(Overlap(Replicate_frame(np.abs(illum)**2,frames.shape[0])))+frames_epsilon)
    
    frames=frames/(np.linalg.norm(frames,axis=0)+frames_epsilon)
    
    return frames
    
####################    

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
        
