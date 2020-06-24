"""
Ptycho operators

"""
import numpy as np
import scipy as sp
import math
import numpy_groupies


def crop_center(img, cropx, cropy):
       y, x = img.shape
       startx = x // 2 - (cropx // 2)
       starty = y // 2 - (cropy // 2)
       return img[starty:starty + cropy, startx:startx + cropx]



def cropmat(img,size):
    left0=math.floor((np.size(img,0)-size[0])/2)
    right0=(size[0]+math.floor((np.size(img,0)-size[0])/2))
    left1=math.floor((np.size(img,1)-size[1])/2)
    right1=(size[1]+math.floor((np.size(img,1)-size[1])/2))
    crop_img= img[left0:right0,left1:right1]
    return crop_img


def make_probe(nx,ny):
    xi=np.reshape(np.arange(1,nx+1)-nx/2,(nx,1))
    rr=np.sqrt(xi**2+(xi.T)**2)
    r1= 0.025*nx*3 #define zone plate circles
    r2= 0.085*nx*3
    Fprobe=np.fft.fftshift((rr>=r1) & (rr<=r2))
    probe=np.fft.fftshift(np.fft.ifft2(Fprobe))
    probe=probe/max(abs(probe).flatten())
    return probe
    
    
def make_translations(Dx,Dy,nnx,nny,Nx,Ny):
    ix,iy=np.meshgrid(np.arange(0,Dx*nnx,Dx)+Nx/2-Dx*nnx/2+1,
                      np.arange(0,Dy*nny,Dy)+Ny/2-Dy*nny/2+1)
    xshift=math.floor(Dx/2)*np.mod(np.arange(1,np.size(ix,1)+1),2)
    ix=np.transpose(np.add(np.transpose(ix),xshift))
    #+1 to account for counting
    return ix,iy
    

def map_frames(translations_x,translations_y,nx,ny,Nx,Ny):

    translations_x=np.reshape(np.transpose(translations_x),(np.size(translations_x),1,1))
    translations_y=np.reshape(np.transpose(translations_y),(np.size(translations_y),1,1))
    
    xframeidx,yframeidx=np.meshgrid(np.arange(nx),np.arange(ny))
    
    spv_x=np.add(xframeidx,translations_x) 
    spv_y=np.add(yframeidx,translations_y) 
    
    mapidx=np.mod(spv_x,Nx)
    mapidy=np.mod(spv_y,Ny)
    mapid=np.add(mapidx,mapidy*Nx) 
    mapid=mapid.astype(int)
    return mapidx,mapidy,mapid
    
    
#def Split(img,col,row):
#    Split=img[row,col]         
#    return Split

def Splitc(img,mapid):
    return (img.ravel())[mapid]

def Overlapc(frames,Nx,Ny, mapid): #check
    accum = np.reshape(numpy_groupies.aggregate(mapid.ravel(),frames.ravel()),(Nx,Ny))
    return accum

def Illuminate_frames(frames,Illumination):
    Illuminated=frames*np.reshape(Illumination,(1,np.shape(Illumination)[0],np.shape(Illumination)[1]))
    return Illuminated
  
def Replicate_frame(frame,nframes):
    Replicated= np.repeat(frame[np.newaxis,:, :], nframes, axis=0)
    return Replicated

def Sum_frames(frames):
    Summed=np.add(frames,axis=0)
    return Summed

def Stack_frames(frames,omega):
    omega=omega.reshape([len(omega),1,1])
    stv=np.multiply(frames,omega)
    return stv

def frames_overlap(translations_x,translations_y,nframes,nx,ny,Nx,Ny):
    #calculates the difference of the coordinates between all frames
    dx=translations_x.ravel(order='F').reshape(nframes,1)
    dy=translations_y.ravel(order='F').reshape(nframes,1)
    dx=np.subtract(dx,np.transpose(dx))
    dy=np.subtract(dy,np.transpose(dy))
    
    #calculates the wrapping effect for a period boundary
    dx=-(dx+Nx*((dx < (-Nx/2)).astype(float)-(dx > (Nx/2)).astype(float)))
    dy=-(dy+Ny*((dy < (-Ny/2)).astype(float)-(dy > (Ny/2)).astype(float)))    
 
    #find the frames idex that overlaps
    col,row=np.where(np.tril(np.logical_and(abs(dy)< nx,abs(dx) < ny)).T)
    
    return col,row,dx,dy

def ket(ystackr,ii,jj,nx,ny,dx,dy): 
#extracts the portion of the left frame that overlaps
    dx=dx.astype(int)
    dy=dy.astype(int)
    ket=ystackr[ii,max([0,dy[ii,jj]]):min([nx,nx+dy[ii,jj]]),
                max([0,dx[ii,jj]]):min([nx,nx+dx[ii,jj]])]
    return ket

def bra(ystackl,ii,jj,nx,ny,dx,dy):
#calculates the portion of the right frame that overlaps
    bra=np.conj(ket(ystackl,jj,ii,nx,ny,dx,dy))
    return bra

def bracket(ystackl,ystackr,ii,jj,nx,ny,dx,dy):
#calculates inner products between the overlapping portion
    bracket=np.sum(np.multiply(bra(ystackl,ii,jj,nx,ny,dx,dy),ket(ystackr,ii,jj,nx,ny,dx,dy)))
    return bracket

def Gramiam(nframes,framesl,framesr,col,row,nx,ny,dx,dy):
 # computes all the inner products between overlaping frames
   
    nnz=len(col)
    val=np.zeros((nnz,1),dtype='complex128')
    
    for ii in range(nnz):
        val[ii]=bracket(framesl,framesr,row[ii],col[ii],nx,ny,dx,dy)

    H=sp.sparse.csr_matrix((val.ravel(), (col, row)), shape=(nframes,nframes))
    
    H=H+(sp.sparse.triu(H,1)).getH()
    
    return H
    
