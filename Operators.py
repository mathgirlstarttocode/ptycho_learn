"""
Ptycho operators

"""
import numpy as np
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
    
    #xx,yy=np.meshgrid(np.arange(1,nx+1)-nx/2,np.transpose(np.arange(1,ny+1)-ny/2))
    #rr=np.sqrt(xx**2 + yy**2) #calculate distance
    r1= 0.025*nx*3 #define zone plate circles
    r2= 0.085*nx*3
    Fprobe=np.fft.fftshift((rr>=r1) & (rr<=r2))
    probe=np.fft.fftshift(np.fft.ifft2(Fprobe))
    probe=probe/max(abs(probe).flatten())
    return probe
    
    
def make_translations(Dx,Dy,nnx,nny,Nx,Ny):
    ix,iy=np.meshgrid(np.arange(0,Dx*nnx,Dx)+Nx/2-Dx*nnx/2,
                      np.arange(0,Dy*nny,Dy)+Ny/2-Dy*nny/2)
    xshift=math.floor(Dx/2)*np.mod(np.arange(1,np.size(ix,1)+1),2)
    ix=np.transpose(np.add(np.transpose(ix),xshift))
    
    return ix,iy
    

def map_frames(translations_x,translations_y,nx,ny,nnx,nny,Nx,Ny):
    translations_x=np.reshape(translations_x,(nnx*nny,1,1))
    translations_y=np.reshape(translations_y,(nnx*nny,1,1))

    xframeidx,yframeidx=np.meshgrid(np.arange(nx),np.arange(ny))
    xframeidx=np.reshape(xframeidx,(1,nx,ny))
    yframeidx=np.reshape(yframeidx,(1,nx,ny))
    
    spv_x=translations_x+xframeidx
    spv_y=translations_y+yframeidx
    
    #spv_x=np.add(xframeidx,np.reshape(np.transpose(translations_x),(np.size(translations_x),1,1))) 
    #spv_y=np.add(yframeidx,np.reshape(np.transpose(translations_y),(np.size(translations_y),1,1))) 
    mapidx=np.mod(spv_x,Nx).astype(int)
    mapidy=np.mod(spv_y,Ny).astype(int)
    #mapid=np.add(mapidx*Nx,mapidy)
    mapid=np.add(mapidx,mapidy*Nx)
    return mapidx,mapidy,mapid
    
    
def Split(img,col,row):
    Split=img[row,col]         
    return Split

def Splitc(img,mapid):
    return (img.ravel())[mapid]

def Overlapc(frames,Nx,Ny, mapid): #check
    accum = np.reshape(numpy_groupies.aggregate(mapid.ravel(),frames.ravel()),(Nx,Ny))
    #idx_list=np.squeeze(np.reshape(mapid,(1,np.size(mapid))).astype(int))
    #weig=np.squeeze(np.reshape(frames,(1,np.size(frames))))
    #accumr=np.bincount(idx_list,weights=weig.real)
    #accumi=np.bincount(idx_list,weights=weig.imag)
    #accum=np.reshape((accumr+1j* accumi), [Nx,Ny])
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
