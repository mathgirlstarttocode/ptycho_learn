nx=16; % frame size
Dx=5; % Step size
nnx=8; % number of frames in x direction

ny=nx; % y dimensions are the same
nny=nnx;
Dy=Dx;
nframes=nnx*nny;

%starting point for simulations...(ground_truth, i.e. solution)
% load an image and make it complex
img0=double(imread('gold_balls.png'))+1j;
% cropping function
cropmat=@(a,siz) a((1:siz(1))+floor((size(a,1)-siz(1))/2),(1:siz(2))+floor((size(a,2)-siz(2))/2));

truth=cropmat(img0,floor([nnx*Dx nny*Dy]));
% dimensions
[Nx,Ny]=size(truth);

 illumination = make_probe(nx,ny);
%illumination = ones(nx,ny);
[translations_x,translations_y] = make_translations(Dx,Dy,nnx,nny,Nx,Ny);


% generate mapidx to transform from image to stack of frames
mapidx= map_frames(translations_x,translations_y,nx,ny,Nx,Ny);

%operators
Split =@(img) img(mapidx);
Overlap=@(frames) reshape(accumarray(mapidx(:),frames(:),[Nx*Ny 1]),Nx,Ny);
Illuminate_frames=@(frames,illumination) bsxfun(@times,frames,illumination);
Replicate_frame=@(frame) frame(:,:,ones(nframes,1));
Sum_frames =@(frames) sum(frames,3);


% generate frames
frames = Illuminate_frames(Split(truth),illumination);

% compute normalization
normalization=Overlap(Replicate_frame(abs(illumination).^2));

% .... skipping phase retrieval.... 

% get the image
img=Overlap(Illuminate_frames(frames,conj(illumination)))./normalization;

% next, randomize framewise phases
phases=exp(1i*rand(nframes,1)*2*pi);
% stack of frames * 1d vector (broadcast multiply)
stv=@(xx,ixx)   bsxfun(@times,xx,reshape(ixx,1,1,numel(ixx)));
frames=stv(frames,phases);

% see if we can reconstruct 
img1=Overlap(Illuminate_frames(frames,conj(illumination)))./normalization;
%% next we phase-synchronize the frames
%%----------------------------------------

% set up the inner products
Gcalc=gramiam(translations_x,translations_y,nx,ny,Nx,Ny);

% inverse of normalization
inormalization=1./normalization;

% compute matrix
H=Gcalc(Illuminate_frames(frames,conj(illumination)),Illuminate_frames(frames,conj(illumination)).*inormalization(mapidx));

% preconditioner
frames_norm=squeeze(sum(sum(abs(frames).^2)));
v=sqrt(1./frames_norm);
D=spdiags(v,0,numel(v),numel(v));
H1=D*H*D;
% make sure it is hermitian
H1=(H1+H1')/2;

% get the largest eigenvalue
opts.v0=ones(nframes,1); % initial for eigs
opts.issym=1; % symmetric H
[omegaeigs2,~]=eigs(H1,2,'lm',opts); % largest 2 eigenvalues
omega=omegaeigs2(:,1); % top eigenvalue
omega=omega./abs(omega); % make it into a phase factor

% synchronize frames
frames = stv(frames,omega);
%%--------

% get the image
img2=Overlap(Illuminate_frames(frames,conj(illumination)))./normalization;
%

%% plot results
figure(1);
subplot(2,2,1)
imagesc(abs(truth));title('truth')
axis image
%figure(2)
subplot(2,2,2)
imagesc(abs(img)); title('reconstruction')
axis image
%figure(3)
subplot(2,2,3)
imagesc(abs(img1)); title('reconstruction with wrong phases')
axis image

% synchronize next...
subplot(2,2,4)
imagesc(abs(img2)); title('reconstruction with phases synchronized')
axis image




