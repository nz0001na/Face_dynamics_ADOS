function [Feature_Vector] = LPQ_TOP(Texture_Volume,weight_vector,decorr,winSizes)
% Funtion Feature_Vector=LPQ_TOP(Texture_Volume,weight_vector,decorr,winSizes)
% calculates the LPQ-TOP feature vector for a dynamic texture Texture_Volume.
%
% Inputs:
% Texture_Volume = M*N*T double, volume of gray scale frames for analyzation.
% weight_vector  = 1*3 double, weight vector for the three orthogonal planes.
% decorr         = 1*2 double, indicates whether decorrelation is used or not.
%                  Possible values are:
%                               0 -> no decorrelation, 
%                   [rho_s rho_t] -> decorrelation (default = [0.1 0.1])
% winSizes       = 1*3 double, size of the local window in each dimension.
%                  winSizes must be a row vector and the indices must be
%                  greater or equal to 3 (default winSizes=[5 5 5]).
%
% Output:
% Feature_Vector = 768*1 double concatenated LPQ-TOP histogram.
%
% Example usage (DynTex++ files):
% load('0001.mat');
% [LPQTOPdesc] = LPQ_TOP(subv,[1 1 1],[0.1 0.1],[5 5 5]);
% figure; bar(LPQTOPdesc);
%
% Version published in 2010 by Juhani Päivärinta, Esa Rahtu, and Janne Heikkilä
% Machine Vision Group, University of Oulu, Finland
%
% Based on the original LBP-TOP algorithm
% Credits to Guoying Zhao & Matti Pietikäinen
%

%% Defaul parameters
% Weight vector for the tree planes
if nargin<2 || isempty(weight_vector)
    weight_vector=[1 1 1];  % default weight vector [1 1 1]
end

% Decorrelation
if nargin<3 || isempty(decorr)   
    deco=1; % use decorrelation by default
    decorr = [0.1 0.1];    % default
end

% Output mode
if nargin<4 || isempty(winSizes)
    winSizes = [5 5 5];     % default window sizes [5 5 5]
end

%% Check input parameters
if size(Texture_Volume,4)~=1
    error('Only a 3-D volume can be used as input.');
end
if size(weight_vector,1)~=1 || size(weight_vector,2)~=3
    error('Weight vector must be a row vector of length 3.');
end
if size(winSizes,1)~=1 || size(winSizes,2)~=3
    error('Window size must be a row vector of length 3.');
end
if winSizes(1)<3 || winSizes(2)<3 || winSizes(3)<3 || ...
    rem(winSizes(1),2)~=1 || rem(winSizes(2),2)~=1 || rem(winSizes(3),2)~=1
        error('Window size indices must be odd numbers and greater than equal to 3.');
end
if decorr == 0
    deco = 0;
else
    deco = 1;
end
if deco ~= 0
    if size(decorr,1)~=1 || size(decorr,2)~=2
        error('decorr parameter must be set to 0->no decorrelation or [rho_s rho_t]. See help for details.');
    end
end

maxsize = size(Texture_Volume,3);
height = size(Texture_Volume,1);
width = size(Texture_Volume,2);

Feature_Matrix = [];

%% For XY Plane
if deco == 1
    [C,D,V]=lpqtopCov_([winSizes(1) winSizes(2)],decorr(1),decorr(2),1);
else
    V=0;
end
for i=1:maxsize       
    G = double(Texture_Volume(:,:,i)); 
    LPQ_XY = (lpqtop_(G,[winSizes(1) winSizes(2)],decorr,'h',1,V))';
    
    if(i==1)
        Hist_XY = zeros(size(LPQ_XY,1),1);
        Hist_XY = Hist_XY + LPQ_XY;
    else
        Hist_XY = Hist_XY + LPQ_XY;
    end
end

%% For XT Plane
if deco == 1
    [C,D,V]=lpqtopCov_([winSizes(1) winSizes(3)],decorr(1),decorr(2),2);
else
    V=0;
end
Temp_Volume_XT = permute(Texture_Volume,[1 3 2]);    
for index1 = 1:size(Temp_Volume_XT,3) 
     G = double(Temp_Volume_XT(:,:,index1));
     LPQ_XT = (lpqtop_(G,[winSizes(1) winSizes(3)],decorr,'h',2,V))';
     
     if(index1==1)
         Hist_XT = zeros(size(LPQ_XT,1),1);
         Hist_XT = Hist_XT + LPQ_XT;
     else
         Hist_XT = Hist_XT + LPQ_XT;
     end
end
    
%% For YT Plane  
if deco == 1
    [C,D,V]=lpqtopCov_([winSizes(2) winSizes(3)],decorr(1),decorr(2),3);
else
    V=0;
end
Temp_Volume_YT = permute(Texture_Volume,[2 3 1]);
for index2 = 1:size(Temp_Volume_YT,3)
     G = double(Temp_Volume_YT(:,:,index2));
     LPQ_YT = (lpqtop_(G,[winSizes(2) winSizes(3)],decorr,'h',3,V))';
     
     if(index2==1)
         Hist_YT = zeros(size(LPQ_YT,1),1);
         Hist_YT = Hist_YT + LPQ_YT;
     else
         Hist_YT = Hist_YT + LPQ_YT;
     end
end


%% Normalize the LPQ histogram of each plane to range [0,1]
Hist_XY = Hist_XY./sum(Hist_XY);
Hist_XT = Hist_XT./sum(Hist_XT);
Hist_YT = Hist_YT./sum(Hist_YT);

Feature_Vector = [weight_vector(1,1).*Hist_XY;weight_vector(1,2).*Hist_XT;weight_vector(1,3).*Hist_YT];




%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Additional functions %%
%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Function for computing LPQ descriptors
function LPQdesc = lpqtop_(img,winSizes,decorr,mode,planeIndx,V)
% Funtion LPQdesc=lpqtop_(img,winSizes,decorr,mode,planeIndx,V) computes the
% Local Phase Quantization (LPQ) descriptor for the input plane img.
% Descriptors are calculated using only valid pixels.
%
% Inputs: (All empty or undefined inputs will be set to default values)
% img       = N*N uint8 or double, format gray scale image to be analyzed.
% winSizes  = 1*2 double, size of the local window. winSizes indices be
%               odd numbers and greater or equal to 3 (default = [3 3]).
% decorr    = 1*2 double, indicates whether decorrelation is used or not.
%               Possible values are:
%                      0 -> no decorrelation, 
%          [rho_s rho_t] -> decorrelation parameters (default = [0.1 0.1])
% mode      = 1*n char, defines the desired output type. Possible choices are:
%         (default) 'nh' -> normalized histogram of LPQ codewords
%                   'h'  -> un-normalized histogram of LPQ codewords
% planeIndx = 1*1 double, index of the plane for which the descriptor is being calculated.
%           (default)  1 -> XY plane
%                      2 -> XT plane
%                      3 -> YT plane
% V         = 8*8 double, a matrix for whitening transform (compute with lpqtopCov).
%
%
% Based on the original LPQ (published by Janne Heikkilä, Esa Rahtu, and Ville Ojansivu)
% Machine Vision Group, University of Oulu, Finland
%

%% Defaul parameters
% Local window size
if nargin<2 || isempty(winSizes)
    winSizes=[3 3]; % default window size 3
end

% Decorrelation
if nargin<3 || isempty(decorr)   
    deco=1; % use decorrelation by default
    decorr = [0.1 0.1]; % default
end

% Output mode
if nargin<4 || isempty(mode)
    mode='nh'; % return normalized histogram as default
end

% plane index
if nargin<5 || isempty(planeIndx)
    planeIndx=1; % assume that we are using the xy plane
end

% Other
convmode='valid'; % Compute descriptor responses only on part that have full neigborhood.


%% Check inputs
if size(img,3)~=1
    error('Only gray scale image can be used as input');
end
if size(winSizes,1)~=1 || size(winSizes,2)~=2
    if winSizes(1)<3 || winSizes(2)<3 || rem(winSizes(1),2)~=1 || rem(winSizes(2),2)~=1
        error('Window size indices must be odd numbers and greater than equal to 3');
    end
end
if decorr == 0
    deco = 0;
else
    deco = 1;
end
if deco ~= 0
    if size(decorr,1)~=1 || size(decorr,2)~=2
        error('decorr parameter must be set to 0->no decorrelation or [rho_s rho_t]. See help for details.');
    end
end
if sum(strcmp(mode,{'nh','h','im'}))==0
    error('mode must be nh, h, or im. See help for details.');
end
if planeIndx<1 || planeIndx >3
    error('plane index planeIndx must be 1, 2 or 3. See help for details.');
end

STFTalpha1=1/winSizes(1);  % alpha in STFT approaches
STFTalpha2=1/winSizes(2);  % alpha in STFT approaches

%% Initialize
img=double(img); % Convert image to double
r1=(winSizes(1)-1)/2; % Get radius from window size
r2=(winSizes(2)-1)/2; % Get radius from window size
x1=-r1:r1; % Form spatial coordinates in window
x2=-r2:r2; % Form spatial coordinates in window

%% Form 1-D filters
% Basic STFT filters
w01=(x1*0+1);
w11=exp(complex(0,-2*pi*x1*STFTalpha1));
w21=conj(w11);
w02=(x2*0+1);
w12=exp(complex(0,-2*pi*x2*STFTalpha2));
w22=conj(w12);

%% Run filters to compute the frequency response in the four points. Store real and imaginary parts separately
% Run first filter
filterResp=conv2(conv2(img,w01.',convmode),w12,convmode);
% Initilize frequency domain matrix for four frequency coordinates (real and imaginary parts for each frequency).
freqResp=zeros(size(filterResp,1),size(filterResp,2),8); 
% Store filter outputs
freqResp(:,:,1)=real(filterResp);
freqResp(:,:,2)=imag(filterResp);
% Repeat the procedure for other frequencies
filterResp=conv2(conv2(img,w11.',convmode),w02,convmode);
freqResp(:,:,3)=real(filterResp);
freqResp(:,:,4)=imag(filterResp);
filterResp=conv2(conv2(img,w11.',convmode),w12,convmode);
freqResp(:,:,5)=real(filterResp);
freqResp(:,:,6)=imag(filterResp);
filterResp=conv2(conv2(img,w11.',convmode),w22,convmode);
freqResp(:,:,7)=real(filterResp);
freqResp(:,:,8)=imag(filterResp);

% Read the size of frequency matrix
[freqRow,freqCol,freqNum]=size(freqResp);

% Reshape frequency response
freqResp=reshape(freqResp,[freqRow*freqCol,freqNum]);

if deco==1
    % Perform whitening transform
    freqResp=freqResp*V;
end

%% Perform quantization and compute LPQ codewords
ffreqResp=freqResp>=0;

LPQdesc = (ffreqResp(:,1))+...
               (ffreqResp(:,2))*2+...
               (ffreqResp(:,3))*4+...
               (ffreqResp(:,4))*8+...
               (ffreqResp(:,5))*16+...
               (ffreqResp(:,6))*32+...
               (ffreqResp(:,7))*64+...
               (ffreqResp(:,8))*128;


%% Histogram if needed
if strcmp(mode,'nh') || strcmp(mode,'h')
    LPQdesc=hist(LPQdesc(:),0:255);
end

%% Normalize histogram if needed
if strcmp(mode,'nh')
    LPQdesc=LPQdesc/sum(LPQdesc);
end






%% Function for computing the model based covariance estimate
function [C,D,V]=lpqtopCov_(winSizes,rho_s,rho_t,planeIndx)
% function [C,D,V]=lpqtopCov(winSizes,rho_s,rho_t,planeIndx) calculates
% a whitening transformation matrix V.
%
% Inputs: (All empty or undefined inputs will be set to default values)
% winSizes  = 1*2 double, size of the local window in each dimension.
%           winSizes must be a row vector and the indices must be
%           greater or equal to 3 (default winSizes=[3 3]).
% rho_s     = 1*1 double, rho in the spatial domain (default rho_s=0.1).
% tho_t     = 1*1 double, rho in the temporal domain (default rho_s=0.1).
% planeIndx = 1*1 double, index of the current plane (1, 2, or 3)
%
% Output:
% C       = (winSizes(1)*winSizes(2))*(winSizes(1)*winSizes(2)) double,
%           a model based covariance matrix.
% D       = 8*8 double, final covariance matrix (D=M*C*M.').
% V       = 8*8 double, a matrix for whitening transform.
%
% Example usage:
% [C,D,V]=lpqtopCov([5 5],0.1,0.1,1);
%

%% Defaul parameters
% Local window size
if nargin<1 || isempty(winSizes)
    winSizes=[3 3]; % default window size 3
end

% Spatial rho
if nargin<2 || isempty(rho_s)
    rho_s = 0.1; % default spatial rho 0.1
end
% Temporal rho
if nargin<3 || isempty(rho_t)
    rho_s = 0.1; % default temporal rho 0.1
end

% Plane index
if nargin<4 || isempty(planeIndx)
    planeIndx=1; % default plane index 1 (xy plane)
end


%% Check inputs
if size(winSizes,1)~=1 || size(winSizes,2)~=2
    error('Window size must be a row vector of length 2 (in lpqtopCov).');
end
if winSizes(1)<3 || winSizes(2)<3 || rem(winSizes(1),2)~=1 || rem(winSizes(2),2)~=1
    error('Window size indices must be odd numbers and greater than equal to 3');
end

if planeIndx<1 || planeIndx >3
    error('plane index planeIndx must be 1, 2 or 3. See help for details.');
end

STFTalpha1=1/winSizes(1);  % alpha in STFT approaches
STFTalpha2=1/winSizes(2);  % alpha in STFT approaches

%% Initialize
r1=(winSizes(1)-1)/2; % Get radius from window size
r2=(winSizes(2)-1)/2; % Get radius from window size
x1=-r1:r1; % Form spatial coordinates in window
x2=-r2:r2; % Form spatial coordinates in window

%% Form 1-D filters
% Basic STFT filters
w01=(x1*0+1);
w11=exp(complex(0,-2*pi*x1*STFTalpha1));
w21=conj(w11);
w02=(x2*0+1);
w12=exp(complex(0,-2*pi*x2*STFTalpha2));
w22=conj(w12);

if planeIndx == 1
    % spatial
    [xp,yp]=meshgrid(1:winSizes(2),1:winSizes(1));
    pp=[xp(:) yp(:)];
    d_s=dist(pp,pp');
    Cs=rho_s.^d_s;
    
    % temporal
    Ct=1;
end
if planeIndx == 2 || planeIndx == 3
    % spatial
    [xp,yp]=meshgrid(ones(1,winSizes(2)),1:winSizes(1));
    pp=[xp(:) yp(:)];
    d_s=dist(pp,pp');
    Cs=rho_s.^d_s;
    
    % temporal
    [xp,yp,zp]=meshgrid(1,ones(1,winSizes(1)),1:winSizes(2));
    pp=[xp(:) yp(:) zp(:)];
    d_t=dist(pp,pp');
    Ct=rho_t.^d_t;
end

% rho_s.^d_s .* rho_t.^d_t
C = Cs.*Ct;

% Form 2-D filters q1, q2, q3, q4 and corresponding 2-D matrix operator M (separating real and imaginary parts)
q1=w01.'*w12;
q2=w11.'*w02;
q3=w11.'*w12;
q4=w11.'*w22;
u1=real(q1); u2=imag(q1);
u3=real(q2); u4=imag(q2);
u5=real(q3); u6=imag(q3);
u7=real(q4); u8=imag(q4);
M=[u1(:)';u2(:)';u3(:)';u4(:)';u5(:)';u6(:)';u7(:)';u8(:)'];
M = fliplr(M);

% Compute whitening transformation matrix V
D=M*C*M';
A=diag([1.000007 1.000006 1.000005 1.000004 1.000003 1.000002 1.000001 1]); % Use "random" (almost unit) diagonal matrix to avoid multiple eigenvalues.
[U,S,V]=svd(A*D*A);





