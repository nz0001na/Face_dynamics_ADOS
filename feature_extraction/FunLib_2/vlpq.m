function [VLPQdesc] = vlpq(vimg,V,winSize,planes,corr,L,norm)
% Function VLPQdesc=vlpq(vimg,V,winSize,planes,corr,L,norm) computes the
% Volume Local Phase Quantization (VLPQ) descriptor for the input volume vimg.
% Descriptors are calculated using only valid pixels.
%
% Inputs: (All empty or undefined inputs will be set to default values)
% vimg    = M*N*T double, volume of gray scale frames for analyzation.
% V       = 26*26 double, matrix for whitening transform. V can be produced
%           using e.g. vlpqCov or left empty when it is automatically generated using
%           the same method as vlpqCov. If V is given, parameter corr is useless.
%               [] -> use the same model based transform as vlpqCov.
% winSize = 1*1 double, size of the neighborhood in spatial direction.
%           Must be odd number and greater or equal to 3 (default winSize=3).
% planes  = 1*1 double, size of the neighborhood in temporal direction.
%           Must be odd nubmer and greater or equal to 3 (default planes=3).
% corr    = 1*2 double, correlation coefficients [rho_s, rho_t]. If V was
%           given, corr is not needed.
%               [] -> use default parameters (default corr=[0.1 0.1])
% L       = 1*1 double, number of eigenvectors to be picked in the
%           dimension reduction (default L=10).
% norm    = 1*1 double, indicates whether the histogram is normalized.
%               1 -> normalization (default)
%               0 -> no normalization
%
% Output:
% VLPQdesc = 1*2^L double VLPQ descriptor histogram.
%
% Example usage (DynTex++ files):
% load('0001.mat');
% [VLPQdesc] = vlpq(subv,[],5,5,[0.1 0.1],10,1);
% figure; bar(VLPQdesc);
%
%
% Version published in 2011 by Juhani Päivärinta, Esa Rahtu, and Janne Heikkilä
% Machine Vision Group, University of Oulu, Finland
%

%% Default settings
% Spatial window size
if nargin<3 || isempty(winSize)
    winSize=3;          % default window size 3
end
% Temporal size of a neighborhood
if nargin<4 || isempty(planes)
    planes=3;           % default number of planes 3
end
% Correlation coefficients
if nargin<5 || isempty(corr)
    corr = [0.8 0.6];   % default [rho_s rho_t] [0.1 0.1]
end
% Number of L in the dimension reduction
if nargin<6 || isempty(L)
    L = 10;             % default L 10
end
% Normalization
if nargin<7 || isempty(norm)
    norm = 1;           % use normalization as default
end

%% Check input parameters
if size(vimg,4)~=1
    error('Only a 3-D volume can be used as input.');
end
if ~isempty(V)
    if size(V,1)~=26 || size(V,2)~=26
        error('Size of V must be 26x26.');
    end
end
if winSize<3 || rem(winSize,2)~=1
    error('Window size must be odd number and greater than equal to 3.');
end
if planes<3 || rem(planes,2)~=1
    error('Number of planes must be odd number and greater than equal to 3.');
end
if size(corr,1)~=1 || size(corr,2)~=2
    error('Parameter corr must be a row vector of length 2.');
end
if L<1 || L>26
    error('Must be 0<L<27.');
end
if sum(norm==[0 1])==0
    error('Parameter norm must be set to 1 -> normalization or 0 -> no normalization.');
end

% Make whitening transformation matrix V if not given as input
if nargin<2 || isempty(V)
    rho_s = corr(1);
    rho_t = corr(2);   
    [C,D,V] = vlpqCov_(winSize,planes,rho_s,rho_t);
end

%% Initialization
vimg = double(vimg);    % convert vimg to double
STFTalpha = 1/winSize;  % alpha in STFT approaches
STFTalpha2 = 1/planes;  % alpha2 in STFT approaches (z-direction)
r = (winSize-1)/2;      % Get radius from window size
x = -r:r;               % Form spatial coordinates in window
r2 = (planes-1)/2;      % Measurement for planes
z = -r2:r2;             % Coordinates in the temporal direction
convmode = 'valid';     % use valid area only

%% Filters
% STFT with uniform window
w0 = double((x*0+1));
w1 = double(exp(complex(0,-2*pi*x*STFTalpha)));
w2 = double(conj(w1));
w0d = double(reshape((z*0+1),1,1,planes));
w1d = double(reshape(exp(complex(0,-2*pi*z*STFTalpha2)),1,1,planes));
w2d = double(reshape(conj(w1d),1,1,planes));

%% Convolutions
t0 = convnc(vimg,w0.',convmode);
t00 = convnc(t0,w0,convmode);
filterResp = convnc(t00,w1d,convmode); % F13
freqResp = zeros(size(filterResp,1),size(filterResp,2),size(filterResp,3),26);
freqResp(:,:,:,25) = real(filterResp); freqResp(:,:,:,26) = imag(filterResp);
clear t00;
t01 = convnc(t0,w1,convmode);
filterResp = convnc(t01,w0d,convmode); % F4
freqResp(:,:,:,7) = real(filterResp); freqResp(:,:,:,8) = imag(filterResp);
filterResp = convnc(t01,w1d,convmode); % F5
freqResp(:,:,:,9) = real(filterResp); freqResp(:,:,:,10) = imag(filterResp);
filterResp = convnc(t01,w2d,convmode); % F6
freqResp(:,:,:,11) = real(filterResp); freqResp(:,:,:,12) = imag(filterResp);
clear t01;
clear t0;
t1 = convnc(vimg,w1.',convmode);
t10 = convnc(t1,w0,convmode);
filterResp = convnc(t10,w0d,convmode); % F1
freqResp(:,:,:,1) = real(filterResp); freqResp(:,:,:,2) = imag(filterResp);
filterResp = convnc(t10,w1d,convmode); % F2
freqResp(:,:,:,3) = real(filterResp); freqResp(:,:,:,4) = imag(filterResp);
filterResp = convnc(t10,w2d,convmode); % F3
freqResp(:,:,:,5) = real(filterResp); freqResp(:,:,:,6) = imag(filterResp);
clear t10;
t11 = convnc(t1,w1,convmode);
filterResp = convnc(t11,w0d,convmode); % F7
freqResp(:,:,:,13) = real(filterResp); freqResp(:,:,:,14) = imag(filterResp);
filterResp = convnc(t11,w1d,convmode); % F8
freqResp(:,:,:,15) = real(filterResp); freqResp(:,:,:,16) = imag(filterResp);
filterResp = convnc(t11,w2d,convmode); % F9
freqResp(:,:,:,17) = real(filterResp); freqResp(:,:,:,18) = imag(filterResp);
clear t11;
t12 = convnc(t1,w2,convmode);
filterResp = convnc(t12,w0d,convmode); % F10
freqResp(:,:,:,19) = real(filterResp); freqResp(:,:,:,20) = imag(filterResp);
filterResp = convnc(t12,w1d,convmode); % F11
freqResp(:,:,:,21) = real(filterResp); freqResp(:,:,:,22) = imag(filterResp);
filterResp = convnc(t12,w2d,convmode); % F12
freqResp(:,:,:,23) = real(filterResp); freqResp(:,:,:,24) = imag(filterResp);
clear t12;
clear t1;
[freqRow,freqCol,freqT,freqNum]=size(freqResp); % Read the size of frequency matrix 
clear filterResp;

%% Dimension reduction and decorrelation
freqResp = reshape(freqResp,[freqRow*freqCol*freqT,26]); % reshape the 26-dimensional space to a matrix
freqResp = freqResp*V(:,1:L);                            % pick only the most important values
[freqRow,freqCol] = size(freqResp);                      % Read the size of frequency matrix

%% Perform quantization and compute LPQ codewords
freqResp=freqResp>=0;
if L == 10                              % a faster calculation if L=10
    VLPQdesc = (freqResp(:,1))+...
               (freqResp(:,2))*2+...
               (freqResp(:,3))*4+...
               (freqResp(:,4))*8+...
               (freqResp(:,5))*16+...
               (freqResp(:,6))*32+...
               (freqResp(:,7))*64+...
               (freqResp(:,8))*128+...
               (freqResp(:,9))*256+...
               (freqResp(:,10))*512;
else                     
    VLPQdesc = double(zeros(freqRow,1)); % Initialize VLPQ code word
    for i=1:L
        VLPQdesc = VLPQdesc+(double(freqResp(:,i)))*(2^(i-1));
    end
end

%% Calculate histogram
VLPQdesc = histc(VLPQdesc(:),0:(2^L)-1).';

%% Normalize histogram if wanted
if norm == 1
    VLPQdesc = VLPQdesc/sum(VLPQdesc(:));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Additional functions %%
%%%%%%%%%%%%%%%%%%%%%%%%%%

function [C,D,V] = vlpqCov_(winSize,planes,rho_s,rho_t)
% function [C,D,V] = vlpqCov(winSize,planes,rho_s,rho_t) calculates a model
% based covariance matrix C and the final covariance matrix D (D=M*C*M.').
% Also the corresponding whitening transformation matrix V is calculated
% using SVD.
%
% Inputs: (All empty or undefined inputs will be set to default values)
% winSize = 1*1 double, size of the window in spatial direction.
%           Must be odd number and greater or equal to 3 (default winSize=3).
% planes  = 1*1 double, size of the neighborhood in temporal direction.
%           Must be odd nubmer and greater or equal to 3 (default
%           planes=3).
% rho_s   = 1*1 double, rho in the spatial domain (default rho_s=0.1).
% tho_t   = 1*1 double, rho in the temporal domain (default rho_t=0.1).
%
% Outputs:
% C       = (planes*winSize^2)*(planes*winSize^2) double, a model based
%           covariance matrix.
% D       = 26*26 double, final covariance matrix of F_x (D=M*C*M.').
% V       = 26*26 double, a matrix for whitening transform.
%
% Example usage:
% [C,D,V] = vlpqCov(5,5,0.1,0.1);
% 

%% Default settings
% Spatial window size
if nargin<1 || isempty(winSize)
    winSize=3;   % default window size 3
end
% Temporal size of a neighborhood
if nargin<2 || isempty(planes)
    planes=3;    % default number of planes 3
end
% Spatial rho
if nargin<3 || isempty(rho_s)
    rho_s = 0.1; % default spatial rho 0.1
end
% Temporal rho
if nargin<4 || isempty(rho_t)
    rho_s = 0.1; % default temporal rho 0.1
end

%% Check input parameters
if winSize<3 || rem(winSize,2)~=1
    error('Window size must be odd number and greater than equal to 3');
end
if planes<3 || rem(planes,2)~=1
    error('Number of planes must be odd number and greater than equal to 3');
end

%% Initialization
STFTalpha = 1/winSize;  % alpha in STFT approaches
STFTalpha2 = 1/planes;  % alpha2 in STFT approaches (z-direction)
r = (winSize-1)/2;      % Get radius from window size
x = -r:r;               % Form spatial coordinates in window
r2 = (planes-1)/2;      % Measurement for planes
z = -r2:r2;             % Coordinates in the temporal direction

%% Filters
% Basic STFT filters
w0 = double((x*0+1));
w1 = double(exp(complex(0,-2*pi*x*STFTalpha)));
w2 = double(conj(w1));
w0d = double(reshape((z*0+1),1,1,planes));
w1d = double(reshape(exp(complex(0,-2*pi*z*STFTalpha2)),1,1,planes));
w2d = double(reshape(conj(w1d),1,1,planes));

%% Compute covariance matrix
% spatial
[xp,yp,zp]=meshgrid(1:winSize,1:winSize,ones(1,planes));
pp=[xp(:) yp(:) zp(:)];
d_s=dist(pp,pp');
Cs=rho_s.^d_s;

% temporal
[xp,yp,zp]=meshgrid(ones(1,winSize),ones(1,winSize),1:planes);
pp=[xp(:) yp(:) zp(:)];
d_t=dist(pp,pp');
Ct=rho_t.^d_t;

C = Cs.*Ct;

%% Compute 3-D filters
% Here we can use convnc with zero-padding to get q1, ..., q13.
% Results could also be achieved using multiplications.
q1 = convnc(convnc(w1.',w0),w0d); % F1
q2 = convnc(convnc(w1.',w0),w1d); % F2
q3 = convnc(convnc(w1.',w0),w2d); % F3
q4 = convnc(convnc(w0.',w1),w0d); % F4
q5 = convnc(convnc(w0.',w1),w1d); % F5
q6 = convnc(convnc(w0.',w1),w2d); % F6
q7 = convnc(convnc(w1.',w1),w0d); % F7
q8 = convnc(convnc(w1.',w1),w1d); % F8
q9 = convnc(convnc(w1.',w1),w2d); % F9
q10 = convnc(convnc(w1.',w2),w0d); % F10
q11 = convnc(convnc(w1.',w2),w1d); % F11
q12 = convnc(convnc(w1.',w2),w2d); % F12
q13 = convnc(convnc(w0.',w0),w1d); % F13


%% Form matrix operator M (separating real and imaginary parts)
u1=real(q1); u2=imag(q1);
u3=real(q2); u4=imag(q2);
u5=real(q3); u6=imag(q3);
u7=real(q4); u8=imag(q4);
u9=real(q5); u10=imag(q5);
u11=real(q6); u12=imag(q6);
u13=real(q7); u14=imag(q7);
u15=real(q8); u16=imag(q8);
u17=real(q9); u18=imag(q9);
u19=real(q10); u20=imag(q10);
u21=real(q11); u22=imag(q11);
u23=real(q12); u24=imag(q12);
u25=real(q13); u26=imag(q13);
M=[u1(:)';u2(:)';u3(:)';u4(:)';u5(:)';u6(:)';u7(:)';u8(:)';u9(:)';u10(:)';...
   u11(:)';u12(:)';u13(:)';u14(:)';u15(:)';u16(:)';u17(:)';u18(:)';u19(:)';...
   u20(:)';u21(:)';u22(:)';u23(:)';u24(:)';u25(:)';u26(:)'];
M=fliplr(M);

%% Compute whitening transformation matrix V
D=M*C*M.';
% Use "random" (almost unit) diagonal matrix to avoid multiple eigenvalues.
A=diag([1.0000013 1.0000012 1.0000011 1.0000010 1.000009 1.000008 1.000007 ...
        1.0000006 1.0000005 1.0000004 1.0000003 1.0000002 1.0000001 1 ...
        0.9999999 0.9999998 0.9999997 0.9999996 0.9999995 0.9999994 0.9999993 ...
        0.9999992 0.9999991 0.9999990 0.9999989 0.9999988]);  
[U,S,V]=svd(A*D*A);







