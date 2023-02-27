function [C,D,V] = vlpqCov(winSize,planes,rho_s,rho_t)
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


