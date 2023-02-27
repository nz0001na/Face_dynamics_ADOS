function [C,D,V]=lpqtopCov(winSizes,rho_s,rho_t,planeIndx)
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

