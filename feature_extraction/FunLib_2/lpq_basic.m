function [h, B] = lpq(T, r)
% Local Phase Quantization (LPQ)
% [h, B] = lpq(T, r)
% where 
%   T = input image
%   r = radius of the operator (default = 3)
%   h = resulting lpq histogram
%   B = labeled image

% Reference: 
% Ojansivu V & Heikkilä J (2008) Blur insensitive texture classification 
% using local phase quantization. Proc. Image and Signal Processing 
% (ICISP 2008), Cherbourg-Octeville, France, 5099:236-243.
%
% Copyright 2008 by Heikkilä & Ojansivu

if nargin==1
    r=3; %default radius
end

T=double(T);

f=1;
x=-r:r;
n=length(x);
rho=0.95;


[xp,yp]=meshgrid(1:n,1:n);
pp=[xp(:) yp(:)];
dd=dist(pp,pp');
C=rho.^dd;

w0=(x*0+1);
w1=exp(-2*pi*j*x*f/n);
w2=conj(w1);

q1=w0.'*w1;
q2=w1.'*w0;
q3=w1.'*w1;
q4=w1.'*w2;

u1=real(q1); u2=imag(q1);
u3=real(q2); u4=imag(q2);
u5=real(q3); u6=imag(q3);
u7=real(q4); u8=imag(q4);

M=[u1(:)';u2(:)';u3(:)';u4(:)';u5(:)';u6(:)';u7(:)';u8(:)'];
D=M*C*M';
[U,S,V]=svd(D);


Qa=conv2(conv2(T,w0.','same'),w1,'same');
Qb=conv2(conv2(T,w1.','same'),w0,'same');
Qc=conv2(conv2(T,w1.','same'),w1,'same');
Qd=conv2(conv2(T,w1.','same'),w2,'same');

Fa=real(Qa); Ga=imag(Qa);
Fb=real(Qb); Gb=imag(Qb);
Fc=real(Qc); Gc=imag(Qc);
Fd=real(Qd); Gd=imag(Qd);
F=[Fa(:) Ga(:) Fb(:) Gb(:) Fc(:) Gc(:) Fd(:) Gd(:)];
G=(V'*F');

t=0;

B=(G(1,:)>=t)+(G(2,:)>=t)*2+(G(3,:)>=t)*4+(G(4,:)>=t)*8+(G(5,:)>=t)*16+(G(6,:)>=t)*32+(G(7,:)>=t)*64+(G(8,:)>=t)*128;
B=reshape(B,size(Fa,1),size(Fa,2));

h=hist(B(:),0:255);
h=h/sum(h);
