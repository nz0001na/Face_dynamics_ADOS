function outCov=Compute_Cov_Features(inImage)

 [SizeX,SizeY,nChannels]=size(inImage);
% 
% 
% imagePosY=repmat((1:SizeY),[SizeX,1])/SizeY;
% imagePosX=repmat((1:SizeX),[SizeY,1])'/SizeX;
% 
% 
%  MAX_VAL=max(max(max(inImage))); %To safe handling color images
% if (MAX_VAL>1)
%     inImage=inImage./MAX_VAL;
% end

covSamples=[];
%Pos information
% covSamples=[covSamples;...
%         reshape(imagePosX,[1,SizeX*SizeY]);...
%         reshape(imagePosY,[1,SizeX*SizeY])];


% inImage = rgb2gray(inImage);

%Grey values
covSamples=[covSamples;reshape(inImage,[1,SizeX*SizeY])];

% %Color information
% covSamples=[covSamples;...
%             reshape(inImage(:,:,1),[1,SizeX*SizeY]);...
%             reshape(inImage(:,:,2),[1,SizeX*SizeY]);...
%             reshape(inImage(:,:,3),[1,SizeX*SizeY])];

%Gradient information
tmpImage_D1_H=abs(imfilter(inImage,[-1 0 1]'));
tmpImage_D1_V=abs(imfilter(inImage,[-1 0 1]));
% tmpImage_D2_H=abs(imfilter(inImage,[-1 2 -1]'));
% tmpImage_D2_V=abs(imfilter(inImage,[-1 2 -1]));

covSamples=[covSamples;...
    reshape(tmpImage_D1_H,[1,SizeX*SizeY]);...
    reshape(tmpImage_D1_V,[1,SizeX*SizeY])];
%     reshape(tmpImage_D2_H,[1,SizeX*SizeY]);...
%     reshape(tmpImage_D2_V,[1,SizeX*SizeY])];


%Computing the covariance matrix
outCov=cov(covSamples');