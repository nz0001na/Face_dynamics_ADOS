%read avi files
filename = 'p1_boxing1.avi';
obj = mmreader(filename);
vidFrames = read(obj);
numFrames = get(obj, 'numberOfFrames');
gVidFrames=zeros(size(vidFrames,1),size(vidFrames,2),size(vidFrames,4));
for f=1:numFrames
    gVidFrames(:,:,f) = rgb2gray(vidFrames(:,:,:,f));
    %A = gVidFrames(:,:,f);
    %fmatrix(f,:) = A(:);
end

save('boxing_frames.mat','gVidFrames');


img1 = imresize(gVidFrames(:,:,1),[30,40]);
%imshow(img1,[]);
X = reshape(img1,[1200,1]);

n = 1200;

D = zeros(n);
D(:,1) = 1/sqrt(n);
for i = 2:n
  v = cos((0:n-1)*pi*(i-1)/n)';
  v = v-mean(v);
  D(:,i) = v/norm(v);
end

D = [D eye(n)];
T=8;
gamma = OMP(D'*X,D'*D,T); 

err = X-D*gamma;

imshow(reshape(D*gamma,[30,40]),[]);

