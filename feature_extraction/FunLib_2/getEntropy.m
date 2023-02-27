function entropy=getEntropy(data)
%get covariance feature
fea = zeros(1,64);
Temp_Volume_XT = permute(data,[1 3 2]);    
Temp_Volume_YT = permute(data,[2 3 1]);
for i=1:100
    combins = zeros(160,160);
    combins(1:60,1:100) = double(Temp_Volume_XT(:,:,i))';
    combins(1:100,1:60) = double(Temp_Volume_YT(:,:,i));
    outCov=Compute_Cov_Features(combins);
    num = getNum(outCov);
    index = num+1;
    fea(index) = fea(index)+1;
end
fea = fea./sum(fea);
id = find(fea==0);
fea(id)=[];
entropy = -sum(fea.*log2(fea));