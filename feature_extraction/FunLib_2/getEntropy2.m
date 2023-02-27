function entropy_num=getEntropy2(data)
%get covariance feature
fea1 = zeros(1,64);
fea2 = zeros(1,64);
Temp_Volume_XT = permute(data,[1 3 2]);    
Temp_Volume_YT = permute(data,[2 3 1]);
for i=1:100
    combins = zeros(160,160);
    combins1 = double(Temp_Volume_XT(:,:,i))';
    combins2 = double(Temp_Volume_YT(:,:,i));
    outCov1=Compute_Cov_Features(combins1);
    outCov2=Compute_Cov_Features(combins1);
    num1 = getNum(outCov1);
     num2 = getNum(outCov2);
    index1 = num1+1;
    fea1(index1) = fea1(index1)+1;
     index2 = num2+1;
    fea2(index2) = fea2(index2)+1;
end
fea1 = fea1./sum(fea1);
id1 = find(fea1==0);
fea1(id1)=[];
entropy1 = -sum(fea1.*log2(fea1));
fea2 = fea2./sum(fea2);
id2 = find(fea2==0);
fea2(id2)=[];
entropy2 = -sum(fea2.*log2(fea2));
entropy_num = sqrt(entropy1^2+entropy2^2);