%% learn_hist_t
function [hist, names] = learn_hist_t(ksvds,voldir)

T=ksvds.k;
n = length(ksvds.dict);

% filescontent = dir([voldir '*.mat']);  
filescontent = dir([voldir '2*.mat']);  
nfiles = length(filescontent);
for i = 1:nfiles
    filename = filescontent(i).name;  
    % load the video data (lpq feature vectors for one video)
    load([voldir filename]);  
    names{i} = filename;
       
    % the lpq feature matrix  (nxm, m is the feature dimention, n is the number of samples) 
    % added by Zhenzhu Zheng
    %imgp = ;    
    imgp = feas; 
    gamma = omp(ksvds.dict'*imgp(:,:)',ksvds.dict'*ksvds.dict,T);  % NaN
    hist{i} = gamma;
end


end


