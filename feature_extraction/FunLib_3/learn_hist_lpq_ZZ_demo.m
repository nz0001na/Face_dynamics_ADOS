function learn_hist_lpq_ZZ_demo

addpath('./ompbox10');

% indir = './Input_lpq/';
% fdir = './Output_Dict_lpq/'; %dict dir
% outFolder = 'Output_Hist_lpq'; 

% get LPQ features
indir = './Res_lpq1/';
% dictionary dir
fdir = './Res_Dict_lpq1/'; 
% output dir
outFolder = 'Res_Hist_lpq1'; 
mkdir(outFolder);
outdir = ['./' outFolder '/']; 

%% separate LPQ features into Train & Test set
trnFolder = 'Trn_lpqtop1'; 
mkdir(trnFolder);
tstFolder = 'Tst_lpqtop1'; 
mkdir(tstFolder);

voldir_trn = ['./' trnFolder '/'];  %train data dir
voldir_tst = ['./' tstFolder '/'];   %test data dir

fList = dir([indir '*.mat' ]); 
% fList = fList(3:end);
% number of training samples
% sNum = round(length(fList)/2)+1;  
sNum = round(length(fList)/2);  

for i = 1 : length(fList)
    fName = fList(i).name;
    
    if i <= sNum
        copyfile( [indir fName], voldir_trn );
    else
        copyfile( [indir fName], voldir_tst );
    end
    
end

%% begin 
% trained dictionary data
filescontent = dir([fdir 'ksvd_trn*.mat']);
nfiles = length(filescontent);

for i=1:nfiles
    filename = filescontent(i).name;         
    load([fdir filename]);
    
    oname = [outdir 'hist' filename(9:end)];    
    [hist_trn, names_trn]= learn_hist_t(ksvd_trn,voldir_trn);
    [hist_tst, names_tst] = learn_hist_t(ksvd_trn,voldir_tst);
    
    save(oname, 'hist_trn','hist_tst','names_trn','names_tst');
    fprintf('%s%d/%d\n', 'Finished:',i,nfiles);
end

end


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



