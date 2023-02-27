clear all; clc; close all

addpath(genpath(pwd))
addpath('./ompbox10'); 
addpath('./ksvdbox13');

%% set parameters
inFolder = 'Res_lpq2';
outFolder = 'Res_Dict_lpq1'; 
mkdir(outFolder);
% k in K-SVD, typically from 1 to 4
Ks = [1:4];           
% Dictionary dimension,typically 100:100:500, due to simplicity use small numbers here
dictDims = [1:1:5];     

%% generate a set of LPQ features 
fList = dir(fullfile(inFolder, '*.mat'));
% fList = fList(3:end);
lpq_data = [];
for i = 1 : length(fList)
    x = load( [ './' inFolder '/' fList(i).name ] );
    feas = x.feas;
    lpq_data = [lpq_data; feas];
end

%% learn the KSVD dict
for k = Ks  
    for m = dictDims  
         outname =  sprintf('ksvd_trn_k=%d_dict=%d.mat',k,m);
                     
         params = struct;
         % Sparse coding target.
         params.Tdata = k;
         % specifies the number of dictionary atoms to train
         params.dictsize = m;
         % Number of training iterations.
         params.iternum = 20;
         %  Memory usage.
         params.memusage = 'high';
         % Training data.
         params.data(:,:) = lpq_data;
         params.data = params.data';
         
         disp('ksvd...');
         % trained dictionary Dksvd
         % the signal representation matrix g
         % and error.
         [Dksvd,g,err] = ksvd(params,'');
         disp('Done.');
         
         % training dictionary data
         ksvd_trn.dict = Dksvd;
         ksvd_trn.g= g;
         ksvd_trn.dictsize = m;
         ksvd_trn.k = k;
    end 
    save(['./' outFolder '/' outname],'ksvd_trn');
end



