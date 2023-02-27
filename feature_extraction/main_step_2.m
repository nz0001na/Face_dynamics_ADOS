clear all; clc;

addpath(genpath(pwd))

% input: face detected images -- from step 1
% output: LPQ feature stored in *.mat file

%%
%im_path = 'J:\lab\despression\depression\dev_faces_9_23\';
%im_path ='J:\lab\despression\dev_face_10_10\';
%im_path ='J:\labs\depression\test_region\';
%im_path ='J:\lab\despression\depression\train_faces_10_3\';

%lpq_path = 'J:\labs\depression\lpq_TOP_Feature_laptop2\test\';
%lpq_path = 'J:\lab\despression\train_lpq_10_16\';

% im_path = 'G:\wen2\depression\dev_face_10_9\';   % use it by LW
% lpq_path = 'G:\wen2\depression\HistEG\dev_lpqtop\';  % use it by LW


%% Parameters
outFolder = 'Res_lpq2'; 
% im_path = '/Users/Erica/Others/LW/G_wen2/wen2/depression/dev_face_10_9/';
% im_path = './Input_img/';
im_path = '/Res_img1/';
im_path = [pwd im_path];

%% Begin
mkdir(outFolder);
% output LPQ feature
out_path = ['./' outFolder '/'];  

im_fold = dir(im_path);

%%  !! IMPORTANT !!!
im_fold(1:2)='';  % for Windows
% im_fold = im_fold(4:end);  % for Mac

%%
for fi= 1:length(im_fold)
    im_list = dir([im_path,im_fold(fi).name,'\*.jpg']);   % for Windows
%     im_list = dir([im_path,im_fold(fi).name,'/*.jpg']);
    feas=[];
    data=[];

    for imi= 1  : length(im_list)
        imi
        im = imread([im_path,im_fold(fi).name,'/',im_list(imi).name]);
        % 60*60 cropped faces -> grayscale 100*100 faces
        im0 = rgb2gray(imresize(im,[100,100]));
        % Enhance contrast using histogram equalization.
        im = histeq(im0);
        data(:,:,imi) = double(im);

        % ORG by Zhenzhu Zheng
        Feature_Vector = LPQ_TOP(data,[1 1 1],[0.1 0.1],[5 5 5]); 
        feas=[feas;Feature_Vector'];
    end
    %1:[Feature_Vector] = LPQ_TOP(data,[0 1 1],[0.1 0.1],[15 15 15]);
    %2 [Feature_Vector] = LPQ_TOP(data,[1 1 1],[0.1 0.1],[15 15 15]);
    %3 patch-based
    
    % Kick out NAN   
    feas(find(isnan(feas))) = eps;  
    save([out_path,im_fold(fi).name,'.mat'],'feas');
end
