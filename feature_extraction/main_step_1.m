%%% transfer video[format: mp4] to frames[format: jpg]
%%% face detection and Crop with 60*60

clear all; clc;

%% add path & make folder
addpath(genpath(pwd))
addpath(genpath('./normalization'))

inFolder = './Input_vid';  
outFolder = 'Res_img1';  
mkdir(outFolder);
inList = dir(fullfile(inFolder,'*.mp4'));

os = computer;
if  strcmp(os, 'PCWIN64')
    inList = inList(3:end);
end

%% load data
% load and visualize model
% Pre-trained model with 146 parts. Works best for faces larger than 80*80
% load face_p146_small.mat

% Pre-trained model with 99 parts. Works best for faces larger than 150*150
% load face_p99.mat

% Pre-trained model with 1050 parts. Give best performance on localization, but very slow
% load multipie_independent.mat

load face_p146_small.mat

%% set parameter

% 5 levels for each octave 
model.interval = 5;  
% threshold
model.thresh = min(-0.65, model.thresh);  
% selected number of frames from the begining, delete the variable if ALL frames are used.
% sNumFrame = 10;  

%% check platform
os = computer;
if strcmp(os, 'MACI64')
    outpath = ['./' outFolder '/']; % for mac directory
else
    outpath = ['.\' outFolder '\'];  % for windows directory        
end
clear os

%% mapping
% define the mapping from view-specific mixture id to viewpoint
if length(model.components)==13 
    posemap = 90:-15:-90;
elseif length(model.components)==18
    posemap = [90:-15:15 0 0 0 0 0 0 -15:-15:-90];
else
    error('Can not recognize this model');
end

%% begin
for i = 1 : length(inList)
        fName = inList(i).name; 
        mkdir(outpath, fName(1:end-4));
                
        disp(['Reading video  '   fName  '...']);
        % use "VideoReader" instead of "mmreader" for later versions of MATLAB
        readerobj = VideoReader(fName, 'tag', 'myreader1');         

        tic
        if exist ('sNumFrame', 'var')
            % Read in PARTIAL video frames.
            vidFrames = read(readerobj, [1 sNumFrame]);             
        else
            % Read in ALL video frames.
            vidFrames = read(readerobj);                            
        end
        toc
        disp('done')

        % Get the number of frames. {ALL frames}
        % numFrames = get(readerobj, 'numberOfFrames');      
        numFrames = size(vidFrames, 4);
        oldCorner=[0,0];

        % Create a MATLAB movie struct from the video frames.
       for k = 1: numFrames                                 
             k
             im = vidFrames(:,:,:,k);
            
             tic;
             [newim, corner, gflag] = getfacefromframe1(im, model,oldCorner);
             gflag
             toc;
             
             nid=num2str(k);
             
             while 5-length(nid)>0
                 nid=['0',nid];
             end
             if gflag
             
             savepath = [outpath fName(1:end-4) '/'];
             imwrite(newim,[savepath,nid,'.jpg'],'JPEG');
             oldCorner = corner;
             end
       end
end


