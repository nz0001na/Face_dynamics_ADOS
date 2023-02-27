clear all; clc;
addpath(genpath(pwd))

%% add path
addpath './MFA'
addpath './libsvm-mat-3.0-3'

% dev_path = '/Users/Erica/Others/LW/G_wen2/wen2/depression/lpq_TOP_Feature_laptop/dev/2/';
% train_path = '/Users/Erica/Others/LW/G_wen2/wen2/depression/lpq_TOP_Feature_laptop/train/2/';

% NOTE: still have not figured out why they are different?
% data_path = '/Users/Erica/Project/Depression Project/Step 4_MFA_SVR_decision fusion/yu_sparsecoding2/hists_lpq1/';
% data_path2 = '/Users/Erica/Project/Depression Project/Step 4_MFA_SVR_decision fusion/yu_sparsecoding2/hists_lpq1223/';  

dev_path = './Res_lpq1/';
train_path = './Trn_lpqtop1/';
data_path = './Res_Hist_lpq1/';
data_path2 = data_path;  % USE NEW DATA IF NECESSARY

label_path = './train_test_label/';

% dev_lpq_list = dir([dev_path,'*.mat']); 
% train_lpq_list = dir([train_path,'*.mat']);
data = dir([data_path,'*.mat']);

%% set parameters
% MAE: mean absolute error
% RMSE: root mean square error
mini_mae=999;
mini_rmse=9999;
ncd=4;
nc=[0.0001,0.1,0.5,1,5,10,15,20,25,30,35,40,45,50,70,100];
bestc=0;
bestg=0;

res=[];
name_list={};

%% begin
for fi=1:length(data)  
    % for simplicity, REMOVE IT TO RUN ON WHOLE DATA
% for fi=1:1                                  
    filename = data(fi).name;
    
    train_test = load([data_path,filename]);
    train_test2 = load([data_path2,filename]);
    
    train_data=[];
    test_data=[];
    
    test_name=train_test.names_tst;
    train_name = train_test.names_trn;
    
    trn = train_test.hist_trn;
    trn2 =  train_test2.hist_trn;
    tst = train_test.hist_tst;
    tst2 =  train_test2.hist_tst;

    train_label=[];
    test_label=[];

    for i = 1:length(train_name)
        name = train_name(i);
        name_str = name{1};
        namei = find(ismember(name_str,'_'));
        name_str(namei(2):end)='';
        name_str = [name_str,'_Depression.csv'];
        label_data = textread([label_path,name_str],'','delimiter',',',...
            'emptyvalue', NaN);
        train_label=[train_label;label_data];
    
%%%%%%lpq_top
       lpq_fea=load([train_path,name{1}]);
       lpqs=(lpq_fea.feas);
       feature=[];
       feature2=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%sparsecoding
        spr = trn{i};
        spr2 = trn2{i};
        fea=[];
        fea2=[];
        spr_coding = full(spr);
        spr_coding2 = full(spr2);
        spr_coding = spr_coding';
        spr_coding2 = spr_coding2';
        avl=floor(size(spr_coding,1)/ncd);
        for t=1:size(spr_coding,1)
%             pa=spr_coding(:,t);          
%             fea=fea+pa;
%             
            %%%%lpq_top
            lpq_feature = lpqs(t,:);
            lpq1=lpq_feature(1:256);
            lpq2=lpq_feature(257:512);
            lpq3=lpq_feature(513:768);
            lpq_num = lpq1;
            lpq_num(find(lpq_num<0.0001))=[];
            entropy_feature1=(-sum(lpq_num.*log2(lpq_num)));
            lpq_num = lpq2;
            lpq_num(find(lpq_num<0.0001))=[];
            entropy_feature2=(-sum(lpq_num.*log2(lpq_num)));
            lpq_num = lpq3;
            lpq_num(find(lpq_num<0.0001))=[];
            entropy_feature3=(-sum(lpq_num.*log2(lpq_num)));
            feature=[feature;lpq1];
            feature2=[feature2;(entropy_feature2^2+entropy_feature3^2)];
        end
        [fea_number,index]=sort(feature2,'ascend');
        index=[1:length(index)];
        sort_sp = spr_coding(index,:);
        sort_sp2 = spr_coding2(index,:);
        ii=0;
        for si=1:avl:length(index)-avl+1
            fea = [fea,mean(sort_sp(si:(si+avl-1),:))];
            fea2 = [fea2,mean(sort_sp2(si:(si+avl-1),:))];
            ii=ii+1;
            if ii==ncd-1
                break;
            end
        end
        fea = [fea,mean(sort_sp(ii*avl+1:end,:))];
        fea2 = [fea2,mean(sort_sp2(ii*avl+1:end,:))];
        fea = [fea,fea2];
   %fea = fea ./ size(spr_coding,2);
    train_data = [train_data;fea];
    end
        disp('test');
    for i = 1:length(test_name)
        name = test_name(i);
        name_str = name{1};
        namei = find(ismember(name_str,'_'));
        name_str(namei(2):end)='';
        name_str = [name_str,'_Depression.csv'];
        label_data = textread([label_path,name_str],'','delimiter',',',...
            'emptyvalue', NaN);
        test_label=[test_label;label_data];
        
       % lpq_top
       lpq_fea=load([dev_path,name{1}]);
       lpqs=(lpq_fea.feas);
       feature=[];
       feature2=[];

        % sparsecoding
        spr = tst{i};
        spr2 = tst2{i};
        fea=[];
        fea2=[];
        spr_coding = full(spr);
        spr_coding = spr_coding';
        spr_coding2 = full(spr2);
        spr_coding2 = spr_coding2';
        avl=floor(size(spr_coding,1)/ncd);
        
        for t=1:size(spr_coding,1)
            % lpq_top
            lpq_feature = lpqs(t,:);            
            lpq1=lpq_feature(1:256);
            lpq2=lpq_feature(257:512);
            lpq3=lpq_feature(513:768);
            
            lpq_num = lpq1;
            lpq_num(find(lpq_num<0.0001))=[];
            entropy_feature1=(-sum(lpq_num.*log2(lpq_num)));
            lpq_num = lpq2;
            lpq_num(find(lpq_num<0.0001))=[];
            entropy_feature2=(-sum(lpq_num.*log2(lpq_num)));
            lpq_num = lpq3;
            lpq_num(find(lpq_num<0.0001))=[];
            entropy_feature3=(-sum(lpq_num.*log2(lpq_num)));
            feature=[feature;lpq1];
            feature2=[feature2;(entropy_feature2^2+entropy_feature3^2)];
        end
        
        [fea_number,index]=sort(feature2,'ascend');
        index=[1:length(index)];
        sort_sp = spr_coding(index,:);
        sort_sp2 = spr_coding2(index,:);
        ii=0;
        
        for si=1:avl:length(index)-avl+1
            fea = [fea,mean(sort_sp(si:(si+avl-1),:))];
             fea2 = [fea2,mean(sort_sp2(si:(si+avl-1),:))];
            ii=ii+1;
            if ii==ncd-1
                break;
            end
        end
        fea = [fea,mean(sort_sp(ii*avl+1:end,:))];
        fea2 = [fea2,mean(sort_sp2(ii*avl+1:end,:))];
        fea = [fea,fea2];
       %fea = fea ./ size(spr_coding,2);
        test_data = [test_data;fea];
    end

    options = [];
    for ink=4:4
        for iek = 1900:1900
            options.intraK = ink;
            options.interK = iek;
            options.PCARatio = 0.99;
            %options.ReducedDim = 40;
            %options.ReducedDim = mfai;
            options.Regu = 1;
            for al = 0.15:0.15
                options.ReguAlpha = al;

                [eigvector, eigvalue] = MFA(train_label, options, train_data);

                train_fea = train_data*eigvector;
                dev_fea =test_data*eigvector;
                train_data =train_fea;
                test_data = dev_fea;

                mini_pred=[]; beste=0;
                
                disp('Testing is in progress')                
            for ci=1:16    
                for log2g = -10:0.1:10
                    cmd = ['-s 3 -t 2   -c ', num2str(nc(ci)), ' -g ', num2str(2^log2g)];
                    model_face=svmtrain(train_label, train_data, cmd);

                    [label_result, accuracy_result, dec_test_result]=svmpredict(test_label,test_data, model_face);

                    mae=mean(abs(label_result-test_label));
                    rmse = sqrt(mean((label_result-test_label).^2)); 
                     if mini_rmse>=rmse %|| mini_mae>=mae
                         mini_mae = mae;
                         mini_rmse = rmse;
                         mini_pred = label_result;

                         bestc=nc(ci);
                         bestg=2^log2g;
                         res=[res;[mini_mae,mini_rmse]];
                         name_list=[name_list;filename];

                         fprintf('MAE1 %f \n', mini_mae);
                         fprintf('rmse %f \n', mini_rmse);
                     end
                end
            end
            end    
        end
    end
end

fprintf('MAE1 %f \n', mini_mae);
fprintf('rmse %f \n', mini_rmse);