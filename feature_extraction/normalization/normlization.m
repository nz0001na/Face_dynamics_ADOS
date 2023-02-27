clear all;
clc;


%list = dir([im_path,'\male','\*.bmp']);
%list = dir(['F:\Wen\lab\crossage\fgnet\FGNET','\*.JPG']);
%fid = fopen('F:\Wen\morph\CD2\MORPH_Album2_EYECOORDS.csv','r');

%data = textscan(fid,'%s%f%f%f%f','Delimiter',',','CollectOutput',1);
%inpath = 'F:\Wen\labs\bmi_new\Testgroup\Testgroup\';
inpath = 'F:\Wen\labs\bmi_new\third\Test\Org\bf\';
%inpath = 'F:\Wen\labs\bmi_new\Testgroup\Testgroup\second\';
list = dir([inpath,'*.jpg']);
k=1;
%outpath = 'F:\Wen\labs\bmi_new\Testgroup\norm\';
outpath = 'F:\Wen\labs\bmi_new\third\Test\norm\';
pointpath = 'F:\Wen\labs\bmi_new\third\Test\point\';
cutpath = 'F:\Wen\labs\bmi_new\third\Test\cutface\';
allipath = 'F:\Wen\labs\bmi_new\third\Test\alli\bf\';
%pointpath = 'F:\Wen\labs\bmi_new\Testgroup\point\';
%pointpath = 'F:\Wen\labs\bmi_new\Testgroup\Testgroup\second\';
for k=1:length(list)
    k
    strin = strcat(inpath,list(k).name);
    strout = strcat(outpath,list(k).name);
    s = list(k).name;
    s(end-3:end)='';
    dataname = strcat(pointpath,s,'.mat');
    if exist(dataname)        
        pointdata = load(dataname,'-ASCII');
    else
        continue;
    end
    im = imread(strin);
    ed = pointdata(37,1)-pointdata(32,1);
    [m n z]=size(im);
    if pointdata(32,2)-1.5*ed>1
        y1 = floor(pointdata(32,2)-1.5*ed);
    else
        y1 = 1;
    end
    if pointdata(32,2)+2.1*ed < m
        y2 = floor(pointdata(32,2)+2.1*ed);
    else
        y2 = m;
    end
    if pointdata(32,1)-ed >1
        x1 = floor(pointdata(32,1)-ed);
    else
        x1 = 1;
    end
    if pointdata(32,1)+1.9*ed < n
        x2 = floor(pointdata(32,1)+1.9*ed);
    else
        x2 = n;
    end
    imcut = im(y1:y2,x1:x2,:);
    imwrite(imcut,[cutpath,list(k).name],'JPEG');
    Lx = double(pointdata(32,1) - x1);
    Ly = double(pointdata(32,2) - y1);
    Rx = double(pointdata(37,1) - x1);
    Ry = double(pointdata(37,2) - y1);
    x1=(pointdata(28,1)+pointdata(30,1))/2;
    y1=(pointdata(28,2)+pointdata(30,2))/2;
    x2=(pointdata(35,1)+pointdata(33,1))/2;
    y2=(pointdata(35,2)+pointdata(33,2))/2;
    om = affin_eye((im),[x1,y1,x2,y2],[160,225,220,225]);%144	218	247	214
    %om = affin_eye((im),[144, 218, 247, 214],[160,225,220,225]);
    imwrite(om,[allipath,list(k).name],'JPEG');
end
%while k<=length(list)
 %   strin = strcat('F:\Wen\morph2\',list(k).name);
  %  strout = strcat('F:\Wen\Nmorph\',list(k).name);
   % s=list(k).name;
    %s(ismember(s,'.JPG'))='';
    %m = find(ismember(data{1},s));
    %a=sprintf('%d %d %d %d',data{2}(m,1),data{2}(m,2),data{2}(m,3),data{2}(m,4));
    %system(['opencv_test_2.exe ',strin,' ',a,' ',strout]);
    %k=k+1;    
%end
%fclose(fid);


% for k = 1:length(list)
%     strin = strcat('F:\Wen\WhiteWomen\1\',list(k).name);
%     strout = strcat('F:\Wen\WhiteWomen\normalization\1\',list(k).name);
%     s=list(k).name;
%     s(ismember(s,'.JPG'))='';
%     m = find(ismember(data{1},s));
%     %a=sprintf('[%d %d %d %d]',data{2}(m,1),data{2}(m,2),data{2}(m,3),data{2}(m,4));
%     if (m > 0)
%     im=imread(strin);
%     om=affin_eye(im,[data{2}(m,1),data{2}(m,2),data{2}(m,3),data{2}(m,4)],[160,225,220,225]);   
%     imwrite(om,strout,'JPEG');
%     end
%     k=k+1;
% end
% fclose(fid);