clear all;
clc;



list = dir(['F:\Wen\lab\crossage\fgnet\FGNET','\*.JPG']);

k=1;
for k = 1:length(list)
    strin = strcat('F:\Wen\lab\crossage\fgnet\FGNET\',list(k).name);
    strout = strcat('F:\Wen\lab\crossage\fgnet\fgnet_normalization\',list(k).name);
    s=list(k).name;
    s(ismember(s,'.JPG'))='';
    s(ismember(s,'.jpg'))='';
    s(ismember(s,'A'))='a';
    str=[s,'.mat']
    points = load(['F:\Wen\lab\crossage\fgnet\points_mat\points\',str]);
    im=imread(strin);
%     Lx=round(points.p.x(4));
%     Rx=round(points.p.x(12));
%     Dy=round((points.p.y(6)+points.p.y(10))/2);
%     %%highest point
%     p79x=(points.p.x(26)+points.p.x(27))/2;
%     p79y=(points.p.y(26)+points.p.y(27))/2;
%     p80x=(points.p.x(20)+points.p.x(21))/2;
%     p80y=(points.p.y(20)+points.p.y(21))/2;
%     p28x = points.p.x(28);
%     p28y = points.p.y(28);
%     p33x = points.p.x(33);
%     p33y = points.p.y(33);
%             
%     a1 = double(p28y-p79y)/double(p28x-p79x);
%     a2 = double(p33y-p80y)/double(p33x-p80x);
%     b1 = p79y - a1 * p79x;
%     b2 = p33y - a2 * p33x;
%     x0 = double(b2-b1)/double(a1-a2);
%     y0 = a1 * x0 + b1;
%     Uy=round(y0);
%     %%
%     im2=im(Uy:Dy,Lx:Rx);
    om=affin_eye(im,[points.p.x(32),points.p.y(32),points.p.x(37),points.p.y(37)],[120,150,170,150]);   
    imwrite(om,strout,'JPEG');
    
    k=k+1;
   
end
