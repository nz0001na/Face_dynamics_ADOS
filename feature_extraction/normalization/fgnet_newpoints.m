clear all;
clc;

in_path = ('F:\Wen\lab\crossage\fgnet\points_mat\points');
out_path = ('F:\Wen\lab\crossage\fgnet\points_normalization');

%list = dir([im_path,'\male','\*.bmp']);
list = dir([in_path,'\*.mat']);

k=1;
while k<=length(list)   
  data =load([in_path '\' list(k).name]);
  
  ox1 = data.p.x(32);
  oy1 = data.p.y(32);
  ox2 = data.p.x(37);
  oy2 = data.p.y(37);
  %130,150,160,150
  nx1 = 120;
  ny1 = 150;
  nx2 = 170;
  ny2 = 150;
  
  scale=sqrt(((nx2-nx1)^2+(ny2-ny1)^2)/((ox2-ox1)^2+(oy2-oy1)^2) );%%%%%the distance of eyes
  angle=-atan( (oy2-oy1)/(ox2-ox1) );

  Trotation=[cos(angle) sin(angle) 0; -sin(angle) cos(angle) 0 ; 0 0 1 ];
  scale_matrix=[scale 0 0; 0 scale 0; 0 0 1];
  Transform_rule=Trotation*scale_matrix;
  
  s=list(k).name;
  s=s(1:(length(s)-4));
  s(ismember(s,'a'))='A';
  im=imread(['F:\Wen\lab\crossage\fgnet\FGNET\',s,'.JPG']);
  [XM XY d3]=size(im);
  %XM = 400;
  %XY =480; %size of original picture
  
  Md = [1 1 1;XM 1 1;XM XY 1;1 XY 1]*Transform_rule;
  x_delta1 = min(Md(:,1));
  y_delta1 = min(Md(:,2));
  
  move_matrix1 = [1 0 0; 0 1 0; -x_delta1 -y_delta1 1];

  MTR = [ox1,oy1,1]*Transform_rule;
  x_delta2 = nx1 - (MTR(1)-x_delta1);
  y_delta2 = ny1 - (MTR(2)-y_delta1);
  
  move_matrix2 = [1 0 0; 0 1 0; x_delta2 y_delta2 1];
  
  i = 1;
  fname = strcat(out_path, '\', list(k).name);
  fid = fopen(fname,'wt');
  while(i <= length(data.p.x))
      d_x = data.p.x(i);
      d_y = data.p.y(i);
      TM = [d_x d_y 1]*Transform_rule*move_matrix1*move_matrix2;      
      fprintf(fid,'%f %f\n',TM(1),TM(2));
      i = i+1;
  end
  fclose(fid);
  k = k+1;
end
