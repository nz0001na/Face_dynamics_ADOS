clear all;
clc;

in_path = ('F:\Wen\WhiteWomen\points\6');
out_path = ('F:\Wen\WhiteWomen\newPoints\6');

%list = dir([im_path,'\male','\*.bmp']);
list = dir([in_path,'\*.mat']);

k=1;
while k<=length(list)   
  data =load([in_path '\' list(k).name]);
  
  ox1 = data.A{2}(15,1);
  oy1 = data.A{2}(15,2);
  ox2 = data.A{2}(20,1);
  oy2 = data.A{2}(20,2);
  
  nx1 = 160;
  ny1 = 225;
  nx2 = 220;
  ny2 = 225;
  
  scale=sqrt(((nx2-nx1)^2+(ny2-ny1)^2)/((ox2-ox1)^2+(oy2-oy1)^2) );%%%%%the distance of eyes
  angle=-atan( (oy2-oy1)/(ox2-ox1) );

  Trotation=[cos(angle) sin(angle) 0; -sin(angle) cos(angle) 0 ; 0 0 1 ];
  scale_matrix=[scale 0 0; 0 scale 0; 0 0 1];
  Transform_rule=Trotation*scale_matrix;
  
  XM = 400;
  XY =480; %size of original picture
  
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
  while(i <= size(data.A{2},1))
      d_x = data.A{2}(i,1);
      d_y = data.A{2}(i,2);
      TM = [d_x d_y 1]*Transform_rule*move_matrix1*move_matrix2;      
      fprintf(fid,'%f %f\n',TM(1),TM(2));
      i = i+1;
  end
  fclose(fid);
  k = k+1;
end
