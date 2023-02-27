%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%author: Xiaolong Wang; Lingyun Wen
%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%description: according to the location of eyes
%%%%%%%%%%%%%%%             scale and rotation, move the eye location to the
%%%%%%%%%%%%%%%             desired position (alignment eyes)
%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%improvement: adjust automatically the size of the output images
%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%input: im: the original image
%%%%%%%%%%%%%%%       eye_original_coordinate: eye position in original image (form: [lefteye_x, lefteye_y, righteye_x, righteye_y])
%%%%%%%%%%%%%%%       eye_specified_coordinate: eye position in the result image (form: [lefteye_x, lefteye_y, righteye_x, righteye_y])
%%%%%%%%%%%%%%%       imsize: the size of the result image: (form: [height width depth])
%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%output: result image  

 function im_output=affin_eye(im,eye_original_coordinate,eye_specified_coordinate, im_size)

%%%%%%%%%%%%%%%example:   im_output=affin_eye(im,[e11,e12,e21,e22],[100,80,130,80],[200,200,3]);

ox1=eye_original_coordinate(1);
oy1=eye_original_coordinate(2);
ox2=eye_original_coordinate(3);
oy2=eye_original_coordinate(4);

nx1=eye_specified_coordinate(1);
nx2=eye_specified_coordinate(3);
ny1=eye_specified_coordinate(2);
ny2=eye_specified_coordinate(4);

scale=sqrt(((nx2-nx1)^2+(ny2-ny1)^2)/((ox2-ox1)^2+(oy2-oy1)^2) );%%%%%the distance of eyes
%angle=-atan( (oy2-oy1)/(ox2-ox1) );
angle=-atan( (oy2-oy1)/(ox2-ox1) );
% if scale >1
%     disp('The image is enlargeing!')
%     return;
% end

Trotation=[cos(angle) sin(angle) 0; -sin(angle) cos(angle) 0 ; 0 0 1 ];
scale_matrix=[scale 0 0; 0 scale 0; 0 0 1];
Transform_rule=Trotation*scale_matrix;
Transm=maketform('affine',Transform_rule);

[im0 x_delta1 y_delta1]=imtransform(im,Transm,'bilinear');


[x11,y11]=tformfwd([ox1,oy1],Transm);
[x21,y21]=tformfwd([ox2,oy2],Transm);

% if abs((x21-x11) -( nx2-nx1)) > 5
%     disp('should define smaller eye distance!');
%     return;
% end
 x1=x11-x_delta1(1);
 y1=y11-y_delta1(1);

%  x_a=round(nx1-x1);
%  y_a=round(ny1-y1);
x_a=round(nx1-x1);
y_a=round(ny1-y1);

% move_matrix=[1 0 0; 0 1 0; x_a y_a 1];

% move_matrix=[1 0 0; 0 1 0;  y_a x_a 1];
% 
% 
% 
% 
x_max = size(im0,1);
y_max = size(im0,2);

x_max_new=(x_max+y_a-1);
y_max_new=(y_max+x_a-1);
% 
%  
% 
% tm=move_matrix;
% tform3 = maketform('affine', tm);
% [im2 x_delta2 y_delta2]=imtransform(im1,tform3,'XData',[1,200],'YData',[1,200],'FillValue',0.5);
% %%
% [im2 x_delta2 y_delta2]=imtransform(im1,tform3,'XData',[1,x_max_new],'YData',[1,y_max_new],'FillValue',0.5);% imtransform(im,tform2,'XData',[1 size(im,1)],'YData',[1 size(im,2)]);

% if im_size(1) < y_max_new || im_size(2) < x_max_new
%     disp('should define a larger result image size!')
%     return;
% end
% if y_a < 1 
%     disp('should define a larger eye position y axis!');
%     return;
% end
% if x_a <1
% disp('should define a larger eye position x axis!');
%     return;
% end
% % delta_y = im_size(1)-y_max_new;
% delta_x = im_size(2)-x_max_new;
 depth = im_size(3);

% im3 = uint8(zeros(y_max_new+delta_y,x_max_new+delta_x,depth)); 
% 
%  im3(1:y_max_new,1:x_max_new,1:depth)=im3(1:y_max_new,1:x_max_new,1:depth)+im2; 
% im3(1:size(im2,1),1:size(im2,2),1:depth)=im3(1:size(im2,1),1:size(im2,2),1:depth)+im2; 
im3 = uint8(zeros(im_size));
y_a=floor(y_a);
if y_a<1
    y_a = 1;
end
[imm, imn, imz]=size(im0);
x_max_new = y_a+imm-1;
x_a = floor(x_a);
if x_a<1
    x_a = 1;
end
y_max_new = x_a+imn-1;
im3(y_a:x_max_new,x_a:y_max_new,1:depth)=im3(y_a:x_max_new,x_a:y_max_new,1:depth)+im0;
im_output=im3;
    





