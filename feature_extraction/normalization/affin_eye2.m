%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%author:  Lingyun Wen
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

 %function im_output=affin_eye(im,eye_original_coordinate,eye_specified_coordinate, im_size)
function im_output=affin_eye(im,eye_original_coordinate,eye_specified_coordinate)

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
angle=-atan( (oy2-oy1)/(ox2-ox1) )+atan((ny2-ny1)/(nx2-nx1));

Trotation=[cos(angle) sin(angle) 0; -sin(angle) cos(angle) 0 ; 0 0 1 ];
scale_matrix=[scale 0 0; 0 scale 0; 0 0 1];
Transform_rule=Trotation*scale_matrix;

Transm=maketform('affine',Transform_rule);

[im0 x_delta1 y_delta1]=imtransform(im,Transm,'bilinear');

%[x11,y11]=tformfwd([(ox1+ox2)/2,(oy1+oy2)/2],Transm);
[x11,y11]=tformfwd([(ox1),(oy1)],Transm);
 x1=x11-x_delta1(1);
 y1=y11-y_delta1(1);

 x_a=(round(nx1-x1));
 y_a=(round(ny1-y1));

depth=size(im0,3);
delta=10;


im3=zeros(abs(y_a)+size(im0,1)+delta,abs(x_a)+size(im0,2)+delta,depth);
im3_y=[];

im3_x=[];

im0_y=[];
im0_x=[];

if y_a >0
    im3_y = 1+y_a:size(im0,1)+y_a;
    im0_y = 1:size(im0,1);
end
if y_a <=0
    im3_y = 1:size(im0,1)+y_a;
    im0_y = 1-y_a:size(im0,1);
end
if x_a >0 
    im3_x = 1+x_a:size(im0,2)+x_a;
    im0_x = 1:size(im0,2);
end
if x_a <= 0
    im3_x = 1:size(im0,2)+x_a;
    im0_x = 1-x_a:size(im0,2);
end

im3(im3_y,im3_x,1:depth)=im0(im0_y,im0_x,1:depth);

im_output=uint8(im3);


    





