%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%author: xiaolong wang
%%%%%%%%%%%%%%%data:0408-2011
%%%%%%%%%%%%%%%description: according to the location of eyes
%%%%%%%%%%%%%%%             scale and rotation,move the eye location to the
%%%%%%%%%%%%%%%             same coordinates
%%%%%%%%%%%%%%%version: 1.0 ---0408
%%%%%%%%%%%%%%%improvement: adjust automatically the size of the output images

 function im_output=affin_eye(im,eye_original_coordiante,eye_specified_coordinate)

% eye_original_coordiante=
% eye_specified_coordinate=[50,100,150,100];


ox1=eye_original_coordiante(1);
ox2=eye_original_coordiante(3);
oy1=eye_original_coordiante(2);
oy2=eye_original_coordiante(4);

nx1=eye_specified_coordinate(1);
nx2=eye_specified_coordinate(3);
ny1=eye_specified_coordinate(2);
ny2=eye_specified_coordinate(4);
data=((nx2-nx1)^2+(ny2-ny1)^2)*1.0/((ox2-ox1)^2+(oy2-oy1)^2);
scale=sqrt(double(data));%%%%%the distance of eyes
angle=-atan( (oy2-oy1)/(ox2-ox1) );

Trotation=[cos(angle) sin(angle) 0; -sin(angle) cos(angle) 0 ; 0 0 1 ];
scale_matrix=[scale 0 0; 0 scale 0; 0 0 1];
Transform_rule=Trotation*scale_matrix;
Transm=maketform('affine',Transform_rule);

[im1 x_delta1 y_delta1]=imtransform(im,Transm);

[x11,y11]=tformfwd([ox1,oy1],Transm);

x1=x11-x_delta1(1);
y1=y11-y_delta1(1);

x_a=round(nx1-x1);
y_a=round(ny1-y1);

move_matrix=[1 0 0; 0 1 0; x_a y_a 1];

if 1

%     Transm1=maketform('affine',move_matrix);
%     [im2 x_del y_del]=imtransform(im1,Transm1);
%     im_output=im2;
%     
x_max=size(im1,1);
y_max=size(im1,2);
   x_a=nx1-x1;
   y_a=ny1-y1;

%    x_max_new=x_max+x_a;
%    y_max_new=y_max+x_a;

% x_max_new=300;
x_max_new=200;

% x_max_new=300;
y_max_new=200;

move_matrix=[1 0 0; 0 1 0; x_a y_a 1];

tform3 = maketform('affine', move_matrix);

[im2 x_delta2 y_delta2]=imtransform(im1,tform3,'XData',[1,x_max_new],'YData',[1,y_max_new],'FillValue',0.5);% imtransform(im,tform2,'XData',[1 size(im,1)],'YData',[1 size(im,2)]);

    
    im_output=im2;
    
else    
    
se = translate(strel(1), [floor(y_a) floor(x_a)] );

im_output=imdilate(im1,se);

end




