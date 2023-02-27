

function [newim, corner, gflag] = getfacefromframe1(im, model,oldCorner)
    oldCorner
    ox=oldCorner(1);
    oy=oldCorner(2);
    
    regionX = 220;
    regionY = 220;
    [m n z]=size(im);
   
    if ox <1
        ox=1;
    end
    if oy <1
        oy=1;
    end
    dx=ox-20;
    dy=oy-20;
    if dx <1
        dx=1;
    end
    if dy <1
        dy=1;
    end
    if ox+regionX>n
        regionX = n-ox-1;
    end
    if oy+regionY>m
        regionY = m-oy-1;
    end
    
    oldRegion = im(dy:oy+regionY,dx:ox+regionX,:);
    bs = detect(oldRegion, model, model.thresh);
 
   flagRegion=true;
    if isempty(bs) || size(bs(1).xy,1)<68
        flagRegion=false;
        bs = detect(im, model, model.thresh);
         if isempty(bs) || size(bs(1).xy,1)<68
             newim=[];
             corner=[];
             gflag=false;             
             return;
         end
    
    end
   

        right_eye.x=((bs(1).xy(11,1)+bs(1).xy(11,3))/2+(bs(1).xy(12,1)+bs(1).xy(12,3))/2+(bs(1).xy(13,1)+bs(1).xy(13,3))/2+(bs(1).xy(14,1)+bs(1).xy(14,3))/2)/4;
        right_eye.y=((bs(1).xy(11,2)+bs(1).xy(11,4))/2+(bs(1).xy(12,2)+bs(1).xy(12,4))/2+(bs(1).xy(13,2)+bs(1).xy(13,4))/2+(bs(1).xy(14,2)+bs(1).xy(14,4))/2)/4;
        left_eye.x=((bs(1).xy(22,1)+bs(1).xy(22,3))/2+(bs(1).xy(23,1)+bs(1).xy(23,3))/2+(bs(1).xy(24,1)+bs(1).xy(24,3))/2+(bs(1).xy(25,1)+bs(1).xy(25,3))/2)/4;
        left_eye.y=((bs(1).xy(22,2)+bs(1).xy(22,4))/2+(bs(1).xy(23,2)+bs(1).xy(23,4))/2+(bs(1).xy(24,2)+bs(1).xy(24,4))/2+(bs(1).xy(25,2)+bs(1).xy(25,4))/2)/4;
       
%back to im

       
          
    
    
    
    
 %fprintf('Detection took %.1f seconds\n', dettime);
  if flagRegion
      
       sx=round(dx+right_eye.x-60);
        lx=round(dx+left_eye.x+60);
        sy=round(dy+right_eye.y-60);
        ly=round(dy+left_eye.y+150);
        if sx<1
            sx=1;
        end
        if lx>n
            lx=n;
        end
        if sy<1
            sy=1;
        end
        if ly>m
           ly=m;
        end
  else
       sx=round(right_eye.x-60);
        lx=round(left_eye.x+60);
        sy=round(right_eye.y-60);
        ly=round(left_eye.y+150);
        if sx<1
            sx=1;
        end
        if lx>n
            lx=n;
        end
        if sy<1
            sy=1;
        end
        if ly>m
           ly=m;
        end
      
     
  end
  subim = im(sy:ly,sx:lx,:);
  
 om = affin_eye3((subim),double([right_eye.x,right_eye.y,left_eye.x,left_eye.y]),[60,80,120,80],[600,600,3]);
 newim = imresize(om(40:180,30:150,:),[60,60]);
 
       
corner = [sx,sy];
gflag=true;
    
 
