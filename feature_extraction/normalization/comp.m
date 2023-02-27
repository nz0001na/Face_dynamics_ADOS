clear all;
clc;


fname = ('F:\\Wen\\WhiteWomen\\newPoints\\COMP.txt');
fid = fopen(fname,'w');

t = 1;
while t <= 32       
  fdata = sprintf('F:\\Wen\\WhiteWomen\\newPoints\\comp%d.mat',t);
  fd = load(fdata,'-ASCII');
  fprintf(fid,'Point %d \r\n',t);
  l = 1;
  while l <= size(fd,1);
  fprintf(fid, '%3.2f %3.2f %3.2f %3.2f\r\n', fd(l,1),fd(l,2),fd(l,3),fd(l,4));
  l = l + 1;
  end
  t = t + 1;
end
fclose(fid);
 
