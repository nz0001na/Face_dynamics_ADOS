clear all;
clc;



dt = 1;
while dt <= 6
    
in_path = sprintf('F:\\Wen\\WhiteWomen\\newPoints\\%d',dt);



list = dir([in_path,'\*.mat']);

k=1;


while k <= length(list)   
  data = load([in_path '\' list(k).name], '-ASCII');
  
  i = 1;
  
  while i <= size(data,1)
      if i > 1
        MX = [MX,data(i,1)];
        MY = [MY,data(i,2)];     
      else
        MX = [data(1,1)];
        MY = [data(1,2)];   
      end
  i=i+1;
  end
      
      if k==1&&dt==1
        SX = MX;
        SY = MY;
      else 
        SX = [SX;MX];
        SY = [SY;MY];
      
      end  
  k = k+1;
end
dt = dt + 1;
end

fname = 'F:\Wen\WhiteWomen\newPoints\total.mat';
fid = fopen(fname,'wt');

t = 1;
while t <= size(SX,2)
  fprintf(fid,' %d %d %d %d\n', mean(SX(:,t)),var(SX(:,t)),mean(SY(:,t)),var(SY(:,t)));
 
  t = t+1;
end
fclose(fid);