function [num] = getNum(outCov)
f1 = outCov(1,1);
f2 = outCov(1,2);
f3 = outCov(1,3);
f5 = outCov(2,2);
f6 = outCov(2,3);
f9 = outCov(3,3);

f=[f1,f2,f3,f5,f6,f9];
a = f;
m=median(f);
id1 = find(f>=m);
id2 = find(f<m);
a(id1) = 1;
a(id2) = 0;

s=num2str(a);
num = bin2dec(s);