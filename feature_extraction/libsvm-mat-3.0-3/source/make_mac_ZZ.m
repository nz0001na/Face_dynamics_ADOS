% This make.m is used under Mac (Linux?)

% add -largeArrayDims on 64-bit machines
% remove them on 32-bit machines

mex -largeArrayDims -O -c svm.cpp
mex -largeArrayDims -O -c svm_model_matlab.c
mex -largeArrayDims -O svmtrain.c svm.o svm_model_matlab.o
mex -largeArrayDims -O svmpredict.c svm.o svm_model_matlab.o

mex -largeArrayDims -O read_sparse.c

mex -largeArrayDims -O libsvmread.c
mex -largeArrayDims -O libsvmwrite.c