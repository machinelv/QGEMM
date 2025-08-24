
all: test.ldmatrix.layout

test.ldmatrix.layout: test.ldmatrix.layout.cu
	nvcc -O3 --gpu-architecture sm_80 -o test.ldmatrix.layout test.ldmatrix.layout.cu