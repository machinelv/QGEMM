

all: ldmatrix movmatrix

ldmatrix: test.ldmatrix.layout.cu
	nvcc -O3 --gpu-architecture ${ARCH} -o test.ldmatrix.layout test.ldmatrix.layout.cu

movmatrix: test.movmatrix.cu
	nvcc -O3 --gpu-architecture ${ARCH} -o test.movmatrix test.movmatrix.cu