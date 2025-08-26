all:
	nvcc -DUSE_CUDA -DLOCAL_TEST -I../../include -I../include -o test_transpose transpose.cu -arch=sm_80 --std=c++17