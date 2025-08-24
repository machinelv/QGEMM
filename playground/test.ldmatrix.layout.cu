/***
 * 
 * 
 */

#include <iostream>
#include <cstdlib>
#include <functional>
#include <vector>
#include <cstring>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
using namespace std;

#define SMEM_M 64
#define SMEM_N 64

__global__ void test_ldmatrix_x4(half* read, half* write, const bool trans_flag=true) {
    __align__(16) __shared__ half smem [8*8*4];
    uint32_t reg[4];

    const int start_id = threadIdx.x * 8;
    (reinterpret_cast<float4*>(&smem[start_id]))[0] = (reinterpret_cast<float4*>(&read[start_id]))[0];
    const int smem_id = threadIdx.x / 16 * 8 + (threadIdx.x % 16) * 16;

    if (trans_flag == true){
        //use ldmatrix.trans
        asm("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];\n\t"
            : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
            : "l"(&smem[smem_id]));
    } else {
        //use ldmatrix.trans
        asm("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];\n\t"
            : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
            : "l"(&smem[smem_id]));
    }

    //reg --> global
    (reinterpret_cast<float*>(&write[start_id + 0]))[0] = (reinterpret_cast<float*>(&reg[0]))[0];
    (reinterpret_cast<float*>(&write[start_id + 2]))[0] = (reinterpret_cast<float*>(&reg[1]))[0];
    (reinterpret_cast<float*>(&write[start_id + 4]))[0] = (reinterpret_cast<float*>(&reg[2]))[0];
    (reinterpret_cast<float*>(&write[start_id + 6]))[0] = (reinterpret_cast<float*>(&reg[3]))[0];
}


__global__ void test_ldmatrix_x2(half* read, half* write, const bool trans_flag=true){
    __align__(16) __shared__ half smem [8*8*4];
    uint32_t reg[4];

    const int start_id = threadIdx.x / 16 * 8 + (threadIdx.x % 16) * 16;
    (reinterpret_cast<float4*>(&smem[start_id]))[0] = (reinterpret_cast<float4*>(&read[start_id]))[0];

    const int smem_id = threadIdx.x / 16 * 8 + (threadIdx.x % 16) * 16;

    if (trans_flag) {
        //use ldmatrix.trans
        asm("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0, %1}, [%2];\n\t"
            : "=r"(reg[0]), "=r"(reg[1])
            : "l"(&smem[smem_id]));
        asm("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0, %1}, [%2];\n\t"
            : "=r"(reg[2]), "=r"(reg[3])
            : "l"(&smem[smem_id + 8]));
    } else {
        asm("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];\n\t"
            : "=r"(reg[0]), "=r"(reg[1])
            : "l"(&smem[smem_id]));
        asm("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];\n\t"
            : "=r"(reg[2]), "=r"(reg[3])
            : "l"(&smem[smem_id + 8]));
    }

    //reg --> global
    (reinterpret_cast<float*>(&write[smem_id + 0]))[0] = (reinterpret_cast<float*>(&reg[0]))[0];
    (reinterpret_cast<float*>(&write[smem_id + 2]))[0] = (reinterpret_cast<float*>(&reg[1]))[0];
    (reinterpret_cast<float*>(&write[smem_id + 4]))[0] = (reinterpret_cast<float*>(&reg[2]))[0];
    (reinterpret_cast<float*>(&write[smem_id + 6]))[0] = (reinterpret_cast<float*>(&reg[3]))[0];
    
}

template<typename T,int N> void init_mem(T (&ptr)[N]){
    for(int i=0;i<N;i++){
        ptr[i] = i;
    }
}

int main(){
    //FP16, (8*8) * 4个
    half host_send[8*8*4];
    half host_receive[8*8*4];
    half* device_read;
    half* device_write;
    init_mem(host_send);

    constexpr int data_size = 8*8*4 * sizeof(half);
    cudaMalloc(&device_read, data_size);
    cudaMalloc(&device_write, data_size);
    
    // auto print_reg = [&]() {
    //     for(int i=0;i<32;i++){
    //         cout << "thread " << i << " holds: ";
    //         for(int j=0;j<8;j+=2){
    //             cout << "(" << j/2 << ")" << " " << __half2float(host_receive[i*8 + j]) << ", "  << __half2float(host_receive[i*8 + (j + 1)]) << ",";
    //         }
    //         cout << endl;
    //     }
    // };

    vector<function<void()>> kernels = {
        [&]() {
            cout << "Instruction: ldmatrix.sync.aligned.m8n8.x4.trans.b16" << endl;
            test_ldmatrix_x4<<<1, 32>>>(device_read, device_write, true);
        },
        [&]() {
            cout << "Instruction: ldmatrix.sync.aligned.m8n8.x4.b16" << endl;
            test_ldmatrix_x4<<<1, 32>>>(device_read, device_write, false);
        },
        [&]() {
            cout << "Instruction: ldmatrix.sync.aligned.m8n8.x2.trans.b16" << endl;
            test_ldmatrix_x2<<<1, 32>>>(device_read, device_write, true);
        },
        [&]() {
            cout << "Instruction: ldmatrix.sync.aligned.m8n8.x2.b16" << endl;
            test_ldmatrix_x2<<<1, 32>>>(device_read, device_write, false);
        }
    };

    for (auto func: kernels) {
        cout << "====================================================================================================" << endl;
        // 每次测试前清零接收数组
        memset(host_receive, 0, data_size);
        cudaMemcpy(device_read, host_send, data_size, cudaMemcpyHostToDevice);
        cudaMemset(device_write, 0, data_size);
        func();
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            cerr << "CUDA error: " << cudaGetErrorString(err) << endl;
        }
        cudaDeviceSynchronize();
        cudaMemcpy(host_receive, device_write, data_size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        // print_reg();
        for(int i=0;i<32;i++){
            cout << "thread " << i << " holds: ";
            for(int j=0;j<8;j+=2){
                cout << "(" << j/2 << ")" << " " << __half2float(host_receive[i*8 + j]) << ", "  << __half2float(host_receive[i*8 + (j + 1)]) << ",";
            }
            cout << endl;
        }
        cout << "====================================================================================================" << endl;
    }

    cudaFree(device_read);
    cudaFree(device_write);
    return 0;
}