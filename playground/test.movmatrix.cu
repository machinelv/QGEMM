#include <iostream>
#include <cstdlib>
#include <functional>
#include <vector>
#include <cstring>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
using namespace std;


#define MOVMAT(dst, src) asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" \
            : "=r"(dst) \
            : "r"(src));

#define LDMATRIX_BF16_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "l"(addr))


__global__ void test_movmatrix_x4(half* read, half* write, const bool trans_flag = true) {
    __align__(16) __shared__ half smem [8*8*4];
    uint32_t reg[4];
    uint32_t reg_trans[4];

    const int start_id = threadIdx.x * 8;
    (reinterpret_cast<float4*>(&smem[start_id]))[0] = (reinterpret_cast<float4*>(&read[start_id]))[0];
    const int smem_id = start_id;

    LDMATRIX_BF16_X4(reg[0], reg[1], reg[2], reg[3], &smem[smem_id]);

    if (trans_flag){
        for (int i{0}; i < 4; i++) {
            MOVMAT(reg_trans[i], reg[i]);
        }
    } else {
        for (int i{0}; i < 4; i++) {
            reg_trans[i] = reg[i];
        }
    }
    //reg --> global
    (reinterpret_cast<float*>(&write[start_id + 0]))[0] = (reinterpret_cast<float*>(&reg_trans[0]))[0];
    (reinterpret_cast<float*>(&write[start_id + 2]))[0] = (reinterpret_cast<float*>(&reg_trans[1]))[0];
    (reinterpret_cast<float*>(&write[start_id + 4]))[0] = (reinterpret_cast<float*>(&reg_trans[2]))[0];
    (reinterpret_cast<float*>(&write[start_id + 6]))[0] = (reinterpret_cast<float*>(&reg_trans[3]))[0];
}


__global__ void test_movmatrix(half* read, half* write, const bool trans_flag = true){
    __align__(16) __shared__ half smem [8*8];
    uint32_t reg;
    uint32_t reg_trans;

    const int start_id = threadIdx.x * 8;
    if (start_id < 64) {
        (reinterpret_cast<float4*>(&smem[start_id]))[0] = (reinterpret_cast<float4*>(&read[start_id]))[0];
    }
    const int smem_id = threadIdx.x / 16 * 8 + (threadIdx.x % 16) * 16;

    asm volatile("ldmatrix.sync.aligned.m8n8.x1.b16 {%0}, [%1];\n" 
                 : "=r"(reg) 
                 : "l"(&smem[smem_id]));

    if (trans_flag){
        MOVMAT(reg_trans, reg);
    } else {
        reg_trans = reg;
    }
    //reg --> global
    if (start_id < 256) {
        (reinterpret_cast<float*>(&write[start_id + 0]))[0] = (reinterpret_cast<float*>(&reg_trans))[0];
    }
}

template<typename T,int N> void init_mem(T (&ptr)[N]){
    for(int i=0;i<N;i++){
        ptr[i] = __float2half(static_cast<float>(i));
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
            cout << "Instruction: movmatrix.sync.aligned.m8n8.trans.b16(x4.no_trans)" << endl;
            test_movmatrix_x4<<<1, 32>>>(device_read, device_write, false);
        },
        [&]() {
            cout << "Instruction: movmatrix.sync.aligned.m8n8.trans.b16(x4.trans)" << endl;
            test_movmatrix_x4<<<1, 32>>>(device_read, device_write, true);
        },
        // [&]() {
        //     cout << "Instruction: movmatrix.sync.aligned.m8n8.trans.b16" << endl;
        //     test_movmatrix<<<1, 32>>>(device_read, device_write);
        // },
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