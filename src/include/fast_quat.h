#pragma once

template<typename FP16T>
__host__ __device__ FP16T fast_f32tob16(float f) {
    union {
        float fp32;
        unsigned int u32;
    } u = {f};
    u.u32 += 0x7fff + ((u.u32 >> 16) & 1);
    auto ret = u.u32 >> 16;
    return reinterpret_cast<FP16T &>(ret);
}