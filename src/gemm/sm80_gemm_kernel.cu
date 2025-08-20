


void bf16_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c) 
{
    const size_t m = a.size(0);
    const size_t n = b.size(0);
    const size_t k = a.size(1); 

    gemm_kernel:: select_BLOCK_TILE_SIZE_M(static_cast<bfloat16_t*>(a.data_ptr()), static_cast<bfloat16_t*>(b.data_ptr()), 
    as.data_ptr<float>(), bs.data_ptr<float>(), static_cast<bfloat16_t*>(c.data_ptr()), m, n, k);
}

void int8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c) 
{
    const size_t m = a.size(0);
    const size_t n = b.size(0);
    const size_t k = a.size(1); 

    select_BLOCK_TILE_SIZE_M(static_cast<float8_t*>(a.data_ptr()), static_cast<float8_t*>(b.data_ptr()), 
    as.data_ptr<float>(), bs.data_ptr<float>(), static_cast<bfloat16_t*>(c.data_ptr()), m, n, k);
}