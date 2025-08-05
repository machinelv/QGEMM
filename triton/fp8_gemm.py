import torch
import triton
import triton.language as tl    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@staticmethod
@triton.jit
def _triton_matmul_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_dm, stride_dn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """Triton kernel for GEMM operation"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute block offsets
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate over blocks of K
    for k in range(0, K, BLOCK_K):
        # Load A block
        a_block_ptr = a_ptr + m_start * stride_am + k * stride_ak
        a_block = tl.load(a_block_ptr, mask=tl.arange(0, BLOCK_M)[:, None] < M - m_start, other=0.0)
        
        # Load B block
        b_block_ptr = b_ptr + k * stride_bk + n_start * stride_bn
        b_block = tl.load(b_block_ptr, mask=tl.arange(0, BLOCK_N)[None, :] < N - n_start, other=0.0)
        
        # Compute matmul for this block
        acc += tl.dot(a_block, b_block)
    
    # Load C block
    c_block_ptr = c_ptr + m_start * stride_cm + n_start * stride_cn
    c_block = tl.load(c_block_ptr, mask=(tl.arange(0, BLOCK_M)[:, None] < M - m_start) & 
                                        (tl.arange(0, BLOCK_N)[None, :] < N - n_start), other=0.0)
    
    # Add C to the result
    result = acc + c_block
    
    # Store the result
    d_block_ptr = d_ptr + m_start * stride_dm + n_start * stride_dn
    tl.store(d_block_ptr, result, mask=(tl.arange(0, BLOCK_M)[:, None] < M - m_start) & 
                                    (tl.arange(0, BLOCK_N)[None, :] < N - n_start))

def _triton_gemm(self, A, B, C, precision: str):
    """Use Triton kernel for GEMM: D = AB + C"""
    # Convert to desired precision
    if precision == PrecisionType.FP32:
        dtype = torch.float32
    elif precision == PrecisionType.FP16:
        dtype = torch.float16
    elif precision == PrecisionType.BF16:
        dtype = torch.bfloat16
    elif precision == PrecisionType.INT8:
        dtype = torch.int8
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    
    A = A.to(dtype)
    B = B.to(dtype)
    C = C.to(dtype)
    
    M, K = A.shape
    K, N = B.shape
    
    # Create output tensor
    D = torch.empty((M, N), device=A.device, dtype=dtype)
    
    # Define the grid
    grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    
    # Launch the Triton kernel
    self._triton_matmul_kernel[grid](
        A, B, C, D,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        D.stride(0), D.stride(1),
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32
    )
    
    return D