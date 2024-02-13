#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h> 
#include <cuda_bf16.h>
#include "dispatch_utils.h"
#include "reduction_utils.cuh"

namespace vllm {


#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800)
#define ENABLE_BF16
#endif

template <typename T_OUT, typename T_IN>
__device__ inline T_OUT cuda_cast(T_IN val)
{
    return val;
}

//half -> float
template <>
__device__ inline float cuda_cast<float, __half>(__half val)
{
    return __half2float(val);
}

//float -> half
template <>
__device__ inline __half cuda_cast<__half, float>(float val)
{
    return __float2half(val);
}

#ifdef ENABLE_BF16
//__nv_bfloat16 -> float
template <>
__device__ inline float cuda_cast<float, __nv_bfloat16>(__nv_bfloat16 val)
{
    return __bfloat162float(val);
}

//float -> __nv_bfloat16
template <>
__device__ inline __nv_bfloat16 cuda_cast<__nv_bfloat16, float>(float val)
{
    return __float2bfloat16(val);
}
#endif


template<typename T> struct packed_type;
template <>          struct packed_type<__half>          { using type = __half2; };
template <>          struct packed_type<__nv_bfloat16>  { using type = __nv_bfloat162; };


template<typename T>
//typename std::enable_if<std::is_same<T, __half>::value || std::is_same<T, __bfloat16>::value, void>::type
void __global__  rms_norm_kernel_e8(float4 *output, const float4 *input,
				       const float4 *weight, const int n, float epsilon, bool use_shmem) {
  
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean;
  float local_sums[1] = {0.0f};
  const int n_8 = n / 8;
  int offset = m_idx * n_8;
  input += offset;
  output += offset;
  
  //if T is __half, packed_t is half2
  //if T is __nv_bfloat16, packed_t is __nv_bfloat162
  using packed_t = typename packed_type<T>::type;

  extern __shared__ __align__(sizeof(float)) char _shmem[];
  float4 * shmem = reinterpret_cast<float4*>(_shmem);

  for (int index = tid; index < n_8; index += bdimx) {
    const float4 local_val = input[index];
    if (use_shmem) {
      shmem[index] = local_val;
    }
    const packed_t *h1 = (packed_t *)&local_val.x;
    const packed_t *h2 = (packed_t *)&local_val.y;
    const packed_t *h3 = (packed_t *)&local_val.z;
    const packed_t *h4 = (packed_t *)&local_val.w;
    local_sums[0] += cuda_cast<float>(h1->x) * cuda_cast<float>(h1->x) +
                     cuda_cast<float>(h1->y) * cuda_cast<float>(h1->y) +
                     cuda_cast<float>(h2->x) * cuda_cast<float>(h2->x) +
                     cuda_cast<float>(h2->y) * cuda_cast<float>(h2->y) +
                     cuda_cast<float>(h3->x) * cuda_cast<float>(h3->x) +
                     cuda_cast<float>(h3->y) * cuda_cast<float>(h3->y) +
                     cuda_cast<float>(h4->x) * cuda_cast<float>(h4->x) +
                     cuda_cast<float>(h4->y) * cuda_cast<float>(h4->y);
  }


  local_sums[0] = blockReduceSum<float>(local_sums[0]);

  
  if (threadIdx.x == 0) {
    s_mean = rsqrtf(local_sums[0] / n + epsilon);
  }
  __syncthreads();

  for (int index = tid; index < n_8; index += bdimx) {
    const float4 local_val = use_shmem?shmem[index]:input[index];
    const float4 weight_val = weight[index];

    const packed_t *l1 = (packed_t *)&local_val.x;
    const packed_t *l2 = (packed_t *)&local_val.y;
    const packed_t *l3 = (packed_t *)&local_val.z;
    const packed_t *l4 = (packed_t *)&local_val.w;

    const packed_t *g1 = (packed_t *)&weight_val.x;
    const packed_t *g2 = (packed_t *)&weight_val.y;
    const packed_t *g3 = (packed_t *)&weight_val.z;
    const packed_t *g4 = (packed_t *)&weight_val.w;

    float4 tmp;
    packed_t *h1 = (packed_t *)&tmp.x;
    packed_t *h2 = (packed_t *)&tmp.y;
    packed_t *h3 = (packed_t *)&tmp.z;
    packed_t *h4 = (packed_t *)&tmp.w;

    h1->x = cuda_cast<T>(cuda_cast<float>(l1->x) * s_mean * cuda_cast<float>(g1->x));
    h1->y = cuda_cast<T>(cuda_cast<float>(l1->y) * s_mean * cuda_cast<float>(g1->y));
    h2->x = cuda_cast<T>(cuda_cast<float>(l2->x) * s_mean * cuda_cast<float>(g2->x));
    h2->y = cuda_cast<T>(cuda_cast<float>(l2->y) * s_mean * cuda_cast<float>(g2->y));
    h3->x = cuda_cast<T>(cuda_cast<float>(l3->x) * s_mean * cuda_cast<float>(g3->x));
    h3->y = cuda_cast<T>(cuda_cast<float>(l3->y) * s_mean * cuda_cast<float>(g3->y));
    h4->x = cuda_cast<T>(cuda_cast<float>(l4->x) * s_mean * cuda_cast<float>(g4->x));
    h4->y = cuda_cast<T>(cuda_cast<float>(l4->y) * s_mean * cuda_cast<float>(g4->y));

    output[index] = tmp;
  }
}

// TODO(woosuk): Further optimize this kernel.
template<typename scalar_t>
__global__ void rms_norm_kernel(
  scalar_t* __restrict__ out,             // [..., hidden_size]
  const scalar_t* __restrict__ input,     // [..., hidden_size]
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size,
  bool use_shmem
  ) {
  __shared__ float s_variance;
  float variance = 0.0f;
  extern __shared__ __align__(sizeof(float)) char _shmem[];
  scalar_t* shmem = reinterpret_cast<scalar_t*>(_shmem);


  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = input[blockIdx.x * hidden_size + idx];
    if (use_shmem) {
      shmem[idx] = x;
    }
    variance += x * x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = use_shmem?shmem[idx]:input[blockIdx.x * hidden_size + idx];
    out[blockIdx.x * hidden_size + idx] = ((scalar_t) (x * s_variance)) * weight[idx];
  }
}

// TODO: Further optimize this kernel.
template<typename scalar_t>
__global__ void fused_add_rms_norm_kernel(
  scalar_t* __restrict__ input,           // [..., hidden_size]
  scalar_t* __restrict__ residual,        // [..., hidden_size]
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size,
  bool use_shmem
  ) {
  __shared__ float s_variance;
  float variance = 0.0f;
  extern __shared__ __align__(sizeof(float)) char _shmem[];
  scalar_t* shmem = reinterpret_cast<scalar_t*>(_shmem);

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float) input[blockIdx.x * hidden_size + idx];
    x += (float) residual[blockIdx.x * hidden_size + idx];
    variance += x * x;
    if (use_shmem) {
      shmem[idx] = x;
    }
    residual[blockIdx.x * hidden_size + idx] = (scalar_t) x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = use_shmem?shmem[idx]:residual[blockIdx.x * hidden_size + idx];
    input[blockIdx.x * hidden_size + idx] = ((scalar_t) (x * s_variance)) * weight[idx];
  }
}

} // namespace vllm


inline int getMaxSharedMemoryPerBlock(const torch::Tensor& input) {
  int max_shmem_size;
  cudaDeviceGetAttribute(&max_shmem_size, cudaDevAttrMaxSharedMemoryPerBlock, input.device().index());
  return max_shmem_size;
}


void rms_norm(
  torch::Tensor& out,      // [..., hidden_size]
  torch::Tensor& input,    // [..., hidden_size]
  torch::Tensor& weight,   // [hidden_size]
  float epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  VLLM_DISPATCH_FLOATING_TYPES(
    input.scalar_type(),
    "rms_norm_kernel",
    [&] {
      bool use_shmem = true;
      //estimate the shared memory size
      int shmem_size = hidden_size * sizeof(scalar_t);

      if (shmem_size > getMaxSharedMemoryPerBlock(input)) {
        shmem_size = 0;
        use_shmem = false;
      }

      dim3 grid(num_tokens);
    
      //if hidden_size is multiple of 8, use vector type
      if (hidden_size % 8 == 0 && (std::is_same<scalar_t, at::Half>::value 
      #ifdef ENABLE_BF16
      || std::is_same<scalar_t, at::BFloat16>::value
      #endif
      )) {

          dim3 block(min(1024, (hidden_size / 8 + 31) / 32 * 32));

          if (input.scalar_type() == at::ScalarType::Half) {
                if (shmem_size >=48 * 1024) {
                    VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(vllm::rms_norm_kernel_e8<__half>, shmem_size);
                }
                 vllm::rms_norm_kernel_e8<__half><<<grid, block, shmem_size, stream>>>(
              (float4*)out.data_ptr(), (float4*)input.data_ptr(), (float4*)weight.data_ptr(), hidden_size, epsilon, use_shmem);
          }
          #ifdef ENABLE_BF16
          else if (input.scalar_type() == at::ScalarType::BFloat16) {
               if (shmem_size >=48 * 1024) {
                    VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(vllm::rms_norm_kernel_e8<__nv_bfloat16>, shmem_size);
               }
               vllm::rms_norm_kernel_e8<__nv_bfloat16><<<grid, block, shmem_size, stream>>>(
              (float4*)out.data_ptr(), (float4*)input.data_ptr(), (float4*)weight.data_ptr(), hidden_size, epsilon, use_shmem);
          } 
          #endif
          else {
              TORCH_CHECK(false, "BUGON: Unsupported type");
          }
              
      } else {
          dim3 block(min(1024, hidden_size));
          if (shmem_size >=48 * 1024) {
            VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(vllm::rms_norm_kernel<scalar_t>, shmem_size);
          }
          vllm::rms_norm_kernel<scalar_t><<<grid, block, shmem_size, stream>>>(
          out.data_ptr<scalar_t>(),
          input.data_ptr<scalar_t>(),
          weight.data_ptr<scalar_t>(),
          epsilon,
          num_tokens,
          hidden_size,
          use_shmem
         );
      }
    });
}

void fused_add_rms_norm(
  torch::Tensor& input,    // [..., hidden_size]
  torch::Tensor& residual, // [..., hidden_size]
  torch::Tensor& weight,   // [hidden_size]
  float epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    input.scalar_type(),
    "fused_add_rms_norm_kernel",
    [&] {

      bool use_shmem = true;
      //estimate the shared memory size
      int shmem_size = hidden_size * sizeof(scalar_t);

      if (shmem_size > getMaxSharedMemoryPerBlock(input)) {
        shmem_size = 0;
        use_shmem = false;
      }
      if (shmem_size >=48 * 1024) {
        VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(vllm::fused_add_rms_norm_kernel<scalar_t>, shmem_size);
      }
      
      vllm::fused_add_rms_norm_kernel<scalar_t><<<grid, block, shmem_size, stream>>>(
        input.data_ptr<scalar_t>(),
        residual.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        epsilon,
        num_tokens,
        hidden_size,
        use_shmem
        );
    });
}