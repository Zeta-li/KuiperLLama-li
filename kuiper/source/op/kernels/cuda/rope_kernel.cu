#include "rope_kernel.cuh"
namespace kernel {

#if defined (LLAMA3_SUPPORT)
__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q, const float* input_k,
                                    const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx > total_pairs) {
    return;
  }

  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;

  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;

  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];

  int rotn = i < kv_dim ? 2 : 1;

  for (int v = 0; v < rotn; v++) {
    float* vec = const_cast<float*>(v == 0 ? input_q : input_k);  // the vector to rotate (query or key)
    float v0 = vec[v0_idx];
    float v1 = vec[v1_idx];
    vec[v0_idx] = fcr * v0 - fci * v1;
    vec[v1_idx] = fcr * v1 + fci * v0;
  }
}

// Batch RoPE kernel for LLAMA3: processes multiple tokens in parallel
// blockIdx.x = token index, threadIdx.x processes head pairs within each token
__global__ void rope_kernel_cu_fp32_batch(int token_num, int dim, int kv_dim, int head_size,
                                          const float* input_q, const float* input_k,
                                          const int32_t* input_pos, const float* sin_cache,
                                          const float* cos_cache) {
  int token_idx = blockIdx.x;
  int idx = threadIdx.x + blockDim.x * blockIdx.y;

  if (token_idx >= token_num) return;

  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx > total_pairs) return;

  int pos = input_pos[token_idx];
  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;

  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;

  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];

  int rotn = i < kv_dim ? 2 : 1;

  float* q_base = const_cast<float*>(input_q + token_idx * dim);
  float* k_base = const_cast<float*>(input_k + token_idx * kv_dim);

  for (int v = 0; v < rotn; v++) {
    float* vec = (v == 0) ? q_base : k_base;
    float v0 = vec[v0_idx];
    float v1 = vec[v1_idx];
    vec[v0_idx] = fcr * v0 - fci * v1;
    vec[v1_idx] = fcr * v1 + fci * v0;
  }
}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int head_dim = idx % head_size;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    float freq = 1.0f / pow(500000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
    float val = static_cast<float>(pos) * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    *(sin_cache + pos * head_size + head_dim) = fci;
    *(cos_cache + pos * head_size + head_dim) = fcr;
  }
}
#elif defined (QWEN2_SUPPORT) || defined (QWEN3_SUPPORT)
__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q, const float* input_k,
                                    const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx > total_pairs) {
    return;
  }

  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;

  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;

  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];

  int rotn = i < kv_dim ? 2 : 1;

  for (int v = 0; v < rotn; v++) {
    float* vec = const_cast<float*>(v == 0 ? input_q : input_k);  // the vector to rotate (query or key)
    float v0 = vec[v0_idx];
    float v1 = vec[v1_idx];
    vec[v0_idx] = fcr * v0 - fci * v1;
    vec[v1_idx] = fcr * v1 + fci * v0;
  }
}

// Batch RoPE kernel: processes multiple tokens in parallel
// blockIdx.x = token index, threadIdx.x processes head pairs within each token
__global__ void rope_kernel_cu_fp32_batch(int token_num, int dim, int kv_dim, int head_size,
                                          const float* input_q, const float* input_k,
                                          const int32_t* input_pos, const float* sin_cache,
                                          const float* cos_cache) {
  int token_idx = blockIdx.x;
  int idx = threadIdx.x + blockDim.x * blockIdx.y;

  if (token_idx >= token_num) return;

  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx > total_pairs) return;

  int pos = input_pos[token_idx];
  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;

  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;

  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];

  int rotn = i < kv_dim ? 2 : 1;

  float* q_base = const_cast<float*>(input_q + token_idx * dim);
  float* k_base = const_cast<float*>(input_k + token_idx * kv_dim);

  for (int v = 0; v < rotn; v++) {
    float* vec = (v == 0) ? q_base : k_base;
    float v0 = vec[v0_idx];
    float v1 = vec[v1_idx];
    vec[v0_idx] = fcr * v0 - fci * v1;
    vec[v1_idx] = fcr * v1 + fci * v0;
  }
}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int head_dim = idx % head_size;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    float freq = 1.0f / pow(1000000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
    float val = static_cast<float>(pos) * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    *(sin_cache + pos * head_size + head_dim) = fci;
    *(cos_cache + pos * head_size + head_dim) = fcr;
  }
}
#else
__device__ void rope_calc(float fcr, float fci, float* vec, int32_t idx) {
  float2* vec_ptr = reinterpret_cast<float2*>(vec + idx);
  float2 vec_value = *vec_ptr;
  *vec_ptr =
      make_float2(vec_value.x * fcr - vec_value.y * fci, vec_value.x * fci + vec_value.y * fcr);
}

__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q, const float* input_k,
                                    const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  idx = idx * 2;
  if (idx >= dim) {
    return;
  }

  int head_dim = idx % head_size;
  float fci = *(sin_cache + pos * head_size + head_dim);
  float fcr = *(cos_cache + pos * head_size + head_dim);

  rope_calc(fcr, fci, const_cast<float*>(input_q), idx);
  if (idx >= kv_dim) {
    return;
  }
  rope_calc(fcr, fci, const_cast<float*>(input_k), idx);
}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int head_dim = idx % head_size;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    float freq = 1.0f / pow(10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
    float val = static_cast<float>(pos) * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    *(sin_cache + pos * head_size + head_dim) = fci;
    *(cos_cache + pos * head_size + head_dim) = fcr;
  }
}
#endif

void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, cudaStream_t stream) {
  CHECK_EQ(sin_cache.is_empty(), false);
  CHECK_EQ(cos_cache.is_empty(), false);
  int threads = head_size;
  if (stream) {
    sin_cos_calc<<<1, threads, 0, stream>>>(head_size, max_seq_len,
                                            const_cast<float*>(sin_cache.ptr<float>()),
                                            const_cast<float*>(cos_cache.ptr<float>()));
  } else {
    sin_cos_calc<<<1, threads>>>(head_size, max_seq_len, const_cast<float*>(sin_cache.ptr<float>()),
                                 const_cast<float*>(cos_cache.ptr<float>()));
  }
}

void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                    const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                    const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                    void* stream) {
  const int32_t pos = *input_pos.ptr<int32_t>(0);
  int threads = 128;
  int blocks = (dim + threads - 1) / threads;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    rope_kernel_cu_fp32<<<blocks, threads, 0, stream_>>>(
        pos, dim, kv_dim, head_size, input_q.ptr<float>(), input_k.ptr<float>(),
        sin_cache.ptr<float>(), cos_cache.ptr<float>());
  } else {
    rope_kernel_cu_fp32<<<blocks, threads>>>(pos, dim, kv_dim, head_size, input_q.ptr<float>(),
                                             input_k.ptr<float>(), sin_cache.ptr<float>(),
                                             cos_cache.ptr<float>());
  }
}

void rope_kernel_cu_batch(int32_t dim, int32_t kv_dim, int32_t head_size, int32_t token_num,
                         const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                         const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
                         const tensor::Tensor& cos_cache, void* stream) {
  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;

  int threads = 128;
  int blocks_y = (total_pairs + threads - 1) / threads;

  dim3 grid(token_num, blocks_y);
  dim3 block(threads);

  // input_pos must be on CPU for the kernel to read
  // We need to copy positions to GPU if not already there
  // For now, assume input_pos is on CPU and we create a temp GPU buffer
  // Actually, the positions are in a CPU tensor, so we need to upload them

  // Create a temporary GPU buffer for positions if needed
  std::shared_ptr<base::DeviceAllocator> gpu_alloc = base::CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor pos_gpu(base::DataType::kDataTypeInt32, token_num, true, gpu_alloc);
  pos_gpu.set_device_type(base::DeviceType::kDeviceCUDA);

  // Copy positions from CPU to GPU
  cudaMemcpy(pos_gpu.ptr<int32_t>(), input_pos.ptr<int32_t>(), token_num * sizeof(int32_t),
             cudaMemcpyHostToDevice);

  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    rope_kernel_cu_fp32_batch<<<grid, block, 0, stream_>>>(
        token_num, dim, kv_dim, head_size, input_q.ptr<float>(), input_k.ptr<float>(),
        pos_gpu.ptr<int32_t>(), sin_cache.ptr<float>(), cos_cache.ptr<float>());
  } else {
    rope_kernel_cu_fp32_batch<<<grid, block>>>(
        token_num, dim, kv_dim, head_size, input_q.ptr<float>(), input_k.ptr<float>(),
        pos_gpu.ptr<int32_t>(), sin_cache.ptr<float>(), cos_cache.ptr<float>());
  }
}
}  // namespace kernel