#include "../cpu/mha_kernel.h"
#include <cfloat>
#include <cuda_runtime_api.h>
#include "../kernels_interface.h"
namespace kernel {
void mha_kernel(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len, int32_t kv_dim,
                int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
                const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                base::DeviceType device_type, CudaConfig* config) {
  int32_t layer_offset = layer_index * seq_len * kv_dim;
  float scale = 1.f / std::sqrt(static_cast<float>(head_size));

  std::shared_ptr<base::DeviceAllocator> allocator;
    if (device_type == base::DeviceType::kDeviceCPU) {
      allocator = base::CPUDeviceAllocatorFactory::get_instance();
    } else {
      allocator = base::CUDADeviceAllocatorFactory::get_instance();
    }
  for (int32_t h = 0; h < head_num; ++h) {
    float* score_head_addr = const_cast<float*>(score_tensor.ptr<float>() + h * seq_len);
    float* query_head_addr = const_cast<float*>(query_tensor.ptr<float>() + h * head_size);

    
    tensor::Tensor query_mat(base::DataType::kDataTypeFp32, head_size, false, nullptr,
                               query_head_addr);
    query_mat.set_device_type(device_type);
    
    for (int32_t t = 0; t <= pos; t++) {
      int32_t cache_offset = t * kv_dim + (h / kv_mul) * head_size;
      const float* key_head_addr = key_cache_tensor.ptr<float>() + layer_offset + cache_offset;
      tensor::Tensor key_mat(base::DataType::kDataTypeFp32, 1, head_size, false, nullptr,
                             const_cast<float*>(key_head_addr));
      
      tensor::Tensor score_mat(base::DataType::kDataTypeFp32, 1, false, nullptr,
                               score_head_addr + t);
      key_mat.set_device_type(device_type);
      score_mat.set_device_type(device_type);
      get_matmul_kernel(device_type)(query_mat, key_mat, score_mat, scale, config);
    }

    tensor::Tensor score_head_tensor(base::DataType::kDataTypeFp32, pos + 1, false, nullptr,
                                     score_head_addr);
    score_head_tensor.set_device_type(device_type);
    get_softmax_kernel(device_type)(score_head_tensor, config ? config->stream : nullptr);

    float* output_head_ptr = const_cast<float*>(mha_out.ptr<float>()) + h * head_size;
    allocator->memset_zero(output_head_ptr, sizeof(float) * head_size,
                              config ? config->stream : nullptr, false);
    tensor::Tensor output_tensor(base::DataType::kDataTypeFp32, head_size, false, nullptr,
                                 output_head_ptr);
    output_tensor.set_device_type(device_type);

    int32_t cache_offset = (h / kv_mul) * head_size;
    float* value_head_addr =
        const_cast<float*>(value_cache_tensor.ptr<float>()) + layer_offset + cache_offset;
    tensor::Tensor value_tensor(base::DataType::kDataTypeFp32, head_size, false, nullptr,
                                value_head_addr);
    get_scale_sum_kernel(device_type)(value_tensor, score_head_tensor, output_tensor, pos,
                                      head_size, kv_dim, config ? config->stream : nullptr);
  }
}

void mha_prefill_kernel(int32_t pos_start, int32_t token_num, int32_t head_num,
                        int32_t layer_index, int32_t seq_len, int32_t kv_dim, int32_t kv_mul,
                        int32_t head_size, const tensor::Tensor& mha_out,
                        const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                        const tensor::Tensor& key_cache_tensor,
                        const tensor::Tensor& value_cache_tensor, base::DeviceType device_type,
                        CudaConfig* config) {
  int32_t layer_offset = layer_index * seq_len * kv_dim;
  float scale = 1.f / std::sqrt(static_cast<float>(head_size));

  std::shared_ptr<base::DeviceAllocator> allocator;
  if (device_type == base::DeviceType::kDeviceCPU) {
    allocator = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    allocator = base::CUDADeviceAllocatorFactory::get_instance();
  }

  // Prefill MHA: query shape [token_num * dim], key/value already in KV cache
  // For each query token q_i at position pos_start + i, it attends to all
  // key/value tokens at positions [0, pos_start + i] (causal mask)
  // Output: we only compute the last token's output for next token prediction,
  // but we need to compute attention for all tokens to fill the KV cache properly.
  // Actually, we need all tokens' MHA output since they feed into subsequent layers.
  // The output is [token_num, dim] with causal attention.

  for (int32_t h = 0; h < head_num; ++h) {
    int32_t kv_head_idx = h / kv_mul;
    int32_t head_offset = kv_head_idx * head_size;

    // For each query token, compute attention scores
    for (int32_t qi = 0; qi < token_num; ++qi) {
      int32_t q_pos = pos_start + qi;
      float* score_head_addr =
          const_cast<float*>(score_tensor.ptr<float>() + (qi * head_num + h) * seq_len);

      // Query for this token and head
      const float* query_head_addr = query_tensor.ptr<float>() + qi * head_num * head_size + h * head_size;

      // Compute attention scores: query * key^T for all positions <= q_pos
      for (int32_t t = 0; t <= q_pos; t++) {
        const float* key_head_addr =
            key_cache_tensor.ptr<float>() + layer_offset + t * kv_dim + head_offset;

        float score = 0.0f;
        for (int32_t d = 0; d < head_size; ++d) {
          score += query_head_addr[d] * key_head_addr[d];
        }
        score_head_addr[t] = score * scale;

        // Apply causal mask: set future positions to -infinity
        // (future positions t > q_pos are not computed since loop goes to q_pos)
      }

      // Softmax over [0, q_pos]
      // Find max for numerical stability
      float max_val = -FLT_MAX;
      for (int32_t t = 0; t <= q_pos; ++t) {
        if (score_head_addr[t] > max_val) {
          max_val = score_head_addr[t];
        }
      }

      float sum = 0.0f;
      for (int32_t t = 0; t <= q_pos; ++t) {
        score_head_addr[t] = expf(score_head_addr[t] - max_val);
        sum += score_head_addr[t];
      }
      for (int32_t t = 0; t <= q_pos; ++t) {
        score_head_addr[t] /= sum;
      }

      // Compute weighted sum of values
      float* output_head_addr =
          const_cast<float*>(mha_out.ptr<float>()) + qi * head_num * head_size + h * head_size;
      for (int32_t d = 0; d < head_size; ++d) {
        output_head_addr[d] = 0.0f;
      }
      for (int32_t t = 0; t <= q_pos; ++t) {
        const float* value_head_addr =
            value_cache_tensor.ptr<float>() + layer_offset + t * kv_dim + head_offset;
        float attn_weight = score_head_addr[t];
        for (int32_t d = 0; d < head_size; ++d) {
          output_head_addr[d] += attn_weight * value_head_addr[d];
        }
      }
    }
  }
}

}  // namespace kernel