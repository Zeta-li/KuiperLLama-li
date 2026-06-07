#ifndef KUIPER_INLCUDE_MHA_H
#define KUIPER_INLCUDE_MHA_H
#include <base/cuda_config.h>
#include "layer.h"
namespace op {

enum class MHA_MODE { PREFILL, DECODE };

class MultiHeadAttention : public op::Layer {
 public:
  explicit MultiHeadAttention(base::DeviceType device_type, int32_t layer_index,
                              int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
                              int32_t head_num, int32_t head_size);

  base::Status check() const override;

  void set_pos(int32_t pos);
  void set_layer_idx(int32_t layer_idx);
  void set_mode(MHA_MODE mode);
  void set_pos_range(int32_t start, int32_t end);
  void set_token_num(int32_t token_num);

  base::Status forward() override;

 private:
  int32_t layer_index_ = 0;
  int32_t pos_ = 0;
  int32_t kv_mul_ = 0;
  int32_t kv_dim_ = 0;
  int32_t seq_len_ = 0;
  int32_t head_num_ = 0;
  int32_t head_size_ = 0;
  MHA_MODE mode_ = MHA_MODE::DECODE;
  int32_t pos_start_ = 0;
  int32_t pos_end_ = 0;
  int32_t token_num_ = 1;
};
}  // namespace op
#endif  // KUIPER_INLCUDE_MHA_H
