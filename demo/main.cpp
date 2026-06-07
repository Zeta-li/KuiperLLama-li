#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include "model/llama3.h"
int32_t generate(const model::LLama2Model& model, const std::string& sentence, int total_steps,
                 bool need_output = false) {
  auto tokens = model.encode(sentence);
  int32_t prompt_len = tokens.size();
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  int32_t pos = 0;
  int32_t next = -1;
  bool is_prompt = true;
  const auto& prompt_embedding = model.embedding(tokens);
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  std::vector<int32_t> words;

  // === Phase 1: Prefill (batch process all prompt tokens) ===
  {
    tensor::Tensor prefill_input = model.fill_input_prefill(prompt_embedding, prompt_len);
    model.prefill_predict(prefill_input, prompt_len, next);

    // Collect all prompt tokens (skip the first one since it's the BOS/special token)
    for (int32_t i = 1; i < prompt_len; ++i) {
      words.push_back(tokens.at(i));
    }
    words.push_back(next);
    pos = prompt_len;
    is_prompt = false;
  }

  // === Phase 2: Decode (generate tokens one at a time) ===
  while (pos < total_steps) {
    tokens = std::vector<int32_t>{next};
    const auto& token_embedding = model.embedding(tokens);
    pos_tensor.index<int32_t>(0) = pos;
    tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
    model.predict(input, pos_tensor, is_prompt, next);

    if (model.is_sentence_ending(next)) {
      break;
    }
    words.push_back(next);
    pos += 1;
  }

  if (need_output) {
    printf("%s ", model.decode(words).data());
    fflush(stdout);
  }
  return std::min(pos, total_steps);
}


int main(int argc, char* argv[]) {
  if (argc != 3) {
    LOG(INFO) << "Usage: ./demo checkpoint path tokenizer path";
    return -1;
  }
  const char* checkpoint_path = argv[1];  // e.g. out/model.bin
  const char* tokenizer_path = argv[2];

  model::LLama2Model model(base::TokenizerType::kEncodeSpe, tokenizer_path,
    checkpoint_path, true);
  auto init_status = model.init(base::DeviceType::kDeviceCUDA);
  if (!init_status) {
    LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
  }
  const std::string& sentence = "hello";

  auto start = std::chrono::steady_clock::now();
  printf("Generating...\n");
  fflush(stdout);
  int steps = generate(model, sentence, 128, true);
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double>(end - start).count();
  printf("\nsteps/s:%lf\n", static_cast<double>(steps) / duration);
  fflush(stdout);
  return 0;
}
