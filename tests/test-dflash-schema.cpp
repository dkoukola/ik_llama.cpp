#include "ggml.h"
#include "llama.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#include <chrono>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

using model_ptr = std::unique_ptr<llama_model, decltype(&llama_free_model)>;

enum class fixture_kind {
    dense,
    ambiguous,
    unknown,
    dflash2,
};

static void add_tensor(
        gguf_context * gguf,
        ggml_context * data,
        const char * name,
        const std::vector<int64_t> & ne) {
    ggml_tensor * tensor = ggml_new_tensor(data, GGML_TYPE_F32, ne.size(), ne.data());
    require(tensor != nullptr, "failed to create fixture tensor");
    ggml_set_name(tensor, name);
    ggml_set_zero(tensor);
    gguf_add_tensor(gguf, tensor);
}

static void write_fixture(const std::filesystem::path & path, fixture_kind kind) {
    gguf_context * gguf = gguf_init_empty();
    require(gguf != nullptr, "failed to create fixture GGUF context");

    ggml_init_params init_params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * data = ggml_init(init_params);
    require(data != nullptr, "failed to create fixture tensor context");

    gguf_set_val_str(gguf, "general.architecture", "dflash");
    gguf_set_val_str(gguf, "general.name", "synthetic-dflash-schema");
    gguf_set_val_u32(gguf, "dflash.block_count", 1);
    gguf_set_val_u32(gguf, "dflash.context_length", 16);
    gguf_set_val_u32(gguf, "dflash.embedding_length", 4);
    gguf_set_val_u32(gguf, "dflash.feed_forward_length", 8);
    gguf_set_val_u32(gguf, "dflash.attention.head_count", 1);
    gguf_set_val_u32(gguf, "dflash.attention.head_count_kv", 1);
    gguf_set_val_u32(gguf, "dflash.rope.dimension_count", 4);
    gguf_set_val_f32(gguf, "dflash.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(gguf, "dflash.block_size", 4);
    gguf_set_val_u32(gguf, "tokenizer.ggml.mask_token_id", 3);
    const uint32_t target_layers[] = { 1 };
    gguf_set_arr_data(gguf, "dflash.target_layers", GGUF_TYPE_UINT32, target_layers, 1);

    const char * tokens[] = { "<unk>", "<s>", "</s>", "<mask>" };
    const float scores[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    const int32_t token_types[] = { 2, 3, 3, 3 };
    gguf_set_val_str(gguf, "tokenizer.ggml.model", "llama");
    gguf_set_arr_str(gguf, "tokenizer.ggml.tokens", tokens, 4);
    gguf_set_arr_data(gguf, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores, 4);
    gguf_set_arr_data(gguf, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, token_types, 4);
    gguf_set_val_u32(gguf, "tokenizer.ggml.unknown_token_id", 0);
    gguf_set_val_u32(gguf, "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(gguf, "tokenizer.ggml.eos_token_id", 2);

    add_tensor(gguf, data, "fc.weight", { 4, 4 });
    if (kind == fixture_kind::dense || kind == fixture_kind::ambiguous) {
        add_tensor(gguf, data, "blk.0.attn_q.weight", { 4, 4 });
    }
    if (kind == fixture_kind::ambiguous) {
        add_tensor(gguf, data, "blk.0.attn_q_a.weight", { 4, 4 });
    }
    if (kind == fixture_kind::dflash2) {
        add_tensor(gguf, data, "selector_hidden.weight", { 4, 4 });
    }

    if (kind == fixture_kind::dense) {
        add_tensor(gguf, data, "output_norm.weight", { 4 });
        add_tensor(gguf, data, "enc.output_norm.weight", { 4 });
        add_tensor(gguf, data, "blk.0.attn_norm.weight", { 4 });
        add_tensor(gguf, data, "blk.0.attn_q_norm.weight", { 4 });
        add_tensor(gguf, data, "blk.0.attn_k.weight", { 4, 4 });
        add_tensor(gguf, data, "blk.0.attn_k_norm.weight", { 4 });
        add_tensor(gguf, data, "blk.0.attn_v.weight", { 4, 4 });
        add_tensor(gguf, data, "blk.0.attn_output.weight", { 4, 4 });
        add_tensor(gguf, data, "blk.0.ffn_norm.weight", { 4 });
        add_tensor(gguf, data, "blk.0.ffn_gate.weight", { 4, 8 });
        add_tensor(gguf, data, "blk.0.ffn_up.weight", { 4, 8 });
        add_tensor(gguf, data, "blk.0.ffn_down.weight", { 8, 4 });
    }

    gguf_write_to_file(gguf, path.string().c_str(), false);
    ggml_free(data);
    gguf_free(gguf);
}

static void require_hparams_rejected(const std::filesystem::path & path) {
    llama_model_loader loader(path.string(), 0, false, false, false, false,
            false, false, false, nullptr, nullptr);
    llama_model model;
    model.arch = loader.get_arch();
    bool rejected = false;
    try {
        llm_load_hparams(loader, model);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "invalid DFlash schema was accepted");
}

int main() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto base = std::filesystem::temp_directory_path() /
            ("llama-dflash-schema-" + std::to_string(nonce));
    const auto dense = base.string() + "-dense.gguf";
    const auto ambiguous = base.string() + "-ambiguous.gguf";
    const auto unknown = base.string() + "-unknown.gguf";
    const auto dflash2 = base.string() + "-dflash2.gguf";

    write_fixture(dense, fixture_kind::dense);
    write_fixture(ambiguous, fixture_kind::ambiguous);
    write_fixture(unknown, fixture_kind::unknown);
    write_fixture(dflash2, fixture_kind::dflash2);

    llama_backend_init();

    llama_model_params params = llama_model_default_params();
    params.use_mmap = false;
    model_ptr model(llama_model_load_from_file(dense.c_str(), params), llama_free_model);
    require(model != nullptr, "official dense DFlash fixture failed to load");
    require(model->arch == LLM_ARCH_DFLASH, "dense DFlash resolved to the wrong architecture");
    require(!model->hparams.dflash_dsv4, "dense DFlash was classified as DSV4");
    require(llama_rope_type(model.get()) == LLAMA_ROPE_TYPE_NEOX,
            "dense DFlash selected the wrong RoPE layout");
    require(model->hparams.dflash_n_target_layers == 1, "dense DFlash target layers were not loaded");
    require(model->hparams.dflash_target_layer_ids[0] == 0, "one-based DFlash target layer was not normalized");
    require(model->hparams.dflash_n_target_features == 4, "DFlash feature width was not derived from fc.weight");

    require_hparams_rejected(ambiguous);
    require_hparams_rejected(unknown);

    llama_model_loader selector_loader(dflash2, 0, false, false, false, false,
            false, false, false, nullptr, nullptr);
    require(selector_loader.get_arch() == LLM_ARCH_DFLASH2,
            "selector_hidden.weight did not select DFlash2");

    model.reset();
    llama_backend_free();
    std::filesystem::remove(dense);
    std::filesystem::remove(ambiguous);
    std::filesystem::remove(unknown);
    std::filesystem::remove(dflash2);
    return 0;
}
