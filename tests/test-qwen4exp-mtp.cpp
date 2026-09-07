#include "llama.h"

#if defined(LLAMA_TEST_SPECULATIVE_BRIDGE)
#include "llama-speculative.h"
#endif

#include "llama-context.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-spec-features.h"
#include "llama-spec-features-dflash.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        std::abort(); \
    } \
} while (0)

void llama_set_mtp_step_idx(struct llama_context * ctx, int32_t mtp_step_idx);

using model_ptr = std::unique_ptr<llama_model, decltype(&llama_free_model)>;
using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;

struct hidden_capture {
    std::vector<float> values;
};

struct selection_capture {
    std::vector<int32_t> values;
    std::vector<float> mask;
};

static int capture_hidden_norm(ggml_tensor * tensor, bool ask, void * user_data) {
    static constexpr const char * name = "mtp_hidden_norm";
    if (std::strncmp(tensor->name, name, std::strlen(name)) != 0) {
        return 0;
    }
    if (ask) {
        return 1;
    }

    CHECK(tensor->type == GGML_TYPE_F32);
    auto & capture = *(hidden_capture *) user_data;
    capture.values.resize((size_t) ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, capture.values.data(), 0, capture.values.size()*sizeof(float));
    return 1;
}

static int capture_qsa_selection(ggml_tensor * tensor, bool ask, void * user_data) {
    static constexpr const char * name = "qsa_mtp_top_k";
    static constexpr const char * mask_name = "qsa_mask";
    const bool is_selection = std::strncmp(tensor->name, name, std::strlen(name)) == 0;
    const bool is_mask = std::strncmp(tensor->name, mask_name, std::strlen(mask_name)) == 0;
    if (!is_selection && !is_mask) {
        return 0;
    }
    if (ask) {
        return 1;
    }

    auto & capture = *(selection_capture *) user_data;
    if (is_selection) {
        CHECK(tensor->type == GGML_TYPE_I32);
        capture.values.resize((size_t) tensor->ne[0]);
        const size_t row = (size_t) std::max<int64_t>(0, tensor->ne[1] - 1);
        ggml_backend_tensor_get(tensor, capture.values.data(), row*tensor->nb[1],
                capture.values.size()*sizeof(int32_t));
    } else {
        capture.mask.resize((size_t) tensor->ne[0]);
        if (tensor->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(tensor, capture.mask.data(), 0,
                    capture.mask.size()*sizeof(float));
        } else {
            CHECK(tensor->type == GGML_TYPE_F16);
            std::vector<ggml_fp16_t> values(capture.mask.size());
            ggml_backend_tensor_get(tensor, values.data(), 0,
                    values.size()*sizeof(ggml_fp16_t));
            std::transform(values.begin(), values.end(), capture.mask.begin(), ggml_fp16_to_fp32);
        }
    }
    return 1;
}

static model_ptr load_model(const char * path, bool mtp) {
    llama_model_params params = llama_model_default_params();
    params.mtp = mtp;
    params.use_mmap = false;
    return model_ptr(llama_model_load_from_file(path, params), llama_free_model);
}

static void test_metadata(const char * path) {
    llama_model_loader loader(path, 0, false, false, false, false,
            false, false, false, nullptr, nullptr);
    loader.mappings.resize(3);
    CHECK(!loader.has_anonymous_mapping());
    if (llama_mmap::SUPPORTED) {
        loader.defer_ple = true;
        loader.mmap_ple = true;
        const auto * ple = loader.get_weight("per_layer_token_embd.weight");
        CHECK(ple != nullptr);
        loader.mmap_ple_file = ple->idx;
        loader.build_ple_tensor_index();
        CHECK(loader.should_defer_ple_mmaps());
        llama_mlocks locks;
        loader.init_mappings(false, &locks, false);
        CHECK(locks.empty()); // dedicated NUMA PLE mappings are not locked
        CHECK(loader.mappings.at(ple->idx) != nullptr);
        CHECK(!loader.has_anonymous_mapping());
        loader.apply_ple_mmap_policy();
    }
    const uint32_t ratios[] = { 0, 2, 2 };
    for (int n_ratios : { 3, 2 }) {
        gguf_set_arr_data(loader.meta, "qwen4exp.attention.compress_ratios",
                GGUF_TYPE_UINT32, ratios, n_ratios);
        for (bool mtp : { false, true }) {
            llama_model model;
            model.arch = LLM_ARCH_QWEN4EXP;
            model.mtp = mtp;
            llm_load_hparams(loader, model);
            CHECK(model.hparams.n_embd_out == 512);
            CHECK(model.hparams.dsv4_compress_ratios[0] == 0);
            CHECK(model.hparams.dsv4_compress_ratios[1] == 2);
            CHECK(model.hparams.dsv4_compress_ratios[2] == 2);
            CHECK(model.hparams.is_qsa(2));
            CHECK(model.hparams.n_layer_kv_from_start == (mtp ? 3 : 2));
        }
    }

    for (uint32_t n_nextn : { 3u, 4u }) {
        gguf_set_val_u32(loader.meta, "qwen4exp.nextn_predict_layers", n_nextn);
        llama_model model;
        model.arch = LLM_ARCH_QWEN4EXP;
        bool rejected = false;
        try {
            llm_load_hparams(loader, model);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        CHECK(rejected);
    }
}

static context_ptr make_context(
        llama_model * model,
        llama_mtp_op_type op,
        ggml_backend_sched_eval_callback callback = nullptr,
        void * callback_data = nullptr,
        int32_t dsa_top_k = -1,
        uint32_t n_ctx = 32,
        bool graph_reuse = true,
        bool flash_attn = true,
        bool fused_idx_topk = true) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = 8;
    params.n_ubatch = 8;
    params.n_seq_max = 1;
    params.n_threads = 2;
    params.n_threads_batch = 2;
    params.embeddings = true;
    params.mtp = true;
    params.mtp_op_type = op;
    params.cb_eval = callback;
    params.cb_eval_user_data = callback_data;
    params.dsa_top_k = dsa_top_k;
    params.graph_reuse = graph_reuse;
    params.flash_attn = flash_attn;
    params.fused_idx_topk = fused_idx_topk;
    return context_ptr(llama_init_from_model(model, params), llama_free);
}

static context_ptr make_plain_context(llama_model * model, bool embeddings = false) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 32;
    params.n_batch = 8;
    params.n_ubatch = 8;
    params.n_seq_max = 1;
    params.n_threads = 2;
    params.n_threads_batch = 2;
    params.embeddings = embeddings;
    return context_ptr(llama_init_from_model(model, params), llama_free);
}

static void decode_one(llama_context * ctx, llama_token token, llama_pos pos, bool logits = true) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 1;
    batch.token[0] = token;
    batch.pos[0] = pos;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = logits;
    const int32_t status = llama_decode(ctx, batch);
    llama_batch_free(batch);
    CHECK(status == 0);
}

static void decode_many(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        llama_pos pos) {
    CHECK(!tokens.empty());
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    batch.n_tokens = (int32_t) tokens.size();
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = pos + (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }
    const int32_t status = llama_decode(ctx, batch);
    llama_batch_free(batch);
    CHECK(status == 0);
}

static std::vector<float> copy_embedding(
        llama_context * ctx,
        int32_t width,
        int32_t index = 0) {
    const float * data = llama_get_embeddings_ith(ctx, index);
    CHECK(data != nullptr);
    std::vector<float> result(data, data + width);
    CHECK(std::all_of(result.begin(), result.end(), [](float value) { return std::isfinite(value); }));
    return result;
}

static std::vector<float> copy_logits(llama_context * ctx, int32_t n_vocab) {
    const float * data = llama_get_logits_ith(ctx, 0);
    CHECK(data != nullptr);
    std::vector<float> result(data, data + n_vocab);
    CHECK(std::all_of(result.begin(), result.end(), [](float value) { return std::isfinite(value); }));
    return result;
}

static void seed_draft_step(
        llama_context             * ctx,
        const std::vector<float>  & hidden,
        llama_token                 token,
        llama_pos                   pos,
        int32_t                     step) {
    CHECK(llama_set_draft_input_hidden_state_copy(ctx, hidden.data(), hidden.size()));
    llama_set_mtp_step_idx(ctx, step);
    decode_one(ctx, token, pos);
}

static void check_close(
        const std::vector<float> & actual,
        const std::vector<float> & expected,
        float absolute_tolerance = 1e-4f,
        float relative_tolerance = 1e-4f) {
    CHECK(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        const float limit = absolute_tolerance +
                relative_tolerance*std::max(std::fabs(actual[i]), std::fabs(expected[i]));
        if (std::fabs(actual[i] - expected[i]) > limit) {
            std::fprintf(stderr,
                    "vectors differ at %zu: actual=%g expected=%g limit=%g\n",
                    i, actual[i], expected[i], limit);
            std::abort();
        }
    }
}

static void check_joint_hidden_norm(llama_model * model, int32_t width) {
    hidden_capture capture;
    context_ptr ctx = make_context(
            model, MTP_OP_DRAFT_GEN, capture_hidden_norm, &capture);
    CHECK(ctx != nullptr);

    std::vector<float> input(width);
    for (int32_t i = 0; i < width/2; ++i) {
        input[i] = (float) (i + 1);
        input[width/2 + i] = (float) (i + 20);
    }
    seed_draft_step(ctx.get(), input, 3, 0, 0);
    CHECK(capture.values.size() == input.size());

    float sum_sq = 0.0f;
    for (float value : input) {
        sum_sq += value*value;
    }
    const float scale = 1.0f/std::sqrt(sum_sq/(float) width + 1e-6f);
    float max_norm_error = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        max_norm_error = std::max(max_norm_error,
                std::fabs(capture.values[i] - input[i]*scale));
    }
    CHECK(max_norm_error < 1e-3f);
}

static void check_qsa_reuse_row(
        llama_context * ctx,
        int32_t expected_mask_width,
        int32_t expected_capacity,
        int32_t expected_tail_size) {
    CHECK(ctx->kv_self.n == (uint32_t) expected_mask_width);
    CHECK(ctx->qwen4_mtp_qsa_tail_capacity == expected_capacity);
    const size_t frozen_size = ctx->qwen4_mtp_qsa_topk.size();
    CHECK(frozen_size + (size_t) expected_capacity <= (size_t) expected_mask_width);

    const auto qsa = std::find_if(ctx->inp_qsa.begin(), ctx->inp_qsa.end(),
            [](const llama_context::qsa_input & input) { return input.reuse_topk != nullptr; });
    CHECK(qsa != ctx->inp_qsa.end());
    CHECK(qsa->reuse_topk->ne[0] == (int64_t) frozen_size + expected_capacity);
    CHECK(qsa->reuse_topk->ne[0] <= expected_mask_width);
    CHECK(ggml_backend_buffer_is_host(qsa->reuse_topk->buffer));

    int32_t actual_tail_size = 0;
    for (const llama_kv_cell & cell : ctx->kv_self.cells) {
        if (!cell.is_empty() && cell.has_seq_id(0) &&
                cell.qsa_pos >= ctx->qwen4_mtp_qsa_captured_len) {
            ++actual_tail_size;
        }
    }
    CHECK(actual_tail_size == expected_tail_size);
    CHECK(actual_tail_size <= expected_capacity);
}

static void check_qsa_reuse_padding_boundary(
        llama_model * model,
        bool graph_reuse) {
    static constexpr int32_t frozen_width = 25;
    static constexpr int32_t top_k = 24;
    static constexpr uint32_t n_ctx = 64;

    context_ptr ctx = make_context(
            model, MTP_OP_UPDATE_ACCEPTED, nullptr, nullptr, top_k, n_ctx,
            graph_reuse, false, false);
    CHECK(ctx != nullptr);
    CHECK(ctx->cparams.graph_reuse == graph_reuse);

    const int32_t width = (int32_t) llama_model_mtp_feature_width(model);
    const int32_t n_vocab = llama_n_vocab(model);
    std::vector<float> hidden((size_t) width, 1.0f);
    for (llama_pos pos = 0; pos < frozen_width; ++pos) {
        seed_draft_step(ctx.get(), hidden, 3 + pos%(n_vocab - 3), pos, 0);
    }

    llama_set_mtp_op_type(ctx.get(), MTP_OP_DRAFT_GEN);
    seed_draft_step(ctx.get(), hidden, 3 + frozen_width%(n_vocab - 3), frozen_width, 0);
    CHECK(!ctx->qwen4_mtp_qsa_pending);
    CHECK(ctx->qwen4_mtp_qsa_topk.size() == (size_t) frozen_width);
    check_qsa_reuse_row(ctx.get(), 32, 7, 1);

    // Position 26 establishes the settled graph. The remaining rows fit the same
    // clipped reserve and must retain that graph when reuse is enabled.
    seed_draft_step(ctx.get(), hidden, 3 + (frozen_width + 1)%(n_vocab - 3), frozen_width + 1, 1);
    const auto * settled_prev = ctx->prev_mtp.get();
    if (graph_reuse) {
        CHECK(settled_prev != nullptr);
    }
    for (llama_pos pos = frozen_width + 2; pos < 32; ++pos) {
        seed_draft_step(ctx.get(), hidden, 3 + pos%(n_vocab - 3), pos, 1);
        if (graph_reuse) {
            CHECK(ctx->prev_mtp.get() == settled_prev);
        }
    }
    check_qsa_reuse_row(ctx.get(), 32, 7, 7);

    // Crossing position 31 -> 32 grows the padded mask and restores the usual reserve.
    seed_draft_step(ctx.get(), hidden, 3 + 32%(n_vocab - 3), 32, 1);
    check_qsa_reuse_row(ctx.get(), 64, 8, 8);

    llama_kv_cache_clear(ctx.get());
    CHECK(ctx->qwen4_mtp_qsa_topk.empty());
    CHECK(ctx->qwen4_mtp_qsa_valid.empty());
    CHECK(ctx->qwen4_mtp_qsa_tail_capacity == 0);
    CHECK(ctx->prev_mtp == nullptr);
}

#if defined(LLAMA_TEST_SPECULATIVE_BRIDGE)
static void check_speculative_status(
        llama_speculative_status status,
        llama_speculative_status expected,
        llama_speculative_error ** error) {
    if (status != expected) {
        std::fprintf(stderr, "unexpected speculative status %d (expected %d): %s\n",
                status, expected, *error != nullptr ? llama_speculative_error_message(*error) : "");
    }
    CHECK(status == expected);
    if (*error != nullptr) {
        llama_speculative_error_free(*error);
        *error = nullptr;
    }
}

static void bridge_decode(
        llama_speculative_session * session,
        const std::vector<llama_token> & tokens,
        llama_pos position) {
    CHECK(!tokens.empty());
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    CHECK(batch.token != nullptr);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = position + (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i + 1 == tokens.size();
    }
    batch.n_tokens = (int32_t) tokens.size();

    llama_speculative_decode_params params = {};
    params.batch = &batch;
    params.expected_position = position;
    params.phase = LLAMA_SPECULATIVE_DECODE_PROMPT;
    llama_speculative_call_result result = {};
    llama_speculative_error * error = nullptr;
    check_speculative_status(
        llama_speculative_session_decode(session, &params, &result, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    CHECK(result.effect == LLAMA_SPECULATIVE_EFFECT_UNCHANGED);
    CHECK(result.resulting_position == position + (llama_pos) tokens.size());
    llama_batch_free(batch);
}

static std::vector<uint8_t> save_bridge_state(
        llama_speculative_session * session,
        llama_speculative_state_info & info) {
    uint64_t required_size = 0;
    llama_speculative_error * error = nullptr;
    check_speculative_status(
        llama_speculative_session_state_size(session, &required_size, &info, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    CHECK(required_size > 0 && required_size <= SIZE_MAX);

    std::vector<uint8_t> state((size_t) required_size);
    uint64_t written_size = 0;
    check_speculative_status(
        llama_speculative_session_state_save(
            session, state.data(), state.size(), &written_size, &info, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    CHECK(written_size == state.size());
    return state;
}

static constexpr size_t MTP_STATE_V1_HEADER_SIZE = 72;
static constexpr size_t MTP_STATE_V1_SECTION_ENTRY_SIZE = 32;
static constexpr size_t MTP_STATE_V1_PREFIX_SIZE = 72;
static constexpr size_t MTP_STATE_V1_DIGEST_OFFSET = 40;
static constexpr size_t MTP_STATE_V1_DIGEST_SIZE = 32;
static constexpr uint32_t MTP_STATE_V1_SECTION_TYPE = 2;
static constexpr uint32_t MTP_STATE_V1_SECTION_REQUIRED = 1;
static constexpr std::array<uint8_t, 8> MTP_STATE_V1_MAGIC = {
    'L', 'L', 'S', 'P', 'S', 'T', 1, 0,
};

struct mtp_state_v1_view {
    uint32_t feature_width;
    uint32_t hidden_count;
    size_t digest_begin;
    size_t digest_end;
    size_t hidden_begin;
    size_t context_begin;
    size_t context_end;
};

static bool checked_add_size(size_t lhs, size_t rhs, size_t & result) {
    if (lhs > std::numeric_limits<size_t>::max() - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

static bool checked_multiply_size(size_t lhs, size_t rhs, size_t & result) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

static bool read_u16_le(
        const std::vector<uint8_t> & bytes,
        size_t offset,
        uint16_t & result) {
    if (offset > bytes.size() || bytes.size() - offset < sizeof(uint16_t)) {
        return false;
    }
    result = (uint16_t) bytes[offset] |
            (uint16_t) ((uint16_t) bytes[offset + 1] << 8);
    return true;
}

static bool read_u32_le(
        const std::vector<uint8_t> & bytes,
        size_t offset,
        uint32_t & result) {
    if (offset > bytes.size() || bytes.size() - offset < sizeof(uint32_t)) {
        return false;
    }
    result = (uint32_t) bytes[offset] |
            (uint32_t) bytes[offset + 1] << 8 |
            (uint32_t) bytes[offset + 2] << 16 |
            (uint32_t) bytes[offset + 3] << 24;
    return true;
}

static bool read_u64_le(
        const std::vector<uint8_t> & bytes,
        size_t offset,
        uint64_t & result) {
    if (offset > bytes.size() || bytes.size() - offset < sizeof(uint64_t)) {
        return false;
    }
    result = 0;
    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
        result |= (uint64_t) bytes[offset + i] << (8*i);
    }
    return true;
}

static mtp_state_v1_view parse_mtp_state_v1(
        const std::vector<uint8_t> & bytes,
        uint32_t expected_feature_width) {
    CHECK(bytes.size() >= MTP_STATE_V1_HEADER_SIZE +
            MTP_STATE_V1_SECTION_ENTRY_SIZE + MTP_STATE_V1_PREFIX_SIZE);
    CHECK(std::equal(MTP_STATE_V1_MAGIC.begin(), MTP_STATE_V1_MAGIC.end(), bytes.begin()));

    uint32_t header_size = 0;
    uint32_t total_size = 0;
    uint32_t container_flags = 0;
    uint32_t section_count = 0;
    CHECK(read_u32_le(bytes, 8, header_size));
    CHECK(read_u32_le(bytes, 12, total_size));
    CHECK(read_u32_le(bytes, 64, container_flags));
    CHECK(read_u32_le(bytes, 68, section_count));
    CHECK(header_size == MTP_STATE_V1_HEADER_SIZE);
    CHECK(total_size == bytes.size());
    CHECK(container_flags == 0);
    CHECK(section_count == 1);

    const size_t entry = MTP_STATE_V1_HEADER_SIZE;
    uint32_t section_index = 0;
    uint32_t section_type = 0;
    uint16_t section_major = 0;
    uint16_t section_minor = 0;
    uint32_t section_flags = 0;
    uint64_t section_offset_wire = 0;
    uint64_t section_size_wire = 0;
    CHECK(read_u32_le(bytes, entry, section_index));
    CHECK(read_u32_le(bytes, entry + 4, section_type));
    CHECK(read_u16_le(bytes, entry + 8, section_major));
    CHECK(read_u16_le(bytes, entry + 10, section_minor));
    CHECK(read_u32_le(bytes, entry + 12, section_flags));
    CHECK(read_u64_le(bytes, entry + 16, section_offset_wire));
    CHECK(read_u64_le(bytes, entry + 24, section_size_wire));
    CHECK(section_index == 0);
    CHECK(section_type == MTP_STATE_V1_SECTION_TYPE);
    CHECK(section_major == 1 && section_minor == 0);
    CHECK(section_flags == MTP_STATE_V1_SECTION_REQUIRED);
    CHECK(section_offset_wire == MTP_STATE_V1_HEADER_SIZE + MTP_STATE_V1_SECTION_ENTRY_SIZE);
    CHECK(section_offset_wire <= bytes.size());
    CHECK(section_size_wire <= bytes.size());

    const size_t section_offset = (size_t) section_offset_wire;
    const size_t section_size = (size_t) section_size_wire;
    size_t section_end = 0;
    CHECK(checked_add_size(section_offset, section_size, section_end));
    CHECK(section_end == bytes.size());
    CHECK(section_size >= MTP_STATE_V1_PREFIX_SIZE);

    const size_t payload = section_offset;
    uint32_t feature_width = 0;
    uint32_t warmed_heads = 0;
    uint32_t hidden_count = 0;
    uint32_t payload_flags = 0;
    uint64_t context_size_wire = 0;
    CHECK(read_u32_le(bytes, payload, feature_width));
    CHECK(read_u32_le(bytes, payload + 4, warmed_heads));
    CHECK(read_u32_le(bytes, payload + 8, hidden_count));
    CHECK(read_u32_le(bytes, payload + 12, payload_flags));
    CHECK(read_u64_le(bytes, payload + 32, context_size_wire));
    CHECK(feature_width == expected_feature_width);
    CHECK(warmed_heads == 1);
    CHECK(hidden_count == feature_width);
    CHECK(payload_flags == 1);
    CHECK(context_size_wire > 0 && context_size_wire <= bytes.size());

    size_t hidden_bytes = 0;
    size_t expected_section_size = MTP_STATE_V1_PREFIX_SIZE;
    CHECK(checked_multiply_size(hidden_count, sizeof(float), hidden_bytes));
    CHECK(checked_add_size(expected_section_size, hidden_bytes, expected_section_size));
    CHECK(checked_add_size(expected_section_size, (size_t) context_size_wire, expected_section_size));
    CHECK(expected_section_size == section_size);

    mtp_state_v1_view result = {};
    result.feature_width = feature_width;
    result.hidden_count = hidden_count;
    CHECK(checked_add_size(payload, MTP_STATE_V1_DIGEST_OFFSET, result.digest_begin));
    CHECK(checked_add_size(result.digest_begin, MTP_STATE_V1_DIGEST_SIZE, result.digest_end));
    CHECK(checked_add_size(payload, MTP_STATE_V1_PREFIX_SIZE, result.hidden_begin));
    CHECK(result.digest_end == result.hidden_begin);
    CHECK(checked_add_size(result.hidden_begin, hidden_bytes, result.context_begin));
    CHECK(checked_add_size(result.context_begin, (size_t) context_size_wire, result.context_end));
    CHECK(result.context_end == section_end);
    return result;
}

static float read_f32_le(
        const std::vector<uint8_t> & bytes,
        size_t offset,
        uint32_t & bits) {
    CHECK(read_u32_le(bytes, offset, bits));
    float result = 0.0f;
    static_assert(sizeof(result) == sizeof(bits), "F32 wire decoding requires 32-bit float");
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint32_t f32_host_bits(float value) {
    uint32_t result = 0;
    static_assert(sizeof(result) == sizeof(value), "F32 comparison requires 32-bit float");
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

static llama_speculative_metrics bridge_metrics(llama_speculative_session * session) {
    llama_speculative_metrics metrics = {};
    llama_speculative_error * error = nullptr;
    check_speculative_status(
        llama_speculative_session_get_metrics(session, &metrics, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    return metrics;
}

static void check_mtp_bridge(
        llama_model * target_model,
        llama_model * companion_model,
        int32_t n_max) {
    CHECK(n_max >= 1 && n_max <= 4);
    llama_speculative_model_binding binding = {};
    binding.name = "draft";
    binding.model = companion_model;
    binding.context_size = 32;
    binding.threads = 2;
    binding.batch_threads = 2;

    llama_speculative_engine_params engine_params = llama_speculative_engine_default_params();
    engine_params.target_model = target_model;
    engine_params.auxiliary_models = &binding;
    engine_params.auxiliary_model_count = 1;
    const std::string stage_expression =
        "mtp:n_max=" + std::to_string(n_max) + ",p_min=0,heads=1";
    engine_params.stage_expression = stage_expression.c_str();

    llama_speculative_error * error = nullptr;
    llama_speculative_engine * engine = nullptr;
    check_speculative_status(
        llama_speculative_engine_create(&engine_params, &engine, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    CHECK(engine != nullptr);

    llama_speculative_engine_capabilities capabilities = {};
    check_speculative_status(
        llama_speculative_engine_get_capabilities(engine, &capabilities, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    CHECK((capabilities.flags & LLAMA_SPECULATIVE_CAP_CANONICAL_STATE) != 0);
    CHECK((capabilities.flags & LLAMA_SPECULATIVE_CAP_PORTABLE_CANONICAL_STATE) != 0);
    CHECK((capabilities.flags & LLAMA_SPECULATIVE_CAP_REQUIRES_TARGET_MTP_OUTPUT) != 0);
    CHECK((capabilities.flags & LLAMA_SPECULATIVE_CAP_TARGET_ONLY_WARMUP) == 0);
    CHECK(capabilities.history_requirement == LLAMA_SPECULATIVE_HISTORY_COMPLETE_PREFIX);
    CHECK(capabilities.configured_max_draft_tokens == (uint32_t) n_max);

    context_ptr plain = make_plain_context(target_model);
    CHECK(plain != nullptr);
    llama_speculative_session_params session_params = {};
    session_params.target_context = plain.get();
    llama_speculative_session * invalid_session = nullptr;
    llama_speculative_call_result call = {};
    check_speculative_status(
        llama_speculative_session_create(engine, &session_params, &invalid_session, &call, &error),
        LLAMA_SPECULATIVE_MODEL_INCOMPATIBLE,
        &error);
    CHECK(invalid_session == nullptr);
    plain.reset();

    context_ptr oversized_target = make_context(
        target_model, MTP_OP_NONE, nullptr, nullptr, -1, 512);
    CHECK(oversized_target != nullptr);
    session_params.target_context = oversized_target.get();
    check_speculative_status(
        llama_speculative_session_create(engine, &session_params, &invalid_session, &call, &error),
        LLAMA_SPECULATIVE_UNSUPPORTED,
        &error);
    CHECK(invalid_session == nullptr);
    oversized_target.reset();

    context_ptr target = make_context(target_model, MTP_OP_NONE);
    CHECK(target != nullptr);
    session_params.target_context = target.get();
    llama_speculative_session * session = nullptr;
    check_speculative_status(
        llama_speculative_session_create(engine, &session_params, &session, &call, &error),
        LLAMA_SPECULATIVE_OK,
        &error);

    const std::vector<llama_token> history = { 3, 4 };
    bridge_decode(session, history, 0);

    llama_speculative_state_info before_info = {};
    const std::vector<uint8_t> before_round = save_bridge_state(session, before_info);
    CHECK(before_info.readiness == LLAMA_SPECULATIVE_CONDITIONING_READY);
    CHECK((before_info.flags & LLAMA_SPECULATIVE_STATE_CAN_DRAFT) != 0);
    CHECK(before_info.next_position == 2);

    llama_speculative_sampler_params sampler_params = llama_speculative_sampler_default_params();
    sampler_params.top_k = -1;
    sampler_params.top_p = NAN;
    sampler_params.min_p = NAN;
    sampler_params.temperature = 0.0f;
    llama_speculative_sampler * sampler = nullptr;
    check_speculative_status(
        llama_speculative_sampler_create(engine, &sampler_params, &sampler, &error),
        LLAMA_SPECULATIVE_OK,
        &error);

    llama_speculative_generation_begin_params generation = {};
    generation.sampler = sampler;
    generation.history.tokens = history.data();
    generation.history.token_count = history.size();
    generation.history.sequence_start_position = 0;
    generation.history.history_start_position = 0;
    generation.history.next_position = 2;
    generation.history.flags = LLAMA_SPECULATIVE_HISTORY_COMPLETE_FROM_SEQUENCE_START;
    check_speculative_status(
        llama_speculative_session_generation_begin(session, &generation, &call, &error),
        LLAMA_SPECULATIVE_OK,
        &error);

    llama_speculative_round_params round_params = {};
    round_params.cooperative_token_allowance = (uint32_t) n_max + 2;
    round_params.generation_token_allowance = (uint32_t) n_max + 2;
    round_params.context_token_allowance = (uint32_t) n_max + 2;
    llama_speculative_round * round = nullptr;
    check_speculative_status(
        llama_speculative_round_begin(session, &round_params, &round, &call, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    llama_speculative_round_view view = {};
    check_speculative_status(
        llama_speculative_round_get_view(round, &view, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    CHECK(view.proposed_draft_tokens > 0);
    if (n_max == 4) {
        CHECK(view.proposed_draft_tokens == 4);
    }
    CHECK(view.committable_count > 0 && view.tokens != nullptr);
    CHECK((view.flags & LLAMA_SPECULATIVE_ROUND_USED_SPECULATION) != 0);
    check_speculative_status(
        llama_speculative_round_abort(&round, &call, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    CHECK(round == nullptr);
    check_speculative_status(
        llama_speculative_session_generation_end(session, &call, &error),
        LLAMA_SPECULATIVE_OK,
        &error);

    llama_speculative_state_info after_abort_info = {};
    CHECK(save_bridge_state(session, after_abort_info) == before_round);

    check_speculative_status(
        llama_speculative_session_generation_begin(session, &generation, &call, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    const llama_speculative_metrics before_commit_metrics = bridge_metrics(session);
    check_speculative_status(
        llama_speculative_round_begin(session, &round_params, &round, &call, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    check_speculative_status(
        llama_speculative_round_get_view(round, &view, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    CHECK(view.committable_count > 0);
    if (n_max == 4) {
        CHECK(view.proposed_draft_tokens == 4);
    }
    const std::vector<llama_token> verification_tokens(
            view.tokens, view.tokens + view.committable_count);
    const llama_token committed_token = view.tokens[0];
    llama_speculative_round_commit_result commit = {};
    check_speculative_status(
        llama_speculative_round_commit(&round, 1, &commit, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    CHECK(commit.call.resulting_position == 3);
    const llama_speculative_metrics after_commit_metrics = bridge_metrics(session);
    // One target decode verifies the round. A base replay of the committed
    // prefix would add another target decode call here.
    CHECK(after_commit_metrics.target_decode_calls == before_commit_metrics.target_decode_calls + 1);
    check_speculative_status(
        llama_speculative_session_generation_end(session, &call, &error),
        LLAMA_SPECULATIVE_OK,
        &error);

    llama_speculative_state_info committed_info = {};
    const std::vector<uint8_t> committed_state = save_bridge_state(session, committed_info);
    CHECK(committed_info.readiness == LLAMA_SPECULATIVE_CONDITIONING_READY);
    CHECK(committed_info.next_position == 3);

    const std::vector<llama_token> restored_history = { 3, 4, committed_token };
    generation.history.tokens = restored_history.data();
    generation.history.token_count = restored_history.size();
    generation.history.next_position = 3;
    check_speculative_status(
        llama_speculative_session_generation_begin(session, &generation, &call, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    check_speculative_status(
        llama_speculative_round_begin(session, &round_params, &round, &call, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    check_speculative_status(
        llama_speculative_round_get_view(round, &view, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    CHECK(view.proposed_draft_tokens > 0);
    CHECK(view.committable_count > 0 && view.tokens != nullptr);
    const uint32_t live_next_proposed = view.proposed_draft_tokens;
    const uint32_t live_next_flags = view.flags;
    const std::vector<llama_token> live_next_tokens(
        view.tokens, view.tokens + view.committable_count);
    check_speculative_status(
        llama_speculative_round_abort(&round, &call, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    check_speculative_status(
        llama_speculative_session_generation_end(session, &call, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    llama_speculative_state_info after_next_abort_info = {};
    CHECK(save_bridge_state(session, after_next_abort_info) == committed_state);

    context_ptr replay_target = make_context(target_model, MTP_OP_NONE);
    CHECK(replay_target != nullptr);
    llama_speculative_session_params replay_params = {};
    replay_params.target_context = replay_target.get();
    llama_speculative_session * replay_session = nullptr;
    check_speculative_status(
        llama_speculative_session_create(engine, &replay_params, &replay_session, &call, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    bridge_decode(replay_session, { 3, 4 }, 0);
    bridge_decode(replay_session, { committed_token }, 2);
    llama_speculative_state_info replay_info = {};
    const std::vector<uint8_t> replay_state = save_bridge_state(replay_session, replay_info);
    if (n_max < 4) {
        CHECK(replay_state == committed_state);
    } else {
        // At five verification rows the CPU backend changes matrix kernels. To
        // distinguish that reduction-order difference from checkpoint damage,
        // compare each wire row bit-exactly with an independent execution of its
        // own shape. The companion context must remain byte-identical.
        const int64_t feature_width = llama_model_mtp_feature_width(target_model);
        CHECK(feature_width > 0 && feature_width <= UINT32_MAX);
        const mtp_state_v1_view committed = parse_mtp_state_v1(
                committed_state, (uint32_t) feature_width);
        const mtp_state_v1_view replay = parse_mtp_state_v1(
                replay_state, (uint32_t) feature_width);
        CHECK(committed_info.container_version == 1);
        CHECK(replay_info.container_version == 1);
        CHECK(committed_info.flags == LLAMA_SPECULATIVE_STATE_CAN_DRAFT);
        CHECK(replay_info.flags == committed_info.flags);
        CHECK(committed_info.section_count == 1 && replay_info.section_count == 1);
        CHECK(committed_info.readiness == LLAMA_SPECULATIVE_CONDITIONING_READY);
        CHECK(replay_info.readiness == committed_info.readiness);
        CHECK(committed_info.sequence_start_position == replay_info.sequence_start_position);
        CHECK(committed_info.next_position == replay_info.next_position);
        CHECK(committed_info.canonical_state_bytes == committed_state.size());
        CHECK(replay_info.canonical_state_bytes == replay_state.size());
        CHECK(committed_info.derived_state_bytes == 0 && replay_info.derived_state_bytes == 0);
        CHECK(committed.feature_width == replay.feature_width);
        CHECK(committed.hidden_count == replay.hidden_count);
        CHECK(committed.digest_begin == replay.digest_begin);
        CHECK(committed.context_begin == replay.context_begin);
        CHECK(committed.context_end == replay.context_end);
        CHECK(std::equal(
                replay_state.begin(), replay_state.begin() + replay.digest_begin,
                committed_state.begin(), committed_state.begin() + committed.digest_begin));
        CHECK(std::equal(
                replay_state.begin() + replay.context_begin,
                replay_state.begin() + replay.context_end,
                committed_state.begin() + committed.context_begin,
                committed_state.begin() + committed.context_end));

        CHECK(verification_tokens.size() == 5);
        context_ptr batched_reference = make_context(target_model, MTP_OP_NONE);
        context_ptr scalar_reference = make_context(target_model, MTP_OP_NONE);
        CHECK(batched_reference != nullptr && scalar_reference != nullptr);
        decode_many(batched_reference.get(), history, 0);
        decode_many(scalar_reference.get(), history, 0);
        decode_many(batched_reference.get(), verification_tokens, 2);
        decode_one(scalar_reference.get(), committed_token, 2);
        const std::vector<float> batched_hidden = copy_embedding(
                batched_reference.get(), (int32_t) feature_width, 0);
        const std::vector<float> scalar_hidden = copy_embedding(
                scalar_reference.get(), (int32_t) feature_width, 0);

        size_t committed_cursor = committed.hidden_begin;
        size_t replay_cursor = replay.hidden_begin;
        for (uint32_t i = 0; i < committed.hidden_count; ++i) {
            uint32_t committed_bits = 0;
            uint32_t replay_bits = 0;
            const float committed_value = read_f32_le(
                    committed_state, committed_cursor, committed_bits);
            const float replay_value = read_f32_le(
                    replay_state, replay_cursor, replay_bits);
            CHECK(std::isfinite(committed_value));
            CHECK(std::isfinite(replay_value));
            CHECK(committed_bits == f32_host_bits(batched_hidden[i]));
            CHECK(replay_bits == f32_host_bits(scalar_hidden[i]));
            committed_cursor += sizeof(float);
            replay_cursor += sizeof(float);
        }
        CHECK(committed_cursor == committed.context_begin);
        CHECK(replay_cursor == replay.context_begin);
    }
    check_speculative_status(
        llama_speculative_session_quiesce_and_detach(replay_session, &call, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    llama_speculative_session_destroy(replay_session);
    replay_target.reset();

    uint8_t quality_key[32] = {};
    llama_speculative_state_inspect_params inspect = {};
    inspect.state = committed_state.data();
    inspect.state_size = committed_state.size();
    inspect.quality_key = quality_key;
    inspect.quality_key_capacity = sizeof(quality_key);
    llama_speculative_state_info inspected = {};
    check_speculative_status(
        llama_speculative_engine_state_inspect(engine, &inspect, &inspected, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    CHECK(inspected.readiness == LLAMA_SPECULATIVE_CONDITIONING_READY);
    CHECK(inspected.next_position == 3);

    std::vector<uint8_t> corrupted = committed_state;
    const int64_t corruption_feature_width = llama_model_mtp_feature_width(target_model);
    CHECK(corruption_feature_width > 0 && corruption_feature_width <= UINT32_MAX);
    const mtp_state_v1_view corruption_view = parse_mtp_state_v1(
            corrupted, (uint32_t) corruption_feature_width);
    CHECK(corruption_view.hidden_begin < corruption_view.context_begin);
    corrupted[corruption_view.hidden_begin] ^= 1;
    inspect.state = corrupted.data();
    check_speculative_status(
        llama_speculative_engine_state_inspect(engine, &inspect, &inspected, &error),
        LLAMA_SPECULATIVE_STATE_CORRUPT,
        &error);

    check_speculative_status(
        llama_speculative_session_quiesce_and_detach(session, &call, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    llama_speculative_session_destroy(session);
    session = nullptr;

    session_params.next_position = 3;
    check_speculative_status(
        llama_speculative_session_create(engine, &session_params, &session, &call, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    check_speculative_status(
        llama_speculative_session_state_load(
            session, committed_state.data(), committed_state.size(), &call, &error),
        LLAMA_SPECULATIVE_OK,
        &error);

    check_speculative_status(
        llama_speculative_session_generation_begin(session, &generation, &call, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    check_speculative_status(
        llama_speculative_round_begin(session, &round_params, &round, &call, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    check_speculative_status(
        llama_speculative_round_get_view(round, &view, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    CHECK(view.proposed_draft_tokens == live_next_proposed);
    CHECK(view.flags == live_next_flags);
    CHECK(view.committable_count == live_next_tokens.size());
    CHECK(view.tokens != nullptr);
    CHECK(std::equal(live_next_tokens.begin(), live_next_tokens.end(), view.tokens));
    check_speculative_status(
        llama_speculative_round_abort(&round, &call, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    check_speculative_status(
        llama_speculative_session_generation_end(session, &call, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    check_speculative_status(
        llama_speculative_session_quiesce_and_detach(session, &call, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
    llama_speculative_session_destroy(session);
    llama_speculative_sampler_destroy(sampler);
    check_speculative_status(
        llama_speculative_engine_destroy(&engine, &error),
        LLAMA_SPECULATIVE_OK,
        &error);
}
#endif

int main(int argc, char ** argv) {
    std::array<const char *, 4> paths = { nullptr, nullptr, nullptr, nullptr };
    std::vector<const char *> rejected_paths;
    size_t n_paths = 0;
    bool bridge_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bridge-only") == 0) {
            bridge_only = true;
        } else if (std::strcmp(argv[i], "--reject-load") == 0 && i + 1 < argc) {
            rejected_paths.push_back(argv[++i]);
        } else if (n_paths < paths.size()) {
            paths[n_paths++] = argv[i];
        } else {
            std::fprintf(stderr,
                    "usage: %s [EMBEDDED [SPLIT-COMPANION [FUSED-COMPANION "
                    "[SHARED-COMPANION]]]] [--reject-load PATH]... [--bridge-only]\n",
                    argv[0]);
            return 1;
        }
    }

    const char * model_path = paths[0] != nullptr
            ? paths[0] : std::getenv("LLAMACPP_TEST_QWEN4EXP_MTP_MODELFILE");
    const char * companion_path = paths[1] != nullptr
            ? paths[1] : std::getenv("LLAMACPP_TEST_QWEN4EXP_MTP_COMPANION");
    const char * fused_path = paths[2] != nullptr
            ? paths[2] : std::getenv("LLAMACPP_TEST_QWEN4EXP_MTP_FUSED_COMPANION");
    const char * shared_path = paths[3] != nullptr
            ? paths[3] : std::getenv("LLAMACPP_TEST_QWEN4EXP_MTP_SHARED_COMPANION");
    if (model_path == nullptr || model_path[0] == '\0') {
        std::fprintf(stderr,
                "test-qwen4exp-mtp: skipped (generate a tiny model with "
                "tests/generate-qwen4exp-mtp-gguf.py and pass its path)\n");
        return 0;
    }
    llama_backend_init();

    test_metadata(model_path);

    for (const char * rejected_path : rejected_paths) {
        CHECK(load_model(rejected_path, true) == nullptr);
    }

    {
        model_ptr target_only = load_model(model_path, false);
        CHECK(target_only != nullptr);
        CHECK(llama_model_n_nextn_layer(target_only.get()) == 1);
        CHECK(llama_model_mtp_package(target_only.get()) == LLAMA_MTP_PACKAGE_TARGET_ONLY);

        context_ptr plain = make_plain_context(target_only.get());
        CHECK(plain != nullptr);
        decode_one(plain.get(), 3, 0);
        const auto expected_logits = copy_logits(plain.get(), llama_n_vocab(target_only.get()));
        CHECK(expected_logits.size() ==
                (size_t) llama_n_vocab(target_only.get()));

#ifdef __linux__
        llama_model_params mmap_params = llama_model_default_params();
        mmap_params.use_mmap = true;
        mmap_params.defer_ple = true;
        model_ptr mapped(llama_model_load_from_file(model_path, mmap_params), llama_free_model);
        CHECK(mapped != nullptr);
        context_ptr mapped_ctx = make_plain_context(mapped.get());
        CHECK(mapped_ctx != nullptr);
        decode_one(mapped_ctx.get(), 3, 0);
        const auto mapped_logits = copy_logits(mapped_ctx.get(), llama_n_vocab(mapped.get()));
        CHECK(mapped_logits.size() == expected_logits.size());
        for (size_t i = 0; i < expected_logits.size(); ++i) {
            CHECK(std::fabs(mapped_logits[i] - expected_logits[i]) < 1e-5f);
        }
#endif

        context_ptr embedding = make_plain_context(target_only.get(), true);
        CHECK(embedding != nullptr);
        decode_one(embedding.get(), 3, 0);
        CHECK(copy_embedding(embedding.get(), llama_model_n_embd(target_only.get())).size() ==
                (size_t) llama_model_n_embd(target_only.get()));
    }

    model_ptr model = load_model(model_path, true);
    CHECK(model != nullptr);
    CHECK(llama_model_mtp_package(model.get()) == LLAMA_MTP_PACKAGE_EMBEDDED);
    CHECK(llama_model_n_nextn_layer(model.get()) == 1);

    const int32_t width = (int32_t) llama_model_mtp_feature_width(model.get());
    const int32_t n_vocab = llama_n_vocab(model.get());
    CHECK(width == 512);
    CHECK(n_vocab == 16);

    const auto load_companion = [width, n_vocab](const char * path) {
        model_ptr result(nullptr, llama_free_model);
        if (path == nullptr || path[0] == '\0') {
            return result;
        }
        result = load_model(path, true);
        CHECK(result != nullptr);
        CHECK(llama_model_mtp_package(result.get()) == LLAMA_MTP_PACKAGE_COMPANION);
        CHECK(llama_model_n_nextn_layer(result.get()) == 1);
        CHECK((int32_t) llama_model_mtp_feature_width(result.get()) == width);
        CHECK(llama_n_vocab(result.get()) == n_vocab);
        return result;
    };

    model_ptr companion = load_companion(companion_path);
    model_ptr fused = load_companion(fused_path);
    model_ptr shared_target(nullptr, llama_free_model);
    model_ptr shared = load_companion(shared_path);

    if (companion) {
        context_ptr companion_checkpoint = make_context(companion.get(), MTP_OP_UPDATE_ACCEPTED);
        CHECK(companion_checkpoint != nullptr);
        CHECK(llama_spec_ckpt_init(
                companion_checkpoint.get(), LLAMA_SPEC_CKPT_PER_STEP, 2) ==
                LLAMA_SPEC_CKPT_NONE);
    }

    if (fused) {
        CHECK(llama_model_speculative_io_mode(fused.get(), model.get()) ==
                LLAMA_DFLASH_IO_MODE_SELF_CONTAINED);
        CHECK(llama_model_share_speculative_io_tensors(fused.get(), model.get()));
    }

    if (shared) {
        CHECK(llama_model_speculative_io_mode(shared.get(), model.get()) ==
                LLAMA_DFLASH_IO_MODE_INVALID);
        CHECK(make_context(shared.get(), MTP_OP_DRAFT_GEN) == nullptr);
        shared_target = load_model(model_path, false);
        CHECK(shared_target != nullptr);
        CHECK(llama_model_share_speculative_io_tensors(
                shared.get(), shared_target.get()));
        CHECK(llama_model_speculative_io_mode(shared.get(), shared_target.get()) ==
                LLAMA_DFLASH_IO_MODE_SHARED);
        model_ptr different_target = load_model(model_path, false);
        CHECK(different_target != nullptr);
        CHECK(!llama_model_share_speculative_io_tensors(
                shared.get(), different_target.get()));
    }

#if defined(LLAMA_TEST_SPECULATIVE_BRIDGE)
    if (bridge_only) {
        model_ptr bridge_target = load_model(model_path, false);
        CHECK(bridge_target != nullptr);
        CHECK(companion != nullptr || fused != nullptr || shared != nullptr);
        for (llama_model * draft : { companion.get(), fused.get() }) {
            if (draft == nullptr) {
                continue;
            }
            check_mtp_bridge(bridge_target.get(), draft, 1);
            check_mtp_bridge(bridge_target.get(), draft, 2);
            check_mtp_bridge(bridge_target.get(), draft, 4);
        }
        if (shared) {
            check_mtp_bridge(shared_target.get(), shared.get(), 1);
            check_mtp_bridge(shared_target.get(), shared.get(), 2);
            check_mtp_bridge(shared_target.get(), shared.get(), 4);
        }
        shared.reset();
        shared_target.reset();
        fused.reset();
        companion.reset();
        model.reset();
        llama_backend_free();
        std::fprintf(stderr, "test-qwen4exp-mtp: bridge success\n");
        return 0;
    }
#else
    CHECK(!bridge_only);
#endif

    check_joint_hidden_norm(model.get(), width);
    if (fused) {
        check_joint_hidden_norm(fused.get(), width);
    }
    if (shared) {
        check_joint_hidden_norm(shared.get(), width);
    }

    std::vector<float> norm_input(width);
    for (int32_t i = 0; i < width/2; ++i) {
        norm_input[i] = (float) (i + 1);
        norm_input[width/2 + i] = (float) (i + 20);
    }

    for (const int32_t top_k : { 0, 1, 3 }) {
        CHECK(make_context(model.get(), MTP_OP_DRAFT_GEN, nullptr, nullptr, top_k) == nullptr);
    }

    check_qsa_reuse_padding_boundary(model.get(), true);
    check_qsa_reuse_padding_boundary(model.get(), false);

    selection_capture selection;
    context_ptr top_k_check = make_context(
            model.get(), MTP_OP_UPDATE_ACCEPTED, capture_qsa_selection, &selection, 2);
    CHECK(top_k_check != nullptr);
    seed_draft_step(top_k_check.get(), norm_input, 3, 0, 0);
    seed_draft_step(top_k_check.get(), norm_input, 4, 1, 0);
    CHECK(selection.values == std::vector<int32_t>({ 0, 1, -1 }));

    selection_capture carried_selection;
    context_ptr carried = make_context(
            model.get(), MTP_OP_UPDATE_ACCEPTED, capture_qsa_selection, &carried_selection, 2);
    CHECK(carried != nullptr);
    seed_draft_step(carried.get(), norm_input, 3, 0, 0);
    CHECK(!carried_selection.values.empty());

    std::vector<int32_t> expected_carry = carried_selection.values;
    expected_carry.erase(std::remove(expected_carry.begin(), expected_carry.end(), -1),
            expected_carry.end());
    llama_set_mtp_op_type(carried.get(), MTP_OP_DRAFT_GEN);
    seed_draft_step(carried.get(), norm_input, 4, 1, 0);
    CHECK(!carried->qwen4_mtp_qsa_pending);
    CHECK(carried->qwen4_mtp_qsa_topk == expected_carry);
    CHECK(carried_selection.mask.size() > 1);
    CHECK(std::isfinite(carried_selection.mask[0]));
    CHECK(std::isfinite(carried_selection.mask[1]));
    seed_draft_step(carried.get(), norm_input, 5, 2, 1);
    CHECK(carried->qwen4_mtp_qsa_topk == expected_carry);
    CHECK(carried_selection.mask.size() > 2);
    CHECK(std::isfinite(carried_selection.mask[0]));
    CHECK(std::isfinite(carried_selection.mask[1]));
    CHECK(std::isfinite(carried_selection.mask[2]));

    context_ptr target = make_context(model.get(), MTP_OP_NONE);
    CHECK(target != nullptr);
    decode_one(target.get(), 3, 0);
    std::vector<float> target_hidden = copy_embedding(target.get(), width);

    context_ptr reuse = make_context(model.get(), MTP_OP_DRAFT_GEN);
    context_ptr recompute = make_context(model.get(), MTP_OP_DRAFT_GEN);
    CHECK(reuse != nullptr);
    CHECK(recompute != nullptr);

    seed_draft_step(reuse.get(), target_hidden, 3, 0, 0);
    seed_draft_step(recompute.get(), target_hidden, 3, 0, 0);
    std::vector<float> next_hidden = copy_embedding(reuse.get(), width);
    std::vector<float> next_hidden_recompute = copy_embedding(recompute.get(), width);
    CHECK(next_hidden == next_hidden_recompute);
    const std::vector<float> first_logits = copy_logits(recompute.get(), n_vocab);

    for (llama_model * candidate : { companion.get(), fused.get(), shared.get() }) {
        if (candidate == nullptr) {
            continue;
        }
        context_ptr external = make_context(candidate, MTP_OP_DRAFT_GEN);
        CHECK(external != nullptr);
        seed_draft_step(external.get(), target_hidden, 3, 0, 0);
        check_close(copy_embedding(external.get(), width), next_hidden_recompute);
        check_close(copy_logits(external.get(), n_vocab), first_logits);
    }

    seed_draft_step(reuse.get(), next_hidden, 4, 1, 1);
    llama_set_mtp_op_type(recompute.get(), MTP_OP_UPDATE_ACCEPTED);
    seed_draft_step(recompute.get(), next_hidden_recompute, 4, 1, 0);

    CHECK(!reuse->qwen4_mtp_qsa_topk.empty());
    CHECK(!reuse->qwen4_mtp_qsa_pending);
    CHECK(reuse->qwen4_mtp_qsa_topk != recompute->qwen4_mtp_qsa_topk);

    recompute.reset();
    reuse.reset();
    target.reset();
    carried.reset();
    top_k_check.reset();
#if defined(LLAMA_TEST_SPECULATIVE_BRIDGE)
    if (companion || fused) {
        model_ptr bridge_target = load_model(model_path, false);
        CHECK(bridge_target != nullptr);
        for (llama_model * draft : { companion.get(), fused.get() }) {
            if (draft == nullptr) {
                continue;
            }
            check_mtp_bridge(bridge_target.get(), draft, 1);
            check_mtp_bridge(bridge_target.get(), draft, 2);
            check_mtp_bridge(bridge_target.get(), draft, 4);
        }
    }
    if (shared) {
        check_mtp_bridge(shared_target.get(), shared.get(), 1);
        check_mtp_bridge(shared_target.get(), shared.get(), 2);
        check_mtp_bridge(shared_target.get(), shared.get(), 4);
    }
#endif
    shared.reset();
    shared_target.reset();
    fused.reset();
    companion.reset();
    model.reset();
    llama_backend_free();
    std::fprintf(stderr, "test-qwen4exp-mtp: success\n");
    return 0;
}
