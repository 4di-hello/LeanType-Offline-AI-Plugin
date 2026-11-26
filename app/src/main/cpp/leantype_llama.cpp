// SPDX-License-Identifier: GPL-3.0-only
// JNI bridge between the plugin and llama.cpp: load a GGUF model, generate text for one prompt.
// Texts cross the bridge as UTF-8 byte arrays (JNI's modified UTF-8 breaks emoji), and are never logged.

#include <jni.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <mutex>
#include <string>
#include <vector>

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "LeanTypeLlama"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#else
#define LOGI(...) do { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } while (0)
#define LOGW(...) LOGI(__VA_ARGS__)
#endif

namespace {

std::mutex g_mutex;
bool g_backend_ready = false;
llama_model * g_model = nullptr;
llama_context * g_ctx = nullptr;
common_chat_templates_ptr g_templates;
int g_n_batch = 512;
std::atomic<bool> g_cancel{false};
std::string g_error;
std::string g_stats;

void log_callback(ggml_log_level level, const char * text, void *) {
    // only problems: llama.cpp does not log texts, but the info output is very long
    if (level != GGML_LOG_LEVEL_WARN && level != GGML_LOG_LEVEL_ERROR) return;
#ifdef __ANDROID__
    __android_log_print(level == GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR : ANDROID_LOG_WARN, LOG_TAG, "%s", text);
#else
    fputs(text, stderr);
#endif
}

bool abort_callback(void *) { return g_cancel.load(); }

double ms_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

std::string to_string(JNIEnv * env, jbyteArray bytes) {
    if (bytes == nullptr) return {};
    const jsize length = env->GetArrayLength(bytes);
    std::string result(static_cast<size_t>(length), '\0');
    if (length > 0) env->GetByteArrayRegion(bytes, 0, length, reinterpret_cast<jbyte *>(&result[0]));
    return result;
}

jbyteArray to_bytes(JNIEnv * env, const std::string & text) {
    jbyteArray result = env->NewByteArray(static_cast<jsize>(text.size()));
    if (result != nullptr && !text.empty()) {
        env->SetByteArrayRegion(result, 0, static_cast<jsize>(text.size()), reinterpret_cast<const jbyte *>(text.data()));
    }
    return result;
}

// incomplete UTF-8 sequence at the end of text (a token can end inside a character): its length
size_t incomplete_utf8_tail(const std::string & text) {
    for (size_t back = 1; back <= 4 && back <= text.size(); back++) {
        const auto c = static_cast<unsigned char>(text[text.size() - back]);
        if ((c & 0xC0) == 0x80) continue; // continuation byte
        size_t needed = 1;
        if ((c & 0xE0) == 0xC0) needed = 2;
        else if ((c & 0xF0) == 0xE0) needed = 3;
        else if ((c & 0xF8) == 0xF0) needed = 4;
        return needed > back ? back : 0;
    }
    return 0;
}

void unload_locked() {
    g_templates.reset();
    if (g_ctx != nullptr) { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model != nullptr) { llama_model_free(g_model); g_model = nullptr; }
}

// decodes tokens starting at position pos, logits only for the last one
bool decode(const std::vector<llama_token> & tokens, llama_pos pos) {
    common_batch batch(g_ctx);
    for (size_t i = 0; i < tokens.size(); i += g_n_batch) {
        const size_t n = std::min(tokens.size() - i, static_cast<size_t>(g_n_batch));
        batch.clear();
        for (size_t j = 0; j < n; j++) {
            batch.add(tokens[i + j], pos + static_cast<llama_pos>(i + j), 0, i + j == tokens.size() - 1);
        }
        const int result = llama_process(g_ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (result != 0) {
            g_error = g_cancel.load() ? "cancelled" : "decoding the prompt failed (" + std::to_string(result) + ")";
            return false;
        }
    }
    return true;
}

} // namespace

extern "C" {

JNIEXPORT void JNICALL
Java_helium314_keyboard_ai_plugin_LlamaNative_nativeInit(JNIEnv * env, jclass, jstring jlib_dir) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_backend_ready) return;
    llama_log_set(log_callback, nullptr);
    const char * dir = env->GetStringUTFChars(jlib_dir, nullptr);
    // the CPU backend comes in variants for different processors, the best one for this phone is chosen
    if (dir != nullptr && dir[0] != '\0') ggml_backend_load_all_from_path(dir);
    else ggml_backend_load_all();
    env->ReleaseStringUTFChars(jlib_dir, dir);
    llama_backend_init();
    g_backend_ready = true;
    LOGI("backend ready: %s", llama_print_system_info());
}

JNIEXPORT jboolean JNICALL
Java_helium314_keyboard_ai_plugin_LlamaNative_nativeLoad(JNIEnv * env, jclass, jstring jpath, jint n_ctx,
                                                         jint n_threads, jboolean use_mmap) {
    std::lock_guard<std::mutex> lock(g_mutex);
    unload_locked();
    g_error.clear();
    const auto start = std::chrono::steady_clock::now();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.load_mode = use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
    const char * path = env->GetStringUTFChars(jpath, nullptr);
    g_model = llama_model_load_from_file(path, model_params);
    env->ReleaseStringUTFChars(jpath, path);
    if (g_model == nullptr) {
        g_error = "the model could not be loaded (file damaged or model type not supported)";
        return JNI_FALSE;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx > 0 ? static_cast<uint32_t>(n_ctx) : 4096;
    ctx_params.n_batch = static_cast<uint32_t>(g_n_batch);
    ctx_params.n_ubatch = static_cast<uint32_t>(g_n_batch);
    ctx_params.n_threads = n_threads > 0 ? n_threads : 4;
    ctx_params.n_threads_batch = ctx_params.n_threads;
    ctx_params.abort_callback = abort_callback;
    ctx_params.abort_callback_data = nullptr;
    g_ctx = llama_init_from_model(g_model, ctx_params);
    if (g_ctx == nullptr) {
        g_error = "no context for the model (not enough memory?)";
        unload_locked();
        return JNI_FALSE;
    }
    try {
        g_templates = common_chat_templates_init(g_model, "");
    } catch (const std::exception & e) {
        LOGW("chat template not usable: %s", e.what());
        g_templates.reset();
    }
    char desc[128];
    llama_model_desc(g_model, desc, sizeof(desc));
    LOGI("loaded %s in %.0f ms, context %u, threads %d, chat template %s", desc, ms_since(start),
         llama_n_ctx(g_ctx), ctx_params.n_threads,
         g_templates && common_chat_templates_was_explicit(g_templates.get()) ? "from the model" : "none");
    return JNI_TRUE;
}

JNIEXPORT jbyteArray JNICALL
Java_helium314_keyboard_ai_plugin_LlamaNative_nativeGenerate(
        JNIEnv * env, jclass, jbyteArray jprompt, jbyteArray jsystem, jbyteArray juser, jboolean thinking,
        jint max_tokens, jfloat temperature, jfloat top_p, jint top_k, jfloat min_p, jobjectArray jstops) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_error.clear();
    g_stats.clear();
    g_cancel.store(false);
    if (g_model == nullptr || g_ctx == nullptr) {
        g_error = "no model loaded";
        return nullptr;
    }
    const llama_vocab * vocab = llama_model_get_vocab(g_model);

    std::vector<std::string> stops;
    if (jstops != nullptr) {
        const jsize n = env->GetArrayLength(jstops);
        for (jsize i = 0; i < n; i++) {
            auto stop = static_cast<jbyteArray>(env->GetObjectArrayElement(jstops, i));
            std::string s = to_string(env, stop);
            if (!s.empty()) stops.push_back(s);
            env->DeleteLocalRef(stop);
        }
    }

    // the prompt: chat messages in the chat format of the model if it has one, otherwise the plain prompt
    std::vector<llama_token> tokens;
    bool used_template = false;
    if (juser != nullptr && g_templates && common_chat_templates_was_explicit(g_templates.get())) {
        try {
            common_chat_templates_inputs inputs;
            const std::string system = to_string(env, jsystem);
            if (!system.empty()) {
                common_chat_msg msg;
                msg.role = "system";
                msg.content = system;
                inputs.messages.push_back(msg);
            }
            common_chat_msg msg;
            msg.role = "user";
            msg.content = to_string(env, juser);
            inputs.messages.push_back(msg);
            inputs.add_generation_prompt = true;
            inputs.use_jinja = true;
            inputs.enable_thinking = thinking == JNI_TRUE;
            const common_chat_params chat = common_chat_templates_apply(g_templates.get(), inputs);
            for (size_t i = 0; i < chat.prompt.parts.size(); i++) {
                const auto & part = chat.prompt.parts[i];
                // special tokens only from the template, never from the text (e.g. pasted "<|im_end|>")
                const auto part_tokens = common_tokenize(vocab, part.text, i == 0, !part.is_input);
                tokens.insert(tokens.end(), part_tokens.begin(), part_tokens.end());
            }
            for (const auto & stop : chat.additional_stops) stops.push_back(stop);
            used_template = !tokens.empty();
        } catch (const std::exception & e) {
            LOGW("chat template failed, using the plain prompt: %s", e.what());
            tokens.clear();
        }
    }
    if (!used_template) {
        tokens = common_tokenize(vocab, to_string(env, jprompt), true, true);
    }
    if (tokens.empty()) {
        g_error = "empty prompt";
        return nullptr;
    }

    const int n_ctx = static_cast<int>(llama_n_ctx(g_ctx));
    const int n_prompt = static_cast<int>(tokens.size());
    int n_predict = max_tokens > 0 ? max_tokens : 1024;
    if (n_prompt + 16 > n_ctx) {
        g_error = "the text is too long for the context (" + std::to_string(n_prompt) + " of " + std::to_string(n_ctx) + " tokens)";
        return nullptr;
    }
    n_predict = std::min(n_predict, n_ctx - n_prompt - 1);

    llama_memory_clear(llama_get_memory(g_ctx), true);
    const auto prompt_start = std::chrono::steady_clock::now();
    if (!decode(tokens, 0)) return nullptr;
    const double prompt_ms = ms_since(prompt_start);

    common_params_sampling sampling;
    sampling.temp = temperature;
    sampling.top_p = top_p;
    sampling.top_k = top_k;
    sampling.min_p = min_p;
    common_sampler * sampler = common_sampler_init(g_model, sampling);
    if (sampler == nullptr) {
        g_error = "sampler could not be created";
        return nullptr;
    }

    std::string output;
    int n_generated = 0;
    llama_pos pos = n_prompt;
    common_batch batch(g_ctx);
    const auto gen_start = std::chrono::steady_clock::now();
    bool stopped = false;
    while (n_generated < n_predict && !stopped) {
        if (g_cancel.load()) { g_error = "cancelled"; break; }
        const llama_token token = common_sampler_sample(sampler, g_ctx, -1);
        common_sampler_accept(sampler, token, true);
        if (llama_vocab_is_eog(vocab, token)) break;
        n_generated++;
        output += common_token_to_piece(g_ctx, token, false);
        for (const auto & stop : stops) {
            // only the end can contain a new match
            const size_t from = output.size() > stop.size() + 64 ? output.size() - stop.size() - 64 : 0;
            const size_t found = output.find(stop, from);
            if (found != std::string::npos) {
                output.resize(found);
                stopped = true;
                break;
            }
        }
        if (stopped || n_generated >= n_predict) break;
        batch.clear();
        batch.add(token, pos++, 0, true);
        if (llama_process(g_ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            if (!g_cancel.load()) g_error = "decoding failed after " + std::to_string(n_generated) + " tokens";
            else g_error = "cancelled";
            break;
        }
    }
    const double gen_ms = ms_since(gen_start);
    common_sampler_free(sampler);
    output.resize(output.size() - incomplete_utf8_tail(output));

    char stats[256];
    snprintf(stats, sizeof(stats), "prompt %d tokens %.0f ms (%.1f t/s), generated %d tokens %.0f ms (%.1f t/s), template %s",
             n_prompt, prompt_ms, prompt_ms > 0 ? n_prompt * 1000.0 / prompt_ms : 0.0,
             n_generated, gen_ms, gen_ms > 0 ? n_generated * 1000.0 / gen_ms : 0.0, used_template ? "yes" : "no");
    g_stats = stats;
    LOGI("%s", stats);
    if (g_error == "cancelled") return nullptr;
    return to_bytes(env, output);
}

JNIEXPORT void JNICALL
Java_helium314_keyboard_ai_plugin_LlamaNative_nativeCancel(JNIEnv *, jclass) {
    g_cancel.store(true);
}

JNIEXPORT void JNICALL
Java_helium314_keyboard_ai_plugin_LlamaNative_nativeUnload(JNIEnv *, jclass) {
    g_cancel.store(true);
    std::lock_guard<std::mutex> lock(g_mutex);
    unload_locked();
}

JNIEXPORT jstring JNICALL
Java_helium314_keyboard_ai_plugin_LlamaNative_nativeLastError(JNIEnv * env, jclass) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return env->NewStringUTF(g_error.c_str());
}

JNIEXPORT jstring JNICALL
Java_helium314_keyboard_ai_plugin_LlamaNative_nativeLastStats(JNIEnv * env, jclass) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return env->NewStringUTF(g_stats.c_str());
}

} // extern "C"
