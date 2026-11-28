// SPDX-License-Identifier: GPL-3.0-only
// JNI bridge between the plugin and llama.cpp: load a GGUF model, generate text for one prompt.
// Texts cross the bridge as UTF-8 byte arrays (JNI's modified UTF-8 breaks emoji), and are never logged.

#include <jni.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <sched.h>
#include <unistd.h>
#include <string>
#include <vector>

#include "chat.h"
#include "common.h"
#include "ggml-cpu.h"
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
std::string g_reasoning; // what the model thought in the last generation, only when thinking was on
FILE * g_model_file = nullptr; // open while the model is loaded
std::mutex g_log_mutex;
std::string g_last_log_error; // the last error llama.cpp logged, says why loading failed

void log_callback(ggml_log_level level, const char * text, void *) {
    // only problems: llama.cpp does not log texts, but the info output is very long
    if (level != GGML_LOG_LEVEL_WARN && level != GGML_LOG_LEVEL_ERROR) return;
    if (level == GGML_LOG_LEVEL_ERROR && text != nullptr) {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        g_last_log_error = text;
        while (!g_last_log_error.empty() && (g_last_log_error.back() == '\n' || g_last_log_error.back() == ' ')) g_last_log_error.pop_back();
    }
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

// own threads on the fastest cores (see fast_cores)
ggml_threadpool * g_threadpool = nullptr;
decltype(ggml_threadpool_free) * g_threadpool_free = nullptr;
std::string g_cores; // which cores compute, for the stats: "7,4,5,6 of 0-7"

/**
 * The [n] fastest cores this process may use, by their highest clock. A phone has cores of different
 * speed; with more threads than fast cores every step waits for a slow one (measured: 4 threads twice
 * as fast as 8). Empty if it can't be told, then the system places the threads.
 */
std::vector<int> fast_cores(int n, std::string & desc) {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return {};
    std::vector<std::pair<long, int>> cpus; // max kHz, cpu
    std::string allowed_list;
    for (int i = 0; i < CPU_SETSIZE && i < GGML_MAX_N_THREADS; i++) {
        if (!CPU_ISSET(i, &allowed)) continue;
        allowed_list += (allowed_list.empty() ? "" : ",") + std::to_string(i);
        long khz = 0;
        std::ifstream file("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/cpufreq/cpuinfo_max_freq");
        file >> khz;
        cpus.emplace_back(khz, i);
    }
    desc = "? of " + allowed_list;
    if (cpus.empty() || std::any_of(cpus.begin(), cpus.end(), [](const auto & c) { return c.first <= 0; })) return {};
    std::stable_sort(cpus.begin(), cpus.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
    std::vector<int> chosen;
    std::string chosen_list;
    for (int i = 0; i < n && i < static_cast<int>(cpus.size()); i++) {
        chosen.push_back(cpus[i].second);
        chosen_list += (chosen_list.empty() ? "" : ",") + std::to_string(cpus[i].second);
    }
    desc = chosen_list + " of " + allowed_list;
    return chosen;
}

/** Threads of llama.cpp only on the fastest cores; without it the system places them */
void attach_fast_threads(int n_threads) {
    std::vector<int> cores = fast_cores(n_threads, g_cores);
    if (cores.empty()) { LOGW("cores not known (%s), the system places the threads", g_cores.c_str()); return; }
    auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    auto * new_fn = reg ? (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new") : nullptr;
    g_threadpool_free = reg ? (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free") : nullptr;
    if (new_fn == nullptr || g_threadpool_free == nullptr) { LOGW("no thread pool in the CPU backend"); return; }
    ggml_threadpool_params params = ggml_threadpool_params_default(n_threads);
    for (int core : cores) params.cpumask[core] = true;
    params.strict_cpu = false; // all threads may use all chosen cores
    g_threadpool = new_fn(&params);
    if (g_threadpool == nullptr) { LOGW("thread pool not created"); return; }
    llama_attach_threadpool(g_ctx, g_threadpool, nullptr);
    LOGI("threads on cores %s", g_cores.c_str());
}

void unload_locked() {
    g_templates.reset();
    if (g_ctx != nullptr) { llama_free(g_ctx); g_ctx = nullptr; }
    // after the context, which used it
    if (g_threadpool != nullptr && g_threadpool_free != nullptr) g_threadpool_free(g_threadpool);
    g_threadpool = nullptr;
    g_cores.clear();
    if (g_model != nullptr) { llama_model_free(g_model); g_model = nullptr; }
    if (g_model_file != nullptr) { fclose(g_model_file); g_model_file = nullptr; }
}

std::string last_log_error() {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    return g_last_log_error;
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
    LOGI("backend ready (%zu devices): %s", ggml_backend_dev_count(), llama_print_system_info());
}

JNIEXPORT jboolean JNICALL
Java_helium314_keyboard_ai_plugin_LlamaNative_nativeLoad(JNIEnv * env, jclass, jstring jpath, jint fd, jint n_ctx,
                                                         jint n_threads, jboolean use_mmap) {
    std::lock_guard<std::mutex> lock(g_mutex);
    unload_locked();
    g_error.clear();
    { std::lock_guard<std::mutex> log_lock(g_log_mutex); g_last_log_error.clear(); }
    const auto start = std::chrono::steady_clock::now();

    llama_model_params model_params = llama_model_default_params();
    if (ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) == nullptr) {
        g_error = "no CPU backend: the libraries of the plugin were not found";
        return JNI_FALSE;
    }
    model_params.n_gpu_layers = 0;
    model_params.load_mode = use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
    if (fd >= 0) {
        // a document picked by the user: read through the descriptor we got for it, the app may not
        // be allowed to open the file by its path
        const int own_fd = dup(fd);
        // the duplicate shares the position with the original, which a failed attempt may have moved
        if (own_fd >= 0) lseek(own_fd, 0, SEEK_SET);
        g_model_file = own_fd >= 0 ? fdopen(own_fd, "rb") : nullptr;
        if (g_model_file == nullptr) {
            if (own_fd >= 0) close(own_fd);
            g_error = "the model file can't be read";
            return JNI_FALSE;
        }
        g_model = llama_model_load_from_file_ptr(g_model_file, model_params);
    } else {
        const char * path = env->GetStringUTFChars(jpath, nullptr);
        g_model = llama_model_load_from_file(path, model_params);
        env->ReleaseStringUTFChars(jpath, path);
    }
    if (g_model == nullptr) {
        const std::string reason = last_log_error();
        g_error = reason.empty() ? "file damaged or model type not supported" : reason;
        unload_locked();
        return JNI_FALSE;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx > 0 ? static_cast<uint32_t>(n_ctx) : 4096;
    ctx_params.n_batch = static_cast<uint32_t>(g_n_batch);
    ctx_params.n_ubatch = static_cast<uint32_t>(g_n_batch);
    ctx_params.n_threads = n_threads > 0 ? n_threads : 4;
    ctx_params.n_threads_batch = ctx_params.n_threads;
    // test: without flash attention (Gemma 4 has different head sizes per layer)
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.abort_callback = abort_callback;
    ctx_params.abort_callback_data = nullptr;
    g_ctx = llama_init_from_model(g_model, ctx_params);
    if (g_ctx == nullptr) {
        g_error = "no context for the model (not enough memory?)";
        unload_locked();
        return JNI_FALSE;
    }
    attach_fast_threads(ctx_params.n_threads);
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
    g_reasoning.clear();
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
    // knows the chat format of the model: separates what it thinks from the answer
    std::unique_ptr<common_chat_session> session;
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
            // thoughts go into reasoning_content, not into the answer (as llama-server does by default)
            inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            session = std::make_unique<common_chat_session>(g_templates.get(), vocab, inputs);
            const auto & parts = session->prompt().parts;
            for (size_t i = 0; i < parts.size(); i++) {
                const auto & part = parts[i];
                // special tokens only from the template, never from the text (e.g. pasted "<|im_end|>")
                const auto part_tokens = common_tokenize(vocab, part.text, i == 0, !part.is_input);
                tokens.insert(tokens.end(), part_tokens.begin(), part_tokens.end());
            }
            for (const auto & stop : session->additional_stops()) stops.push_back(stop);
            used_template = !tokens.empty();
        } catch (const std::exception & e) {
            LOGW("chat template failed, using the plain prompt: %s", e.what());
            tokens.clear();
            session.reset();
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
    if (session) session->apply_sampling(sampling);
    common_sampler * sampler = common_sampler_init(g_model, sampling);
    if (sampler == nullptr) {
        g_error = "sampler could not be created";
        return nullptr;
    }

    std::string output;
    // the same with the tokens of the chat format (e.g. where thinking ends), for the session
    common_chat_input chat_output;
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
        if (session) chat_output.append(common_token_to_piece(g_ctx, token, sampling.preserved_tokens.count(token) > 0), token);
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

    // the answer without the thoughts; the plain text as before if the chat format can't parse it
    bool thinking_unfinished = false;
    if (session && !stopped && g_error.empty()) {
        try {
            const common_chat_msg & msg = session->finish(chat_output);
            g_reasoning = msg.reasoning_content;
            if (!msg.content.empty()) {
                output = msg.content;
            } else if (!g_reasoning.empty()) {
                // it was still thinking when the tokens ran out
                g_error = "the model was still thinking after " + std::to_string(n_generated) +
                          " tokens: allow longer answers or switch thinking off";
                thinking_unfinished = true;
            }
        } catch (const std::exception & e) {
            LOGW("chat output not parsed, using the plain text: %s", e.what());
        }
    }

    char stats[320];
    snprintf(stats, sizeof(stats), "prompt %d tokens %.0f ms (%.1f t/s), generated %d tokens %.0f ms (%.1f t/s), template %s, cores %s",
             n_prompt, prompt_ms, prompt_ms > 0 ? n_prompt * 1000.0 / prompt_ms : 0.0,
             n_generated, gen_ms, gen_ms > 0 ? n_generated * 1000.0 / gen_ms : 0.0, used_template ? "yes" : "no",
             g_cores.empty() ? "system" : g_cores.c_str());
    g_stats = stats;
    LOGI("%s", stats);
    if (g_error == "cancelled" || thinking_unfinished) return nullptr;
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

JNIEXPORT jbyteArray JNICALL
Java_helium314_keyboard_ai_plugin_LlamaNative_nativeLastReasoning(JNIEnv * env, jclass) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return to_bytes(env, g_reasoning);
}

JNIEXPORT jstring JNICALL
Java_helium314_keyboard_ai_plugin_LlamaNative_nativeLastStats(JNIEnv * env, jclass) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return env->NewStringUTF(g_stats.c_str());
}

} // extern "C"
