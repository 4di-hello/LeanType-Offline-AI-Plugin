// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.ai.plugin

import android.content.Context
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.util.Log
import helium314.keyboard.latin.ai.IOfflineAiProvider

/**
 * Offline AI with llama.cpp (see [LlamaNative]).
 *
 * [generate] takes the plain prompt as before. Additionally the keyboard can pass the chat messages
 * as params "chat_system" and "chat_user" (and "enable_thinking"): then the chat format stored in
 * the model file is used, which works for every model family (Gemma, Qwen, Llama, ...). Keyboards
 * that don't pass them keep working with the plain prompt.
 */
class OfflineAiProviderImpl : IOfflineAiProvider {
    companion object {
        private const val TAG = "OfflineAiProviderImpl"
        private val DEFAULT_STOPS = listOf("\nInput:", "\nInstruction:", "\nOutput:", "\nCorrected:", "<|im_end|>")
    }

    private val lock = Any()
    private var appContext: Context? = null
    private var initialized = false
    @Volatile private var loaded = false
    private var currentModelPath: String? = null
    /** open while the model is loaded: llama.cpp maps the file through it */
    private var modelFile: ParcelFileDescriptor? = null

    override fun getInterfaceVersion(): Int = 1

    override fun init(context: Context) {
        appContext = context.applicationContext ?: context
        LlamaNative.ensureInitialized()
        initialized = true
        Log.i(TAG, "initialized")
    }

    override fun isAvailable(): Boolean = initialized

    override fun isModelLoaded(): Boolean = loaded

    override fun loadModel(context: Context, modelPath: String, threads: Int, nCtx: Int): Boolean = synchronized(lock) {
        if (loaded && currentModelPath == modelPath) return true
        unloadLocked()
        LlamaNative.ensureInitialized()
        val nThreads = if (threads > 0) threads else Runtime.getRuntime().availableProcessors().coerceAtMost(4)
        val contextSize = if (nCtx > 0) nCtx else 4096
        try {
            var path = modelPath
            var fd = -1
            if (!modelPath.startsWith("/")) {
                // a document picked by the user: llama.cpp reads it through our file descriptor
                val resolver = (appContext ?: context).contentResolver
                val pfd = resolver.openFileDescriptor(Uri.parse(modelPath), "r")
                    ?: throw IllegalArgumentException("model file can't be opened")
                modelFile = pfd
                fd = pfd.fd
                path = ""
            }
            val start = System.currentTimeMillis()
            // mapped: loads fast and the system can share the memory; read completely if mapping fails
            var ok = LlamaNative.nativeLoad(path, fd, contextSize, nThreads, true)
            if (!ok) {
                Log.w(TAG, "loading mapped failed (${LlamaNative.nativeLastError()}), reading the file instead")
                ok = LlamaNative.nativeLoad(path, fd, contextSize, nThreads, false)
            }
            if (!ok) {
                // the keyboard shows the last line with "failed" from the log
                Log.e(TAG, "model loading failed: ${LlamaNative.nativeLastError()}")
                unloadLocked()
                return false
            }
            Log.i(TAG, "model loaded in ${System.currentTimeMillis() - start} ms, threads $nThreads, context $contextSize")
            currentModelPath = modelPath
            loaded = true
            true
        } catch (e: Throwable) {
            Log.e(TAG, "model not loaded", e)
            unloadLocked()
            false
        }
    }

    override fun unloadModel() {
        // stops a running generation, then frees the model
        LlamaNative.nativeCancel()
        synchronized(lock) { unloadLocked() }
    }

    private fun unloadLocked() {
        try {
            LlamaNative.nativeUnload()
        } catch (e: Throwable) {
            Log.w(TAG, "error unloading the model", e)
        }
        try {
            modelFile?.close()
        } catch (_: Exception) {}
        modelFile = null
        currentModelPath = null
        loaded = false
    }

    override fun generate(prompt: String, params: Map<String, Any>?): String {
        if (!loaded) throw IllegalStateException("Model not loaded in AI plugin")
        val temperature = (params?.get("temperature") as? Number)?.toFloat() ?: 0.2f
        val topP = (params?.get("top_p") as? Number)?.toFloat() ?: 0.95f
        val topK = (params?.get("top_k") as? Number)?.toInt() ?: 40
        val minP = (params?.get("min_p") as? Number)?.toFloat() ?: 0.05f
        val maxTokens = (params?.get("max_tokens") as? Number)?.toInt() ?: 1024
        val system = params?.get("chat_system") as? String
        val user = params?.get("chat_user") as? String
        val thinking = params?.get("enable_thinking") as? Boolean ?: false
        val stops = (params?.get("stop") as? List<*>)?.filterIsInstance<String>() ?: DEFAULT_STOPS

        val result = synchronized(lock) {
            LlamaNative.nativeGenerate(
                prompt.toByteArray(Charsets.UTF_8),
                system?.toByteArray(Charsets.UTF_8),
                user?.toByteArray(Charsets.UTF_8),
                thinking, maxTokens, temperature, topP, topK, minP,
                stops.map { it.toByteArray(Charsets.UTF_8) }.toTypedArray()
            )
        }
        // only numbers: the texts are private
        Log.i(TAG, LlamaNative.nativeLastStats())
        if (result == null) throw RuntimeException("Inference error: ${LlamaNative.nativeLastError()}")
        return String(result, Charsets.UTF_8).trim()
    }

    override fun proofread(text: String, instruction: String?): String {
        if (text.isBlank()) return text
        val instr = instruction ?: "Fix grammar, spelling, and punctuation errors. Return ONLY the corrected text without preamble."
        val system = "You are a professional proofreader. Correct grammatical, spelling, and punctuation errors while preserving original formatting."
        val user = "Instruction: $instr\nText: $text"
        val prompt = "<|im_start|>system\n$system<|im_end|>\n<|im_start|>user\n$user<|im_end|>\n<|im_start|>assistant\n"
        val tokensNeeded = maxOf(512, (text.length / 2.5).toInt() + 128)
        val out = generate(prompt, mapOf("temperature" to 0.1, "max_tokens" to tokensNeeded, "chat_system" to system, "chat_user" to user))
        return cleanOutput(out)
    }

    override fun translate(text: String, sourceLang: String, targetLang: String): String {
        if (text.isBlank()) return text
        val system = "You are a professional translator. Translate from $sourceLang to $targetLang. Output ONLY the translation without preamble."
        val user = "Text: $text"
        val prompt = "<|im_start|>system\n$system<|im_end|>\n<|im_start|>user\n$user<|im_end|>\n<|im_start|>assistant\n"
        val tokensNeeded = maxOf(512, (text.length / 2.5).toInt() + 128)
        val out = generate(prompt, mapOf("temperature" to 0.2, "max_tokens" to tokensNeeded, "chat_system" to system, "chat_user" to user))
        return cleanOutput(out)
    }

    private fun cleanOutput(text: String): String {
        return text
            .replace(Regex("<think>[\\s\\S]*?</think>", RegexOption.IGNORE_CASE), "")
            .replace(Regex("<thinking>[\\s\\S]*?</thinking>", RegexOption.IGNORE_CASE), "")
            .replace(Regex("<thought>[\\s\\S]*?</thought>", RegexOption.IGNORE_CASE), "")
            .replace(Regex("<reasoning>[\\s\\S]*?</reasoning>", RegexOption.IGNORE_CASE), "")
            .replace(Regex("<details>[\\s\\S]*?</details>", RegexOption.IGNORE_CASE), "")
            .trim()
    }

    override fun cleanup() {
        unloadModel()
        appContext = null
        initialized = false
    }
}
