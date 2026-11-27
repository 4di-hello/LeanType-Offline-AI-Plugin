// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.ai.plugin

import android.util.Log
import dalvik.system.BaseDexClassLoader
import java.io.File

/** llama.cpp, see leantype_llama.cpp. Texts are passed as UTF-8 bytes. */
internal object LlamaNative {
    private const val TAG = "LlamaNative"
    private const val LIBRARY = "leantype_llama"
    /** loaded before [LIBRARY], in case the linker doesn't find them next to it */
    private val DEPENDENCIES = listOf("c++_shared", "ggml-base", "ggml", "llama", "llama-common")

    @Volatile private var initialized = false

    /** Loads the native libraries and the CPU backend that fits this phone; once */
    @Synchronized
    fun ensureInitialized() {
        if (initialized) return
        // the keyboard extracts the libraries of the plugin into one folder and loads the plugin with it
        val loader = LlamaNative::class.java.classLoader as? BaseDexClassLoader
        val libDir = loader?.findLibrary(LIBRARY)?.let { File(it).parentFile }
        for (name in DEPENDENCIES) {
            val file = libDir?.let { File(it, System.mapLibraryName(name)) }
            try {
                if (file != null && file.exists()) System.load(file.absolutePath)
            } catch (e: Throwable) {
                Log.w(TAG, "could not preload $name", e)
            }
        }
        System.loadLibrary(LIBRARY)
        nativeInit(libDir?.absolutePath ?: "")
        initialized = true
    }

    @JvmStatic external fun nativeInit(libDir: String)
    @JvmStatic external fun nativeLoad(path: String, fd: Int, nCtx: Int, nThreads: Int, useMmap: Boolean): Boolean
    @JvmStatic external fun nativeGenerate(
        prompt: ByteArray, system: ByteArray?, user: ByteArray?, thinking: Boolean,
        maxTokens: Int, temperature: Float, topP: Float, topK: Int, minP: Float, stops: Array<ByteArray>
    ): ByteArray?
    @JvmStatic external fun nativeCancel()
    @JvmStatic external fun nativeUnload()
    @JvmStatic external fun nativeLastError(): String
    @JvmStatic external fun nativeLastStats(): String
    /** what the model thought in the last generation (UTF-8), empty if thinking was off */
    @JvmStatic external fun nativeLastReasoning(): ByteArray
}
