# Keep the plugin entry point and interfaces
-keep interface helium314.keyboard.latin.ai.IOfflineAiProvider {
    <methods>;
}
-keep class helium314.keyboard.latin.ai.** { *; }
-keep interface helium314.keyboard.latin.ai.** { *; }

-keep class helium314.keyboard.ai.plugin.OfflineAiProviderImpl {
    public <init>();
    <methods>;
    public *;
}

# JNI: the native functions are found by class and method name
-keep class helium314.keyboard.ai.plugin.LlamaNative {
    native <methods>;
}
