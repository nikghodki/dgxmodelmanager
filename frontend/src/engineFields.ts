export type FieldType = "int" | "bool" | "text" | "select";
export interface FieldDef { key: string; label: string; type: FieldType; options?: string[]; help?: string; advanced?: boolean }

export const ENGINE_FIELDS: Record<string, FieldDef[]> = {
  vllm: [
    { key: "max_model_len", label: "Context length (tokens)", type: "int", help: "Max tokens per request (prompt + output)." },
    { key: "max_num_seqs", label: "Concurrent sequences", type: "int", help: "Requests decoded in parallel." },
    { key: "kv_cache_dtype", label: "KV cache dtype", type: "select", options: ["auto", "fp8", "fp8_e4m3", "fp8_e5m2"], help: "fp8 halves KV memory." },
    { key: "kv_tokens", label: "KV pool tokens (optional)", type: "int", help: "Total KV tokens shared by all sequences. Empty = min(context × sequences, 2 × context)." },
    { key: "max_num_batched_tokens", label: "Max batched tokens", type: "int", advanced: true },
    { key: "enable_prefix_caching", label: "Prefix caching", type: "bool" },
    { key: "enable_chunked_prefill", label: "Chunked prefill", type: "bool", advanced: true },
    { key: "enable_sleep_mode", label: "Sleep mode (fast switching)", type: "bool" },
    { key: "enable_auto_tool_choice", label: "Auto tool choice", type: "bool", advanced: true },
    { key: "tool_call_parser", label: "Tool-call parser", type: "text", advanced: true },
    { key: "reasoning_parser", label: "Reasoning parser", type: "text", advanced: true },
    { key: "served_model_name", label: "Served model name", type: "text", advanced: true },
    { key: "trust_remote_code", label: "Trust remote code", type: "bool", advanced: true },
    { key: "safetensors_load_strategy", label: "Safetensors load strategy", type: "select", options: ["", "lazy", "eager", "prefetch"], advanced: true },
  ],
  llamacpp: [
    { key: "ctx_size", label: "Context size (all slots)", type: "int" },
    { key: "parallel", label: "Parallel slots", type: "int" },
    { key: "cache_type_k", label: "K cache type", type: "select", options: ["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "f32"] },
    { key: "cache_type_v", label: "V cache type", type: "select", options: ["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "f32"] },
    { key: "flash_attn", label: "Flash attention", type: "select", options: ["auto", "on", "off"] },
    { key: "n_gpu_layers", label: "GPU layers", type: "int", advanced: true },
    { key: "n_batch", label: "Batch size", type: "int", advanced: true },
    { key: "n_ubatch", label: "Micro-batch size", type: "int", advanced: true },
    { key: "alias", label: "Served model name", type: "text", advanced: true },
  ],
};
