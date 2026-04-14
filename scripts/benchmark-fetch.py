from mlenergy_data.records import LLMRuns

runs = LLMRuns.from_hf()

# Llama 3.1 on H100 and B200
llama = runs.model_id("meta-llama/Llama-3.1-70B-Instruct")
print(llama)
for gpu in ["H100", "B200"]:
    print(f"\n=== {gpu} ===")
    for r in llama.gpu_model(gpu):
        print(
            f"batch={r.max_num_seqs:4d} | "
            f"{r.energy_per_token_joules:.4f} J/tok | "
            f"{r.output_throughput_tokens_per_sec:6.1f} tok/s | "
            f"task={r.task}"
        )
