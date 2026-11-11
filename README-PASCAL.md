# llama.cpp - Pascal (sm_61 / Tesla P40) branch

Experimental fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) tuned for NVIDIA Pascal GPUs (Tesla P40, GTX 10xx: DP4A, no tensor cores, slow FP16).
Goal: keep Pascal fast and buildable while upstream moves on (CUDA 13 no longer targets sm_61).

**Not an upstream contribution.** This branch is not proposed to ggml-org. Use at your own risk.

## What is in this branch

Base: upstream `aa75ff3` (2026-11-10).

1. Backport of three open upstream PRs (credit to their authors):
   - [#27721](https://github.com/ggml-org/llama.cpp/pull/27721) CUDA graphs on Pascal - @ArtyomITA
   - [#25479](https://github.com/ggml-org/llama.cpp/pull/25479) MMVQ nwarps for Pascal DP4A - @animeshsri14
   - [#31352](https://github.com/ggml-org/llama.cpp/pull/31352) Pascal DP4A MMQ I tile - @rynwang
2. A Pascal-only MMVQ path (`ggml/src/ggml-cuda/mmvq.cu`): one warp per block, several rows per block, also used for 2-4 columns (speculative / MTP verification), plus reuse of the q8_1-quantized activations between consecutive MUL_MAT.

Runtime switches:

| Variable | Values | Effect |
|---|---|---|
| `GGML_CUDA_PASCAL_MMVQ_ROWS` | unset = auto, 0, 1, 2, 4 | rows per block for batch 1; 0 = upstream kernel |
| `GGML_CUDA_PASCAL_MMVQ_MULTI` | 0, 1 (default) | use the new kernel for 2-4 columns |
| `GGML_CUDA_PASCAL_Q8_CACHE` | 0, 1 (default) | reuse q8_1 activations |

`GGML_CUDA_PASCAL_MMVQ_ROWS=0 GGML_CUDA_PASCAL_Q8_CACHE=0 GGML_CUDA_DISABLE_GRAPHS=1` gives back upstream runtime behaviour.

## Build

CUDA 12.x (tested 12.4, driver 580):

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Results (Tesla P40 24 GB, Qwen3.6-35B-A3B Q4_0)

Correctness: `test-backend-ops` MUL_MAT / MUL_MAT_ID 3484/3484, perplexity identical to upstream (7.1616).

llama-bench, `-fa off -ub 2048`:

| | upstream | this branch |
|---|---|---|
| pp4096 | 1578 t/s | 1797 t/s (+14%) |
| tg128 | 63.7 t/s | 68.7 t/s (+8%) |

llama-server, real French prompt, 600 generated tokens, temp 0, `-fa on`, MTP draft (`-md mtp-...Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 2`):

| | 2k prompt | 19k prompt |
|---|---|---|
| upstream, no MTP | 61 t/s | 55 t/s |
| upstream + MTP | 80 t/s | 63 t/s |
| this branch + MTP | 86 t/s | 70 t/s |

Most of the batch-1 gain comes from the three backported PRs; the multi-column kernel mainly helps MTP on long contexts. `--spec-draft-n-max 3` and `--spec-draft-p-min` were slower. Full tables: [pascal/results-kernels.md](pascal/results-kernels.md), [pascal/results-server.md](pascal/results-server.md); benchmark script: [pascal/bench.sh](pascal/bench.sh).

## How this was made

The kernel changes and the benchmark script were written by AI coding agents (Anthropic Claude), then measured on a rented P40. Treat it as experimental code.
