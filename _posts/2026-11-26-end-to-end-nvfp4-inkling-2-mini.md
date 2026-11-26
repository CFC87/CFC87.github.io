---
layout: post
title: "Beyond Training Precision: Serving Inkling 2 Mini in End-to-End NVFP4"
summary: "Using our quantization-aware distillation (QAD) recipe, we serve Inkling 2 Mini beyond its training precision—quantizing its remaining MXFP8 expert layers to NVFP4 near-losslessly to unlock gains in KV cache, throughput, and latency."
---

> **TL;DR**
> - Training precision does not have to dictate serving precision: we serve natively NVFP4-trained Inkling 2 Mini in **end-to-end NVFP4** by quantizing its last 1/6 of expert layers (4 of 24) from MXFP8 to NVFP4, freeing **1.8 GiB** of GPU memory.
> - Across the 11 benchmarks we evaluate, direct PTQ degrades accuracy on tasks such as LiveCodeBench Pro and AutomationBench, while **our QAD recipe** recovers near-lossless performance with only **0.048B training tokens**.
> - On a single RTX 5090, the reclaimed memory expands KV cache capacity by **44%–70%**, boosting concurrent throughput by up to **39%**, halving time to first token (TTFT) across four concurrent agent sessions, and unlocking DSpark speculative decoding for long-context agents (**285 tok/s**, +77%).
> - More broadly, **developing post-training adaptation recipes like QAD** has the potential to bridge the accuracy gap and unlock more aggressive low-precision serving configurations than training can stably support, with significant gains in efficiency.
{: .tldr}

Inkling 2 Mini is an open-weight multimodal model released by Thinking Machines Lab on November 20, 2026, under the Apache 2.0 license. It is a looped mixture-of-experts transformer with 24 layers and 128 routed experts per layer (8 active per token), where the middle 12 layers run three times with shared weights. In total, it has 26.4B parameters—3.4B active per token (5.4B counting the looped passes)—and supports up to a 1M-token context window.

The model is built for NVFP4, NVIDIA's 4-bit floating-point format, which stores each value in 4 bits with an FP8 scale shared per block of 16 values (3.6× fewer bits than BF16). Blackwell GPUs execute NVFP4 matrix multiplies natively. Because Thinking Machines trained Inkling 2 Mini with an NVFP4 forward pass from pretraining through final RL rollouts <a class="cite" href="#ref-1">[1]</a>, the released checkpoint **ships directly in NVFP4** with no post-training quantization (PTQ) step. At roughly 20 GB in a single file, it fits on a DGX Spark, a 32 GB Mac, or a 32 GB GPU such as the RTX 5090.

Training every layer in NVFP4 is unstable, however, so Thinking Machines keeps the final layers in higher precision:

> "We keep the last 16.6% of layers in higher precision, which we compute in MXFP8 rather than BF16." <a class="cite" href="#ref-1">[1]</a>

In Inkling 2 Mini, that 1/6 corresponds to the last four of its 24 layers: the experts in layers 0–19 are NVFP4, while those in layers 20–23 remain MXFP8.

That higher precision exists to keep 4-bit training stable. Serving has no backward pass to destabilize, so those four layers can run in NVFP4 too—freeing 1.8 GiB of GPU memory—provided we can recover the resulting quantization error. We call this **end-to-end NVFP4**: every expert layer, from 0 to 23, runs in NVFP4. Converting them directly with PTQ degrades accuracy. Below, we show that the QAD recipe from our upcoming paper <a class="cite" href="#ref-3">[3]</a> recovers that loss with only 0.048B training tokens, measure what the reclaimed memory delivers on a single RTX 5090, and illustrate how post-training adaptation recipes like QAD can unlock more aggressive low-precision serving configurations than training can stably support.

## End-to-end NVFP4 and what it costs

The official release already defines the NVFP4 weight-quantization rule: re-running it on layers 0–19 reproduces the shipped weights bit for bit. Applying the same rule to the experts in layers 20–23 shrinks the checkpoint from **19.53 GB to 17.61 GB**. Attention, embeddings, and the KV cache format remain unchanged.

That naive conversion comes at a clear accuracy cost. Across our 11-benchmark suite spanning reasoning, code, long context, and agentic tasks, direct PTQ loses **2.1 points (3.3% relative) on LiveCodeBench Pro**, along with 0.5 to 0.7 points on SuperGPQA, LiveCodeBench v6, LongBench Pro, and AutomationBench (Table 1, *End-to-end NVFP4* column).

## Recovering accuracy with QAD

QAD trains a quantized student model to match a higher-precision teacher <a class="cite" href="#ref-2">[2]</a>. Here, the official release serves as the teacher, the end-to-end NVFP4 model is the student, and **only the four converted expert layers are updated**. Following the recipe from our upcoming paper <a class="cite" href="#ref-3">[3]</a>, recovery takes **0.048B training tokens**; we will open-source the code and data soon.

<figure>
  <div class="fig-art" role="img" aria-label="Benchmark scores of end-to-end NVFP4 with QAD, end-to-end NVFP4 without it, and the official release">{% include blog/end-to-end-nvfp4/table-benchmarks.svg %}</div>
  <figcaption><strong>Table 1. After QAD, end-to-end NVFP4 is near-lossless across reasoning, code, long context, and agentic tasks.</strong> Blue marks the highest score in each row and grey the second highest (ties share a shade). Red arrows show the relative drop of end-to-end NVFP4 without QAD below the official release. All runs sample at temperature 1.0, averaging 16 samples per problem for GPQA Diamond, 4 for SuperGPQA and LiveCodeBench Pro, 2 for LongBench Pro and MRCR, and 8 for the rest. The first math row pools 433 problems from the 2025–2026 contests (HMMT: Feb 2025, Nov 2025, Feb 2026, and further rounds), and the second pools 246. SuperGPQA uses the science subset, BFCL the multi-turn split, and τ²-bench covers all three domains.</figcaption>
</figure>

Overall, **end-to-end NVFP4 + QAD matches or beats the official release on 8 of 11 benchmarks**. It completely erases the LiveCodeBench Pro regression (**62.0% → 64.1%**), turns the PTQ dips on the AIME/HMMT set and LongBench Pro into slight gains, and widens the lead on BeyondAIME/OlymMATH-Hard/Apex and τ²-bench. Only three benchmarks remain marginally below baseline: SuperGPQA (−0.5), LiveCodeBench v6 (−0.4), and AutomationBench (−0.1).

## Deploying on an RTX 5090

On a 32 GB GPU, the 1.8 GiB freed by end-to-end NVFP4 goes straight into the KV cache. SGLang's Inkling 2 Mini cookbook provides two RTX 5090 configurations: a **Standard** configuration and a **w/ DSpark** (speculative decoding) configuration. All measurements below use the cookbook's launch commands <a class="cite" href="#ref-4">[4]</a> verbatim, except for the context-limit adjustment under DSpark noted below.

### KV cache capacity

Because DSpark reserves extra memory for speculative decoding, its baseline KV cache budget is tighter—so the same 1.8 GiB savings produces an even larger relative jump: **+70%** with DSpark (47,360 → 80,512 tokens) versus **+44%** in Standard mode (81,792 → 118,016 tokens) (Figure 1).

<figure>
  <div class="fig-art" role="img" aria-label="KV cache capacity of the official release and the end-to-end NVFP4 model in the Standard and w/ DSpark configurations">{% include blog/end-to-end-nvfp4/fig1-kv.svg %}</div>
  <figcaption><strong>Figure 1. End-to-end NVFP4 holds 44% more KV cache, and 70% more with DSpark.</strong> KV cache capacity on a single RTX 5090 with the cookbook's two configurations, Standard and w/ DSpark (speculative decoding).</figcaption>
</figure>

### Throughput and queuing under load

For a single request, extra KV cache makes no difference: both models generate 150 to 165 tokens per second, since the four converted layers are a small fraction of the weights read per token. The gap opens up once requests share the GPU. Past 4 to 8 concurrent requests, the official release runs out of KV cache, its throughput plateaus, and incoming requests begin to queue, whereas end-to-end NVFP4 continues scaling to **up to 39% higher output throughput** (Figure 2).

<figure>
  <div class="fig-art" role="img" aria-label="Output throughput against concurrent requests for three input and output lengths">{% include blog/end-to-end-nvfp4/fig2-throughput.svg %}</div>
  <figcaption><strong>Figure 2. The official release stops scaling at 4 to 8 concurrent requests; end-to-end NVFP4 keeps going, up to 39% higher.</strong> Output throughput against concurrent requests, Standard configuration, random-token prompts at three input and output lengths.</figcaption>
</figure>

That queuing translates directly into wait time. At 8 concurrent requests, median TTFT drops sharply across all three prompt/output regimes:

- **1K in / 1K out:** 2.3 s → 0.13 s (~18× lower)
- **1K in / 8K out:** 14.9 s → 2.1 s (~7× lower)
- **8K in / 1K out:** 12.2 s → 3.3 s (~3.7× lower)

### Long-context agent workloads

Agentic sessions are especially sensitive to KV cache capacity because every turn carries the full conversation history; in the AgentX traces we replayed, every session is at least 44K tokens long.

In the Standard configuration, both models can fit these sessions, and the extra cache pays off under concurrency. With four agent sessions sharing the GPU, end-to-end NVFP4 **cuts mean TTFT in half** (3.85 s → 1.87 s), reduces mean request latency by **25%** (7.83 s → 5.89 s), and completes **9% more requests** within 15 minutes (Figure 3).

<figure>
  <div class="fig-art" role="img" aria-label="Time to first token and request latency with four agent sessions sharing the GPU">{% include blog/end-to-end-nvfp4/fig3-agent-latency.svg %}</div>
  <figcaption><strong>Figure 3. With four agent sessions sharing the GPU, end-to-end NVFP4 cuts time to first token by more than half and request latency by 25% to 38%.</strong> AgentX sessions replayed four at a time for 15 minutes, Standard configuration.</figcaption>
</figure>

With DSpark enabled—raising the context limit from the cookbook's 32,768 tokens to 65,536 so these >44K-token sessions can run—the extra memory becomes a make-or-break difference. The official release's 47,360-token KV cache cannot hold them and fails on the first session that exceeds it (50,700 tokens). End-to-end NVFP4 fits them comfortably and serves them one at a time at **285 tok/s per user (p90)—77% faster** than the best the official release can achieve (Figure 4).

<figure>
  <div class="fig-art" role="img" aria-label="Per-user output speed on AgentX sessions for each model and configuration">{% include blog/end-to-end-nvfp4/fig4-dspark.svg %}</div>
  <figcaption><strong>Figure 4. With DSpark, only end-to-end NVFP4 can serve these agent sessions, 77% faster than the best the official release can do.</strong> Per-user output speed (p90) replaying AgentX sessions one at a time. The official release's best option is the Standard configuration.</figcaption>
</figure>

## Takeaway

The precision a model needs for stable 4-bit training is not the precision it needs for serving. To our knowledge, this is the first demonstration that a natively NVFP4-trained model can be served with a more aggressive NVFP4 scheme than it was trained with, while lightweight QAD recovers near-lossless accuracy.

Even in a single-GPU deployment of Inkling 2 Mini, converting just its last four MXFP8 layers already unlocks large gains in KV cache capacity, throughput, and latency. More broadly, **developing post-training adaptation recipes like QAD has the potential to bridge the accuracy gap and unlock more aggressive low-precision serving configurations than training can stably support—with significant gains in efficiency.**

## References

<ol class="refs">
  <li id="ref-1">Jack Cook and Thinking Machines Lab. <a href="https://thinkingmachines.ai/blog/end-to-end-4-bit-training/">End-to-End 4-Bit Training</a>. Thinking Machines Lab blog, November 2026.</li>
  <li id="ref-2">Meng Xin, Sweta Priyadarshi, et al. <a href="https://arxiv.org/abs/2601.20088">Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery</a>. arXiv:2601.20088, 2026.</li>
  <li id="ref-3">Fanchao Chen, Minghao Yan, Jiantao Jiao, Shivaram Venkataraman. Deriving a Quantization-Aware Distillation Recipe for NVFP4 Inference Accuracy Recovery. 2026 (to be released).</li>
  <li id="ref-4">SGLang. <a href="https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling-2-Mini#hw=gb300&quant=nvfp4&strategy=balanced">Inkling-2-Mini cookbook</a>. SGLang documentation.</li>
</ol>
