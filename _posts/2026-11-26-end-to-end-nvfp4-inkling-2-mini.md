---
layout: post
title: "Serving Inkling 2 Mini in NVFP4 End to End, Near-Losslessly"
summary: "Inkling 2 Mini keeps its last four expert layers in MXFP8 because end-to-end 4-bit training needs them for stability. Converting them to NVFP4 for deployment costs accuracy; the QAD recipe from our upcoming paper makes it near-lossless with 0.048B training tokens, and on one RTX 5090 end-to-end NVFP4 gives 44% to 70% more KV cache and up to 39% more throughput."
---

> **TL;DR**
> - Inkling 2 Mini keeps the experts of its last four layers in MXFP8 because 4-bit training needs them for stability. Deployment does not.
> - Converting them to NVFP4 costs up to 2.1 points (LiveCodeBench Pro). QAD with our recipe wins it back with 0.048B training tokens: the result is at or above the official release on 7 of 11 benchmarks and at most 0.5 points below it on the other four.
> - On one RTX 5090, end-to-end NVFP4 holds 44% more KV cache (70% with DSpark speculative decoding), delivers up to 39% more throughput under load, and halves the time to first token when four agent sessions share the GPU.
{: .tldr}

Inkling 2 Mini is the open-weight multimodal model Thinking Machines Lab released on November 20, 2026, under the Apache 2.0 license. It is a looped mixture-of-experts transformer: 24 layers, each with 128 routed experts of which 8 are active per token, and the middle 12 layers run three times with shared weights. That gives 26.4B parameters, 3.4B of them active per token (5.4B counting the extra passes), and a context window of up to 1M tokens.

The model is built to be deployed in NVFP4, NVIDIA's 4-bit floating-point format. NVFP4 stores each value in 4 bits, with an FP8 scale shared by every block of 16 values, 3.6 times fewer bits than BF16 <a class="cite" href="#ref-1">[1]</a>. Blackwell GPUs, from the GB300 down to the RTX 5090, run NVFP4 matrix multiplies natively. Thinking Machines trained Inkling 2 Mini with an NVFP4 forward pass from pretraining through the final RL rollout <a class="cite" href="#ref-1">[1]</a>, so the released checkpoint is NVFP4 as shipped, with no post-training quantization step. It is a single file of about 20 GB, small enough for a DGX Spark, a 32 GB Mac, or a 32 GB GPU such as the RTX 5090.

Training entirely in NVFP4 is unstable, though, so the recipe, like NVIDIA's, keeps the last layers in higher precision:

> "We keep the last 16.6% of layers in higher precision, which we compute in MXFP8 rather than BF16." <a class="cite" href="#ref-1">[1]</a>

For Inkling 2 Mini that is the last four of its 24 layers. The experts in layers 0 to 19 are NVFP4, and the experts in layers 20 to 23 are MXFP8.

That higher precision is there for training. A deployment runs no backward pass, so the four layers can be NVFP4 as well, which frees 1.8 GiB of memory. We call the result end-to-end NVFP4: the experts of every layer, from 0 to 23, are NVFP4. Quantizing them directly, with post-training quantization (PTQ), costs accuracy: up to 2.1 points on our benchmarks.

We recover that loss with the quantization-aware distillation (QAD) recipe from our upcoming paper <a class="cite" href="#ref-3">[3]</a>, using 0.048B training tokens, and then measure what end-to-end NVFP4 buys on an RTX 5090. To our knowledge, this is the first demonstration that a model trained natively in NVFP4 can be served with a more aggressive NVFP4 scheme than the one it was trained with, quantizing the layers its training kept in higher precision, while QAD recovers the accuracy this costs.

## End-to-end NVFP4 and what it costs

The official release already defines how its weights are quantized to NVFP4: rerun on layers 0 to 19, its rule reproduces the shipped weights bit for bit. We applied the same rule to the experts of layers 20 to 23. The checkpoint shrinks from 19.53 GB to 17.61 GB. Attention, embeddings and the KV cache format stay as shipped.

The conversion is not free. On our 11-benchmark suite, covering reasoning, code, long context and agentic tasks, the converted model loses 2.1 points on LiveCodeBench Pro, and 0.4 to 0.7 points on SuperGPQA, AutomationBench, LiveCodeBench and LongBench Pro (Table 1, End-to-end NVFP4 column).

## Recovering the accuracy with QAD

QAD trains a quantized model to match its higher-precision teacher <a class="cite" href="#ref-2">[2]</a>. Here the teacher is the official release itself, the student is the end-to-end NVFP4 model, and only the four converted layers are trained. We follow the recipe from our paper <a class="cite" href="#ref-3">[3]</a>, which we will open-source with its code and data soon. The whole run uses 0.048B training tokens.

<figure>
  <div class="fig-art" role="img" aria-label="Benchmark scores of end-to-end NVFP4 with QAD, end-to-end NVFP4 without it, and the official release">{% include blog/end-to-end-nvfp4/table-benchmarks.svg %}</div>
  <figcaption><strong>Table 1. After QAD, end-to-end NVFP4 is near-lossless across reasoning, code, long context and agentic tasks.</strong> Blue: the highest of the three scores in a row; grey: the second highest (equal scores share a shade). Red arrows: how far end-to-end NVFP4 without QAD falls below the official release, relative to the official release's score. All runs sample at temperature 1.0, and scores average 16 samples per problem for GPQA Diamond, 4 for SuperGPQA and LiveCodeBench Pro, 2 for LongBench Pro and MRCR, and 8 for the rest. The first math row pools 433 problems from the 2025 and 2026 contests (HMMT: February 2025, November 2025, February 2026 and further rounds), the second 246 problems. SuperGPQA is its science subset, BFCL its multi-turn split, and τ²-bench covers its three domains.</figcaption>
</figure>

QAD removes the LiveCodeBench Pro loss (62.0 to 64.1, the official release's score). On the AIME/HMMT set and on LongBench Pro, where the conversion had dropped below the official release, it finishes above it, and it widens the lead on the BeyondAIME/OlymMATH/Apex set and τ²-bench. It stays slightly under the official release on SuperGPQA (0.5 points), LiveCodeBench (0.4) and AutomationBench (0.1).

## Deploying on an RTX 5090

On a 32 GB card, the 1.8 GiB that end-to-end NVFP4 frees goes straight to KV cache. SGLang's Inkling 2 Mini cookbook has two configurations for the RTX 5090: a Standard configuration, and one with DSpark speculative decoding. All measurements below use its launch commands <a class="cite" href="#ref-4">[4]</a> unchanged, with one exception: for the agent sessions with DSpark we raise the context limit from the cookbook's 32,768 tokens to 65,536, because every session is longer than 32K.

<figure>
  <div class="fig-art" role="img" aria-label="KV cache capacity of the official release and the end-to-end NVFP4 model in the Standard and w/ DSpark configurations">{% include blog/end-to-end-nvfp4/fig1-kv.svg %}</div>
  <figcaption><strong>Figure 1. End-to-end NVFP4 holds 44% more KV cache, and 70% more with DSpark.</strong> KV cache capacity on one RTX 5090 with the cookbook's two configurations, Standard and w/ DSpark (speculative decoding).</figcaption>
</figure>

More cache does nothing for a single request: both models generate 150 to 165 tokens per second. The four converted layers are a small share of the weights read for each token, so a lone request barely notices them. The difference appears when requests share the GPU. Past about 8 concurrent requests, the official release's throughput stops rising and new requests start to queue, while end-to-end NVFP4 keeps rising (Figure 2).

<figure>
  <div class="fig-art" role="img" aria-label="Output throughput against concurrent requests for three input and output lengths">{% include blog/end-to-end-nvfp4/fig2-throughput.svg %}</div>
  <figcaption><strong>Figure 2. The official release stops scaling at about 8 concurrent requests; end-to-end NVFP4 keeps going, up to 39% higher.</strong> Output throughput against concurrent requests, Standard configuration, random-token prompts at three input and output lengths.</figcaption>
</figure>

The queue shows up as waiting. With 8 concurrent requests of 1K input and 1K output tokens, the median time to first token is 2.3 s on the official release and 0.13 s on end-to-end NVFP4; with 8K-token outputs it is 14.9 s against 2.1 s, and with 8K-token inputs 12.2 s against 3.3 s.

Agent workloads have long contexts, because every request carries the whole session so far; in the AgentX sessions we replayed, no session is shorter than 44K tokens. In the Standard configuration both models can hold these sessions, and the extra cache shows when agents share the GPU. With four sessions running at once, end-to-end NVFP4 halves the mean time to first token (3.85 s to 1.87 s), cuts mean request latency by 25% (7.83 s to 5.89 s), and finishes 9% more requests within 15 minutes (Figure 3).

<figure>
  <div class="fig-art" role="img" aria-label="Time to first token and request latency with four agent sessions sharing the GPU">{% include blog/end-to-end-nvfp4/fig3-agent-latency.svg %}</div>
  <figcaption><strong>Figure 3. With four agent sessions sharing the GPU, end-to-end NVFP4 cuts time to first token by more than half and request latency by 25% to 38%.</strong> AgentX sessions replayed four at a time for 15 minutes, Standard configuration.</figcaption>
</figure>

With DSpark (context raised to 65,536 tokens), the official release's 47,360-token cache is too small for these sessions, and the run fails at the first one longer than that (50,700 tokens), while end-to-end NVFP4 serves them, one at a time, at 285 tokens per second per user (Figure 4).

<figure>
  <div class="fig-art" role="img" aria-label="Per-user output speed on AgentX sessions for each model and configuration">{% include blog/end-to-end-nvfp4/fig4-dspark.svg %}</div>
  <figcaption><strong>Figure 4. With DSpark, only end-to-end NVFP4 can serve these agent sessions, 77% faster than the best the official release can do.</strong> Per-user output speed (p90) replaying AgentX sessions one at a time. The official release's best option is the Standard configuration.</figcaption>
</figure>

## Takeaway

The NVFP4 scheme a model needs for stable training is not the scheme it needs for serving. With QAD, a model trained natively in NVFP4 can be served with a more aggressive NVFP4 scheme, every expert layer in NVFP4, at near-lossless accuracy. To our knowledge, Inkling 2 Mini is the first model where this has been shown.

## References

<ol class="refs">
  <li id="ref-1">Jack Cook and Thinking Machines Lab. <a href="https://thinkingmachines.ai/blog/end-to-end-4-bit-training/">End-to-End 4-Bit Training</a>. Thinking Machines Lab blog, November 2026.</li>
  <li id="ref-2">Meng Xin, Sweta Priyadarshi, et al. <a href="https://arxiv.org/abs/2601.20088">Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery</a>. arXiv:2601.20088, 2026.</li>
  <li id="ref-3">Fanchao Chen, Minghao Yan, Jiantao Jiao, Shivaram Venkataraman. Deriving a Quantization-Aware Distillation Recipe for NVFP4 Inference Accuracy Recovery. 2026 (to be released).</li>
  <li id="ref-4">SGLang. <a href="https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling-2-Mini#hw=gb300&quant=nvfp4&strategy=balanced">Inkling-2-Mini cookbook</a>. SGLang documentation.</li>
</ol>
