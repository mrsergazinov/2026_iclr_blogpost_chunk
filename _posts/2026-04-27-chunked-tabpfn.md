---
layout: distill
title: "ChunkTabPFN: Training-free Long Context"
description: Tabular foundation models struggle with large datasets due to the quadratic attention. While methods like FlashAttention promise scalability, practical challenges persist in their application to tabular foundation models. Our work resolves these hurdles, enabling efficient attention, and reveals that contrary to the eariler reports, TabPFN's performance improves with larger contexts, highlighting its inherent robustness and minimal fine-tuning needs when scaling to complex, long datasets from the TabArena benchmark.
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

# Anonymize when submitting
authors:
  - name: Anonymous

# authors:
#   - name: Renat Sergazinov*
#     affiliations:
#       name: Department of Statistics, Texas A&M University
#   - name: Shao-An Yin*
#     affiliations:
#       name: Department of Electrical and Computer Engineering, University of Minnesota
# *equal contribution note will appear in-body so we don't leak identity in metadata if anonymized version is needed

# must be the exact same name as your blogpost (no extension)
# and should exist at assets/bibliography/2026-04-27-chunked-tabpfn.bib
bibliography: 2026-04-27-chunked-tabpfn.bib

# Table of contents for the right-hand nav
toc:
  - name: 1. Introduction
  - name: 2. Methodology
  - name: 3. Experiments
  - name: 4. Conclusion

# Optional per-post styles (fine to delete if you don't need it)
_styles: >
  .inline-math {
    font-family: var(--sans-serif);
    font-weight: 500;
    background: rgba(0,0,0,0.03);
    padding: 0 .25em;
    border-radius: .25em;
  }
  .attn-figure-caption {
    font-size: 0.9rem;
    color: rgba(0,0,0,0.6);
    margin-top: 0.5rem;
    text-align: center;
  }

---

## 1. Introduction
<span id="sec:introduction"></span>

Large language models leverage **in-context learning (ICL)** by adapting their predictions at inference time based solely on provided examples, without requiring any gradient updates. Building on this idea, recent work on **tabular foundation models**, such as TabPFN, TabICL, Mitra, and Limix, extends the same paradigm to tabular data <d-cite key="hollmann2022tabpfn,hollmann2025accurate,qu2025tabicl,zhang2025mitra,zhang2025limix"></d-cite>. These models are trained once on synthetic tasks drawn from a prior, allowing them to approximate the posterior predictive distribution  

$$
p(y_{*} \mid x_*, D_{\text{train}})
$$  

in a single forward pass by supplying the training set as context, without any dataset-specific fine-tuning, without fine-tuning on each new dataset <d-cite key="hollmann2022tabpfn,hollmann2025accurate"></d-cite>. This approach is compelling because it contrasts with most deep tabular models—like TabNet, FT-Transformer, NODE, TabM, or retrieval-style models such as TabR and ModernNCA, which typically require dataset-specific training or fine-tuning <d-cite key="arik2021tabnet,gorishniy2021revisiting,popov2019neural,gorishniy2024tabm,gorishniy2023tabr,ye2024modern"></d-cite>. That dependency undermines the ideal of a true "drop-in foundation model."

ICL-based tabular models move closer to this ideal. However, they face a major practical limitation: **context length**. Transformer attention scales quadratically with sequence length, and current public TabPFN implementations are constrained to around 3,000 samples in the original work to 10,000<d-footnote>At the time of writing, the new TabPFN v2.5 model has just been released, which is supposed to have pushed the context limit further to 50,000.</d-footnote> in later versions <d-cite key="hollmann2022tabpfn,hollmann2025accurate"></d-cite>. Many real-world tabular datasets far exceed these limits.

To address this, researchers have experimented with **shrinking the context**, such as by clustering, partitioning, or retrieving only subsets of the data. Examples include random-forest partitioning <d-cite key="hollmann2025accurate"></d-cite>, the Mixture of In-Context Prompters (MICP) <d-cite key="xu2024mixture"></d-cite>, and KNN-style retrieval <d-cite key="thomas2024retrieval"></d-cite>. Others, like TuneTables <d-cite key="feuer2024tunetables"></d-cite>, compress the data into learned representations.

While these methods can be effective, they come with two drawbacks:

- They often require **dataset-specific tuning** or even retraining, which contradicts the zero-shot, pure ICL philosophy.
- They don’t use the **entire training set**, which is a core assumption of TabPFN’s Bayesian approximation. Replacing full data with summaries introduces conceptual inaccuracy.

Hence, we ask the following question:

> Can we fit **all training examples** into the context (no pruning, no KNN) without learnable compression while staying within GPU memory?

In this work, we focus specifically on TabPFN, though we believe the conclusions extend to other ICL-based tabular models. Our answer is a resounding **yes**. Indeed, TabPFN’s native implementation already supports this on some devices via **FlashAttention** <d-cite key="dao2022flashattention,dao2023flashattention,shah2024flashattention"></d-cite>. But as we’ll show in this blogpost, there are important caveats:

- FlashAttention and similar efficient mechanisms can **fail** when batch or head sizes exceed 65,535.
- These optimizations are **unsupported** on older or consumer-grade GPUs.

To resolve this, we introduce a **simple patch**:

- For efficient attention, we **chunk inputs** along head or batch dimensions to avoid hitting the 65,536 limit.
- For older GPUs, we implement a **chunked forward pass** in pure PyTorch using the **incremental log-sum-exp trick**.

This patch yields results **identical to standard attention** (up to floating-point associativity), without any approximations, fine-tuning, or pre-filtering. 

Empirically, we then test TabPFN out-of-the-box scalability by evaluating it on the full **TabArena** benchmark <d-cite key="tabarena"></d-cite>. We specifically analyze TabPFN performance on datasets with **long contexts** (> 10,000). Key findings include:

- **Accuracy improves** with more data, often up to 100,000+ rows (measured in AUC for classification and RMSE for regression).
- On smaller contexts (<10,000), our chunked version **matches the original**—no hidden degradation.
- The runtime stays **practical** even on commodity GPUs.


## 2. Methodology
<span id="sec:methodology"></span>

Let `(X, y)` be the input to the TabPFN model. The typical dimensions of the feature tensor are `[B, L, F]`, where `B` is the number of datasets in the batch, `L` is the (padded) sample size, and `F` is the number of features. The first thing TabPFN does is group features `X` and embed them, which yields the following shape: `[B, L, G, D]`, where `G` is the number of feature groups and `D` is the embedding size. In the rest of the blog, we assume `X` already has this post-embedding shape.

The labels `y` are similarly embedded and then concatenated with the features along the group dimension, producing an input of shape `[B, L, G + 1, D]`. A keen reader might notice that `y` and `X` effectively have different “logical” lengths: `X` includes both train and test samples, while `y` is only provided for the training split. This is handled by padding the label embeddings for test samples with a dummy embedding. A variable `single_eval_pos` in the original code holds the index where train and test samples are concatenated, and this logic can be seen in the `transformer.py` file of the original TabPFN repository.

The core of TabPFN is the attention mechanism, whose logic is primarily implemented in `layer.py`. TabPFN, like many Transformer-style models, uses attention in two ways: **between samples** and **between features**. The between-sample attention has both self- and cross-attention components: self-attention among training samples and cross-attention from test samples to train samples. Following the TabPFN implementation, we assume attention layers expect input of shape `[batch, seq_len, input_size]`. In the code, the leading dimensions before `(seq_len, input_size)` are collapsed via `_rearrange_inputs_to_flat_batch`. For between-feature attention this yields an effective batch size of `L * B`, whereas for between-item (between-sample) attention it yields `(G + 1) * B`.

Recall that efficient attention implementations in PyTorch (such as the fused CUDA kernels backing `torch.nn.functional.scaled_dot_product_attention`) tile work across the **batch** and **head** dimensions. On NVIDIA GPUs of Ampere architecture and below, this effectively limits the product `B * num_heads` to at most `65535` CUDA blocks; when it reaches `65536` the kernel can fail with  `CUDA error: invalid configuration argument` (see the corresponding [PyTorch GitHub issue](https://github.com/pytorch/pytorch/issues/133976) for a minimal example where `65535` works but `65536` fails). In TabPFN, large sample sizes `L` or a large number of feature groups `G` can easily push these flattened batch sizes (`L * B` or `(G + 1) * B`) past this limit.

A simple practical fix is to loop over the flattened batch dimension in chunks, so that each call to `scaled_dot_product_attention` stays within the kernel’s limits. This keeps the rest of the model unchanged while avoiding the `invalid configuration` errors at large `L` or `G`. Conceptually, this is can be done via the following patch to the attention computation.

```python
outputs = []

for q_chunk, k_chunk, v_chunk in zip(
    torch.split(q, batch_size, dim=0),
    torch.split(k_b, batch_size, dim=0),
    torch.split(v_b, batch_size, dim=0),
):
    # (B_chunk, Lq, H, D) -> (B_chunk, H, Lq, D)
    Q = q_chunk.permute(0, 2, 1, 3).contiguous()
    K = k_chunk.permute(0, 2, 1, 3).contiguous()
    V = v_chunk.permute(0, 2, 1, 3).contiguous()

    out = F.scaled_dot_product_attention(
        Q,
        K,
        V,
        attn_mask=None,
        dropout_p=dropout_p if dropout_p is not None else 0.0,
        is_causal=False,
        scale=softmax_scale,
    )  # (B_chunk, H, Lq, D)

    # -> (B_chunk, Lq, H, D)
    outputs.append(out.permute(0, 2, 1, 3).contiguous())

attention_head_outputs = torch.cat(outputs, dim=0)
```

A different issue is **hardware support** for efficient attention kernels. PyTorch’s `scaled_dot_product_attention` can dispatch to several backends on CUDA: FlashAttention, memory-efficient attention, or a plain math implementation in C++. The availability of these specialized kernels varies across GPU generations. For educational purposes, and for those who wish to implement these kernels on older or unsupported devices, we refer to [this repository](https://github.com/lucidrains/memory-efficient-attention-pytorch/tree/main). We provide a brief sketch of how the chunking works to reduce the memory footprint below.

```python
def chunked_attention(q, k, v, q_chunk, kv_chunk, scale):
    """
    q: (..., Lq, D)
    k: (..., Lk, D)
    v: (..., Lk, Dv)
    q_chunk: size of query tiles (l)
    kv_chunk: size of key/value tiles (r)
    """
    Lq, Lk = q.size(-2), k.size(-2)
    outputs = []

    for qs in range(0, Lq, q_chunk):
        qe = min(qs + q_chunk, Lq)
        q_tile = q[..., qs:qe, :]                            # (..., l, D)

        # running stats per query row
        mu = q_tile.new_full(q_tile.shape[:-1], -float("inf"))  # (..., l)
        s  = torch.zeros_like(mu)                               # (..., l)
        a  = torch.zeros(*mu.shape, v.size(-1),
                         device=q.device, dtype=q.dtype)        # (..., l, Dv)

        for ks in range(0, Lk, kv_chunk):
            ke = min(ks + kv_chunk, Lk)
            k_tile = k[..., ks:ke, :]                           # (..., r, D)
            v_tile = v[..., ks:ke, :]                           # (..., r, Dv)

            logits = torch.matmul(q_tile, k_tile.transpose(-2, -1)) * scale
            local_max = logits.max(dim=-1).values               # (..., l)
            new_mu = torch.maximum(mu, local_max)

            # rescale old aggregates
            alpha = torch.exp(mu - new_mu)
            s *= alpha
            a *= alpha[..., None]

            # accumulate current tile
            exp_logits = torch.exp(logits - new_mu[..., None])
            s += exp_logits.sum(dim=-1)                         # sum_k e^{z_k}
            a += torch.matmul(exp_logits, v_tile)               # sum_k e^{z_k} v_k

            mu = new_mu

        outputs.append(a / s[..., None])                        # softmax = a / s

    return torch.cat(outputs, dim=-2)                           # (..., Lq, Dv)
```

In this implementation, the key components are:

* It tiles queries into chunks `q_chunk` instead of processing all `Lq` at once.
* It streams over keys/values in chunks `kv_chunk`, computing only `l × r` logits at a time.
* It maintains per-row running statistics `(mu, s, a)` using a numerically stable log-sum-exp merge, so the final output matches full attention as if we had formed the entire `Lq × Lk` score matrix in one go.

## 3. Experiments

<span id="sec:experiments"></span>

We evaluate the TabPFN v2 model with chunking enabled on **TabArena** <d-cite key="tabarena"></d-cite>, which includes 51 tabular datasets spanning classification and regression tasks. We report scaling statistics for memory and runtime in Figure 1, and overall performance on TabArena in Figure 2. Note that in the original and subsequent reports of TabPFN, LIMIX, and TabICL on TabArena, the authors have typically imputed values that exceeded the context length for their respective methods. This might have created a distorted view of model capabilities. In Figure 2, we use only directly measured (non-imputed) results.

{% include figure.liquid path="assets/img/2026-04-27-chunked-tabpfn/tabarena_long_results.png" class="img-fluid" %}

<div class="attn-figure-caption"> Figure 1. Scaling TabPFN v2 to long contexts. Chunked TabPFN matches baseline accuracy where both fit, and extends inference to 100K+ examples. </div> 

{% include figure.liquid path="assets/img/2026-04-27-chunked-tabpfn/elo_vs_baselines.png" class="img-fluid" %}

<div class="attn-figure-caption">
Figure 2. Elo and normalized score across TabArena. Striped bars denote prior imputed TabPFN runs (filled with Random Forest fallbacks when OOM); our chunked TabPFN reports direct measurements.
</div>

Separately, we evaluate TabPFN v2 on the same long-context datasets while varying the context length. Specifically, we sample `num_samples` points from each dataset and then report performance, memory, and runtime in Figure 3. To better understand how context length affects TabPFN’s performance, we perform a *scaling study* on the 15 “long-context” datasets from TabArena. For each dataset, we subsample the training set to progressively larger sizes (3,000 &rarr; 5,000 &rarr; 10,000 &rarr; 20,000 &rarr; 50,000 &rarr; 100,000) and compare baseline TabPFN v2 against our Chunked TabPFN.

* Chunked TabPFN maintains *exact equivalence* to baseline TabPFN while extending feasible context length by roughly 10×.
* Empirical scaling shows either plateau or monotonic improvement—never catastrophic degradation.
* Memory and runtime growth are linear in chunk size, enabling inference on 100 K+ examples with a single GPU.

These findings reinforce that **TabPFN’s in-context generalization truly extends beyond its training limit**, and that the primary bottleneck was *implementation-level memory*, not *model-level capacity*.

{% include figure.liquid path="assets/img/2026-04-27-chunked-tabpfn/tabarena_long_results_per_dataset.png" class="img-fluid" %}

<div class="attn-figure-caption">
Figure 4. Scaling curves for long-context datasets. Each plot shows RMSE, AUC, wall-clock inference time (s), and peak GPU memory (MB).  
Chunked TabPFN tracks baseline accuracy exactly up to 10 K examples and continues scaling to 100 K without degradation.
</div>

## 4. Conclusion

<span id="sec:conclusion"></span>

We presented **Chunked TabPFN**, an exact tiling strategy that enables TabPFN to process *long-context* tabular datasets (100 K+ rows) without retraining, fine-tuning, or any pre-processing such as clustering or compression.

Our main results show:

1. **Exactness without approximation.**
   The chunked attention computation is mathematically identical to the original transformer attention—only the evaluation order changes.
   Predictions match baseline TabPFN bit-for-bit (within floating-point tolerance) for all short-context cases.

2. **Memory scalability.**
   Peak GPU memory scales linearly with tile size instead of quadratically with context length.
   This removes the practical 10 K-sample ceiling and allows inference on 100 K+ rows using 24–32 GB GPUs.

3. **Training-free generalization.**
   Chunked TabPFN retains the spirit of in-context learning: no dataset-specific training, no hyperparameter search, no adaptation steps.
   Despite its simplicity, it matches or surpasses tuned deep tabular models on the long-context slice of TabArena.

4. **Empirical insights.**
   Many datasets continue to improve with larger contexts—suggesting that the PFN prior generalizes beyond its nominal pre-training length.