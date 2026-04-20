<h1 align="center">Beyond CLIP</h1>
<p align="center"><i>A First-Principles Field Guide to Multimodal Embedding — From CLIP to the Post-Embedding Agent Stack</i></p>

<p align="center">
  <a href="https://bigballon.github.io/BeyondCLIP/"><img alt="Read Online" src="https://img.shields.io/badge/read-online%20%E2%86%92-4A90E2?style=flat-square&logo=githubpages&logoColor=white"></a>
  <a href="#citation"><img alt="Updated" src="https://img.shields.io/badge/updated-Apr%202026-2ea44f?style=flat-square"></a>
  <a href="#part-vii--reference"><img alt="Papers" src="https://img.shields.io/badge/papers-100%2B-ff9f1c?style=flat-square"></a>
  <a href="https://github.com/BIGBALLON/BeyondCLIP/pulls"><img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-ec4899?style=flat-square"></a>
</p>

> [!NOTE]
> **This is not a neutral survey.** It is a field guide for people who build, train, or deploy multimodal retrieval at production scale. It takes positions, names the trade-offs other surveys hide, and flags where the community is fooling itself. Every quantitative claim is either tied to a primary source or explicitly marked as a snapshot.

---

### Contents

- [TL;DR — four claims](#tldr--four-claims)
- [Part I — The C–L–I Framework](#part-i--the-cli-framework)
- [Part II — The Two Unlocks: MLLM Encoders and Late Interaction](#part-ii--the-two-unlocks-mllm-encoders-and-late-interaction)
- [Part III — Data, Instruction, Benchmarks: Why the Numbers Lie](#part-iii--data-instruction-benchmarks-why-the-numbers-lie)
- [Part IV — Deployment Economics](#part-iv--deployment-economics)
- [Part V — Open Weights, Closed APIs, and the Moats That Will Remain](#part-v--open-weights-closed-apis-and-the-moats-that-will-remain)
- [Part VI — Forecast 2026 → 2030: Near, Mid, Far](#part-vi--forecast-2026--2030-near-mid-far)
- [Part VII — Reference](#part-vii--reference)
- [Citation](#citation)

---

## TL;DR — four claims

**1. Every architectural debate since CLIP is a point on the C–L–I Pareto surface.** *Compression* (how much semantics fit in a single vector), *Localization* (how finely sub-units align), and *Instruction adaptivity* (how task-conditioned the encoder is) are the three axes on which every model trades. "Dual-tower vs MLLM", "single-vector vs late interaction", "zero-shot vs instructed" are not three independent debates — they are three projections of the same triangle. A production stack is never one model; it is a C-axis first stage, an L-axis structured stage for documents, and an I-axis reasoner as rerank. The triangle is not a choice, it is a stack.

**2. MLLM-as-embedder and late interaction are the two unlocks that define Era 3 (2025–2026).** On MMEB-V2, an 8B Qwen3-VL-Embedding sits at **77.82** (Apr 2026 snapshot), roughly ten points above any pure dual-tower — a C + I win that costs an order of magnitude more indexing and query latency. On ViDoRe V3, Nemotron-ColEmbed-V2-8B leads at **63.42 NDCG@10**, a pure L win on visual documents that was economically impractical until **MUVERA (Google, NeurIPS 2024)** compressed multi-vector ~32× with ~10% higher recall and ~90% lower latency than PLAID on BEIR subsets, and Weaviate 1.31 / Qdrant / Vespa / LanceDB shipped the DB plumbing. Both unlocks are real; neither is universal.

**3. Benchmarks are now the rate-limiting honesty problem, not the rate-limiting capability problem.** The MMEB-V2 top-20 span is under one architectural generation. Most "new SOTA" is instruction-format tuning on data that overlaps with training. On held-out splits, public numbers drop several points. A single MMEB-V2 number, quoted without the CSV-vs-paper distinction and without held-out verification, is meaningless; gaps below ~2 points are noise. The next real breakthrough in the field is **evaluation**, not models.

**4. The compute is migrating from pretraining to inference, and then from retrieval to rerank.** Every 2025–2026 winner (MoCa, CoCoA, Think-Then-Embed, UniME-V2, RzenEmbed, AutoThinkRAG) spends compute in places the 2023 generation ignored: bidirectional continual pretraining, EOS-reconstruction pretext, MLLM-as-judge for hard negatives, reasoning traces *before* the embedding pass. The 2603.14635 analysis on BRIGHT makes the allocation explicit — extra thinking at **rerank** buys **+7.5 NDCG@10**; at **query expansion**, **+1.1**; inside the retriever, **≈0**. Scaling the encoder is over; the next five years are about *where, in the pipeline, the compute lives*.

---

## Part I — The C–L–I Framework

A multimodal retriever maps a heterogeneous item (image, text, page, clip, audio, …) into a geometry in which *semantically similar* ≈ *geometrically close*. Every architectural choice trades along three near-orthogonal axes:

| Axis | Name | What it controls | Cheap if… | Expensive if… |
|---|---|---|---|---|
| **C** | Compression | Semantics fit into a single fixed-dim vector | Corpus semantics are simple per item; queries are global | Items are dense (pages, long video, multi-entity scenes) |
| **L** | Localization | Granularity of within-item alignment (single-vector ↔ multi-vector ↔ token-level MaxSim) | Items have few sub-units; MaxSim compute is cheap | Items hold many retrievable sub-units (words on pages, frames in video) |
| **I** | Instruction adaptivity | Dependence of the encoder's output on a per-query task instruction | Workload is uniform | Workload is heterogeneous RAG with per-query task semantics |

Four representative systems, projected on the triangle:

| System family | C | L | I | Sweet spot |
|---|---|---|---|---|
| CLIP / SigLIP / jina-clip / MetaCLIP 2 | **high** (one 512–1024d vec) | none | none | Uniform photo retrieval, classification, fast first stage |
| ColPali / ColQwen3 / Nemotron-ColEmbed | low (1000+ vecs/item) | **high** (token-patch MaxSim) | low–mid | Visual documents, slides, dense pages |
| E5-V / VLM2Vec / GME / Qwen3-VL-Embedding / Seed1.6 | medium (one 2–4k-d vec) | low | **high** (task-conditioned) | Instructed retrieval, heterogeneous RAG, composition |
| Think-Then-Embed / UniME-V2 / AutoThinkRAG | medium | low–mid | **very high** (externalized reasoning) | Reasoning-heavy, ambiguous, or compositional queries |

**Three load-bearing properties of the frame, each with its falsifier.** *The axes are near-orthogonal but not independent* — pushing C weakly helps L and I; pushing I barely helps C or L; most papers concentrate gains on one axis. A single architectural change that lifts all three by comparable magnitudes at fixed budget would refute this; none has been reported. *Every benchmark selects an axis.* MMEB-V2 is I-heavy; ViDoRe is L-heavy; MIEB tries to balance C across eight categories; Flickr/MSCOCO weight C almost alone. A benchmark whose per-task scorecard cleanly separates C/L/I contributions would falsify the selectivity claim; MIEB gestures toward this but does not close it. *Production stacks always hybridize.* The 2026 retrieval pipeline is **C-axis first stage (SigLIP 2 / jina-clip-v2 / MetaCLIP 2) + L-axis structured stage for document corpora (ColQwen3 / Nemotron-ColEmbed-V2) + I-axis reranker or reasoner (Qwen3-VL-Embedding / Think-Then-Embed / MLLM-as-judge)**. A single-model deployment that dominates heterogeneous workloads without a separate reranker or dual index would falsify this; its appearance would reorganize the field.

### Five years of history, in three moves

**Era 1 · 2021–2023 · push C, ignore L and I.** CLIP (2021) → OpenCLIP → SigLIP → EVA-CLIP-18B. Single-vector dual-tower, contrastive on web image-text pairs. It worked, and it hit a ceiling: no localization, no instruction conditioning, saturation on anything compositional (ARO, MMVP), long-text (Long-CLIP), or high-resolution (PS3).

**Era 2 · 2023–2025 · keep C, seed L and I.** Three parallel waves: *data* replaces architecture as the primary lever (VeCLIP, LaCLIP, DAC, DFN, DCI, DreamLIP, Long-CLIP — alt-text → VLM-written long captions, +5–10 R@1 at roughly fixed architecture); *L* gets a foothold with **ColPali** (June 2024), the first credible demonstration that patch-level MaxSim beats global-vector retrieval on visual documents; *I* gets a foothold with UniIR, MMEB-V1, MagicLens, M-BEIR defining the instructed-retrieval task format, then E5-V and VLM2Vec turning an MLLM into an instruction-conditioned embedder.

**Era 3 · 2025–2026 · I becomes first-class, C is unlocked by MLLMs, L gets compressed to practicality.** MLLM contrastive fine-tunes produce 2–4k-d vectors with materially higher capacity than any Era-1 dual-tower; instruction conditioning becomes table stakes; MUVERA plus 2025–2026 DB integration wave turns late interaction from a niche tool into a default-able primitive for large, dynamic corpora; and a new family of reasoning-augmented embedders (Think-Then-Embed, UniME-V2, AutoThinkRAG) moves compute to inference time, ahead of the embedding itself. The three axes are now being pushed simultaneously, by different techniques that no longer fit in one model. This is why Era-3 retrieval is a **stack**, not a model.

> [!IMPORTANT]
> From here on, read every section as answering a single question: *which axis is being pushed, at what cost to the other two?*

---

## Part II — The Two Unlocks: MLLM Encoders and Late Interaction

Era 3 is defined by two structural changes that happen to be near-orthogonal on C–L–I: a causal decoder turning out to be a better C+I encoder than any bidirectional alternative, and late interaction turning out to be affordable at scale. Each solves a problem the other cannot touch. The modality gap — a long-standing worry about the CLIP geometry — is dissolved as a side effect of both.

### 2.1 Why a causal MLLM is secretly a good encoder

On paper, using a causal decoder for retrieval is strange. Attention is triangular; the last token sees everything but nothing sees the last token; retrieval similarity is symmetric and causal LMs are asymmetric by construction. Yet E5-V, VLM2Vec, GME, Seed1.6-Embedding, Qwen3-VL-Embedding, KaLM-Embedding, MoCa, RzenEmbed — all follow the same recipe (run a forward pass, pool the last hidden state, done) and beat every bidirectional alternative at comparable scale. Four mechanisms explain why; a fifth explains where it breaks.

**The last-token bottleneck is a feature, not a bug.** Next-token prediction forces *every preceding token to contribute signal to the final position* — otherwise the predictor cannot be coherent. A well-pretrained causal LM therefore already contains, hidden inside it, an encoder whose output is the last hidden state: a task-agnostic compressed summary. Contrastive fine-tuning rotates this summary into retrieval-friendly subspace; it rarely has to build the summary from scratch. This is Jiang et al.'s argument for text (LLM2Vec); it generalizes to multimodal inputs because cross-modal tokens pass through the same residual stream.

**Cross-modal interleaving shares the compression subspace across modalities.** In CLIP, image and text hidden states are produced by *different networks* glued by a shared projection. In an MLLM embedder, they pass through the *same* transformer stack, interleaved in a single residual stream, with attention free to mix them at every layer. The functional subspace used for retrieval is therefore shared by construction. This is the C-axis property that dissolves the **modality gap** (Liang et al., NeurIPS 2022) at the pooled token: CLIP's narrow disjoint cones are a free-energy minimum of the dual-tower contrastive loss landscape — no gradient reason to close them. MLLMs do not abolish the gap (hidden states still cluster by modality layer-by-layer) but they *relocate* it to the subspace used for retrieval, which is all retrieval needs. The papers that explicitly address modality geometry — MoCa, UniME, Kuaishou's Modality Curation — pick up additional points on cross-modal compositional queries where the residual gap still bites.

**Instruction prompts act as soft task heads.** The E5-V prompt `Summarize the above image in one word:` does the job of a "task head" in CLIP's text tower — it tells the model which retrieval task it is performing and empirically routes information through different parts of the last hidden state. MMEB-style per-task instructions generalize this: the encoder learns a small, fast router from instruction to retrieval subspace. This is pure I-axis capability with no CLIP analogue.

**The minority report: bidirectional beats causal at the margin.** Two independent groups now show that flipping continual pretraining to bidirectional attention plus a masked or EOS-reconstruction pretext recovers measurable points. **MoCa** ([arXiv 2506.23115](https://arxiv.org/abs/2506.23115), Renmin U. + Microsoft Research Asia) runs a modality-aware continual pretraining stage with bidirectional attention + text MLM + image MAE, then contrastive fine-tuning: MoCa-7B reaches 71.5 on MMEB-V1 vs 67.1 for a bidirectional-only baseline (≈ +4.4) and vs 69.8 for mmE5. **CoCoA** ([arXiv 2603.01471](https://arxiv.org/abs/2603.01471), ICT/CAS + Baidu) replaces the pretext with collaborative attention + EOS-based reconstruction, reaching 70.6 at 7B. The effect is real, reproducible across objectives, and cheap — roughly one extra pretraining epoch. Open base models (Qwen3-VL, InternVL3) are still purely causal. This is low-hanging fruit; see §VI for why it lands by Q4 2026.

**The 2026 turn: compute before the embedding.** A newer family shows the next axis beyond bigger encoders. **Think-Then-Embed (TTE)** ([arXiv 2510.05014](https://arxiv.org/abs/2510.05014), Meta + UCF + NYU, ICLR 2026) first generates a reasoning trace with a small MLLM reasoner, then conditions the embedder on both the query and the trace — reported 7% absolute MMEB improvement over recently proposed baselines on the 78-task average. **UniME-V2** ([arXiv 2510.13515](https://arxiv.org/abs/2510.13515), Alibaba, AAAI 2026) replaces the brittle similarity heuristic in contrastive hard-negative mining with MLLM-as-a-judge soft scoring. **RzenEmbed** ([arXiv 2510.27350](https://arxiv.org/abs/2510.27350), Qihoo 360) adds hardness-weighted contrastive loss; **AutoThinkRAG** ([arXiv 2603.05551](https://arxiv.org/abs/2603.05551)) routes by query complexity and reports 82.13% on DocBench with 18.9% fewer per-query tokens than naïve multimodal RAG. Each is a migration of compute from training to inference, from encoder to reasoner — the same dynamic that played out in chat LLMs with o1/Claude reasoning modes, arriving in retrieval roughly one year later. The 2026–2027 recipe almost certainly combines *causal MLLM backbone* + *bidirectional continual pretraining* + *inference-time reasoning trace* + *MLLM-as-judge hard-negative mining*. Anyone selling one lever is selling a quarter of the answer.

### 2.2 Late interaction as within-item localization

ColPali / ColQwen3 / ColNomic / Nemotron-ColEmbed-V2 dominate visual-document retrieval (Nemotron-ColEmbed-V2-8B leads ViDoRe V3 at **63.42 NDCG@10** per NVIDIA's Feb-2026 pipeline snapshot) and barely move Flickr30K / MSCOCO text↔image retrieval. Under C–L–I this is exactly what you expect: a PDF page rendered to pixels produces ~1,000 patches, each aligned to a retrievable sub-unit (word, cell, axis label). MaxSim over query tokens against these patch tokens is ColBERT on a document that happens to be pixels. On photographs the same patches correspond to texture and lighting — units that are not individually retrievable. Page queries are multi-concept ("*Q3 revenue in Europe*") and MaxSim aggregates multi-token relevance; photo queries are typically single-concept and a single global vector losslessly compresses them.

This is a framing point the community keeps missing: **the architecture community calls this "late interaction"; the retrieval community should call it *within-item localization*.** It is a different problem that happens to share the dense-vector toolkit. L-axis is useful only when items have queryable sub-units; on anything else, the ~1000× storage tax is unrecoverable.

### 2.3 The MUVERA era

Late interaction's Achilles' heel was always storage. At 10M pages × ~1,030 tokens × 128-d fp16 ≈ **~2.6 TB** of raw vectors before any compression, and MaxSim compute scales linearly in tokens. Three waves of compression closed the gap:

| Technique | Year | Effect | Residual cost |
|---|---|---|---|
| PLAID centroid clustering (ColBERT-v2) | 2023 | ~8–10× storage reduction; candidate-filter + rerank | Slightly slower query path |
| Binary / int8 quantization of MaxSim tensors | 2024 | ~4–8× storage; recall –1 to –3 pts | QAT helps |
| **MUVERA: Fixed Dimensional Encodings** ([arXiv 2405.19504](https://arxiv.org/abs/2405.19504), **NeurIPS 2024**) | 2024 → DB 2025–26 | Combined with PQ-256-8: **~32× storage** vs raw at negligible quality loss. Vs PLAID on 6 BEIR subsets: **~10% higher recall**, **~90% lower latency**. Weaviate's blog reports **~70% memory reduction on LoTTE**. | Requires DB support |

MUVERA's mechanism is worth stating plainly: it projects a variable-length multi-vector set into a single *fixed-dimensional* vector compatible with any standard ANN index, preserving MaxSim-style relevance under inner product. The breakthrough was not published in 2024 — it was that 2025–2026 DB integration turned the model-side choice into a configuration toggle. Weaviate 1.31 exposes it as a first-class `Encoding.muvera(...)` option; Qdrant's FastEmbed line treats it as an encoder-side post-processing step; Vespa provides native tensor MaxSim; LanceDB and Milvus are catching up. Before this wave, L-axis retrieval was a tool for medium, static, high-value document corpora. After it, L-axis retrieval is a candidate for large, dynamic corpora — and the C-axis single-vector default starts looking weaker on document-heavy workloads.

The 2026 practitioner defaults fall out cleanly. *Natural-image retrieval* (photos, social content, products) stays single-vector; late interaction offers no material gain and still costs ~30× storage even post-MUVERA. *Visual-document retrieval* (PDFs, slides, UI screenshots, scans, charts, diagrams) is now decisively late-interaction; ColQwen3-7B or Nemotron-ColEmbed-V2-8B is mainstream where it was niche 18 months ago. *Mixed corpora* (a web page with both a photo and a table) demand a two-track index — single-vector for the photo, multi-vector for the document segments. No single model handles both optimally, and MUVERA does not change that — it just makes the different problem affordable at scale.

---

## Part III — Data, Instruction, Benchmarks: Why the Numbers Lie

This section merges three debates that are usually treated separately but are in fact one: *how do we know a model is good, and what does "good" reward?* Data, instruction conditioning, and benchmark design are the three knobs through which a paper's headline number gets manufactured, and all three are now contaminated enough that a single score without provenance is indistinguishable from a marketing claim.

### 3.1 The data stratification, and its ceiling

Every paper since 2024 attributes >80% of gains to data engineering. This is correct in a narrow regime and overclaimed everywhere else. The decomposition that holds across regimes:

| Regime | Dominant driver | Representative evidence |
|---|---|---|
| < 500M image-text pairs | Architecture | SigLIP sigmoid loss, EVA-CLIP masking, FLIP token drop — each delivers 3–8 pt gains regardless of data volume |
| 500M – 5B pairs | Data cleaning + re-captioning | VeCLIP / DreamLIP / LaCLIP / DAC / DFN / DCI / Long-CLIP: alt-text → VLM caption lifts R@1 by 5–10 pts |
| 5B – ~50B pairs, decent captions | Synthetic hard negatives + task mixtures | MegaPairs / mmE5 / Modality Curation / B3 — data *shape* (mixture across modalities and tasks) beats data *volume* |
| > 50B tokens, rich task formats | Inference-time techniques | Seed1.6 / Qwen3-VL territory — instruction engineering, MLLM-as-judge hard-negative mining (UniME-V2), reasoning augmentation (TTE), prompt routing |

This resolves the apparent contradiction between "scale is all you need" (DataComp, LAION) and "data quality is all you need" (DFN, Seed1.6). Both are right at different altitudes. The **MegaPairs → mmE5 → Modality-Curation arc** is the one that matters for 2026 practitioners: MegaPairs (BAAI, Dec 2024) synthesized hundreds of millions of image-image-text triplets with no human labelers, ending the "labelling-bounded" myth; mmE5 (Renmin U. + Microsoft, Feb 2025) showed multilinguality is nearly free if you synthesize; Modality Curation (NEU + Kuaishou, May 2025) showed batch composition matters, yielding free MMEB points. The collective — and rarely stated — implication: **the data pipeline has replaced the architecture as the primary IP of an embedding lab.** "Reproduce from our weights and code" has quietly become the gold standard, because "reproduce from scratch" is no longer feasible outside well-funded labs.

The ceiling nobody quite admits: once the 78-task MMEB mixture is saturated with synthetic + mined pairs, the marginal point gets expensive enough that inference-time compute (reasoning, MLLM-as-judge rerank, test-time retrieval) is now cheaper per point than more training data. This is the regime TTE, UniME-V2, AutoThinkRAG exploit. Expect the 2026–2027 paper balance to tilt back toward *architecture + inference*, away from *data volume* — a full cycle from 2022.

### 3.2 Instruction conditioning: product feature disguised as modeling technique

Every competitive 2025–2026 model exposes an `instruction` field. MMEB-V2 bakes 78-task instructions into the eval. The quiet, ugly consequence: **the model learns to branch on the instruction**, and leaderboard numbers start reflecting how well it *switches tasks* rather than how well it *represents content*. Strip the instruction and many 2026 SOTA models drop several points on zero-shot — a gap that rarely appears in press releases. Small prompt rewordings can move NDCG@5 by several points; this is under-reported. Three observations that practitioners keep relearning the hard way: *instructions help only if your workload actually provides them*; *zero-shot and instructed retrieval are different skills* (MagicLens, UniME, ColPali do both well; heavily-instructed models lean on the prompt branch); *style drift is a production risk*. Under C–L–I, instruction conditioning is a pure I-axis lever — valuable where your workload has task heterogeneity, a net loss where it does not.

### 3.3 Benchmarks: saturation, contamination, and what a score actually means

From MMEB-V1 (Nov 2024) to MMEB-V2 (Apr 2026), the top score moved explosively in the first year and is now saturating. Three symptoms. *Per-task ceiling effects* — on the easier MMEB-V2 tasks, the top-20 models are within 1–2 points, and further progress is noise. *Instruction-format overfitting* — several submissions gained points by tuning the MMEB task-prompt formatting without retraining the encoder. *Data leakage* — the current generation is trained on corpora that plausibly include MMEB-related train splits (MMEB reuses M-BEIR, UniIR, and public retrieval sets). Not fraud; the normal hygiene problem every benchmark eventually faces (ImageNet, GLUE, SuperGLUE). But in a field moving this fast, **leaderboards overstate generalization**; an external production deployment typically sees the public MMEB-V2 score *minus 3–5 points*.

Under C–L–I, the differences between benchmarks stop being mysterious:

| Benchmark | Primary axis | Winning architecture (Apr 2026) |
|---|---|---|
| MMEB-V2 (TIGER-Lab) | **I** (instruction + composition + task-switching) | Large MLLM + rich instruction data (Qwen3-VL-Embedding-8B, Seed1.6-Embedding) |
| MIEB (MTEB team) | Balanced **C** (encoder quality across 8 categories) | Balanced MLLMs + strong classical encoders |
| ViDoRe V3 (Illuin) | **L** (localization-heavy visual-doc retrieval) | Late interaction (Nemotron-ColEmbed-V2, ColQwen3) |
| M-BEIR (TIGER-Lab) | **I** (instruction-aware multimodal retrieval) | Reasoning-augmented retrievers (TTE, RAR) |
| MTEB v2 (MTEB team) | **C** on text, partial **I** | Text-only giants (Harrier-OSS-27B, KaLM-Embedding-Gemma3-12B) |
| CoIR (ACL 2025) | Code retrieval **C** | SFR-Embedding-Code-2B_R (≈ 67.4 aggregate, per model card) |

**Cross-leaderboard snapshot — April 2026.** Every number is a single-configuration snapshot from a primary source (leaderboard CSV, model card, or technical report). They move week-to-week. Gaps <2 points are noise.

| Model | MMEB-V2 (Overall) | MTEB v2 (Multilingual) | ViDoRe V3 NDCG@10 | License | Dominant axis |
|---|---:|---:|---:|---|---|
| Qwen3-VL-Embedding-8B | **77.82** (CSV #1) | 67.9 | — | Apache-2.0 | C + I |
| Seed1.6-Embedding-1215 | 75.08 (CSV #2) | — | — | Closed API | C + I |
| IFM-TTE-7B (Think-Then-Embed) | 73.06 (CSV #5) / 74.1 (paper) | — | — | Open | I + inference |
| RzenEmbed-v2-7B | 71.12 (CSV #7) / ~72.9 (paper) | — | — | Weights on HF | C + I |
| UniME-V2 (LLaVA-OV-7B) | 71.2 (paper) / 59.12 (CSV) | — | — | Open | I |
| VLM2Vec-V2-2B | 58–59 (paper / TTE Table 12) | — | — | Apache-2.0 | C + I |
| GME-Qwen2-VL-7B | 57.8 (TTE Table 12) | — | — | Apache-2.0 | C + I |
| Nemotron-ColEmbed-V2-8B | — | — | **63.42** (#1) | Open | **L** |
| Nemotron-ColEmbed-V2-4B | — | — | 61.54 | Open | L |
| Harrier-OSS-v1-27B | — | **74.3** (model card) | — | Apache-2.0 | C (text) |
| KaLM-Embedding-Gemma3-12B | — | 72.32 (self-reported, 2511 snapshot) | — | MIT | C (text, multilingual) |
| Gemini-Embedding-001 | — | 68.37 (MMTEB, per KaLM card) | — | Closed API | C (T+I+V+Audio+PDF breadth) |

Why two MMEB-V2 columns for some rows? The leaderboard CSV penalizes models that skip subsets more than the authors' own Table numbers do. For a fair within-column comparison, pick the CSV. For a paper-claim comparison, pick the Table. Always name which. Corrections to community folklore: RzenEmbed and UniME-V2 do **not** sit at MMEB-V2 ~77; TTE is mid-70s, not 76; the 76.9 sometimes quoted for Seed1.6 is the Qwen3 paper's 78-task recomputation, not a MIEB ranking. **Never quote a single number without naming the axis, the CSV-vs-paper distinction, and whether the evaluation used held-out splits.** This is not pedantry — it is the only defense against a saturating benchmark.

---

## Part IV — Deployment Economics

Leaderboards never show the numbers that dominate a production budget. What follows is a single-configuration baseline on one H100 SXM, fp16, batch tuned per model, CUDA 12.x, realistic image distribution. Treat ±20% as noise.

| Model | Axis | VRAM | Time to index 10M imgs | p99 latency | Storage (10M × vec) |
|---|---|---:|---:|---:|---:|
| SigLIP 2 SO400M-L | C | ~18 GB | ~6 h | ~8 ms | ~23 GB (1152d) |
| MetaCLIP 2 G/14 | C | ~40 GB | ~16 h | ~15 ms | ~26 GB (1280d) |
| jina-clip-v2 | C | ~24 GB | ~8 h | ~12 ms | ~20 GB (1024d) |
| GME-Qwen2-VL-2B | C + I | ~36 GB | ~40 h | ~55 ms | ~31 GB (1536d) |
| GME-Qwen2-VL-7B v2 | C + I | ~60 GB | ~5 d | ~120 ms | ~72 GB (3584d) |
| Qwen3-VL-Embedding-8B | C + I | ~60 GB | ~6–8 d | ~140 ms | ~72 GB (3584d) |
| ColPali v1.3 | L | ~22 GB | ~10 h (+~1000× storage) | ~30 ms + MaxSim | ~2.6 TB (raw) |
| ColQwen3-7B | L | ~48 GB | ~6 d (+~1000×) | ~110 ms + MaxSim | ~2.6 TB (raw) |
| Nemotron-ColEmbed-V2-8B | L | ~56 GB | ~7 d (+~1000×) | ~130 ms + MaxSim | ~2.6 TB (raw) |
| *↳ + MUVERA / FDE* | L (compressed) | same | same | replaced by single inner product | **~80 GB (~32× smaller)** |

Three rules of thumb. *Every 2× in encoder parameters → roughly 2.5–3× in indexing and 1.5–2× in query latency* — MLLMs pay in dynamic resolution and variable-length attention. *Multi-vector without compression ≈ 1000× storage vs single-vector*; MUVERA cuts this to ~30×, which is the regime where hybrid C+L stacks become a default rather than an exotic choice. *Query latency in MLLM encoders is decode-bound* — they are still autoregressively producing special embedding tokens, and vLLM/SGLang-style batched decoding offers several-× headroom that most stacks do not yet use. This is the first target of 2026-H2 inference-infra work (see §VI-Near).

**Matryoshka + binary is not a free lunch.** The marketing claim — "cut storage 64× with <1% quality loss" — holds only on the easiest workloads. MRL truncation (3584 → 512) costs ~1–2% on easy tasks and materially more on compositional / multi-entity / visual-document workloads; 1-bit binary quantization adds percentage points more. Published combined-loss-under-5% results typically come from classification-style probes, not retrieval on hard heterogeneous data. The way to recover the marketing claim is not to deny it — it is to **binary-recall + full-precision rerank**: retrieve top-100 with MRL+binary, re-score those 100 with fp16. Hybrid Recall@10 sits within ~1% of full precision at ≥32× smaller storage. Matryoshka + binary is a *first-stage* technique, not a full-stack technique; if you are not reranking, do not binarise.

**What breaks when you put ColPali in production.** Storage explosion is mitigated by MUVERA but the compressed index still costs ~30× a single-vector corpus — budget accordingly. Serving MaxSim is the DB question, not the model question: Qdrant ships native multivector + MaxSim and is the default for ColPali/ColQwen deployments; Weaviate 1.31 adds native MUVERA via `Encoding.muvera(...)`; Vespa exposes tensor MaxSim; LanceDB has multi-vector columns with client-side MaxSim; Milvus 2.5 supports multi-vector fields with MaxSim as a `group_by` workaround, with 3.0 on the roadmap; FAISS reduces client-side. Pick the DB on *this* feature, not on headline benchmarks. Freshness is hard — each document is thousands of inserts, and bulk-reindex cadences that worked for single-vector break here, although MUVERA's compressed encoding updates incrementally like a normal vector. Chart and screenshot brittleness dominate over model choice on low-DPI inputs; input preprocessing matters more than which ColQwen you picked. And MaxSim at high QPS can become the bottleneck, not the encoder — MUVERA's ~90% latency reduction on the BEIR subsets is the only reason this is tractable at scale.

> [!CAUTION]
> ColPali is the right answer for *medium-sized, high-value, static or slow-changing* document collections. Post-MUVERA, it is also defensible for *large, dynamic* corpora. For *casual web-scale PDFs, ticket attachments, low-value user uploads*, an OCR + text-embedding pipeline is often cheaper end-to-end. Choose architecture on corpus economics, not benchmark scores.

---

## Part V — Open Weights, Closed APIs, and the Moats That Will Remain

The top of almost every 2026 multimodal leaderboard is populated by Chinese labs — Alibaba (Qwen3-VL, GME, UniME), ByteDance (Seed1.6), Tencent (KaLM, LLaVE), BAAI (MegaPairs, BGE), Ant (M2-Encoder, GroupRank), Kuaishou (Modality Curation, UniECS), Xiaohongshu (LamRA), Qihoo 360 (RzenEmbed). This is structural, not cyclical. Three reasons, stated without hyperbole. *Strong open-weights MLLM bases in the 2–14B class* — Qwen2.5-VL, Qwen3-VL, InternVL3, GLM-4.1V, DeepSeek-VL2 are strong open starting points, and embedding quality inherits directly from base quality. *Vertical integration of data pipelines* — ByteDance, Alibaba, Kuaishou, Xiaohongshu operate among the world's largest image-text interactive corpora; "data > architecture" structurally favors them. *Incentive asymmetry* — Chinese labs tend to publish open weights to stake international-reputation claims, while several Western labs gate their best models behind APIs. Public leaderboards therefore *systematically over-represent open-weights Chinese output and undercount closed Western models* (OpenAI, Anthropic, Cohere) that do not submit.

The narrative has to be stated carefully. The Western open-weights scene is not empty. Meta released **Llama 4 (Scout / Maverick)** as natively multimodal open weights in April 2025, and Llama 3.2 Vision earlier; a Llama-4-based embedder is obvious next. **LLaVA-OneVision-1.5** ([arXiv 2509.23661](https://arxiv.org/abs/2509.23661)) competes with Qwen2.5-VL on several benchmarks. Microsoft's **Harrier-OSS-27B** tops text MTEB v2 at 74.3. Meta FAIR / NVIDIA / Google DeepMind still lead on narrow fronts (Perception Encoder, SigLIP 2, PS3, Nemotron-ColEmbed-V2). Salesforce + Waterloo (VLM2Vec), IBM (DAC), Apple (DFN, VeCLIP), Jina AI, Snowflake (Arctic-Embed 2.0), Cohere (Embed v4), Voyage (voyage-multimodal-3), Nomic all ship production-grade multimodal or text embedders outside China. The honest summary: **the 2–14B open-weights multimodal embedding space is disproportionately Chinese, and the production-API breadth space is disproportionately Western. Neither dominates the other end.**

Where closed APIs will keep their moats into 2027–2028 is the set of modalities that are data-gated rather than compute-gated. *Audio embeddings and unified T+I+V+Audio+PDF*: Gemini Embedding is the only credible unified multimodal product as of April 2026 — the experimental `gemini-embedding-exp-03-07` reaches mean-task ≈ 68.32 on the Multilingual MTEB leaderboard, and no open-weights peer matches its modality breadth. *Long-context PDFs (>200 pages, mixed text + figures)* — Gemini and Voyage-multimodal-3 have bespoke tokenizers; open weights lag. *3D / CAD / point clouds* — almost entirely absent from the open ecosystem. *Tabular + image joint retrieval* — Cohere Embed v4 handles text + images + mixed PDFs; truly native tabular encoding is rare on both sides. These are smaller markets than photo search, but they are where the closed-API moat compounds, because the *training corpora themselves are not open-sourceable*.

> [!IMPORTANT]
> The Chinese-dominance narrative is half structural, half benchmark design. If Anthropic submitted a Claude multimodal embedder, or OpenAI submitted `text-embedding-3-multimodal`, the top of several leaderboards would shuffle. They haven't, because they don't compete on public benchmarks. If your use case involves audio, long PDFs, 3D, or tabular+visual joint retrieval, **do not wait for open weights to catch up** — they may not before 2028.

---

## Part VI — Forecast 2026 → 2030: Near, Mid, Far

Forecasts are only useful if they are falsifiable. Each bet below names a *mechanism*, a *signal to track*, and a *falsifier*. Probabilities are subjective and calibrated against the author's track record in adjacent fields. **Brier-score them in 2027, 2028, 2030.** Many bets in the far tier are load-bearing for the thesis of the guide itself: if the far tier is wrong, the C–L–I frame is probably over-stated.

### Near (12–18 months · 2026 Q3 – 2027 Q4)

This horizon is about *where compute moves* and *which plumbing ships*. Bets here are dominated by mechanisms already demonstrated in papers and awaiting productization.

| # | Bet | P(true) | Signal / falsifier |
|---:|---|---:|---|
| N1 | Bidirectional continual pretraining is standard for top-5 MLLM embedders | ~75% | At least one of {Qwen3-VL-Embedding v2, InternVL3-Embedding, Seed1.7-Embedding} ships with *bidirectional continual* or *MLM-MAE pretext* language in the card. Falsifier: all stay purely causal and still win by >2 pts |
| N2 | Test-time compute is allocated at **rerank**, not query expansion, not retriever-internal CoT | ~85% | LlamaIndex / Haystack / DSPy default templates read "single-vector first + multi-vector middle + reasoning rerank on top-k". Falsifier: a published system shows >5 NDCG@10 gain from retriever-internal CoT on a held-out bench |
| N3 | MUVERA-class FDE is the default for new multi-vector deployments on the major DBs | ~70% | Weaviate / Qdrant / Vespa / LanceDB default deployment docs recommend FDE. Falsifier: <30% adoption in late-interaction workloads by end-2026 |
| N4 | An open-source Embed-vLLM / SGLang-embed stack yields ≥3× throughput on Qwen3-VL-Embedding vs HF Transformers | ~65% | Public release with reproducible benchmark. Falsifier: none by mid-2027 |
| N5 | Reasoning-augmented embedders (TTE/UniME-V2 family) hold ≥2 top-5 MMEB-V2 slots | ~70% | MMEB-V2 CSV top-5 in Dec 2026. Falsifier: ≤1 TTE-style entry |
| N6 | A first alpha of a consolidated multimodal benchmark (MTEB-MM or equivalent) lands, with held-out splits and C/L/I scorecard | ~55% | MTEB team RFC. Falsifier: nothing by Q2 2027 |
| N7 | Llama-4-based or Claude-Haiku-based open-weights embedder enters MMEB-V2 top-10 | ~55% | Specifically, a Western 8–14B open-weights VLM-based embedder appears in the top-10 | Falsifier: no Western open-weights entry in the top-10 through end-2026 |

**The load-bearing implication of the near tier** is that the 2026 recipe (causal MLLM + bidirectional continual + reasoning rerank + MUVERA) crystallizes as *the default* rather than *an edge configuration*. After that happens, further gains on MMEB-V2 are bounded by benchmark hygiene, not by model capability.

### Mid (2–3 years · 2027 – 2028 Q4)

This horizon is about *what replaces the leaderboard and what squeezes out of long-context LLMs*. Bets here are structural, not model-level.

**M1 · Public multimodal benchmarks bifurcate into a "trust tier" (held-out, private split, C/L/I scorecard) and a "marketing tier" (public static leaderboards).** Mechanism: saturation + contamination + instruction-format gaming have crossed the threshold where a score is uninterpretable without provenance; institutional buyers (cloud vendors, regulated industries) pay for trusted eval. *P ≈ 65%*. Falsifier: public MMEB / MTEB remain the referenced source for enterprise RFPs through 2028.

**M2 · Long-context LLMs (>4M tokens) absorb a meaningful slice of "retrieval" tasks that are currently RAG.** Mechanism: the point where dumping a 500-page document into context beats retrieving three chunks has moved from "theoretical" to "Gemini 2.5 in production" — the remaining economic constraint is inference cost, which halves roughly every 9–12 months. *P ≈ 55% that >20% of current small-corpus RAG workloads migrate to in-context by end-2028.* Falsifier: token-per-dollar inference cost fails to drop by 4× vs April 2026.

**M3 · The line between "retrieval" and "tool use" dissolves at the agent layer.** Mechanism: MCP-style tool protocols already treat "search the KB" as one capability among many; reasoning-augmented retrievers (§II.1) already run inference-time logic that is indistinguishable from a tool call. *P ≈ 70% that the default agent framework by 2028 treats retrieval as a *capability* rather than a separable *pipeline stage*.* Falsifier: standalone RAG frameworks (LlamaIndex, Haystack) continue as dominant paradigm and agent frameworks stay separate.

**M4 · A unified open-weights T + I + V + Audio embedder reaches within 2 pts of Gemini Embedding's multilingual MTEB score on held-out splits.** Mechanism: the data gap closes slowly but the model-side gap is one pretraining run. *P ≈ 45% by end-2028.* Falsifier: no open-weights model matches Gemini on any unified eval by 2028.

**M5 · Embodied retrieval — for robots, AR glasses, and offline agents — becomes a distinct sub-field.** Mechanism: the scene in front of a wearable device is a new kind of "corpus" (spatially indexed, time-streamed, identity-bound), and existing embedding stacks do not model the spatial or temporal structure. Early signals: Meta Aria, visual SLAM + VLM hybrids, Twelve Labs' video work. *P ≈ 60% that a benchmark for this lands by 2028.* Falsifier: no such benchmark exists and the use case remains niche research.

**M6 · "Retrieval as preference alignment" — the retriever itself is RLHF'd on human or LLM judge signal, not just contrastive-trained.** Mechanism: MLLM-as-judge in UniME-V2 is already a weak version; the next step is closing the loop with user-interaction data. Large consumer platforms (search, social, e-commerce) have the data moat; small labs do not. *P ≈ 70%* for at least one major retrieval paper per year in this direction starting 2027. Falsifier: contrastive InfoNCE remains the de-facto training objective across all top-10 papers through 2028.

**M7 · Chip and energy geography splits the embedding stack into two non-interoperable lineages.** Mechanism: export controls, national-AI policies, and domestic chip supply lines (Ascend / Cambricon vs NVIDIA) combine with sovereign data rules (EU AI Act, China's generative-AI provisions) to make cross-border deployment of certain embedder + corpus combinations legally or operationally infeasible. *P ≈ 55%* that by end-2028 a "sovereign-embedding" tier of vendors explicitly markets on compliance + locality. Falsifier: one global open-weights stack continues to dominate without legal fragmentation.

### Far (3–5+ years · 2029 – 2030+)

This horizon is where the C–L–I frame itself comes under pressure. Far-tier bets are deliberately large-scope — falsifying them would reshape the field, not just reorder a leaderboard.

**F1 · Embedding as a standalone artifact enters its late phase; "retrieval" becomes a subset of agent memory.** Mechanism: four convergent trends kill the standalone embedder for new projects while leaving installed fleets alive for years. *First*, long-context LLMs swallow small-corpus RAG. *Second*, reasoning-augmented retrievers push the embedder into the reasoner's latent state — Think-Then-Embed generalizes to "embedding is a function of the reasoner's working memory at the moment of retrieval", not of the document alone. *Third*, agent memory architectures (episodic + semantic + procedural, à la Voyager, MemGPT, Letta) treat dense vectors as one substrate among several — structured KV, symbolic indices, code-as-memory. *Fourth*, MLLM-as-judge reranking eats the top of the quality distribution, leaving the embedder as a cheap candidate-gen layer. The practical effect: by 2030, greenfield retrieval projects will look like *agent + tool registry + hybrid memory* rather than *encoder + ANN + reranker*; "embedding model" becomes a commodity inside the agent, not the center of the stack. *P ≈ 60%.* Falsifier: standalone embedding APIs (OpenAI, Cohere, Voyage, Jina) grow their embedding-specific revenue share through 2030.

**F2 · The CLIP / SigLIP / ColPali generation becomes "classic layer" infrastructure — paid for, under-maintained, everywhere.** Mechanism: even as the frontier moves to agent memory, the installed base of vector indices, product search engines, recommendation pipelines, and academic benchmarks running on single-vector and ColPali encoders is measured in exabytes of persisted vectors. Migration cost is gigantic. Every previous ML generation (word2vec, ELMo, BERT base, ResNet-50) survived a decade past its SOTA obsolescence in production; this one will too. *P ≈ 85%.* Falsifier: a migration event (a major cloud deprecating CLIP-era encoders, a forced standardization) wipes the installed base by 2030.

**F3 · Evaluation moves from static public leaderboards to *living* benchmarks and private evals; publication norms follow.** Mechanism: (M1) plus the gradual realization that any public test set, once named in a tweet, is compromised within one training cycle. Living benchmarks — dynamically generated queries against held-out corpora, with periodic rotation — become the only reputationally defensible eval. Academic papers start citing *distributions of scores* rather than single numbers. *P ≈ 55%.* Falsifier: MMEB-V4 / MTEB v3 dominate publications in 2029 on a static-leaderboard model.

**F4 · Governance and auditability become first-class features of embedding stacks.** Mechanism: embeddings leak memorized content (membership inference is easier than on generative LLMs); vector databases hold personal data; multimodal embedders have copyright exposure on training images. Enterprise procurement already asks "is your training data auditable"; regulators will. By 2030, a production embedder ships with a training-data provenance document, a membership-inference bound, and a differential-privacy story, or it does not get bought by regulated industry. *P ≈ 70%.* Falsifier: no major enterprise procurement in 2029–2030 requires audit trails for embedding models.

**F5 · The "one embedding per item" assumption dies for a large class of items.** Mechanism: a single-vector representation of "a video", "a codebase", "a person", "a scientific paper" is a dimensional straitjacket. The successor is *a bundle of representations* (spatial, temporal, semantic, stylistic, metadata) whose relevant subspace is selected dynamically by the querying agent — late interaction generalized to arbitrary attribute axes. MUVERA-style compression extends from "multi-token" to "multi-facet". *P ≈ 50%.* Falsifier: single-vector remains the dominant format for 90%+ of new production deployments in 2030.

**F6 · An embedding-free retrieval architecture beats a dense one on a major benchmark.** Mechanism: hybrid stacks of structured indices (code ASTs, scene graphs, symbolic memory), LLM-generated query-specific indices, and neural attention over raw corpora — rather than pre-computed dense vectors — reclaim the top of some leaderboard. Precedent: SPLADE in text showed sparse can beat dense; the multimodal analogue has not landed but is plausible. *P ≈ 40%.* Falsifier: every top-5 entry on every major retrieval benchmark through 2030 uses pre-computed dense vectors as the primary primitive.

**F7 · Embedding geography bifurcates into distinct technical lineages, not just distinct vendors.** Mechanism: M7 compounds over five years. By 2030, the Western lineage optimizes for API breadth + governance; the Chinese lineage optimizes for open-weights scale + domestic deployment; a third lineage (EU / sovereign clouds) optimizes for auditability + privacy. Interoperability exists at the interface level (vectors are vectors) but not at the training-data or compliance level. *P ≈ 45%.* Falsifier: a single global embedding stack (most likely a hyperscaler's) dominates all three geographies in 2030.

**F8 · Human-in-the-loop retrieval is the dominant consumer paradigm; vector similarity is invisible to the user.** Mechanism: the "search bar returns ten links" UX is already eroding — Perplexity, ChatGPT Search, Google AI Mode replace it with a conversational reformulation loop. In that loop, embedding is a background primitive, not a product surface; the *product* is the conversation. *P ≈ 75%.* Falsifier: classical search UX (query → ranked list) remains the dominant consumer modality in 2030.

### Bets I am not making

For calibration, three non-bets for the 12–24 month window. **No single MLLM encoder will "rule them all"** — the C–L–I triangle is real and stacks will hybridize, not collapse. **No >82 MMEB-V2 model will win on all tasks equally** — saturation plus contamination make this a diminishing-returns problem; specialised models will win specialised axes. **Open-weights audio-text embeddings will not reach Gemini Embedding parity by end-2027** — the data gap is too wide.

### Where the field is actually stuck — the open problems that would move the far tier

If you want a paper that is read in 2028 rather than filed, pick one of these and actually solve it. These are not "gaps" — they are unsolved problems whose solutions would change the probabilities in the far-tier table above.

1. **Long-video retrieval (>30 min).** R@1 on LongVideoBench-Retrieval is materially below short-video baselines. Temporal reasoning, scene-change detection, efficient video-token compression are all open. Closed APIs (Twelve Labs Marengo-2.7, Pegasus-1.2) are partial answers.
2. **Fine-grained visual grounding at retrieval time.** ColPali localizes within a document; nothing localizes within a photo or a long video frame well. Snappy ([arXiv 2512.02660](https://arxiv.org/abs/2512.02660)) is a patch-to-region propagation partial answer.
3. **Calibration across modalities.** Cosine similarity of T-T, I-I, and T-I pairs is not on the same scale even after training. Thresholds must be per-modality. No paper formalizes this.
4. **Incremental / streaming multi-vector indexing.** MUVERA helps, but "add 10K pages to a 10M collection" is still thousands of inserts per document and requires periodic recentering.
5. **Compositional generalization evaluation.** ARO, MMVP, SPEC are small and somewhat adversarial; a real-world compositional-generalization benchmark would cost more to build than any single lab has funded.
6. **Embodied / spatial retrieval.** No consensus representation for "the scene in front of me, indexed by location and time".
7. **Preference-aligned retrievers.** Contrastive InfoNCE is a weak proxy for user utility. Replacing it with a judge-in-the-loop or RL-from-interaction objective at scale is open.

---

## Part VII — Reference

### 7.1 Paper list — Era 1–2 foundations (2021 – mid-2025)

| arXiv | Paper | Abbr. | Affiliation |
|---|---|---|---|
| [2103.00020](https://arxiv.org/abs/2103.00020) | Learning Transferable Visual Models From Natural Language Supervision | CLIP | ![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white) |
| [2202.06767](https://arxiv.org/abs/2202.06767) | Wukong: A 100 Million Large-scale Chinese Cross-modal Pre-training Benchmark | Wukong | ![Huawei](https://img.shields.io/badge/Huawei-FF0000?style=flat-square) |
| [2205.01917](https://arxiv.org/abs/2205.01917) | CoCa: Contrastive Captioners are Image-Text Foundation Models | CoCa | ![Google Research](https://img.shields.io/badge/Google-4285F4?style=flat-square&logo=google&logoColor=white) |
| [2210.01936](https://arxiv.org/abs/2210.01936) | When and Why Vision-Language Models Behave Like Bags-of-Words, and What to Do About It? | ARO | ![Stanford](https://img.shields.io/badge/Stanford-8C1515?style=flat-square) |
| [2210.08402](https://arxiv.org/abs/2210.08402) | LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models | LAION-5B | ![LAION](https://img.shields.io/badge/LAION-1A1A1A?style=flat-square) |
| [2211.01335](https://arxiv.org/abs/2211.01335) | Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese | Chinese CLIP | ![Alibaba](https://img.shields.io/badge/Alibaba-FF6A00?style=flat-square) |
| [2211.06679](https://arxiv.org/abs/2211.06679) | AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities | AltCLIP | ![BAAI](https://img.shields.io/badge/BAAI-8B5CF6?style=flat-square) |
| [2212.00794](https://arxiv.org/abs/2212.00794) | Scaling Language-Image Pre-training via Masking | FLIP | ![Meta FAIR](https://img.shields.io/badge/Meta%20FAIR-0081FB?style=flat-square&logo=meta&logoColor=white) |
| [2212.07143](https://arxiv.org/abs/2212.07143) | Reproducible Scaling Laws for Contrastive Language-Image Learning (CVPR 2023) | OpenCLIP | ![LAION](https://img.shields.io/badge/LAION-1A1A1A?style=flat-square) |
| [2303.15343](https://arxiv.org/abs/2303.15343) | Sigmoid Loss for Language Image Pre-Training | SigLIP | ![Google](https://img.shields.io/badge/Google-4285F4?style=flat-square) |
| [2303.15389](https://arxiv.org/abs/2303.15389) | EVA-CLIP: Improved Training Techniques for CLIP at Scale | EVA-CLIP | ![BAAI](https://img.shields.io/badge/BAAI-8B5CF6?style=flat-square) |
| [2304.14108](https://arxiv.org/abs/2304.14108) | DataComp: In Search of the Next Generation of Multimodal Datasets | DataComp | ![consortium](https://img.shields.io/badge/consortium-708090?style=flat-square) |
| [2305.19595](https://arxiv.org/abs/2305.19595) | Dense and Aligned Captions (DAC) Promote Compositional Reasoning in VL Models | DAC | ![IBM](https://img.shields.io/badge/IBM-0530AD?style=flat-square&logo=ibm&logoColor=white) |
| [2305.20088](https://arxiv.org/abs/2305.20088) | Improving CLIP Training with Language Rewrites | LaCLIP | ![Google](https://img.shields.io/badge/Google-4285F4?style=flat-square) |
| [2309.17425](https://arxiv.org/abs/2309.17425) | Data Filtering Networks | DFN | ![Apple](https://img.shields.io/badge/Apple-000000?style=flat-square&logo=apple&logoColor=white) |
| [2310.07699](https://arxiv.org/abs/2310.07699) | VeCLIP: Improving CLIP Training via Visual-enriched Captions | VeCLIP | ![Apple](https://img.shields.io/badge/Apple-000000?style=flat-square) |
| [2310.13355](https://arxiv.org/abs/2310.13355) | SILC: Improving Vision Language Pretraining with Self-Distillation | SILC | ![Google](https://img.shields.io/badge/Google-4285F4?style=flat-square) |
| [2311.17136](https://arxiv.org/abs/2311.17136) | UniIR: Training and Benchmarking Universal Multimodal Information Retrievers | UniIR | ![Waterloo](https://img.shields.io/badge/Waterloo-FFD54F?style=flat-square&logoColor=black) |
| [2312.00081](https://arxiv.org/abs/2312.00081) | Synthesize, Diagnose, and Optimize: Towards Fine-Grained Vision-Language Understanding | SPEC | ![Fudan](https://img.shields.io/badge/Fudan-002966?style=flat-square) |
| [2312.08578](https://arxiv.org/abs/2312.08578) | Revisiting the Role of Language Priors in Vision-Language Models (DCI) | DCI | ![Meta FAIR](https://img.shields.io/badge/Meta%20FAIR-0081FB?style=flat-square) |
| [2401.06209](https://arxiv.org/abs/2401.06209) | Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs | MMVP | ![NYU](https://img.shields.io/badge/NYU-57068C?style=flat-square) |
| [2401.09865](https://arxiv.org/abs/2401.09865) | Improving Fine-grained Understanding in Image-Text Pre-training | SPARC | ![DeepMind](https://img.shields.io/badge/DeepMind-4285F4?style=flat-square) |
| [2401.15896](https://arxiv.org/abs/2401.15896) | M2-Encoder: Advancing Bilingual Image-Text Understanding by Large-scale Efficient Pretraining | M2-Encoder | ![Ant Group](https://img.shields.io/badge/Ant%20Group-1677FF?style=flat-square) |
| [2402.03216](https://arxiv.org/abs/2402.03216) | BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation | BGE M3 | ![BAAI](https://img.shields.io/badge/BAAI-8B5CF6?style=flat-square) |
| [2402.04252](https://arxiv.org/abs/2402.04252) | EVA-CLIP-18B: Scaling CLIP to 18 Billion Parameters | EVA-CLIP-18B | ![BAAI](https://img.shields.io/badge/BAAI-8B5CF6?style=flat-square) |
| [2403.15378](https://arxiv.org/abs/2403.15378) | Long-CLIP: Unlocking the Long-Text Capability of CLIP | Long-CLIP | ![Shanghai AI Lab](https://img.shields.io/badge/Shanghai%20AI%20Lab-4A90E2?style=flat-square) |
| [2403.17007](https://arxiv.org/abs/2403.17007) | DreamLIP: Language-Image Pre-training with Long Captions | DreamLIP | ![multi-institution](https://img.shields.io/badge/ZJU%20%2B%20Ant%20%2B%20SJTU%20%2B%20USTC%20%2B%20EIT%20%2B%20NEU-708090?style=flat-square) |
| [2403.19651](https://arxiv.org/abs/2403.19651) | MagicLens: Self-Supervised Image Retrieval with Open-Ended Instructions | MagicLens | ![Google](https://img.shields.io/badge/Google-4285F4?style=flat-square) |
| [2404.04125](https://arxiv.org/abs/2404.04125) | No "Zero-Shot" Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance | — | ![Tübingen](https://img.shields.io/badge/T%C3%BCbingen-8F2D56?style=flat-square) |
| [2405.13777](https://arxiv.org/abs/2405.13777) | No Filter: Cultural and Socioeconomic Diversity in Contrastive Vision-Language Models | — | ![DeepMind](https://img.shields.io/badge/DeepMind-4285F4?style=flat-square) |
| [2405.16915](https://arxiv.org/abs/2405.16915) | Multilingual Diversity Improves Vision-Language Representations | — | ![UW](https://img.shields.io/badge/UW-4B2E83?style=flat-square) |
| [2405.19504](https://arxiv.org/abs/2405.19504) | MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings (NeurIPS 2024) | MUVERA | ![Google Research](https://img.shields.io/badge/Google-4285F4?style=flat-square) |
| [2406.11251](https://arxiv.org/abs/2406.11251) | Unifying Multimodal Retrieval via Document Screenshot Embedding | DSE | ![Waterloo / JHU](https://img.shields.io/badge/Waterloo%20%2F%20JHU-FFD54F?style=flat-square) |
| [2407.01449](https://arxiv.org/abs/2407.01449) | ColPali: Efficient Document Retrieval with Vision Language Models | ColPali | ![illuin-tech](https://img.shields.io/badge/illuin--tech-FF6F00?style=flat-square) |
| [2407.01523](https://arxiv.org/abs/2407.01523) | MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations | MMLongBench-Doc | ![HKU / Alibaba](https://img.shields.io/badge/HKU%20%2F%20Alibaba-002060?style=flat-square) |
| [2407.02883](https://arxiv.org/abs/2407.02883) | CoIR: A Comprehensive Benchmark for Code Information Retrieval Models | CoIR | ![consortium](https://img.shields.io/badge/consortium-708090?style=flat-square) |
| [2407.12580](https://arxiv.org/abs/2407.12580) | E5-V: Universal Embeddings with Multimodal Large Language Models | E5-V | ![BUAA + Microsoft](https://img.shields.io/badge/BUAA%20%2B%20Microsoft-5E5E5E?style=flat-square) |
| [2410.05160](https://arxiv.org/abs/2410.05160) | VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks | VLM2Vec | ![Waterloo + Salesforce](https://img.shields.io/badge/Waterloo%20%2B%20Salesforce-00A1E0?style=flat-square) |
| [2411.01106](https://arxiv.org/abs/2411.01106) | SV-RAG: LoRA-Contextualizing Adaptation of MLLMs for Long Document Understanding | SV-RAG | ![—](https://img.shields.io/badge/%E2%80%94-CCCCCC?style=flat-square) |
| [2411.02571](https://arxiv.org/abs/2411.02571) | MM-Embed: Universal Multimodal Retrieval with Multimodal LLMs | MM-Embed | ![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?style=flat-square&logo=nvidia&logoColor=white) |
| [2412.01720](https://arxiv.org/abs/2412.01720) | LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant (CVPR 2025) | LamRA | ![SJTU + Xiaohongshu](https://img.shields.io/badge/SJTU%20%2B%20Xiaohongshu-FE2C55?style=flat-square) |
| [2412.04378](https://arxiv.org/abs/2412.04378) | VladVA: Discriminative Fine-tuning of LVLMs | VladVA | ![Samsung AI](https://img.shields.io/badge/Samsung%20AI-1428A0?style=flat-square) |
| [2412.08802](https://arxiv.org/abs/2412.08802) | jina-clip-v2: Multilingual Multimodal Embeddings for Text and Images | jina-clip-v2 | ![Jina](https://img.shields.io/badge/Jina-009191?style=flat-square) |
| [2412.14475](https://arxiv.org/abs/2412.14475) | MegaPairs: Massive Data Synthesis for Universal Multimodal Retrieval | MegaPairs | ![BAAI](https://img.shields.io/badge/BAAI-8B5CF6?style=flat-square) |
| [2412.16855](https://arxiv.org/abs/2412.16855) | GME: Improving Universal Multimodal Retrieval by Multimodal LLMs | GME | ![Alibaba](https://img.shields.io/badge/Alibaba-FF6A00?style=flat-square) |
| [2502.08468](https://arxiv.org/abs/2502.08468) | mmE5: Improving Multimodal Multilingual Embeddings via High-quality Synthetic Data | mmE5 | ![RUC + Microsoft](https://img.shields.io/badge/RUC%20%2B%20Microsoft-5E5E5E?style=flat-square) |
| [2502.14786](https://arxiv.org/abs/2502.14786) | SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features | SigLIP 2 | ![Google](https://img.shields.io/badge/Google-4285F4?style=flat-square) |
| [2503.04812](https://arxiv.org/abs/2503.04812) | LLaVE: Large Language and Vision Embedding Models with Hardness-Weighted Contrastive Learning | LLaVE | ![Tencent](https://img.shields.io/badge/Tencent-0052D9?style=flat-square) |
| [2503.07891](https://arxiv.org/abs/2503.07891) | Gemini Embedding: Generalizable Embeddings from Gemini | Gemini Embedding | ![Google](https://img.shields.io/badge/Google-4285F4?style=flat-square) |
| [2503.19900](https://arxiv.org/abs/2503.19900) | CAFe: Unifying Representation and Generation with Contrastive-Autoregressive Finetuning | CAFe | ![Meta](https://img.shields.io/badge/Meta-0081FB?style=flat-square) |
| [2503.19903](https://arxiv.org/abs/2503.19903) | Scaling Vision Pre-Training to 4K Resolution | PS3 | ![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?style=flat-square) |
| [2504.01017](https://arxiv.org/abs/2504.01017) | Scaling Language-Free Visual Representation Learning | LF-Vision | ![Meta FAIR](https://img.shields.io/badge/Meta%20FAIR-0081FB?style=flat-square) |
| [2504.10471](https://arxiv.org/abs/2504.10471) | MIEB: Massive Image Embedding Benchmark (130 tasks, 38 languages) | MIEB | ![MTEB team](https://img.shields.io/badge/MTEB-FFD21E?style=flat-square&logoColor=black) |
| [2504.13181](https://arxiv.org/abs/2504.13181) | Perception Encoder: The Best Visual Embeddings Are Not at the Output of the Network | Perception Encoder | ![Meta FAIR](https://img.shields.io/badge/Meta%20FAIR-0081FB?style=flat-square) |
| [2504.17432](https://arxiv.org/abs/2504.17432) | Breaking the Modality Barrier: Universal Embedding Learning with Multimodal LLMs | UniME | ![Alibaba](https://img.shields.io/badge/Alibaba-FF6A00?style=flat-square) |
| [2505.11293](https://arxiv.org/abs/2505.11293) | B3: Breaking the Batch Barrier for Vision Language Model Contrastive Learning | B3 | ![Duke](https://img.shields.io/badge/Duke-001A57?style=flat-square) |
| [2505.11651](https://arxiv.org/abs/2505.11651) | MIRACL-VISION: A Large, Multilingual, Visual Document Retrieval Benchmark | MIRACL-VISION | ![—](https://img.shields.io/badge/%E2%80%94-CCCCCC?style=flat-square) |
| [2505.19650](https://arxiv.org/abs/2505.19650) | Modality Curation: Building Universal Embeddings for Advanced Multimodal Information Retrieval | Modality Curation | ![NEU + Kuaishou](https://img.shields.io/badge/NEU%20%2B%20Kuaishou-FF4906?style=flat-square) |
| [2506.04997](https://arxiv.org/abs/2506.04997) | Towards Storage-Efficient Visual Document Retrieval: An Empirical Study on Reducing Patch-Level Embeddings | Light-ColPali | ![—](https://img.shields.io/badge/%E2%80%94-CCCCCC?style=flat-square) |
| [2506.05176](https://arxiv.org/abs/2506.05176) | Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models | Qwen3-Embedding | ![Alibaba](https://img.shields.io/badge/Alibaba-FF6A00?style=flat-square) |
| [2506.18902](https://arxiv.org/abs/2506.18902) | jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval | jina-embeddings-v4 | ![Jina](https://img.shields.io/badge/Jina-009191?style=flat-square) |
| [2506.23115](https://arxiv.org/abs/2506.23115) | MoCa: Modality-aware Continual Pre-training Makes Better Bidirectional Multimodal Embeddings | MoCa | ![RUC + Microsoft](https://img.shields.io/badge/RUC%20%2B%20Microsoft-5E5E5E?style=flat-square) |
| [2507.04590](https://arxiv.org/abs/2507.04590) | VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents | VLM2Vec-V2 | ![Waterloo + Salesforce](https://img.shields.io/badge/Waterloo%20%2B%20Salesforce-00A1E0?style=flat-square) |
| [2507.22062](https://arxiv.org/abs/2507.22062) | MetaCLIP 2: A Worldwide Scaling Recipe | Meta CLIP 2 | ![Meta FAIR](https://img.shields.io/badge/Meta%20FAIR-0081FB?style=flat-square) |

### 7.2 Paper list — Era 3: MLLM-native + reasoning-augmented (Aug 2025 – Apr 2026)

| Reference | Paper / Model | Abbr. | Affiliation |
|---|---|---|---|
| [2508.13843](https://arxiv.org/abs/2508.13843) | UniECS: Unified Multimodal E-Commerce Search Framework with Gated Cross-modal Fusion | UniECS | ![Kuaishou](https://img.shields.io/badge/Kuaishou-FF4906?style=flat-square) |
| [2509.23661](https://arxiv.org/abs/2509.23661) | LLaVA-OneVision-1.5: Fully Open Framework for Democratized Multimodal Training | LLaVA-OV-1.5 | ![EvolvingLMMs](https://img.shields.io/badge/EvolvingLMMs-708090?style=flat-square) |
| [2510.05014](https://arxiv.org/abs/2510.05014) | Think-Then-Embed: Generative Context Improves Multimodal Embeddings (ICLR 2026) | TTE | ![Meta + UCF + NYU](https://img.shields.io/badge/Meta%20%2B%20UCF%20%2B%20NYU-0081FB?style=flat-square) |
| [2510.13515](https://arxiv.org/abs/2510.13515) | UniME-V2: MLLM-as-a-Judge for Universal Multimodal Embedding Learning (AAAI 2026) | UniME-V2 | ![Alibaba](https://img.shields.io/badge/Alibaba-FF6A00?style=flat-square) |
| [2510.27350](https://arxiv.org/abs/2510.27350) | RzenEmbed: Towards Comprehensive Multimodal Retrieval | RzenEmbed | ![Qihoo 360](https://img.shields.io/badge/Qihoo%20360-708090?style=flat-square) |
| [2511.21121](https://arxiv.org/abs/2511.21121) | Beyond Patch Aggregation: 3-Pass Pyramid Indexing for Vision-Enhanced Document Retrieval | VisionRAG | ![Academic](https://img.shields.io/badge/Academic-1E3A8A?style=flat-square) |
| [2512.02660](https://arxiv.org/abs/2512.02660) | Spatially-Grounded Document Retrieval via Patch-to-Region Relevance Propagation | Snappy | ![Academic](https://img.shields.io/badge/Academic-1E3A8A?style=flat-square) |
| [2601.04720](https://arxiv.org/abs/2601.04720) | Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking | Qwen3-VL-Embedding | ![Alibaba Qwen](https://img.shields.io/badge/Alibaba%20Qwen-FF6A00?style=flat-square) |
| [2602.03442](https://arxiv.org/abs/2602.03442) | A-RAG: Agentic Retrieval-Augmented Generation with Dynamic Tool Use | A-RAG | ![Academic](https://img.shields.io/badge/Academic-1E3A8A?style=flat-square) |
| [2602.03992](https://arxiv.org/abs/2602.03992) | Nemotron-ColEmbed-V2: Scaling Late-Interaction Multimodal Embedders for Visual Document Retrieval | Nemotron-ColEmbed-V2 | ![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?style=flat-square) |
| [2602.07125](https://arxiv.org/abs/2602.07125) | Reasoning-Augmented Representations for Multimodal Retrieval | RAR | ![Academic](https://img.shields.io/badge/Academic-1E3A8A?style=flat-square) |
| [2603.01471](https://arxiv.org/abs/2603.01471) | CoCoA: Collaborative Attention with EOS-Reconstruction for Bidirectional Multimodal Embedding | CoCoA | ![ICT/CAS + Baidu](https://img.shields.io/badge/ICT%2FCAS%20%2B%20Baidu-1E3A8A?style=flat-square) |
| [2603.05551](https://arxiv.org/abs/2603.05551) | AutoThinkRAG: Adaptive Complexity-Aware Reasoning for Multimodal Retrieval-Augmented Generation | AutoThinkRAG | ![Academic](https://img.shields.io/badge/Academic-1E3A8A?style=flat-square) |
| [2603.14635](https://arxiv.org/abs/2603.14635) | Compute Allocation for Reasoning-Intensive Retrieval: Where Extra Thinking Actually Helps | ComputeAlloc | ![Academic](https://img.shields.io/badge/Academic-1E3A8A?style=flat-square) |
| [HF](https://huggingface.co/nvidia/llama-embed-nemotron-8b) | llama-embed-nemotron-8B | llama-embed-nemotron | ![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?style=flat-square) |
| [HF](https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511) | KaLM-Embedding-Gemma3-12B | KaLM-Gemma3 | ![Tencent](https://img.shields.io/badge/Tencent-0052D9?style=flat-square) |
| [HF](https://huggingface.co/microsoft/harrier-oss-v1-27b) | Harrier-OSS-v1-27B (MTEB v2 = 74.3) | Harrier-OSS | ![Microsoft](https://img.shields.io/badge/Microsoft-5E5E5E?style=flat-square) |
| [Seed](https://seed1-6-embedding.github.io/) | Seed1.6-Embedding (closed API) | Seed1.6-Embedding | ![ByteDance](https://img.shields.io/badge/ByteDance-000000?style=flat-square) |
| [Google](https://blog.google/technology/developers/gemini-embedding-2/) | Gemini Embedding 2 (closed API, T+I+V+Audio+PDF) | Gemini Embedding 2 | ![Google](https://img.shields.io/badge/Google-4285F4?style=flat-square) |
| [HF blog](https://huggingface.co/blog/nvidia/nemotron-colembed-v2) | Nemotron-ColEmbed-V2 (late-interaction) | Nemotron-ColEmbed-V2 | ![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?style=flat-square) |
| [HF](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-7B-Instruct) | GME-Qwen2-VL-7B-Instruct (v2) | GME-v2 | ![Alibaba DAMO](https://img.shields.io/badge/Alibaba%20DAMO-FF6A00?style=flat-square) |
| [ViDoRe V2](https://huggingface.co/spaces/vidore/vidore-leaderboard-v2) | ColQwen2.5-multilingual-v1.0 | ColQwen2.5-mling | ![illuin-tech](https://img.shields.io/badge/illuin--tech-FF6F00?style=flat-square) |
| [ViDoRe V3](https://huggingface.co/spaces/vidore/vidore-leaderboard-v3) | ColQwen3-7B | ColQwen3 | ![illuin-tech](https://img.shields.io/badge/illuin--tech-FF6F00?style=flat-square) |
| [HF](https://huggingface.co/vidore/SauerkrautLM-ColQwen3-2B-v0.1) | SauerkrautLM-ColQwen3-2B | SauerkrautLM-ColQwen3 | ![VAGO](https://img.shields.io/badge/VAGO-7C3AED?style=flat-square) |
| [HF](https://huggingface.co/nomic-ai/nomic-embed-multimodal-7b) | Nomic Embed Multimodal 7B | Nomic-Embed-MM | ![Nomic AI](https://img.shields.io/badge/Nomic-5E2CA5?style=flat-square) |
| [Cohere docs](https://docs.cohere.com/docs/embed-v4) | Cohere Embed v4 | Embed v4 | ![Cohere](https://img.shields.io/badge/Cohere-39594D?style=flat-square) |
| [Voyage docs](https://docs.voyageai.com/docs/multimodal-embeddings) | Voyage Multimodal 3 | Voyage-MM-3 | ![Voyage](https://img.shields.io/badge/Voyage-8B5CF6?style=flat-square) |
| [HF blog](https://huggingface.co/blog/embeddinggemma) | EmbeddingGemma-300M | EmbeddingGemma | ![Google](https://img.shields.io/badge/Google-4285F4?style=flat-square) |
| [HF](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0) | Snowflake Arctic-Embed 2.0 | Arctic-Embed 2.0 | ![Snowflake](https://img.shields.io/badge/Snowflake-29B5E8?style=flat-square) |
| [OpenAI docs](https://platform.openai.com/docs/guides/embeddings) | OpenAI text-embedding-3 | OpenAI v3 | ![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square) |
| [MMEB](https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard) | MMEB-V2 Leaderboard | MMEB-V2 | ![TIGER-Lab](https://img.shields.io/badge/TIGER--Lab-FF8A00?style=flat-square) |
| [MTEB](https://huggingface.co/spaces/mteb/leaderboard) | MTEB v2 Leaderboard | MTEB v2 | ![MTEB](https://img.shields.io/badge/MTEB-FFD21E?style=flat-square) |

### 7.3 Ecosystem matrix — choose by corpus × modality × freshness × axis

**Vector DBs & ANN indexes.** The question that matters is "multi-vector + MUVERA", not headline QPS.

| System | Version (Apr 2026) | Multi-vector / MaxSim | MUVERA / FDE | Quantization | When to pick |
|---|---|---|---|---|---|
| FAISS | 1.11 | Client-side only | No | Full + **RaBitQ** (new in 1.11) | Embedded libraries, research, <50M items |
| Milvus | 2.5 GA | Multi-vector fields; MaxSim via `group_by`; native MaxSim on 3.0 roadmap | 3.0 roadmap | Binary / int8 / PQ / GPU-CAGRA | Cloud-native, >100M items, hybrid search |
| Qdrant | 1.14 | **Native multivector + MaxSim** (pre-dates 1.14) | FastEmbed post-processing | Binary / SQ / PQ / MRL truncate | **Default for ColPali / ColQwen** |
| Weaviate | 1.31 | MaxSim rerank + **native MUVERA** | **Yes** (`Encoding.muvera(...)`) | Binary / SQ / PQ / RQ | Generative-search-heavy apps; post-MUVERA ColPali |
| Vespa | 8.4xx | **Native tensor MaxSim** | Partial (tensor ops) | Full stack | Largest-scale late-interaction (>100M docs) |
| LanceDB | 0.24 | Multi-vector columns + client-side MaxSim | No official page | Binary / int8 / PQ | Video / blob-heavy lakehouse |
| pgvector / pgvectorscale | 0.8 / 0.6 | Client-side MaxSim | No | HNSW / IVFFlat / StreamingDiskANN (0.6 removed `sbq_speedup`) | Postgres-first + >10M vectors |

**Rerankers — use one, always.**

| Reranker | License | Niche |
|---|---|---|
| Cohere Rerank 3.5 | Closed API | Multilingual, long-context, tabular & code |
| Jina Reranker v2-multimodal | Partial open | Cross-modal text ↔ image |
| Voyage rerank-2 / lite | Closed API | BEIR-competitive |
| mxbai-rerank-large-v2 | Apache-2.0 | Strong open-weights generalist |
| BGE reranker v2-m3 / v2-gemma / v2-minicpm-layerwise | Open | Speed/quality knob via layerwise |
| MLLM-as-reranker (GPT-4o/5, Claude 3.5/4, Gemini 2.x, Qwen2.5-VL, InternVL3) | API / local | Top-k visual rerank, RankGPT-style |
| RankLLM | Apache-2.0 | Listwise LLM rerank library |
| **Think-Then-Embed / UniME-V2 as reranker** | Open | Reasoning-aware rerank — the 2026 new entry |

**Multimodal RAG frameworks.**

| Framework | Highlight |
|---|---|
| LlamaIndex 0.12 / 0.13 | `MultiModalVectorStoreIndex`, tightest DB integrations |
| Haystack 2.12 | `MultiModalTextEmbedder`, Cohere / Jina rerankers |
| DSPy 2.6 / 3.0 preview | Multimodal signatures + MIPROv2 / BootstrapFewShot |
| Byaldi 0.0.8+ | ColPali / ColQwen2 / ColSmol wrapper |
| PyLate | Training + inference for late-interaction (MUVERA-ready) |
| RAGatouille | User-friendly ColBERT / PLAID |
| Morphik 1.x | OSS multimodal RAG server; ColPali + VLM routing |
| RAGFlow 0.16+ | Document-centric RAG with OCR/layout + ColPali |
| Pixeltable | Declarative multimodal tables with CLIP / YOLO / VLM UDFs |

**Stack recipes (April 2026).** Each annotated with the axes it pushes.

- **100K product images, millions of queries/day.** (C only) jina-clip-v2 → FAISS `IndexHNSWFlat` → mxbai-rerank-large-v2 → LLM answer. Skip MLLM encoders and ColPali.
- **100K enterprise PDFs, chart/table/slide retrieval.** (L + I) ColQwen2.5-multilingual / Nemotron-ColEmbed-V2 → Qdrant multivector (or Weaviate 1.31 with MUVERA) → Claude / GPT-4o/5 rerank → VLM answer. The killer app, now affordable at scale.
- **100M social-media images, compositional queries.** (C first, I on rerank) SigLIP 2 first stage (MRL + binary) → Qwen3-VL-Embedding-8B rerank on top-1K → VLM answer. Reserve the MLLM encoder for rerank.
- **Heterogeneous instruction-heavy RAG traffic.** (I dominant) Think-Then-Embed or UniME-V2 reasoning rerank on top of Qwen3-VL-Embedding / RzenEmbed first stage. Monitor zero-shot vs instructed drift.
- **Long-form videos (≥1 hr).** (C + L open) Marengo-2.7 / Pegasus-1.2 (Twelve Labs) for now; open weights not yet competitive. Re-evaluate Q3 2026.
- **Mixed audio + PDF + image.** (C, breadth) Gemini Embedding is the credible unified option in April 2026. Budget for API spend.

### 7.4 Methodology & epistemic hygiene

*Primary sources*: arXiv papers (linked), Hugging Face model cards, MMEB / MTEB / ViDoRe / MIEB public leaderboards (snapshotted April 2026), official vendor docs. *Secondary*: author-run benchmarks on an internal H100 SXM node for §IV; single-configuration, ±20% noise. *Tertiary*: vendor blogs, release notes, conference talks — linked, used only where primary references are unavailable.

**Load-bearing vs illustrative.** The C–L–I frame and the §VI forecasts are load-bearing — the thesis depends on them. Specific numeric claims about individual models (MMEB scores, latency, storage) are illustrative — accurate to the snapshot, aging within 6–12 months.

| Type | Examples | Epistemic status |
|---|---|---|
| Load-bearing | C–L–I frame; late interaction = within-item localization; saturation + contamination understate generalization by several points; the forecast bets | Stands behind these; expects reasonable aging |
| Directionally correct | MUVERA turns L-axis into a first-class primitive; test-time compute belongs in rerank; 2–14B open-weights multimodal disproportionately Chinese | Strongly held but context-dependent |
| Illustrative / snapshot | Specific benchmark numbers; exact VRAM; DB version numbers | Ages within 6–12 months; re-verify |
| Speculative | Far-tier bets (F1–F8) | Calibrated opinions, not fact |

**Known gaps.** A guide that hides its gaps is worse than one that names them. As of April 2026, this document does not cover: *adversarial robustness* (typographic attacks, watermark poisoning, OOD shift — relevant for moderation, T&S, content-auth pipelines); *privacy / membership-inference* on contrastive models; *audio-text embedding* in the depth applied to vision; *3D / point-cloud / CAD*; *on-device / edge embedding*; *non-search uses of embeddings* (RL reward models, moderation, dedup, near-dup); *legal / licensing analysis* of training-data provenance. If you build in any of these areas, supplement this guide; do not substitute it.

**Freshness contract.** Embeddings move fast; so do benchmarks. Treat every number older than this edition as stale. Living leaderboards linked in §7.1–7.2 are authoritative; this guide is a frame for reading them, not a replacement.

> [!IMPORTANT]
> **Disclosure.** The author works in this field and has production stakes. Readers should discount any claim that conveniently justifies a deployment choice the author has already committed to. The guide aims for a forecasting track record that can be graded publicly (§VI); that is the cleanest correction mechanism we have.

---

## Acknowledgements

Thanks to the authors of every paper, model, and leaderboard linked here. Particular debt to the MTEB / MMEB / MIEB / ViDoRe benchmark teams for transparent leaderboards; the ColBERT / ColPali lineage for making late interaction legible; the MUVERA authors for making it affordable; the LLM2Vec / E5-V / VLM2Vec / GME / Qwen-VL-Embedding / Seed1.6 teams for the MLLM-as-encoder recipe; and the many practitioners whose deployment post-mortems informed §IV.

Errors and opinions are the author's alone.

---

## Citation

```bibtex
@misc{li2026beyondclip,
  author       = {Wei Li},
  title        = {Beyond CLIP: A First-Principles Field Guide to Multimodal Embedding, from CLIP to the Post-Embedding Agent Stack},
  howpublished = {\url{https://github.com/BIGBALLON/BeyondCLIP}},
  year         = {2026},
  note         = {2nd edition, April 2026}
}
```
