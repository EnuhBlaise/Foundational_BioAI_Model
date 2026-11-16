# Universal constraints in single-cell representation learning

This document records the failure modes that recur across groups building
single-cell foundation models, and states where each one is handled in this
repository. It is the design rationale: most of the unusual choices in the
code exist because of something on this list.

The organizing claim is that the hard problems here are **not architectural**.
Transformer design is largely a solved, transferable technology. What limits
single-cell foundation models is data structure, confounding, and evaluation
— and those are where a repository can actually help, by making the honest
thing the default thing.

---

## 1. The benchmark problem: the baselines are strong and usually win

The most important empirical fact in this field is that classical pipelines
are hard to beat.

- Zero-shot, Geneformer and scGPT embeddings were found to underperform
  simple baselines on cell clustering and batch integration
  (Kedzierska et al., *Genome Biology* 2025).
- For cell-type annotation, L1-regularized logistic regression was
  competitive with scBERT and scGPT, and ablating pretraining from scBERT
  barely changed its performance (Boiarsky et al., *Nature Machine
  Intelligence*).
- For perturbation prediction, five foundation models and two other deep
  models all failed to beat deliberately simple linear baselines
  (Ahlmann-Eltze, Huber & Anders, *Nature Methods* 2025).

The implication for a research repository is structural, not rhetorical: a
model number is meaningless without baseline numbers computed on the same
split with the same probe. So in this repo **baselines are not optional**.
`src/runners/evaluate.py` runs PCA, scaled PCA, raw HVG and a random
projection unconditionally, and the config cannot switch them off. The report
prints a `VERDICT` line that states plainly whether the model beat its best
control, and refuses to call a win a win when the confidence intervals
overlap.

This is deliberately uncomfortable. A harness that can only produce good news
is not a research instrument.

> **Where:** `src/eval/baselines.py`, `src/runners/evaluate.py`,
> `src/eval/report.py::verdict`

---

## 2. Splitting: random splits leak, and the leak is large

Cells are not independent samples. Cells from one donor, one 10x lane, one
dissociation share technical and genetic structure. A random cell-level split
therefore places near-replicates of every test cell into training, and a
model can score well by recognizing the donor rather than the biology.

This is the single easiest way to publish an inflated number, and it is
common because it is the default behavior of every generic ML splitter.

This repo makes the honest split the default and the dishonest one loud:

- `DonorHeldOutSplitter` is the default; no donor spans two splits.
- `StudyHeldOutSplitter` is the harder regime — generalization to an unseen
  dataset and protocol.
- The splitter's seed chooses *which* groups are held out, so repeating over
  seeds resamples held-out donors (the only way to see donor-level variance).
- `RandomSplitter` exists only as a **leakage control**, logs a warning, and
  the report labels the split regime so a random-split number can never be
  mistaken for a held-out one.
- `audit_leakage()` runs unconditionally before training and **raises** on
  donor overlap. Overriding it requires explicitly setting
  `data.fail_on_leakage=false`.
- A splitter refuses to run when its grouping column is constant, rather than
  silently degenerating into a random split with a rigorous-sounding name.

`configs/experiment/leakage_ablation.yaml` measures the size of the leak
directly: same model, same data, random split. The gap against the
donor-held-out run is the inflation a random split would have bought.

> **Where:** `src/data/splits.py`, `configs/experiment/leakage_ablation.yaml`

---

## 3. Fit-on-train discipline

Related but distinct from splitting: every quantity estimated from data must
be estimated on the training split only. In single-cell pipelines the usual
offenders are

- HVG selection computed over the pooled dataset,
- the scaler in a `scale -> PCA` pipeline,
- the corpus-wide gene medians a rank tokenizer needs,
- label encoders built from all observed classes.

Each leaks test information into the representation. The `SingleCellDataModule`
splits **first**, then fits HVGs, the gene vocabulary, the tokenizer and the
label maps on train alone. `RankTokenizer.fit` raises if called twice, so a
tokenizer cannot be quietly re-fitted across a split boundary. Labels unseen
in training map to `-100` and are ignored by the loss rather than shifting
class indices.

> **Where:** `src/data/datamodule.py::setup`, `src/data/tokenizers/rank.py`

---

## 4. Batch effects are the dominant axis of variation

In most real datasets, donor and protocol explain more variance than cell
type. Three consequences:

**HVG selection is confounded.** Genes selected over a pooled multi-study
dataset are preferentially batch-driven, and that choice then propagates into
everything downstream. `select_hvg` ranks genes *within* each batch and
averages ranks. It scores genes by the variance of analytic Pearson residuals
(Lause, Berens & Kobak 2021) rather than variance/mean, because under
negative-binomial noise variance/mean rises with the mean and mostly returns
the most abundant genes. The old scoring is kept as `flavor="dispersion"` for
comparison; `Notebooks/02` measures both. On the synthetic data the
difference is real but modest and disappears once `n_top` is generous.

**`log1p` re-introduces depth.** Depth scaling removes most depth signal from
a PCA embedding; the log transform that follows can bring much of it back,
because a shallow cell keeps more zeros and its non-zero values are scaled
up. `Notebooks/02` shows this directly, and it is the reason
`library_size_r2` is always in the evaluation panel.

**Composition differs by batch.** Donors do not contribute cell types in
equal proportion. Any batch metric computed without conditioning on cell type
rewards an embedding that merges biology. `batch_asw` is therefore computed
within each cell type.

**Over-correction is as bad as under-correction.** Removing all
batch-predictive signal removes real biology when composition is confounded
with batch. The adversarial head exists but is **disabled by default**, with
the trade documented at its definition. Luecken et al. observed the same
dynamic at the pipeline level: scaling pushes integration methods to
prioritize batch removal over biological conservation.

> **Where:** `src/data/preprocessing.py::select_hvg`,
> `src/eval/metrics.py::batch_asw`, `src/models/heads/batch_adversarial.py`

---

## 5. Metrics are individually gameable; report both families

Biological-conservation metrics and batch-correction metrics are each
trivially maximizable at the other's expense:

- Gaussian noise scores near-perfectly on every batch-mixing metric.
- An embedding that encodes batch identity can score well on clustering
  whenever composition differs across batches.

The repo's test suite asserts exactly this (`test_metrics.py::
test_noise_scores_well_on_batch_and_badly_on_biology`), and
`evaluate_embedding` always returns both families plus a stated 60/40
aggregate — stated, because any single number is a weighting choice rather
than a fact. The `random_projection` baseline is in the mandatory set for the
same reason: any metric on which it scores well is not measuring
representation quality.

One diagnostic here is not in the standard panel. `library_size_r2` regresses
log sequencing depth on the embedding. An embedding can look acceptable on
clustering while substantially encoding depth; nothing in the usual scIB set
surfaces that directly.

> **Where:** `src/eval/metrics.py`, `tests/test_metrics.py`

---

## 6. Count statistics: model the noise you actually have

UMI counts are close to negative-binomial with mean-dependent
overdispersion. The widespread belief in pervasive technical "dropout" is
largely wrong for droplet UMI protocols: the observed zeros are consistent
with count sampling (Svensson 2020; Townes et al. 2019), and where zero
inflation does occur, its origins are mostly biological (Choi et al. 2020).

Practical consequences encoded here:

- `ZINBDecoderHead` defaults to **plain NB** (`zero_inflated=False`).
- The synthetic generator's `extra_dropout` defaults to `0.0`; the realistic
  sparsity comes from NB sampling and a heavy-tailed expression distribution,
  not from an imposed dropout process.
- `is_raw_counts()` guards against the silent, serious error of fitting a
  count likelihood to log-normalized values — a frequent mistake because many
  public `.h5ad` files store normalized values in `.X` and counts in a layer.
  `H5ADSource` refuses to load rather than proceed.

> **Where:** `src/models/heads/zinb.py`, `src/data/preprocessing.py`,
> `src/data/sources/h5ad_source.py`

---

## 7. Tokenization is an open question, not a settled one

Genes are an unordered set with no natural sequence, so "tokenizing" a cell
means choosing what to discard:

| scheme | discards | keeps |
|---|---|---|
| rank (Geneformer-style) | magnitude entirely | ordering; robust to depth and protocol |
| binned (scGPT-style) | within-bin resolution, absolute level | per-cell relative level; invariant to rescaling, **not** to real depth loss |
| continuous | nothing | exact value; no discrete vocabulary to predict |

All three are implemented and swappable by config, and `continuous` exists
specifically as the control: a claimed benefit of discretization has to be
demonstrated against simply feeding the value. `test_binned_is_invariant_to_
depth_scaling` pins invariance to *rescaling*. Thinning real counts is a
different matter: it changes which genes are detected and how many are tied,
so bins and ranks shift (`Notebooks/04`, section 4). Tied values share a bin,
which makes bin occupancy uneven in shallow data; masked-bin accuracy must be
read against the majority-bin floor, not `1/n_bins`.

Because genes are a set, the backbone uses **no positional encoding**, and
`test_embedding_is_permutation_invariant` enforces it. This is not a detail:
it is what lets one checkpoint accept a different gene panel at inference.

> **Where:** `src/data/tokenizers/`, `src/models/backbones/set_transformer.py`

---

## 8. Scale limits that bite in practice

- **Attention is O(L²) in genes per cell.** This is the real reason
  single-cell transformers truncate to 1–2k genes instead of modelling the
  full ~20k transcriptome. The truncation is itself a modelling decision,
  since it keeps the highly expressed genes and drops the long tail where
  much cell-type-specific signal lives.
- **Gene-space harmonization across datasets.** Token ids must mean the same
  gene everywhere. `GeneVocabulary` is frozen at construction and saved with
  the checkpoint; `harmonize_genes` reports coverage and warns below 50%,
  because embeddings from a poorly covered panel are not comparable to the
  pretraining corpus.
- **Corpus duplication.** CELLxGENE Census contains overlapping re-analyses
  of the same studies. Omitting `is_primary_data == True` silently duplicates
  cells across splits — a leak that looks like nothing at all. The config
  carries the filter with a comment explaining why.

> **Where:** `src/data/preprocessing.py::GeneVocabulary`,
> `configs/data/census.yaml`

---

## 9. Uncertainty is donor-level, not cell-level

With hundreds of thousands of cells, cell-level confidence intervals are
tiny and meaningless: cells within a donor are highly correlated, so the
effective sample size is closer to the number of held-out **donors**, which
is often fewer than ten.

`bootstrap_ci` resamples cells and its docstring says plainly that this
understates true uncertainty. The report counts held-out groups and prints an
explicit warning below three. A ranking built on two held-out donors is
provisional and the output says so rather than leaving the reader to work it
out.

> **Where:** `src/eval/probes.py::bootstrap_ci`, `src/eval/report.py`

---

## 10. Simulation is for debugging, never for claims

Synthetic data is genuinely useful for one thing: validating that a metric
measures what it claims, because ground truth is known and effects can be
switched off one at a time. Generate with `batch_effect=0` and a batch metric
must read near zero; if it does not, the metric is measuring something else.

It is useless for claims about biology. Every synthetic `AnnData` is stamped
`uns['synthetic'] = True`, the flag propagates into `ComparisonReport`, and
the printed report carries a banner that cannot be suppressed from config.

> **Where:** `src/data/synthetic/`, `src/eval/report.py`

---

## Summary table

| # | Constraint | Default behavior here |
|---|---|---|
| 1 | Baselines usually win | mandatory controls + explicit verdict |
| 2 | Random splits leak | donor-held-out default; audit raises |
| 3 | Fit-on-train discipline | split before any fitting; refuse re-fit |
| 4 | Batch dominates variance | batch-aware HVG; within-type batch ASW |
| 5 | Metrics gameable alone | both families always; random-projection control |
| 6 | NB, not ZINB, for UMI | plain NB default; counts guard |
| 7 | Tokenization unsettled | three schemes; continuous as control |
| 8 | O(L²) and gene-space drift | frozen vocab; coverage warnings |
| 9 | Donor-level uncertainty | group counts + explicit CI caveat |
| 10 | Simulation ≠ evidence | unsuppressable synthetic banner |

See `docs/BIBLIOGRAPHY.md` for sources.
