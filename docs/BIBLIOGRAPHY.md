# Bibliography

Sources for the ideas implemented in this repository. Each entry says what
was taken from it and where that lands in the code. Citation details were
verified against publisher records; where a work exists as both preprint and
journal article, both are given because the titles differ.

---

## Single-cell foundation models (the designs this repo generalizes)

**Theodoris, C. V., Xiao, L., Chopra, A., Chaffin, M. D., Al Sayed, Z. R.,
Hill, M. C., Mantineo, H., Brydon, E., Zeng, Z., Liu, X. S., & Ellinor, P. T.
(2023). Transfer learning enables predictions in network biology.**
*Nature*, 618(7965), 616–624. doi:10.1038/s41586-023-06139-9

> Geneformer. Source of the **rank-value encoding**: genes are ordered by
> expression normalized against corpus-wide medians, and only the ordering is
> kept, which deprioritizes ubiquitous housekeeping genes and confers
> robustness to depth and protocol at the cost of discarding magnitude. Also
> the ~15% masked-token pretraining ratio.
>
> → `src/data/tokenizers/rank.py`, `mask_ratio` default in `configs/data/`

**Cui, H., Wang, C., Maan, H., Pang, K., Luo, F., Duan, N., & Wang, B.
(2024). scGPT: toward building a foundation model for single-cell multi-omics
using generative AI.** *Nature Methods*, 21, 1470–1480.
doi:10.1038/s41592-024-02201-0

> Source of the **value-binning** scheme (expression discretized into
> per-cell bins) and the gene-token + value-token additive embedding used in
> the default stem.
>
> → `src/data/tokenizers/binned.py`, `src/models/stems/expression_stem.py`

**Yang, F., Wang, W., Wang, F., Fang, Y., Tang, D., Huang, J., Lu, H., &
Yao, J. (2022). scBERT as a large-scale pretrained deep language model for
cell type annotation of single-cell RNA-seq data.** *Nature Machine
Intelligence*, 4, 852–866. doi:10.1038/s42256-022-00534-z

> Earlier BERT-style treatment of expression; the binning lineage and the
> masked-expression objective.

**Hao, M., Gong, J., Zeng, X., Liu, C., Guo, Y., Cheng, X., Wang, T., Ma, J.,
Zhang, X., & Song, L. (2024). Large-scale foundation model on single-cell
transcriptomics.** *Nature Methods*, 21, 1481–1491.
doi:10.1038/s41592-024-02305-7

> scFoundation; read-depth-aware pretraining, one of the models included in
> the perturbation benchmark below.

---

## Critical evaluations (why this repo is built the way it is)

These are the most load-bearing references here. They are the reason
baselines are mandatory rather than optional.

**Kedzierska, K. Z., Crawford, L., Amini, A. P., & Lu, A. X. (2025).
Zero-shot evaluation reveals limitations of single-cell foundation models.**
*Genome Biology*, 26, 101. doi:10.1186/s13059-025-03574-x
*(preprint: "Assessing the limits of zero-shot foundation models in
single-cell biology", bioRxiv 2023.10.16.561085)*

> Evaluated Geneformer and scGPT **without fine-tuning** on cell clustering
> and batch integration, and found they underperformed simpler baselines.
> Argues zero-shot evaluation is the setting that actually tests whether
> pretraining produced transferable biological understanding, since much
> single-cell work is exploratory and lacks labels to fine-tune on.
>
> → the mandatory-baselines design, `src/runners/evaluate.py`

**Boiarsky, R., Singh, N. M., Buendia, A., Getz, G., & Sontag, D. Deeper
evaluation of a single-cell foundation model.** *Nature Machine
Intelligence*. *(preprint: "A Deep Dive into Single-Cell RNA Sequencing
Foundation Models", bioRxiv 2023.10.19.563100)*

> L1-regularized logistic regression was competitive with scBERT and scGPT on
> cell-type annotation including few-shot; ablating pretraining from scBERT
> barely changed results, while for scGPT it mattered. Results were
> dataset-dependent, which is itself the point: single-dataset claims do not
> generalize.
>
> → `src/eval/probes.py` (logistic-regression probe as the primary metric)

**Ahlmann-Eltze, C., Huber, W., & Anders, S. (2025). Deep-learning-based gene
perturbation effect prediction does not yet outperform simple linear
baselines.** *Nature Methods*, 22(8), 1657–1661.
doi:10.1038/s41592-025-02772-6

> Five foundation models and two other deep models compared against
> deliberately simple baselines for single and double perturbation
> prediction; none outperformed the baselines.
>
> → `src/eval/report.py::verdict`, which states a loss plainly

**Miller, J. et al. Deep learning-based genetic perturbation models do
outperform uninformative baselines on well-calibrated metrics.** bioRxiv.

> The counterpoint, included deliberately. Argues that part of the
> baseline-beats-model result is a **metric artifact**: naive metrics reward
> predicting the mean shift from control, so the choice of metric partly
> determines the conclusion. The honest reading of this literature is that
> evaluation design is contested, not that deep models are known to be
> useless.
>
> → why `ComparisonReport` reports a panel rather than one headline number

---

## Count statistics and normalization

**Svensson, V. (2020). Droplet scRNA-seq is not zero-inflated.**
*Nature Biotechnology*, 38(2), 147–150. doi:10.1038/s41587-019-0379-5

> Using negative-control droplet data (uniform RNA solution, no biological
> variation expected), showed the observed zero counts are consistent with
> ordinary count statistics — so a zero-inflation component is not needed for
> UMI droplet data.
>
> → `ZINBDecoderHead(zero_inflated=False)` default; `extra_dropout=0.0` in
> the simulator

**Townes, F. W., Hicks, S. C., Aryee, M. J., & Irizarry, R. A. (2019).
Feature selection and dimension reduction for single-cell RNA-Seq based on a
multinomial model.** *Genome Biology*, 20, 295. doi:10.1186/s13059-019-1861-6

> Multinomial / Poisson-NB treatment of UMI counts; independent support for
> the no-zero-inflation conclusion, and the argument against log-normalizing
> before dimension reduction.

**Choi, K., Chen, Y., Skelly, D. A., & Churchill, G. A. (2020). Bayesian
model selection reveals biological origins of zero inflation in single-cell
transcriptomics.** *Genome Biology*, 21, 183. doi:10.1186/s13059-020-02103-2

> The nuance: zero inflation does exist for some genes, but its causes are
> mostly **biological** rather than technical, and plain NB remains the
> recommended reference model. This is why the ZINB head is available but off
> by default rather than absent.

**Hafemeister, C., & Satija, R. (2019). Normalization and variance
stabilization of single-cell RNA-seq data using regularized negative binomial
regression.** *Genome Biology*, 20, 296. doi:10.1186/s13059-019-1874-5

> sctransform. Source of the mean-dependent overdispersion relation the
> simulator uses (`phi ≈ a + b/mu`) and of variance-stabilizing HVG ranking.
>
> → `src/data/synthetic/generator.py`, `src/data/preprocessing.py::select_hvg`

**Lopez, R., Regier, J., Cole, M. B., Jordan, M. I., & Yosef, N. (2018).
Deep generative modeling for single-cell transcriptomics.** *Nature Methods*,
15, 1053–1058. doi:10.1038/s41592-018-0229-2

> scVI. The count-likelihood decoder formulation (library-size-scaled
> composition, per-gene dispersion, optional zero-inflation) reimplemented in
> the ZINB head.
>
> → `src/models/heads/zinb.py`

---

## Benchmarking, integration and evaluation metrics

**Luecken, M. D., Büttner, M., Chaichoompu, K., Danese, A., Interlandi, M.,
Mueller, M. F., Strobl, D. C., Zappia, L., Dugas, M., Colomé-Tatché, M., &
Theis, F. J. (2022). Benchmarking atlas-level data integration in single-cell
genomics.** *Nature Methods*, 19(1), 41–50. doi:10.1038/s41592-021-01336-8

> scIB. The source of this repo's metric panel — ARI, NMI, cell-type ASW,
> batch ASW, graph iLISI, graph connectivity, kBET, principal-component
> regression — and of the split into biological-conservation vs
> batch-correction families. Also the finding that **scaling pushes methods
> to prioritize batch removal over biological conservation**, which is the
> documented trade in the adversarial head.
>
> → `src/eval/metrics.py` (reimplemented directly rather than importing scib,
> to keep every definition inspectable)

**Büttner, M., Miao, Z., Wolf, F. A., Teichmann, S. A., & Theis, F. J.
(2019). A test metric for assessing single-cell RNA-seq batch correction.**
*Nature Methods*, 16, 43–49. doi:10.1038/s41592-018-0254-1

> kBET: chi-square test of local batch composition against global
> composition, reported as an acceptance rate.
>
> → `src/eval/metrics.py::kbet_acceptance`

**Korsunsky, I., Millard, N., Fan, J., Slowikowski, K., Zhang, F., Wei, K.,
Baglaenko, Y., Brenner, M., Loh, P.-R., & Raychaudhuri, S. (2019). Fast,
sensitive and accurate integration of single-cell data with Harmony.**
*Nature Methods*, 16, 1289–1296. doi:10.1038/s41592-019-0619-0

> LISI (local inverse Simpson's index), the basis of the neighbourhood-mixing
> metric here.
>
> → `src/eval/metrics.py::knn_mixing`

**Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis
in machine-learning-based science.** *Patterns*, 4(9), 100804.
doi:10.1016/j.patterns.2023.100804

> The general taxonomy of data leakage across scientific ML. The
> group-structure case is precisely the donor/batch problem here.
>
> → `src/data/splits.py::audit_leakage`

---

## Data resources

**CZI Cell Science Program et al. (2025). CZ CELLxGENE Discover: a single-cell
data platform for scalable exploration, analysis and modeling of aggregated
data.** *Nucleic Acids Research*, 53(D1), D886–D900. doi:10.1093/nar/gkae1142

> CELLxGENE Discover and Census. The default real-data source: harmonized
> ontology terms for `donor_id`, `assay`, `tissue` and `cell_type` across
> tens of millions of cells — the metadata the honest splitters require and
> that most individual datasets lack. Note the `is_primary_data` flag, which
> exists because Census contains overlapping re-analyses.
>
> → `src/data/sources/census_source.py`, `configs/data/census.yaml`

**The Tabula Sapiens Consortium (2022). The Tabula Sapiens: A multiple-organ,
single-cell transcriptomic atlas of humans.** *Science*, 376(6594), eabl4896.
doi:10.1126/science.abl4896

> Multi-donor, multi-organ atlas with the donor structure needed for
> donor-held-out evaluation; a good second corpus via the `h5ad` source.

**Regev, A. et al. (2017). The Human Cell Atlas.** *eLife*, 6, e27041.
doi:10.7554/eLife.27041

> The program whose data model the obs schema here follows.

---

## Technical artifacts the simulator reproduces

**Young, M. D., & Behjati, S. (2020). SoupX removes ambient RNA
contamination from droplet-based single-cell RNA sequencing data.**
*GigaScience*, 9(12), giaa151. doi:10.1093/gigascience/giaa151

**Fleming, S. J., Chaffin, M. D., Arduini, A., Akkad, A.-D., Banks, E.,
Marioni, J. C., Philippakis, A. A., Ellinor, P. T., & Babadi, M. (2023).
Unsupervised removal of systematic background noise from droplet-based
single-cell experiments using CellBender.** *Nature Methods*, 20, 1323–1335.
doi:10.1038/s41592-023-01943-7

> Ambient RNA ("soup"): a fraction of each droplet's counts comes from a
> pooled background profile rather than the cell.
>
> → `ambient_fraction` in `src/data/synthetic/generator.py`

**McGinnis, C. S., Murrow, L. M., & Gartner, Z. J. (2019). DoubletFinder:
Doublet detection in single-cell RNA sequencing data using artificial nearest
neighbors.** *Cell Systems*, 8(4), 329–337.
doi:10.1016/j.cels.2019.03.003

> Doublet rate and the artificial-doublet construction (summing two cells'
> counts) used by the simulator.
>
> → `doublet_rate` in `src/data/synthetic/generator.py`

---

## Architecture and training

**Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need.**
*NeurIPS 30*.

**Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R., &
Smola, A. (2017). Deep Sets.** *NeurIPS 30*.

**Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
Set Transformer: A framework for attention-based permutation-invariant neural
networks.** *ICML 36*, 3744–3753.

> Together: the justification for treating a cell as an unordered **set** of
> genes and omitting positional encodings, which is what makes the backbone
> accept a different gene panel at inference.
>
> → `src/models/backbones/set_transformer.py`,
> `tests/test_model.py::test_embedding_is_permutation_invariant`

**Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT:
Pre-training of deep bidirectional transformers for language understanding.**
*NAACL-HLT*, 4171–4186.

> The masked-token objective adapted as masked-expression modelling.
>
> → `src/models/heads/masked_value.py`

**Ganin, Y., & Lempitsky, V. (2015). Unsupervised domain adaptation by
backpropagation.** *ICML 32*, 1180–1189.

> Gradient reversal, used to make batch identity undecodable from the cell
> embedding.
>
> → `src/models/heads/batch_adversarial.py`

**Loshchilov, I., & Hutter, I. (2019). Decoupled weight decay
regularization.** *ICLR*.

> AdamW, with cosine decay and linear warmup in the training loop.
>
> → `src/runners/pretrain.py`

---

## A note on reading this list

Sections 1–2 of `docs/CONSTRAINTS.md` lean on the critical-evaluation
references, and those papers disagree with the model papers in the first
section. That disagreement is live and unresolved — the Miller et al. entry
is included precisely so the repository does not present one side of a
contested methodological question as settled. The design response taken here
is not to assume either conclusion but to make the comparison cheap,
automatic and hard to avoid reporting.
