> Historical planning document (7 November 2026). For completed results, current inference fixes and usage, see the project README.

# Named Entity Recognition System — Project Proposal

**Program:** Instant AI & Data Analysis Training\
**Sprint:** 03 — RNN & NLP\
**Project:** 02 — Named Entity Recognition System\
**Date:** 7 November 2026\
**Status:** Scope and initial approach defined. Dataset files, embeddings, and model weights have not been downloaded or audited for this project. No training results are claimed.

## 1. Project overview

The project develops an English sequence-labeling system that locates named entities in text and assigns each detected span an entity type. Unlike whole-text complaint classification, a single input can contain several entities of different types, along with words that are not entities.

For example, in “Alice works at Microsoft in London,” the intended outputs are Alice as a person, Microsoft as an organization, and London as a location. This is an illustrative example rather than a claim about a trained model's predictions.

The study compares four required architectures: LSTM, BiLSTM, BiLSTM + CRF, and a fine-tuned Hugging Face transformer. The best validation performer becomes the default model in a Gradio application that highlights entity spans in the original text with distinct colors.

The intended delivery includes cloud dataset analysis and a small, self-contained Kaggle notebook containing download, preparation, training, evaluation, and inference-export stages. Only the notebook needs to be uploaded to Kaggle; prepared datasets, external source scripts, embedding files, and tokenizers are acquired or generated inside Kaggle.

## 2. Assignment scope

The authoritative source is **INSTANT AI & DATA ANALYSIS TRAINING.pdf**, pages **20–22**, under **Named Entity Recognition System**. Its local location is `C:\Users\user\Documents\Antigravity\Instant_training\INSTANT AI & DATA ANALYSIS TRAINING.pdf`.

| Area | Assignment requirement |
|---|---|
| Dataset | Explore CoNLL-2003 or an equivalent NER dataset |
| Tags | Use an IOB entity-boundary tagging scheme |
| Unseen words | Handle out-of-vocabulary words through character embeddings or subword features |
| Alignment | Align tokenized inputs with the corresponding entity labels |
| Word representations | Use pretrained GloVe or FastText word embeddings |
| Custom architectures | Train LSTM, BiLSTM, and BiLSTM + CRF models |
| Transformer | Fine-tune BERT or DistilBERT using AutoTokenizer, AutoModelForTokenClassification, and Hugging Face Trainer |
| Evaluation | Use seqeval for entity precision, recall, and F1, including PER, ORG, LOC, and MISC |
| Analysis | Compare architectures and explain where CRF improves entity-boundary detection |
| Visualization | Display predicted spans with color-coded entity labels |
| Deployment | Gradio application accepting free text and highlighting entities in real time |

### 2.1 Label-scope discrepancy

The introduction mentions people, organizations, locations, and dates. The detailed evaluation and application requirements instead explicitly name **PER, ORG, LOC, and MISC**, matching the CoNLL-2003 taxonomy. The project follows that explicit taxonomy. A standalone DATE label is outside the initial scope and will not be represented as a learned capability without matching training data.

### 2.2 Required and proposed choices

The four architecture families, pretrained embeddings, OOV handling, seqeval evaluation, Hugging Face Trainer, and Gradio highlighting are assignment requirements. The preferred dataset mirror, character CNN, 50-dimensional GloVe vectors, cased DistilBERT checkpoint, selection rule, runtime strategy, and artifact organization below are implementation choices.

## 3. Dataset and source strategy

### 3.1 Primary dataset

The assignment links to [eriktks/conll2003 on Hugging Face](https://huggingface.co/datasets/eriktks/conll2003). A preferred modern loading candidate is [BramVanroy/conll2003](https://huggingface.co/datasets/BramVanroy/conll2003), a Parquet-format mirror whose card identifies it as a duplicate of the linked dataset. Its exact revision and files remain to be verified during the project audit.

The intended source remains **CoNLL-2003 English**, not an unannounced substitute dataset. The mirror, original source attribution, dataset terms, file hashes, label metadata, and resolved revision will be recorded. A modern data-only loading route avoids relying on execution of an old dataset loading script. Source access failures or unexpected schema changes are reported rather than hidden through silent fallback.

The dataset already provides entity annotations. Manual annotation, AI-generated replacement labels, and pseudo-labeling are not part of the initial scope.

### 3.2 Input and target

- **Input:** the ordered word-token sequence for each sentence.
- **Target:** one IOB tag for each original word.
- **Predictive features:** words, their characters, and transformer subwords.
- **Excluded as predictive input:** gold entity tags, POS tags, chunk tags, identifiers, or any field revealing annotation decisions.

The expected entity types are PER, ORG, LOC, and MISC. With IOB2 notation, the semantic tag inventory is O plus B-/I- variants of those four types. Numeric label IDs will be obtained from dataset metadata rather than assumed from this ordering.

### 3.3 Audit outputs

The dataset audit will include:

- Exact source revision, file checksums, schema, attribution, and split definitions.
- Sentence and token counts for train, validation, and test.
- Token/label length agreement, empty sentences, malformed tags, and invalid label IDs.
- Observed IOB convention and counts of any tag conversions or repairs.
- Entity counts by type, B-/I-/O token distribution, and entity-length distributions.
- Sentence lengths in original words and transformer subwords.
- Repeated sentences within and across splits, including annotation differences.
- Training-vocabulary coverage and pretrained-embedding coverage.
- Unknown-word and unknown-character rates on held-out splits as descriptive audit statistics.
- Representative annotated examples and a small span visualization.

No exact retained counts or hashes are asserted before inspecting the actual files.

## 4. Split policy and leakage controls

The primary comparison preserves the official train, validation, and test boundaries. It does not merge the corpus and produce a new random split.

Official NER corpora can contain repeated short sentences or tokens. Their overlap will be measured and reported rather than automatically applying the complaint project's global deduplication policy, which would silently change the benchmark. Malformed records or necessary exclusions require an explicit ledger identifying the affected split and reason. A separate deduplicated study is optional and will be labeled as a different experiment.

Word and character vocabularies, learned representations, and any training-dependent statistics use training data only. External pretrained GloVe vectors and the pretrained transformer are permitted pretrained resources, with that provenance disclosed. Validation data guides hyperparameters, early stopping, checkpoint choice, and deployment selection. Test data is reserved for the final frozen evaluation phase.

## 5. NER-specific text preparation

The original tokens, sentence boundaries, casing, punctuation, and ordering are preserved. Complaint-classification preprocessing is inappropriate here: removing stop words or punctuation, stemming, or globally lowercasing text can damage label alignment and erase useful entity cues.

The canonical representation stores sentence IDs, original tokens, original and normalized tag strings where conversion is needed, label IDs, and split membership. Each token has exactly one target tag.

IOB1 inputs, if encountered, may be deterministically converted to IOB2 with conversion rules and counts recorded. Malformed annotations are investigated rather than blindly reinterpreted. Evaluation uses the documented normalized convention consistently across all architectures.

### 5.1 Recurrent inputs

The training-only word vocabulary has explicit padding and unknown tokens. A separate lowercase lookup can match uncased GloVe vectors while original token casing remains available in character features; this does not replace the original tokens or displayed text.

Character sequences are derived from the same original words. Their vocabulary is training-only with padding and unknown-character handling. Batch padding is dynamic and excluded from recurrent states, token loss, CRF sequence scoring, and metrics.

### 5.2 Transformer inputs and labels

A fast, cased tokenizer processes the original word lists with `is_split_into_words=True`. Its word-ID alignment maps subwords back to original tokens. Special tokens, padding, and non-first subword labels are assigned the ignore index, conventionally -100, for token-classification loss. The first subword receives the corresponding original word label.

Predictions are reconstructed into one tag per original word before entity evaluation. Token/subword alignment and ignore-index behavior will be tested using split words, punctuation, padding, and multiword entities.

### 5.3 Length and coverage

Maximum word, character, and subword lengths depend on the training audit and pilot memory use. The initial transformer candidate is a 256-subword cap, increased within the checkpoint's supported limit if necessary to preserve full sentences.

The full evaluation population remains shared across all models. Long sequences must not silently lose tail entities. If full sentences cannot fit, a documented word-boundary chunking and reconstruction policy will preserve original word identity and avoid splitting gold entities where feasible. Coverage counts and any unavoidable exclusions are reported, and the application follows the exported length policy.

## 6. Pretrained embeddings and unseen words

The initial recurrent representation combines **50-dimensional pretrained GloVe word vectors** and **learned character-CNN features**.

A compact acquisition candidate is [Gensim-data's glove-wiki-gigaword-50 resource](https://github.com/piskvorky/gensim-data), derived from the pretrained Wikipedia/Gigaword GloVe vectors. Direct streaming of its published vector file is an acceptable alternative to installing the Gensim runtime solely for conversion. The actual resource, checksum, format, dimension, and download path will be validated in the cloud.

Only the vector rows needed by the training word vocabulary are retained in the task embedding matrix. Missing words receive a reproducibly initialized unknown representation plus character features. Padding has a fixed zero representation and remains masked. The embedding layer is initially fine-tunable, consistently across the three recurrent models; changing this is a recorded experimental choice.

Character features provide spelling and morphology signals for unseen names, satisfying the OOV requirement in the custom models. Transformer subword tokenization supplies the corresponding mechanism for DistilBERT.

Random initialization of all word embeddings is not an acceptable silent fallback when a pretrained-vector download fails. Large FastText downloads are unnecessary for this initial route.

## 7. Required architectures

| Model | Initial design | Main comparison |
|---|---|---|
| LSTM | GloVe + character features → unidirectional LSTM → dropout → per-word linear head | Sequential baseline |
| BiLSTM | Same input features → bidirectional LSTM → dropout → per-word linear head | Effect of bidirectional context |
| BiLSTM + CRF | Same BiLSTM encoder and emission head → linear-chain CRF with sequence decoding | Effect of structured decoding |
| DistilBERT | Pretrained cased DistilBERT → token-classification head | Fine-tuned transformer benchmark |

The initial transformer checkpoint is [distilbert/distilbert-base-cased](https://huggingface.co/distilbert/distilbert-base-cased), with its resolved revision pinned during analysis. Fine-tuning uses **AutoTokenizer, AutoModelForTokenClassification, and Trainer**, as explicitly required by the assignment. The encoder receives gradients; pretrained inference alone or a frozen encoder does not replace this experiment.

### 7.1 Fair recurrent comparisons

The recurrent models share embeddings, character encoder design, training vocabulary, split membership, and preprocessing. BiLSTM and BiLSTM + CRF use matching encoders so that the CRF comparison is interpretable. Direction count and resulting parameter differences are reported.

The CRF uses sequence log-likelihood and Viterbi decoding, with numerically stable log-sum-exp computations. Valid masks and real tag IDs are supplied to the CRF; the transformer ignore index is not passed as a CRF tag. Padding cannot contribute to scores. Boundary-transition constraints, if used, are explicitly documented rather than assumed to exist in an unconstrained CRF library.

A compatible tested CRF dependency or a small documented implementation is acceptable. Its loss, decoding, mask handling, and gradients require verification. CRF scoring/decoding can remain float32 while other eligible computations use mixed precision.

### 7.2 Initial training budget

| Setting | Initial candidate, subject to pilot verification |
|---|---|
| Seed | 42 |
| Recurrent depth | One LSTM layer |
| Hidden units | 128 per direction |
| Character representation | 30-dimensional learned character embeddings; small CNN producing approximately 50 features per word |
| Dropout | 0.3 outside the recurrent layer |
| Recurrent optimizer | Adam; initial learning rate 0.001 |
| Transformer optimizer | Trainer-managed AdamW; initial learning rate 0.00002 |
| Recurrent epoch ceiling | 15, early stopping with patience 3 |
| Transformer epoch ceiling | 5, early stopping with patience 2 |
| Initial batch sizes | Recurrent 32; transformer 16 with accumulation if required by memory |
| Best-checkpoint metric | Validation strict entity-level micro F1 |
| Stability | Gradient clipping; supported mixed precision; explicit masks |

These are starting settings rather than measured optimal values. One sensible full configuration per architecture is the initial scope. Data reduction, extensive tuning, and repeated seed sweeps are not default shortcuts or requirements.

Default token classification uses masked cross-entropy for LSTM, BiLSTM, and the transformer, and sequence loss for CRF. O-tag prevalence will be analyzed, but class weights and resampling are not automatically copied from the complaint task. Any imbalance intervention requires validation evidence and clear documentation.

## 8. Evaluation and model selection

The primary metric is **exact entity-span-and-type micro F1**, computed with seqeval in strict IOB2 mode after the agreed tag normalization. Precision and recall use the same entity definition. Per-type precision, recall, F1, and support are reported for PER, ORG, LOC, and MISC. Macro entity F1 and token accuracy are supplementary; a high O-token accuracy is not a sufficient result.

All models are evaluated at original-word/sentence resolution rather than counting each transformer subword as a separate annotation. Sentence boundaries, masks, and padding removal remain correct throughout scoring.

The selected checkpoint maximizes validation entity micro F1. For architectures within 0.001 absolute F1 of the best, the proposed tie rule prefers fewer trainable parameters, then measured validation inference latency, with the rule frozen before test evaluation.

Once configurations and deployment selection are frozen, all four best checkpoints receive a final held-out test evaluation. Test scores do not select the winner or justify retuning within the same claimed held-out experiment.

Reports include learning curves, losses, selected epochs, inference timing methodology, parameter counts, checkpoint sizes, and elapsed training durations. Concurrent runs are described as such; their wall times are not presented as isolated hardware benchmarks.

### 8.1 CRF and error analysis

The comparison investigates exact matches, boundary-only errors, type confusions, missed entities, spurious entities, invalid BIO transitions, and multiword entities. Paired BiLSTM versus BiLSTM + CRF examples identify corrections and regressions rather than assuming CRF must win.

Prediction normalization or repair, if any, is shared and documented across models. Asymmetric cleanup cannot be used to inflate one architecture's score. Additional boundary/type diagnostic measures are labeled separately from the required exact entity metric. Token-level confusion matrices, if supplied, remain supplementary.

## 9. Gradio application and inference

The application accepts ordinary free text and highlights predicted PER, ORG, LOC, and MISC spans with distinct colors. A span-aware tokenizer and stored original character offsets map predictions back to exact substrings, preserving punctuation and whitespace. A deterministic shared span decoder handles entity boundaries; repeated mentions and adjacent entities are tested.

Near-real-time updates can use a short debounce or an explicit Analyze action plus responsive highlighting. Inference latency is measured before describing the app as real time; overlapping callbacks will not create an unbounded inference queue.

The app includes examples, a label legend, an optional entity table, and explicit handling of blank input, no-entity input, unusually long text, punctuation, line breaks, and unsupported text. The primary scope is English and four CoNLL entity types. Nested/overlapping entity labeling, general date extraction, and multilingual NER are not claimed.

Confidence scores are optional, not an assignment requirement. If displayed, their definition and calibration limits are stated; softmax emissions are not presented as CRF sequence probabilities.

The selected model export includes weights, architecture, pretrained-embedding initialization metadata where relevant, trained word/character embedding weights, vocabularies, tokenizer, label mappings, tag convention, offset/span-decoding policy, length settings, dependency requirements, and a clean inference entry point. The app does not need the training dataset, a full GloVe download, or training at startup.

A ready-to-run application ZIP is part of the training notebook's outputs. Publishing a hosted Gradio app is a separate delivery step. Hugging Face Spaces is a proposed host rather than a prescribed assignment requirement.

## 10. Cloud analysis and all-in-one Kaggle workflow

Cloud preparation analysis inspects real sources, validates representations, tests implementation, and produces reference evidence before expensive training. Large assets are acquired in that confirmed cloud environment or in Kaggle, not downloaded to the local machine first.

The notebook repeats the preparation reproducibly inside Kaggle rather than requiring a prepared-data upload. It contains source acquisition, EDA and visualizations, preparation, validation checks, all four training paths, final evaluation, export, and output packaging. External files needed at runtime are fetched from documented pinned sources.

Expected normal Kaggle actions are:

1. Import the notebook.
2. Enable Internet.
3. Select GPU T4 ×2, or a single available GPU.
4. Save Version and Run All.

Reference fingerprints generated by a real cloud audit may be embedded as small metadata. Their expected values are not invented before that audit. Compatible caches are reused only after source, code, dependency, tokenizer, and configuration checks. Mismatches stop with a diagnostic report.

### 10.1 Two-GPU use

Independent processes run at most one architecture on each GPU. A transformer worker starts alongside a recurrent worker, and the next model is scheduled on a free GPU. GPU visibility is set before PyTorch and Trainer initialization; each worker sees one device. Trainer must not accidentally replicate a model onto both GPUs while another worker occupies one.

This strategy preserves each architecture's effective batch size and optimizer meaning; two available GPUs do not imply doubling the learning rate. Validation selection and final reporting wait for all four compatible completed experiments. Single-GPU fallback remains supported.

Logs expose physical assignment, current model, epoch or step progress, and failures. Worker failure stops other active workers and blocks final selection while retaining compatible completed results. The notebook records CUDA memory and pilot throughput without promising a fixed runtime or a twofold speedup.

Project 2's cloud analysis can proceed while Project 1 trains. Access to two T4s does not imply that two separate T4×2 sessions or four GPUs are available simultaneously; training starts when the account's available allocation permits it.

### 10.2 Dependency and restart policy

Project 1 exposed a startup loop caused by checking preloaded version conflicts before installing the required versions, followed by confusion between factory resets and kernel restarts. The new notebook avoids that pattern.

The project dependency set includes the Hugging Face token-classification stack, Trainer/Accelerate, seqeval, and a verified CRF implementation. Versions are selected and checked together rather than blindly copied or downgraded across Kaggle's global environment. Unused vision/audio packages are not allowed to break text-model imports through optional integrations.

Where preloaded notebook libraries prevent a clean import, a project-local isolated runtime for preparation and worker execution is preferable, while preserving access to a compatible CUDA PyTorch build. A narrow kernel-restart step is an alternative if it is implemented and verified clearly. Factory reset is not a required workaround for applying an installation.

The successful path, installed versions, fresh-process imports, and any required restart behavior are documented. An automatic process exit must not sit in a cell that repeatedly runs during Run All.

### 10.3 Completion and interruption behavior

Same-session reruns reuse compatible completed preparation and model artifacts. New Kaggle sessions do not automatically restore `/kaggle/working/`; restoration from saved outputs is explicit.

Completed-model reuse is distinct from resuming interrupted optimization. Unless optimizer, scheduler, scaler, random state, and progress are actually persisted and tested, an interrupted model is described as restarting rather than resuming. Incompatible artifacts are rejected, and changing hyperparameters creates a new experiment identity.

## 11. Validation plan

| Area | Required verification evidence |
|---|---|
| Download | Actual source, embedding, and tokenizer access in the preparation cloud, with hashes/revisions and access requirements |
| Data | Token/tag alignment, splits, IOB validity, counts, vocabularies, embedding coverage, and overlap audit |
| Transformer alignment | Split words, special tokens, ignored subwords, padding, word reconstruction, and long-input coverage |
| Recurrent models | Padding invariance, OOV/character handling, nonzero gradients, masked token losses, and correct output lengths |
| CRF | Small known-path decoding check, finite loss/gradients, mask handling, and padding-safe sequence scoring |
| Metrics | Known exact-span examples distinguishing correct boundaries from wrong type, missing entity, and partial match |
| GPU scheduling | Overlap across devices, serialization on one device, worker failure, cleanup, and single-GPU fallback |
| Integration | Tiny train/validation-only runs for all four architectures, checkpoint reload, and export |
| Application | Exact offsets for punctuation, multiword and repeated entities; blank/no-entity/long inputs; clean CPU reload |
| Notebook | Schema/syntax, every embedded module, correct cell order, and clean-process or native-kernel execution where supported |

Full mode is the delivery default. Smoke mode performs tiny functional runs without claiming model-quality evidence or opening the real held-out test set. Preparation-only verification is a separate clear mode if provided.

If a cloud has no GPU or blocks native kernel execution, CPU functionality and clean-process checks are still useful, but they do not establish actual two-T4 behavior. Verification reports identify what was executed, what was simulated, and what remains for Kaggle. No full-training metrics are fabricated.

## 12. Deliverables and organization

| Location or output | Purpose |
|---|---|
| `docs/Project_Proposal.md` | This descriptive project proposal |
| `docs/Cloud_Work_Prompt.md` | Separate handoff prompt for the cloud implementation task |
| `notebooks/named_entity_recognition_all_in_one.ipynb` | Small self-contained Kaggle notebook |
| `src/` and `configs/`, where retained | Reusable implementation and recorded configuration; all runtime pieces remain embedded in the notebook |
| `reports/` | Dataset/embedding audit, verification evidence, comparison, figures, and error examples |
| Lightweight reports archive | Results review without model weights or source assets |
| Complete experiment archive | Four best checkpoints, configuration, environment record, source fingerprints, and reports |
| Gradio application archive | Selected trained model and standalone inference application |
| `README.md` or notebook instructions | Exact reproduction, rerun, output-retention, and deployment steps |

Large datasets, pretrained assets, caches, and trained artifacts stay in cloud storage where practical. Local project storage initially contains documentation and later lightweight implementation/results. Embedding source files and unneeded optimizer/cache files are excluded from deployment bundles.

## 13. Milestones and acceptance criteria

1. **Source audit:** dataset and embedding sources are genuinely accessible, revisions/hashes are recorded, and the explicit label scope is confirmed.
2. **Representation:** correct IOB handling, original-word alignment, padding, OOV features, embeddings, and span offsets are verified.
3. **Functional pipeline:** all four models train briefly on real training examples and save/reload correctly; tests use no real held-out test predictions.
4. **Training notebook:** one importable notebook acquires all assets, prepares data, and schedules all required models on available Kaggle GPUs.
5. **Comparison:** validation-based checkpoint selection and a frozen final test comparison provide the required entity metrics and CRF analysis.
6. **Inference:** a clean selected-model export reproduces predictions and the Gradio app highlights exact original-text spans.
7. **Delivery evidence:** caches, environment versions, runtime limitations, all verification claims, and saved output archives are reviewable.

The chosen route is sufficiently defined to begin cloud analysis without additional user preferences. Dataset-specific lengths, counts, exact revisions, dependency versions, coverage policies, and pilot batch sizes are audit-dependent settings rather than unknown product goals. Failure of a required public source or a material conflict with the assignment is reported explicitly.

## 14. References

- **INSTANT AI & DATA ANALYSIS TRAINING.pdf**, pages 20–22: assignment requirements.
- [Assignment-linked CoNLL-2003 dataset](https://huggingface.co/datasets/eriktks/conll2003).
- [Proposed Parquet CoNLL-2003 mirror and provenance](https://huggingface.co/datasets/BramVanroy/conll2003).
- [Cased DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-cased).
- [Hugging Face token-classification guide](https://huggingface.co/docs/transformers/tasks/token_classification).
- [Gensim-data pretrained-vector resources](https://github.com/piskvorky/gensim-data).
- [seqeval evaluation implementation and strict-mode documentation](https://github.com/chakki-works/seqeval).

Web source metadata was checked during proposal preparation. Actual data bytes, embedding files, runtime compatibility, and trained performance still require the project audit and execution stages.
