> Historical planning document (7 November 2026). For completed results, current inference fixes and usage, see the project README.

# Cloud implementation handoff — Named Entity Recognition System

Use the attached `Project_Proposal.md` as the specification for Project 2 of Sprint 03: Named Entity Recognition System. Complete the cloud data analysis and implementation through delivery of one self-contained Kaggle notebook. The proposal describes the project; this message requests execution of that work.

## Execution environment and final workflow

First verify where your execution tools run. My local storage and internet bandwidth are limited. Large datasets, embeddings, pretrained weights, and model execution belong in your confirmed cloud sandbox or Kaggle, not on my computer. If your execution is connected only to my local machine, explain that limitation before downloading large assets or running models locally.

The final notebook should need only four normal user actions: import it into Kaggle, enable Internet, select GPU T4×2, and Save & Run All. Kaggle itself downloads the dataset, pretrained embeddings, tokenizer, and model weights and repeats all preparation. No prepared-data ZIP, source scripts, project folder, tokenizer files, or manual dataset upload should be required. Embed all runtime implementation and small reference metadata in the notebook, but not large data or model weights.

## Data and representations

Inspect real CoNLL-2003 English data in the cloud. The assignment links to https://huggingface.co/datasets/eriktks/conll2003. Prefer a modern data-only loading route, such as the Parquet mirror https://huggingface.co/datasets/BramVanroy/conll2003, after verifying its provenance, schema, split identity, tag metadata, and resolved revision. Do not silently substitute another dataset or execute an obsolete loading script to force compatibility.

Preserve official train/validation/test boundaries. Audit token/tag lengths, empty/malformed samples, IOB conventions, entity counts, O-tag prevalence, sentence/subword lengths, embedding coverage, OOV rates, and repeated sentences within/across splits. Report overlap without silently applying Project 1's global deduplication to this official benchmark. Learn task word/character vocabularies only from training data. Do not use POS/chunk labels as predictive features.

The explicit target types are PER, ORG, LOC, and MISC. The proposal explains the introduction's conflicting mention of dates; do not invent a DATE class. Inspect numeric IDs from metadata. Normalize an observed IOB1 convention to IOB2 only through documented rules and recorded conversions.

Preserve casing, punctuation, sentence boundaries, words, and alignment. Do not copy complaint-style stop-word removal, stemming, or global lowercasing. Use pretrained GloVe word vectors plus learned character features for custom-model OOV handling. Start with the compact GloVe-50 route in the proposal, verify the actual vector source and checksum, and retain only necessary rows in the task embedding matrix. Do not silently replace failed pretrained embeddings with an entirely random word embedding layer.

For cased DistilBERT, align subwords to original words, ignore special/padded/non-first subword targets, and reconstruct one prediction per original word. Audit length limits; preserve full evaluation coverage and original sentence identity. Do not silently truncate away test entities. Use original character offsets for app highlighting.

## Models and execution

Implement and genuinely train the required four architectures:

1. LSTM.
2. BiLSTM.
3. BiLSTM + linear-chain CRF.
4. Fine-tuned `distilbert/distilbert-base-cased`, using AutoTokenizer, AutoModelForTokenClassification, and Hugging Face Trainer.

Use the proposal's initial budget, with audit/pilot adjustments documented. Match the BiLSTM encoder in the CRF comparison. Verify CRF negative log-likelihood, Viterbi decoding, masks, padding-safe valid tag IDs, finite float32 sequence computations, and actual gradients. Do not feed -100 into CRF tag indexing. Do not claim a plain independent token classifier is a CRF.

Use dynamic padding, masked losses, best validation checkpoints, early stopping, clipping, and supported mixed precision. Avoid large searches, silent dataset reduction, or unnecessary model families.

Use two independent GPU-isolated workers: one transformer and one recurrent model at a time, with remaining architectures queued for the next free device. Set GPU visibility before importing PyTorch or initializing Trainer. Trainer must see only its worker's one GPU. Keep effective batch sizes and learning rates meaningful rather than automatically scaling them by GPU count. Provide single-GPU fallback, useful logs/progress, CPU thread limits, failure cancellation, and completed-model reuse. Do not claim optimizer-state resume unless it is implemented and tested.

## Dependencies and first-cell reliability

Project 1 repeatedly hit startup errors because a guard demanded a restart before installing missing versions, and factory resets undid installations. Avoid recreating that loop. Validate one compatible package set, including Trainer/Accelerate, seqeval, CRF, and the embedding loader. Keep Kaggle's working CUDA PyTorch where compatible.

Prefer an isolated project runtime for preparation/training workers if notebook-preloaded packages conflict with requirements. If a kernel restart is unavoidable, installation must happen first and the exact one-time action must be documented and checked. Never leave a recurring process-kill/restart cell in the normal Run All path. Do not blindly downgrade the entire shared Kaggle environment or assume a factory reset is a kernel restart.

## Evaluation and application exports

Select models on validation strict IOB2 exact entity-level micro F1 with seqeval and the proposal's predeclared tie rule. Freeze the winner before final test evaluation of all four checkpoints. Report entity precision/recall/F1 and support for PER, ORG, LOC, MISC, plus relevant learning curves, runtime, memory, parameter/checkpoint sizes, and diagnostic errors. Token accuracy alone is insufficient.

Analyze BiLSTM versus BiLSTM + CRF boundary corrections and regressions honestly. Do not assume CRF wins or apply asymmetric label repair to inflate results. Use identical documented scoring and span-decoding conventions across models.

Export the selected model with exact preprocessing, vocabularies, character encoder, tokenizer where applicable, labels, length policy, and original-text span reconstruction. Build a standalone Gradio app with colored entity highlights, a legend and examples, and robust blank/no-entity/long-input handling. A deployment app should not redownload the full training dataset or full GloVe resource. Hosting/publication is a separate step.

Generate a lightweight reports archive, an archive retaining all four best checkpoints and experiment evidence, and a Gradio app archive.

## Verification and delivery

Perform the real source/embedding download and full preparation in your cloud if supported. Generate reference counts and fingerprints from those actual observations and compare Kaggle-side preparation against them. Do not fabricate expected statistics or hashes.

Check subword/word alignment, IOB normalization, padding, OOV features, CRF decoding/loss, exact-span metrics, offset preservation, repeated entities, and clean checkpoint reloads. Run tiny real train/validation-only neural smoke checks for all four models. Test scheduler overlap, isolation, cancellation, failure handling, and single-GPU fallback. Verify output archives and app inference.

Validate the notebook's schema, code, embedded files, cell order, and clean execution as far as the cloud supports. Default to full mode with bounded checks before expensive training; provide smoke mode that does not evaluate the real test partition or claim final metrics. Test data is for the frozen final reporting phase.

Distinguish static checks, simulated scheduler checks, CPU execution, native-kernel execution, real CUDA execution, and full training in the verification report. Do not describe an unexecuted Kaggle or two-GPU path as fully tested. If a real limitation blocks a check, state it and provide the smallest useful workaround.

Deliver the finished `.ipynb` file, concise Kaggle start instructions, and a clear summary of executed verification and remaining runtime checks. Complete the implementation and available tests rather than returning only a plan. No further user preference is needed for routine choices already covered by the proposal; investigate and document audit-dependent settings.
