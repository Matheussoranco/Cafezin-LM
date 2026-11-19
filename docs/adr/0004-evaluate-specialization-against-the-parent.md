# Evaluate specialization against the parent checkpoint

Compare Cafezin against the pinned unadapted S5-nano checkpoint on held-out Brazilian Portuguese language modeling, Portuguese downstream tasks, and multilingual benchmarks as the minimum evidence package. Success also requires generative evaluation in Brazilian Portuguese, including Brazilian culture and knowledge; report cross-lingual retention or regression separately so a Portuguese gain cannot hide a multilingual loss.

Use `lm-evaluation-harness` as the foundation and add tasks or evaluators where coverage is missing. Report per-token perplexity with tokenizer context and use a tokenizer-normalized metric such as bits per UTF-8 byte for direct comparisons across vocabularies. Choose exact datasets, decontamination rules, scoring rubrics, and thresholds before evaluating release candidates.
