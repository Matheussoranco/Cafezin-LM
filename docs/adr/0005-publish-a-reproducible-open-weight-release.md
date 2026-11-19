# Publish a reproducible open-weight release

Publish Cafezin weights on Hugging Face as a versioned artifact distinct from the upstream checkpoint. The minimum package includes SafeTensors weights, matching tokenizer assets, model and generation configs, and a model card recording parent revision, data provenance and curriculum, tokenizer changes, training hyperparameters and compute, evaluation results, and limitations.

Start with an explicitly invoked Python uploader rather than auto-publishing from CI. Before publication, reload the package in a clean environment, record the evaluation evidence, and confirm weight and data provenance, redistribution terms, and required notices; the source-code license alone does not settle the release terms for every artifact.
