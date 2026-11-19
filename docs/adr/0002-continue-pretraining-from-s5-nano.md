# Continue pretraining from S5-nano

Start from the pinned S5-nano checkpoint rather than pretraining a new base model from scratch or treating instruction tuning as the main specialization mechanism. Use existing Portuguese corpora, curated for Brazilian relevance and documented provenance, and mix them with multilingual replay: start at 50/50 and move toward 90/10 Portuguese/multilingual by token count to balance specialization against forgetting.

Initialize from model weights, not a trainer resume checkpoint: start fresh Adam state and a new warmup and decay schedule. Keep the existing experimental MTP loss weight of `0.3` for the first run; this is a Cafezin setting, not a claim about upstream training. On one GPU, treat context lengths of 2,048 and 4,096 as baseline and stretch targets, respectively, not as verified fits. Measure peak memory and throughput before committing to the run budget, optimizer configuration, or peak learning rate.
