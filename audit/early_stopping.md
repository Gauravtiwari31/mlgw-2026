# Early-stopping audit

Numbers from `python -m src.audit_early_stopping --config configs/L2_lgbm_base1-flow1_s0.yaml`
(raw output in `audit/early_stopping.json`). Features: `base_v1` + `flow_v1`, view `half_s0`,
train Jul+Aug, score Sep, seed 0.

**Verdict: early stopping is not losing score.** The Sep macro-F1 is flat from 10 to 500 rounds
(0.431 to 0.437 with the current settings). Stopping at 22 rounds is as good as any other point.
No model setting tried moves the score by more than 0.008, which is the same size as the spread
between seeds. The limit is what the features can see, not the trainer.

## Sep macro-F1 by number of rounds (plain argmax)

| Variant | 10 | 20 | 30 | 50 | 80 | 120 | 180 | 250 | 350 | 500 |
|---|---|---|---|---|---|---|---|---|---|---|
| current (lr 0.08, 63 leaves) | 0.4311 | 0.4352 | 0.4347 | 0.4344 | 0.4342 | 0.4334 | 0.4341 | 0.4352 | 0.4355 | 0.4370 |
| lr 0.03 | 0.0826 | 0.4240 | 0.4373 | 0.4417 | 0.4424 | 0.4410 | 0.4398 | 0.4403 | 0.4407 | 0.4402 |
| smaller trees (15 leaves, 200 per leaf) | 0.4239 | 0.4364 | 0.4381 | 0.4358 | 0.4339 | 0.4341 | 0.4324 | 0.4305 | 0.4307 | 0.4296 |
| smaller trees, lr 0.03 | 0.0860 | 0.4028 | 0.4314 | 0.4387 | 0.4419 | 0.4413 | 0.4409 | 0.4397 | 0.4395 | 0.4372 |
| no class weights | 0.3861 | 0.4025 | 0.4056 | 0.4082 | 0.4084 | 0.4083 | 0.4089 | 0.4084 | 0.4100 | 0.4126 |
| no class weights, smaller trees, lr 0.03 | 0.1459 | 0.3060 | 0.3763 | 0.4095 | 0.4206 | 0.4232 | 0.4237 | 0.4255 | 0.4261 | 0.4278 |
| all legit rows (no downsampling) | 0.4291 | 0.4346 | 0.4363 | 0.4353 | 0.4349 | 0.4371 | 0.4373 | 0.4369 | 0.4368 | 0.4336 |

## What the weighted argmax adds

"Weighted" is cross-fitted inside Sep: weights fitted on one random half, scored on the other,
both ways, averaged.

| Variant | Best plain | Best weighted | Gain |
|---|---|---|---|
| current | 0.4370 | 0.4371 | +0.000 |
| lr 0.03 | 0.4424 | 0.4426 | +0.000 |
| smaller trees | 0.4381 | 0.4379 | -0.000 |
| smaller trees, lr 0.03 | 0.4419 | 0.4422 | +0.000 |
| no class weights | 0.4126 | 0.4220 | +0.009 |
| no class weights, smaller trees, lr 0.03 | 0.4278 | 0.4394 | +0.012 |
| all legit rows | 0.4373 | 0.4379 | +0.001 |

With the class weights used in training (power 0.5), the plain argmax is already at the best
operating point, so the weighted argmax has nothing to add. It only helps a model trained
without class weights, and that model still ends below the weighted ones.

## Learning rate 0.03 against 0.08, three seeds, through `src.train`'s own early stopping

| Seed | lr 0.08 | rounds | lr 0.03 | rounds | Difference |
|---|---|---|---|---|---|
| 0 | 0.4359 | 22 | 0.4431 | 72 | +0.0072 |
| 1 | 0.4377 | 29 | 0.4432 | 76 | +0.0055 |
| 2 | 0.4436 | 31 | 0.4448 | 52 | +0.0012 |

Mean difference +0.0046. The lower rate is slightly better on every seed and steadier across
seeds (0.4431 to 0.4448 against 0.4359 to 0.4436), but the gain is small.

## What this means

- Every variant predicts 1.4% to 1.7% of Sep rows as fraud; the labels say 3.66%. The models
  agree on which fraud is findable and stop there. Recall per fraud class is 0.2 to 0.4.
- Do not spend time tuning rounds, tree size or class weights. New features that expose more
  fraud rows are where the score is.
- For later runs: learning rate 0.03 is a safe default, and seed averaging matters, because one
  seed moves the score by up to 0.008.
- Log loss on Sep keeps falling for 500 rounds in the weighted variants. That is the model
  getting more confident on legit rows, not better at ranking fraud. Do not use it to pick rounds.
