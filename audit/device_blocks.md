# Device-id blocks (L2, 14 Nov 03:20; block end, length prior and linked rows added 04:00)

The generator hands out new phone ids from a counter: fraud first, class by class, then ordinary new phones.
This was hidden from our models because we never fed raw ids (and train and test use different id ranges, so
a raw id would not transfer anyway).

## What the ids look like

| Phone ids | What they are | Evidence (train) |
|---|---|---|
| 0 to 419999 | each account's own phone (id = account number) | 92% of rows |
| 420000 to 422999 | bot pool, reused in every month | every row velocity_bot |
| 423000 to 428454 | phones first used inside a fraud episode | 88.4% fraud-labelled; the rest is the unreported share |
| 428455 to 473631 | ordinary new phones, each used exactly once, no id skipped | fraud of any kind: 0.07% |
| 473632 to 474551 | **test month's fraud block** (separate run, a sixth of the train block) | found without labels |
| 474552 to 482013 | test month's ordinary new phones | each used once, no id skipped |

Inside the fraud block the zones follow the label order. Train (true zones from labels):

| Zone | Ids | Phones seen | Rows | Rows per phone |
|---|---|---|---|---|
| mule_account | 423000-423655 | 350 | 350 | 1.0 |
| velocity_bot | 423658-424152 | 175 | 175 | 1.0 |
| layering_chain | 424155-424624 | 213 | 213 | 1.0 |
| sim_swap | 424628-425032 | 289 | 570 | 1.97 |
| account_takeover | 425039-426762 | 454 | 1165 | 2.57 |
| phishing_collect | 426769-427164 | 226 | 226 | 1.0 |
| qr_swap | 427165-427846 | 230 | 230 | 1.0 |
| refund_abuse | 427848-428066 | 108 | 108 | 1.0 |
| fake_cashback | 428069-428199 | 74 | 74 | 1.0 |
| loan_app_extortion | 428201-428301 | 51 | 51 | 1.0 |
| merchant_collusion | 428304-428442 | 65 | 65 | 1.0 |

96.5% of fraud-labelled rows on block phones carry their zone's class.

## What it means

- The "ordinary-looking" fraud is generated inside each class's loop, not sprinkled on legit rows afterwards:
  when such a row gets a new phone (about 3%, the same rate as legit rows) the id comes from the fraud counter.
  Those rows are now identifiable, with their class. The other 97% still are not.
- sim_swap and account_takeover are not swapped labels. A phone in the sim_swap zone is a sim_swap phone even
  when the row shows the "new phone" signature.
- Legit rows on ordinary new phones are never fraud.

## How zones are found without labels (`src/device_block.py`)

Ordered segmentation of the block's phones with three cues, parameters from the train block:
a cross-fitted row classifier, the gap to the previous seen id (zones use their ids at different densities:
sim_swap 71% of ids seen, takeover 26%), and whether a phone has more than one row.
In train this recovers the true zone for **98.5% of rows** (velocity 79%, cashback 92%, all others 98-100%).

## Effect (`src/zone_apply.py`)

| | Before | After |
|---|---|---|
| Sep macro-F1, raw labels | 0.4749 | 0.5006 (+0.026) |
| Sep, unreported block rows counted as fraud | 0.4816 | 0.5117 (+0.030) |
| Test rows changed (of 829 on block or pool phones) | | 295 |

Every fraud class improves on Sep; sim_swap +0.09, takeover +0.05, the others +0.013 to +0.023.
Test changes: 70 takeover -> sim_swap, 211 legit -> fraud (56 mule, 28 velocity, 27 phishing, 24 qr, 20 layering,
18 refund, 26 sim/takeover, 10 collusion, 2 cashback), 14 other type changes.

Submission: `submissions/1114-0319__L2_blend8_zones.csv` (the 0.5019 file with those 295 rows changed).

## Block end (corrected 14 Nov 04:00)

The first rule (three dense windows) cut the block 9 ids early. Ordinary new phones use every id, so the block
ends right after the last id that no row uses: 428454 in train, 474551 in test. The five train rows gained
(428446, 428450 to 428453) are four merchant_collusion and one legit label. In test this adds 4 rows.

## Linked rows (`src/zone_apply.py --propagate`)

A block phone settles the class of one payment; other payments of the same episode share that class.
Checked in train on rows off the block that the blend already calls fraud:

| Link to a block row | Rows where link and blend disagree | Link right | Blend right | Used |
|---|---|---|---|---|
| same sender within 2 h, both among sim_swap / takeover | 24 | 100% | 0% | yes (R2) |
| same personal receiver within 24 h, blend mule -> link velocity | 45 | 98% | 2% | yes (R1) |
| same personal receiver within 24 h, blend layering -> link mule | 24 | 46% | 0% | yes (R1) |
| same personal receiver within 24 h, link says sim_swap / takeover | 30 | 0 to 20% | 47% | no |
| same sender within 24 h, same receiver within 7 days | | worse than the blend | | no |
| rows the blend calls legit that link to a block row | | 3% fraud | | never flipped |

Sep raw macro-F1: 0.5006 (zones) -> 0.5014 (zones + linked rows). Test: 11 more rows change.
Per class on Sep (before -> after): sim_swap 0.379 -> 0.473, takeover 0.377 -> 0.434, collusion 0.387 -> 0.409,
mule 0.397 -> 0.418, phishing 0.448 -> 0.468, loan 0.387 -> 0.407, cashback 0.353 -> 0.372, the rest +0.015 to +0.017.

## How sure are the test zones (`python -m src.device_block --runs ... --simulate 100`)

The 98.5% above is for the whole train block. The test block is 6 times shorter, so 100 test-sized blocks were
cut out of the train block (each zone a random slice of the right length) and segmented with the cross-fitted
classifier. Rows that get their true zone:

| Method | Lengths vary like Poisson | Lengths vary twice as much |
|---|---|---|
| no prior on zone length (the 03:19 file) | 94.7% (worst tenth of blocks 90.2%) | 94.1% |
| zone length ~ train share, variance = mean (used now) | 96.7% (worst tenth 94.4%) | 96.4% |
| same, variance = 3 x mean | 96.0% | 95.7% |

Weakest zones with the prior: velocity 0.88, loan 0.89, refund 0.91, cashback 0.93, sim_swap 0.94; all others
0.96 to 1.00. The first five test zones come out within 7% of the train proportions with no prior at all, which
is why the prior is reasonable: the generator keeps class shares fixed.

Test zones now (ids): mule 473632-473749, velocity 473750-473838, layering 473839-473917, sim_swap 473920-473987,
takeover 473990-474279, phishing 474281-474359, qr_swap 474361-474460, refund 474462-474500, cashback
474502-474512, loan 474517-474524, collusion 474525-474549. 23 of 564 rows have zone probability under 0.8.

## Files

| File | What | Rows changed vs the 0.5019 file | Expected public |
|---|---|---|---|
| `submissions/1114-0319__L2_blend8_zones.csv` | zones, first block-end rule | 295 | about 0.526 |
| `submissions/1114-0359__L2_blend8_zones2.csv` | corrected block end + length prior + linked rows | 310 | about 0.528 |

Expected change computed two ways: Sep out-of-fold with label-free zones (+0.0265), and from first principles on
the test file (each changed row scored with 0.884 x 0.965 x zone accuracy; +0.027). About 93 of the changed rows
fall in the public 30%, so the public number can land 0.006 either side.

## What was checked and holds nothing (14 Nov, 03:30 to 04:00)

- Transaction ids: uniform random 32-bit, no arithmetic progression, no hash of a counter, no common seed.
- Account and merchant id ranges: fraud share flat across id bins.
- Rows dated after the month end (147 in train, 88% fraud; 46 in test): the blend already calls them fraud
  with the right type.
- Hidden fraud (the 63%): same as legit rows on per-account consistency of account age, device age, SIM age,
  app, city, balance chain; their senders and receivers are not the accounts of visible fraud.
- Unreported rows (real fraud labelled legit, 12.6% of confident fraud predictions): not predictable
  (AUC 0.516), independent coin flips inside an episode.
- Type mix-ups that remain (Sep: velocity called mule 125, mule called layering 98, takeover called sim_swap
  92, cashback / phishing 129): whole episodes generated in one class's loop with another class's pattern.
  19 velocity-labelled fan-in episodes against 185 mule ones: no episode statistic separates them (best AUC
  0.65 on 19 cases). Only a block phone inside the episode tells.
- Larger blends (up to 31 runs): +0.0003 to +0.0009 on Sep. The base model is saturated.
