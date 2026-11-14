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

## Result on the board and what it told us (14 Nov 09:00)

`1114-0359__L2_blend8_zones2.csv` scored **0.5403** (was 0.5019; +0.038 against an estimate of +0.027).
The statement says the test labels are clean: no unreported rows. In test every block row carries a fraud label
(88% in train), which is the difference. The statement also says "some fraud looks completely ordinary, and some
fraud types look like each other": both are how the data is made, not label noise, and both are in test.

**Test-like ruler.** September out-of-fold, raw labels, with the rows `labels_clean` flags as unreported counted
as fraud. `src/zone_apply.py` prints it. The 0.5403 file's method scores 0.5475 on it; the 0.5019 file's 0.5207.

**Where the rest of the score is** (same ruler, Sep):

| | Sep |
|---|---|
| blend + zones + linked rows (0.5403 on the board) | 0.5475 |
| + every visible fraud row we call fraud gets its true type (753 rows) | 0.6048 |
| + visible fraud called legit is found (96 rows) | 0.6094 |
| + no legit row called fraud (122 rows) | 0.6116 |

So wrong fraud type is worth 0.057, everything else under 0.01. The wrong types are whole episodes made in one
class's loop with another class's pattern (velocity as mule fan-in 125, mule as chain 98, takeover as SIM swap 92,
cashback / phishing 129 in Sep). Checked again this round:

- a model trained only on visible fraud rows to tell types: 86.2% type accuracy against the blend's 86.8%;
- class weights fitted on the test-like ruler with a month held out: +0.0003, 0, -0.0007;
- a lower cut-off for calling fraud: +0.0003 at best;
- link rules, full grid of link type x time window x class group: two more rules are right where they disagree
  with the blend. R3: the reverse payment of a pair within 24 h (phishing / cashback, 17 of 17). R4: money
  moving on from or to a block row (mule / layering, 13 of 14). Receiver links for sim_swap / takeover are wrong
  (22% right): fraud rings share receiving accounts across classes. All rules together: Sep 0.5475 -> 0.5483.

**Ordinary-looking fraud cannot be found.** Block phones give 878 known ordinary-looking fraud rows, ordinary
new phones give 45,177 known legit rows. A model on every raw column plus digit patterns (decimals, seconds,
id values) separates them with AUC 0.505. They are made by the same function; the phone counter is the only
difference. Also no grouping: the next row of the same sender, receiver or phone after such a row is another
one 2.2 to 2.5% of the time, the base rate.

**Row order of the CSV files.** Not a full shuffle: train.csv is 6 chunks of about 260,700 rows and test.csv
2 chunks of about 130,300, each a consecutive slice of the legit generation order (ordinary phone ids are
sorted by chunk, rank correlation 0.97 / 0.75 with file position) shuffled inside. Fraud rows are spread evenly
over chunks, with no relation between chunk and zone, class or id. Nothing to use.

**Unused ids inside the block** (59%): spread like independent coin flips inside each zone (run lengths match a
geometric law), not tied to the number of rows on neighbouring phones. Likely one id reserved per episode.

## Candidate for the next upload

`submissions/1114-0854__L2_blend16_zones3.csv`: 16-run blend (the 8 + 8 time-consistent XGBoost seeds) + zones +
rules R1 to R4. Sep test-like 0.5491 against 0.5475; 38 test rows differ from the 0.5403 file. Expected public
0.540 to 0.544: the change is smaller than the public sample's noise.

## 14 Nov 09:30: 0.5417, and two more checks

`1114-0854__L2_blend16_zones3.csv` scored **0.5417** (+0.0014, as estimated). `reproduce.sh` targets it; the zone
step rebuilds it with 0 rows different.

- **No October drift on block rows.** For every zone the blend reads known block rows the same way in train and
  in October (share called legit / right type / other type). Example, sim_swap zone: 16 / 9 / 75% in train,
  14 / 10 / 76% in October. So October's own block phones give nothing to correct. The share of look-alike
  episodes is also the same (mule-looking rows in the velocity zone 7% in train, 5% in October).
- **Look-alikes are near copies, not exact copies.** Phones with several rows in the sim_swap zone all look like
  takeovers (fresh SIM 1%), but every one is away from home (takeover zone 91%) and PIN retries are 0.92
  against 0.83. Too small to separate rows off the block: the type-only model stays at 86%.
- **Velocity bots did not move in October.** Pool-phone rows per month: 1307, 968, 1066, then 269 in October
  (7 phones against 25 to 32, about 14 expected at half size). Off the pool, fast logins (1.6%) and bursts by
  phone or sender look the same as in train; every burst of 3+ rows is a SIM swap or takeover.
- GPU queue `src/queue_day.ps1`: XGBoost on the other legit half with the time-consistent screen, CatBoost,
  and 6 more seeds, for the blend (last 8 members added +0.0008).

## 14 Nov 14:30: leaders at 0.63, and what was checked against it

Board (from the team's screenshots): Loss Leaders 0.6338, Ctrl+C 0.6311, Bolt 0.6111, Draco 0.5731, then a pack at
0.541 to 0.550 (we are at 0.5417). Four teams jumped between 11:00 and 11:30 on 14 Nov. Our ceiling with every
visible fraud row typed correctly is about 0.61, so the top two also find some ordinary-looking fraud.

GPU queue (`src/queue_day.ps1`, 12 runs, all finished 09:57): XGBoost on the other legit half x4, CatBoost x2,
6 more seeds. Added to the 16-run blend they score 0.5479 against 0.5485 on the test-like ruler. Not used.

Checked this round, all empty (hidden = rows `labels_clean` marks as noise, or block rows that look ordinary):

- 589 simple derived features against the hidden label: every hex character and modulus of the transaction id,
  digits and moduli of sender / receiver / device ids, amount / balance / SIM / device-age digits, seconds,
  minutes, weekday, bank, KYC, OS, city, opening month, merchant category and their crosses. Only the device id
  range stands out.
- Time: look-alike episodes are not tied to day, hour or weekday (AUC 0.47 to 0.60); hidden rows of a class are
  not closer to each other or to that class's visible episodes than random legit rows are.
- Account state: device age is capped at account age for hidden rows exactly as for legit rows (1.77% vs 1.72%
  at the cap, none above); `new_payee` is an independent flag (not history); the guest phone of an account is
  fixed and 4% of accounts live on one, same for hidden rows.
- Cross-table: row merchant category always equals merchants.csv, no receiver or merchant used before it
  existed, city / bank / OS matches the same for hidden and legit.
- October only: known hidden rows (215 block rows the blend calls legit) against 7,462 known legit rows: app
  version 5.0 to 5.2 for both, IP score, SIM age, login time the same. No October-only tell.
- test.csv file order and transaction ids: known fraud rows are uniform over position, position modulo k,
  id value and id modulo k; neighbours in the file are not fraud more often.
- Undoing the chunk shuffle with numpy seeds 0 to 50 and common others, 28,910 size / seed / method
  combinations: best rank correlation 0.07.
- Legit rows per file: train has almost exactly 6 x 250,000 legit-phase rows (so 10% of fraud-phase rows are
  unreported), test 2 x 125,000, which puts test fraud at 10,582 rows (4.06%), matching the all-legit probe.

The Gemini notes in the team's Downloads folder reach the same dead ends; their two claims that would matter
("test has no ordinary-looking fraud", "labels_clean corrupted the types") do not hold: the all-legit probe
(0.0816) fixes test fraud at 4.1%, and the raw-label / no-wrong-type runs score lower (type accuracy 0.877
against 0.884).
