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

## 14 Nov 15:00: class weights on a corrected ruler (`src/weighted_base.py`)

The test-like ruler typed unreported rows with the blend's own guess, which penalised every type change. With
those rows left out, class weights on the argmax (fitted on two months, scored on the third) gain +0.0018,
+0.0015, +0.0006. The same three move every time: phishing down, cashback and loan up. Fixed weights kept:
legit 0.67, phishing 0.37, cashback 1.92, loan 1.57 (per month: +0.0018, +0.0016, +0.0010). On test 66 rows
change before the zones (38 phishing -> cashback, 17 phishing -> legit).
File: `submissions/1114-1420__L2_blend16_w_zones3.csv`. Expected public 0.542 to 0.544.

Also empty this round: transaction ids as a seeded draw without replacement at the exact row counts (1,624
combinations); how values are written in the raw CSV (63 text features); hidden share by phone kind x device
age / SIM age / login / amount / balance / IP bands; pair models per look-alike cell with the Gemini
episode-graph features (AUC 0.6 to 0.9 inside cells, but flips with out-of-month thresholds lose 0.0008).

Manual review files for people (train rows only): `work/manual/README.md`.

## 14 Nov 16:00: weighted file scored 0.5411; what the board implies; more empty checks

`1114-1420__L2_blend16_w_zones3.csv` scored 0.5411 (best stays 0.5417). Board at 14:15, best upload per team:
0.6338, 0.6311, 0.6111, 0.6075, 0.5818, 0.5731, then 0.5547 down to us at 0.5417 (14th). One team reached 0.5818
with 6 entries, so it is not leaderboard probing.

**Arithmetic.** Test has 10,582 fraud rows (260,582 minus 250,000 legit). We call about 3,990 rows fraud, so at
most 38% of fraud rows are reachable for us. With every one of them found and typed right the score is about
0.58. The top four are above that: they find several hundred to a thousand ordinary-looking fraud rows, with the
right class.

Checked since the last entry, all empty:

- file chunk against 181 row attributes, train and test (a sharding rule would expose fraud rows): nothing;
- how October's known hidden rows connect (receiver and sender activity, money moving on, reverse pairs)
  against known legit rows, with September as reference: the same;
- a type model given every raw column, every id and all account / merchant attributes: type accuracy 0.879,
  the same as the blend (0.905 even when rows of one episode leak across folds);
- pairs of same-class hidden rows sharing an exact key (28 keys: same second, sender, receiver, amount, balance,
  SIM age, IP score, city and minute, id prefix...): 0.96 to 1.05 times chance on every key with enough pairs;
- October against September rows that look legit (model told apart only by general drift in amounts and IP
  score, AUC 0.636): no extra population in October (0.2% above September's 99.9% level);
- test.csv against train.csv: same columns, formats, categories, id ranges; every sender, receiver and merchant
  exists in the side tables; the only test-only values are app versions 5.0 to 5.2, on every row.

In train the ordinary-looking fraud rows are independent draws from the legit generator. Nothing in the four
files, as far as these checks go, marks them in test either, apart from the phone counter.

## 15 Nov 00:30 to 05:30: last round before the deadline (best stays 0.5417)

Board at midnight: 0.6338, 0.6320, 0.6311, 0.6111, 0.6075, 0.5731, 0.5672, then 0.5547 down to us (17th, 0.5417).
Three files built in a separate session were uploaded on 14 Nov evening and are logged in `uploads_log.csv`:
type flips on look-alike pairs scored 0.5320 (104 rows changed) and 0.5407; `submission2.csv` scored 0.4727.
Flipping types without a block phone in the episode loses points on the board too.

**Why velocity_bot and merchant_collusion probe low on test** (F1 0.35 and 0.27 against 0.55 and 0.45 in
September). Ordinary-looking fraud is a fixed stream per class and month (hidden velocity rows per train month:
1,723 / 1,798 / 1,757), while visible rows come in episodes. October has 7 bot-phone bursts (269 rows) where the
train rate gives 13 to 14, and 5 collusion merchants (85 rows) against 6 to 7. With 269 visible rows out of about
1,240 velocity rows the best possible F1 is 0.36, which is what the probe showed. It is not a miss of ours:
- no bursts off the bot pool in any month: a detector for 10+ small payments into one merchant within 15 minutes
  (far above the merchant's own rate) finds 730 / 476 / 637 / 154 window starts per month with the bot-pool rows
  in, and 0 with them out; the same for 6+ payments by one account;
- share of rows with login-to-payment under 2 to 15 s off the pool is the same in every month;
- mule-typed fan-in episodes in October have the same size, amounts, login times and hours as in train.

**Hidden rows, checked again with different tools, all empty:**
- value support: no numeric value, category or format occurs in hidden rows and not in legit rows; receiver type
  x txn type mix of the hidden rows of every class equals the legit mix (no class-specific conditioning);
- links: hidden rows share senders, receivers and phones with visible fraud 1.03 to 1.15 times as often as legit
  rows do, and with each other exactly as often as random legit rows;
- time: gap to the nearest visible row of the same class, hour, weekday, day of month, rows per day (chi-square
  70.8 over 92 days) all equal to legit;
- models (44 raw, id and side-table features, 4-fold): hidden against legit AUC 0.4996 overall and 0.49 to 0.51
  for every class; class among hidden rows: accuracy 0.188 against 0.193 for always guessing the biggest class;
- October only: ordinary-looking rows on block phones against legit new-phone rows: same app versions (5.0 to
  5.2 on every test row), apps, types, amounts, ages; block rows sit slightly later in the month (rank test
  p = 0.001, 186 rows), with no trend in train and nothing to act on;
- `sample_submission.csv` has the ids of `test.csv` in the same order, all `legit`;
- transaction ids are not consecutive Mersenne-Twister outputs in file order (state rebuilt from 624 ids at
  eight offsets, both byte orders, 0 matches).

**Zones.** Test zone spans equal the train spans times 920/5455 (mule 118 against 110.6, takeover 290 against
290.8, layering 79 against 79.3, sim 68 against 68.3). Only cashback (11 against 22) and loan (8 against 17) are
short; their visible rows (REFUND rows up to 474495, cashback at 474505, loan at 474517 to 474519, collusion at
474529) fix the boundaries, so they stay. Time does not follow phone id inside a zone (Spearman about 0), so the
class loops do not run in date order and the date says nothing about a boundary.

**File chunks, exact.** Each train chunk is 250,000 legit rows of one batch plus the fraud rows drawn into it
(sizes 260,742 / 260,631 / 260,481 / 260,696 / 260,870 / 260,640), shuffled; test is 2 x 125,000 legit plus
10,582 fraud rows. Train has 64,060 fraud rows, 57,648 of them labelled: exactly 10.0% unreported. No column
differs between chunks (KS at noise level for 19 columns) and none follows generation order inside a chunk.
A fixed shuffle seed would give the order back: for chunk 0 of train (n = 260,731 to 260,753) every seed from 0 to
20,000 plus 51 common ones was tried with numpy's legacy and PCG64 permutations, in both directions, and checked
against the generation order of the 7,411 legit new-phone rows: no seed fits (24 minutes on 11 processes).

No organiser hint was found: no e-mail after 13 Nov, nothing on the public event page; the team workspace
needs a login this session does not have.
