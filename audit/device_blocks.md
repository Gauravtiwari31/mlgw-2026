# Device-id blocks (L2, 14 Nov 03:20)

The generator hands out new phone ids from a counter: fraud first, class by class, then ordinary new phones.
This was hidden from our models because we never fed raw ids (and train and test use different id ranges, so
a raw id would not transfer anyway).

## What the ids look like

| Phone ids | What they are | Evidence (train) |
|---|---|---|
| 0 to 419999 | each account's own phone (id = account number) | 92% of rows |
| 420000 to 422999 | bot pool, reused in every month | every row velocity_bot |
| 423000 to 428445 | phones first used inside a fraud episode | 88.4% fraud-labelled; the rest is the unreported share |
| 428446 to 473631 | ordinary new phones, each used exactly once | fraud of any kind: about 0.1% |
| 473632 to 474543 | **test month's fraud block** (separate run, half the size) | found without labels |
| 474544 to 482013 | test month's ordinary new phones | each used once |

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
