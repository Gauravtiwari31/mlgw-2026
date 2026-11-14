# Manual review (train rows only)

**Rule first.** The competition forbids labelling test rows by hand. Everything here is TRAIN data. The job is
to spot a pattern a person can describe as a rule ("column X is always Y for class A"); the rule then goes into
code. Do not open test.csv to mark rows.

Every file has `TRUE_CLASS` in the first column. `model_says` is what our model predicts for the row.
Open in Excel, turn on filters, sort or filter by `TRUE_CLASS`, and compare the groups column by column.

| File | The two groups | What our model cannot tell |
|---|---|---|
| `1_fanin_mule_vs_velocity.csv` | fan-in episodes (many senders pay one person): `mule_account` vs `velocity_bot` | about 10% of these episodes are velocity_bot; we call all of them mule |
| `2_simswap_look_sim_vs_takeover.csv` | bursts from an account with a fresh SIM: `sim_swap` vs `account_takeover` | about 27% are takeover; we call all of them sim_swap |
| `3_phishing_vs_cashback_look.csv` | collect requests and small-then-big payments: `phishing_collect` vs `fake_cashback` (some `loan_app_extortion`) | 10 to 20% are the other class |
| `4_chain_look_layering_vs_mule.csv` | money hopping on within minutes: `layering_chain` vs `mule_account` | about 12% are mule; we call all of them layering |
| `5_ordinary_looking_fraud_vs_legit.csv` | payments on brand-new phones: known FRAUD that looks ordinary vs known LEGIT | nothing we tried separates them |

Rows with the same `episode` value belong to one fraud episode.

## What to look for

Anything, but these have not been checked by eye:

- a value that appears in only one group (a city, bank, app, merchant, amount ending, time of day, weekday);
- a relation between two columns (amount against balance, device city against the two home cities, opening
  dates against the payment date);
- inside an episode: number of rows, spacing in time, order of the amounts, repeated senders;
- ids: do the sender, receiver or merchant numbers of one group fall in a range or share digits.

## How to report

One line per finding: the file, the rule, and how many rows of each group it covers. Example:
"File 2: all account_takeover rows have pin_retries >= 1; 180 of 407 takeover rows, 12 of 1082 sim_swap rows."

## Already checked by code and empty (no need to repeat)

Averages and distributions of every single column, digits and remainders of all ids, transaction-id characters,
row order in the CSV, timestamps, and how values are written in the file.
