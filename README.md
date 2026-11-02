# Customer Churn Prediction and Retention Analytics

We are estimating which telco customers are likely to leave, comparing a neural network with two simpler models, and looking at what a retention campaign might do with those scores. The data is the public Telco Customer Churn file from Kaggle.

This repository is a portfolio project, built one step at a time so each choice can be reviewed before the next piece of code lands.

## How this was built

This project was developed through human-AI collaboration. I set the problem, made the design decisions, and reviewed the results. An AI assistant helped with implementation, debugging, and explanations.

The write-up stays in the first person plural where we did the work together. Where a choice is still open, the code says so instead of pretending it is settled.

## Where we are

Step 1, loading and cleaning, is in `train.py`. We have not split the data, encoded categories, or trained a model. There are no accuracy numbers yet, and there should not be.

## What we found in the raw file

The file is `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

- 7,043 customers and 21 columns. One row per customer.
- 1,869 customers are labeled `Churn = Yes` (26.5%). 5,174 are `No`.
- `customerID` is unique. There are no duplicate rows.
- `TotalCharges` is stored as text. Eleven cells are a single space, not a number. Pandas does not count a space as missing, so a normal missing-value check reports zero gaps.
- Those 11 customers all have `tenure = 0` and `Churn = No`. They look like people who have just joined and have not been billed.
- Phone and internet add-on labels match the parent service. We counted the disagreements and found none. We did not rewrite any of those labels.
- `SeniorCitizen` is already 0/1. The other yes/no fields are still the words `Yes` and `No`.

The full counts from the run are in `outputs/01_cleaning_summary.json`.

## Open decision: blank TotalCharges

We need a rule for those 11 blank bills before we treat this as finished.

| Option | What it does | Tradeoff |
| --- | --- | --- |
| `zero` (recommended) | Set `TotalCharges` to 0 and keep the rows | 0 matches "not billed yet", and brand-new customers stay in the data |
| `drop` | Remove the 11 rows | No filled value at all. We also remove 11 non-churners, so the churn rate moves from 26.5% (1,869 / 7,043) to 26.6% (1,869 / 7,032) |

We do not fill the blank with `MonthlyCharges`. That would invent a payment the file does not contain.

`BLANK_TOTAL_CHARGES` in `train.py` is set to `"zero"` so the script can run. That is the recommendation, not a confirmed decision. Change the constant and run the script again if you want `"drop"`. The summary file records which option the run used.

Why this is not a leakage problem: the rule does not compute an average or a frequency from other rows. It is the same action for every blank cell. Later, when we one-hot encode or scale, we will fit those steps on the training split only.

## What this step leaves alone

- `customerID` stays in the cleaned file so we can trace a row. It is an identifier, not a predictor, so it will stay out of the model matrix.
- Yes/no fields stay as text until we encode them on the training split.
- `SeniorCitizen` stays 0/1.

## How to run this step

From the project folder, with Python 3.9 or newer:

```bash
pip install -r requirements.txt
python train.py
```

That writes:

- `outputs/telco_churn_cleaned.csv` — the cleaned table. It is not committed, because `train.py` rebuilds it from the raw file.
- `outputs/01_cleaning_summary.json` — the counts from the run. This file is committed.

We ran this on Python 3.12.3. We did not have Python 3.9 on the machine, so that exact version was not executed here. The code uses only syntax and pandas operations that Python 3.9 and pandas 2.2 support.

## Limitations we already know

- The file is one snapshot. There is no date for when someone joined or left, so we cannot study churn as a sequence of events.
- Later findings will be associations. A contract type or a bill showing up next to churn does not prove it caused the customer to leave.
- Nothing has been trained yet. Scores on training rows, once we have them, will look better than scores on held-out rows, and we will say so when we report them.

The revenue-at-risk piece is not in this step. When we add it, the assumptions (using `MonthlyCharges` as revenue, a fixed number of months, treating probabilities as calibrated, and a retention success rate and contact cost that we choose rather than estimate) will be written down next to the code.
