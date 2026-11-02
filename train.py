"""Load and clean the Telco Customer Churn data.

This is step 1. We read the Kaggle file, repair the one column that is
not stored as a real number, and save a cleaned table plus a short
summary. We do not split the rows or train a model here.

The repair is a fixed rule, not a statistic learned from the rows, so
it does not use information from a test set. Anything we later *fit*
(encoders, scalers, the models) will be fit on the training split only.

The blank-bill choice below is a recommendation so this script can run.
It stays open until it is confirmed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Paths hang off this file, so the script still finds the data if the
# terminal is sitting in another folder.
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = ROOT / "outputs"
CLEANED_PATH = OUTPUT_DIR / "telco_churn_cleaned.csv"
SUMMARY_PATH = OUTPUT_DIR / "01_cleaning_summary.json"

# Cleaning does not use randomness. We keep the seed here so the later
# split and the models start from the same number.
RANDOM_SEED = 42

# Open decision: blank TotalCharges.
#
# Eleven customers have tenure 0 and a single space in TotalCharges.
# All eleven are labeled Churn = No. They look like people who have just
# joined and have not been billed yet.
#
#   "zero" — set TotalCharges to 0. Recommended. Zero is the honest total
#            for someone who has not been billed, and we keep a real group
#            of brand-new customers in the file.
#   "drop" — remove these 11 rows. Reasonable if we would rather not fill
#            a cell at all. It also removes 11 non-churners, so the churn
#            rate moves by a very small amount.
#
# We are not filling these cells with MonthlyCharges. That would create a
# payment the file does not contain.
#
# Please confirm this value or change it. The summary records which one
# the run actually used.
BLANK_TOTAL_CHARGES = "zero"

EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

# TotalCharges arrives as text because of the blank cells. We strip it
# with the other text columns, then turn it into a number.
TEXT_COLUMNS = [
    "customerID",
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "TotalCharges",
    "Churn",
]

INTERNET_ADD_ONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the Kaggle CSV and check that it has the columns we expect."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find the data file at {path}. "
            "We expect the telco churn CSV inside the data folder."
        )

    frame = pd.read_csv(path)
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    extra = [column for column in frame.columns if column not in EXPECTED_COLUMNS]
    if missing or extra:
        raise ValueError(
            "The CSV columns do not match the telco churn file. "
            f"Missing: {missing or 'none'}. Extra: {extra or 'none'}."
        )

    # A stable column order makes the cleaned file easier to compare.
    return frame.loc[:, EXPECTED_COLUMNS].copy()


def _churn_snapshot(frame: pd.DataFrame) -> dict:
    """Count churn labels. We keep the counts, not just a rounded rate."""
    counts = frame["Churn"].value_counts(dropna=False).to_dict()
    unexpected = sorted(str(label) for label in counts if label not in {"Yes", "No"})
    if unexpected:
        raise ValueError(f"Churn has unexpected labels: {unexpected}.")

    yes = int(counts.get("Yes", 0))
    no = int(counts.get("No", 0))
    rows = yes + no
    return {
        "yes": yes,
        "no": no,
        "rows": rows,
        "churn_rate": round(yes / rows, 6) if rows else None,
    }


def _service_mismatch_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Count rows whose phone or internet add-ons disagree with the parent service.

    A customer with no phone should be marked "No phone service", and a
    customer with no internet should be marked "No internet service" on
    each add-on. We count disagreements. We do not rewrite them.
    """
    no_phone = frame["PhoneService"].eq("No")
    has_phone = frame["PhoneService"].eq("Yes")
    phone_mismatch = (
        no_phone & frame["MultipleLines"].ne("No phone service")
    ) | (has_phone & frame["MultipleLines"].eq("No phone service"))

    no_internet = frame["InternetService"].eq("No")
    has_internet = frame["InternetService"].ne("No")
    mismatches = {"MultipleLines": int(phone_mismatch.sum())}
    for column in INTERNET_ADD_ONS:
        addon_mismatch = (
            no_internet & frame[column].ne("No internet service")
        ) | (has_internet & frame[column].eq("No internet service"))
        mismatches[column] = int(addon_mismatch.sum())
    return mismatches


def clean_customers(
    raw: pd.DataFrame,
    blank_total_charges: str = BLANK_TOTAL_CHARGES,
) -> tuple[pd.DataFrame, dict]:
    """Strip text, repair blank TotalCharges, and return a report of what changed.

    blank_total_charges is "zero" or "drop". See the note on
    BLANK_TOTAL_CHARGES at the top of this file.
    """
    if blank_total_charges not in {"zero", "drop"}:
        raise ValueError(
            "blank_total_charges must be 'zero' or 'drop'. "
            "See BLANK_TOTAL_CHARGES in train.py."
        )

    cleaned = raw.copy()
    duplicate_ids_before = int(cleaned["customerID"].duplicated().sum())
    duplicate_rows_before = int(cleaned.duplicated().sum())

    # A plain missing-value count is not enough. Pandas does not treat a
    # single space as missing, and that is exactly how the blank bills
    # are stored.
    for column in TEXT_COLUMNS:
        cleaned[column] = cleaned[column].astype("string").str.strip()

    if cleaned["customerID"].eq("").any() or cleaned["customerID"].isna().any():
        raise ValueError("A row has an empty customerID. We did not drop it automatically.")

    duplicate_ids = int(cleaned["customerID"].duplicated().sum())
    duplicate_rows = int(cleaned.duplicated().sum())
    if duplicate_ids or duplicate_rows:
        raise ValueError(
            "Duplicate customers or rows showed up. We did not drop them "
            "automatically, because we have not agreed a rule for that."
        )

    allowed_senior = {0, 1}
    senior_values = set(cleaned["SeniorCitizen"].dropna().unique().tolist())
    if senior_values != allowed_senior:
        raise ValueError(
            f"SeniorCitizen should be 0 or 1 only. Found {sorted(senior_values)}."
        )

    if (cleaned["tenure"] < 0).any():
        raise ValueError("tenure has negative values. We did not clip them.")
    if (cleaned["MonthlyCharges"] <= 0).any():
        raise ValueError("MonthlyCharges has a value that is not positive.")

    blank_mask = cleaned["TotalCharges"].eq("") | cleaned["TotalCharges"].isna()
    if bool(blank_mask.any()):
        all_tenure_zero = bool(cleaned.loc[blank_mask, "tenure"].eq(0).all())
    else:
        all_tenure_zero = True
    blank_but_billed = blank_mask & cleaned["tenure"].ne(0)
    if int(blank_but_billed.sum()) > 0:
        raise ValueError(
            "Some rows have a blank TotalCharges but tenure is not 0. "
            "Filling those with 0 would be a guess, so we stop."
        )

    # Anything that is not blank must already be a plain number.
    parsed_present = pd.to_numeric(
        cleaned.loc[~blank_mask, "TotalCharges"], errors="coerce"
    )
    if parsed_present.isna().any():
        raise ValueError("TotalCharges has non-blank values that are not numbers.")

    blank_ids = cleaned.loc[blank_mask, "customerID"].astype(str).tolist()
    blank_churn = _churn_snapshot(cleaned.loc[blank_mask]) if blank_ids else {
        "yes": 0,
        "no": 0,
        "rows": 0,
        "churn_rate": None,
    }
    churn_before = _churn_snapshot(cleaned)

    # The "what if we dropped them" counts are always reported, so we can
    # compare the two options without a second run.
    kept_if_dropped = cleaned.loc[~blank_mask].copy()
    churn_if_dropped = _churn_snapshot(kept_if_dropped)

    rows_before = int(len(cleaned))
    if blank_total_charges == "zero":
        cleaned.loc[blank_mask, "TotalCharges"] = "0"
        rows_removed = 0
    else:
        cleaned = kept_if_dropped
        rows_removed = rows_before - int(len(cleaned))

    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    # A plain float, rather than pandas' nullable Float64, matches what we
    # will hand to scikit-learn and Keras later.
    cleaned["TotalCharges"] = cleaned["TotalCharges"].astype("float64")
    if cleaned["TotalCharges"].isna().any():
        raise ValueError("TotalCharges still has values we could not read as numbers.")
    if (cleaned["TotalCharges"] < 0).any():
        raise ValueError("TotalCharges has a negative value after cleaning.")

    if blank_total_charges == "zero" and blank_ids:
        fixed = cleaned.loc[cleaned["customerID"].isin(blank_ids)]
        if len(fixed) != len(blank_ids) or not (
            fixed["TotalCharges"].eq(0) & fixed["tenure"].eq(0)
        ).all():
            raise AssertionError("The blank TotalCharges rows were not set to 0.")

    mismatches = _service_mismatch_counts(cleaned)
    if sum(mismatches.values()) > 0:
        raise ValueError(
            "Phone or internet add-on labels disagree with the parent service. "
            f"Counts: {mismatches}. We did not rewrite them."
        )

    cleaned = cleaned.reset_index(drop=True)
    churn_after = _churn_snapshot(cleaned)
    remaining_missing = int(cleaned.isna().sum().sum())
    if remaining_missing:
        raise ValueError(
            f"Cleaning left {remaining_missing} missing values. "
            "We expected none on this file."
        )

    report = {
        "step": "load_and_clean",
        "source": str(DATA_PATH.relative_to(ROOT)),
        "random_seed_for_later_steps": RANDOM_SEED,
        "rows_raw": rows_before,
        "columns": int(cleaned.shape[1]),
        "duplicate_customer_ids_before_strip": duplicate_ids_before,
        "duplicate_rows_before_strip": duplicate_rows_before,
        "duplicate_customer_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "blank_total_charges": {
            "rows": len(blank_ids),
            "customer_ids": blank_ids,
            "all_have_tenure_zero": all_tenure_zero,
            "churn": blank_churn,
            "how_they_are_stored": "a single space, which pandas does not count as missing",
        },
        "blank_total_charges_policy": blank_total_charges,
        "policy_status": (
            "Recommendation in use so the script can run. "
            "Not a confirmed decision yet."
        ),
        "rows_removed": rows_removed,
        "rows_after_cleaning": int(len(cleaned)),
        "churn_before": churn_before,
        "churn_after": churn_after,
        "if_blank_rows_were_dropped": {
            "rows": churn_if_dropped["rows"],
            "churn": churn_if_dropped,
        },
        "ranges_after_cleaning": {
            "tenure_min": int(cleaned["tenure"].min()),
            "tenure_max": int(cleaned["tenure"].max()),
            "monthly_charges_min": float(cleaned["MonthlyCharges"].min()),
            "monthly_charges_max": float(cleaned["MonthlyCharges"].max()),
            "total_charges_min": float(cleaned["TotalCharges"].min()),
            "total_charges_max": float(cleaned["TotalCharges"].max()),
        },
        "service_label_mismatches": mismatches,
        "remaining_missing_values": remaining_missing,
        "dtypes_after_cleaning": {column: str(dtype) for column, dtype in cleaned.dtypes.items()},
        "left_unchanged_on_purpose": [
            "customerID is kept so a row can be traced back. It is an id, not a predictor.",
            "Yes/No fields are still text. Encoding waits until the training split.",
            "SeniorCitizen stays 0/1, which is how the file stores it.",
        ],
    }
    return cleaned, report


def write_outputs(cleaned: pd.DataFrame, report: dict) -> None:
    """Save the cleaned table and the summary. The summary is the record of this run."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEANED_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _print_review(report: dict) -> None:
    """Print the few numbers we want to look at before moving on."""
    blank_rows = report["blank_total_charges"]["rows"]
    before = report["churn_before"]
    after = report["churn_after"]
    alternative = report["if_blank_rows_were_dropped"]["churn"]
    policy = report["blank_total_charges_policy"]

    print(f"Raw rows: {report['rows_raw']}")
    print(
        f"Churn in the raw file: {before['yes']} of {before['rows']} "
        f"({before['churn_rate']:.1%})."
    )
    print(
        f"Blank TotalCharges: {blank_rows} rows, all with tenure 0, "
        f"all labeled Churn = No."
    )
    print(f"Policy used this run: {policy!r}.")
    print(
        f"Rows after cleaning: {report['rows_after_cleaning']} "
        f"(removed {report['rows_removed']})."
    )
    print(
        f"Churn after cleaning: {after['yes']} of {after['rows']} "
        f"({after['churn_rate']:.1%})."
    )
    print(
        "If we dropped the blank rows instead: "
        f"{alternative['yes']} of {alternative['rows']} "
        f"({alternative['churn_rate']:.1%})."
    )
    print(f"Wrote {CLEANED_PATH.relative_to(ROOT)}")
    print(f"Wrote {SUMMARY_PATH.relative_to(ROOT)}")
    print()
    print("Step 1 stops here. We have not split the data or trained a model.")
    print(
        "Open decision: BLANK_TOTAL_CHARGES in train.py "
        f"is {BLANK_TOTAL_CHARGES!r}. Confirm it, or switch it to the other option."
    )


def main() -> None:
    raw = load_raw()
    cleaned, report = clean_customers(raw)
    write_outputs(cleaned, report)
    _print_review(report)


if __name__ == "__main__":
    main()
