import unittest

import numpy as np

from microlearn.metrics import roc_auc_score
from microlearn.model_selection import train_test_split, StratifiedKFold
from train import generate_synthetic_data, load_real_data
from app import engineer_features


class LoanProjectTests(unittest.TestCase):
    def test_auc_handles_tied_scores(self):
        y = np.array([0, 1, 0, 1])
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(roc_auc_score(y, scores), 0.5)

    def test_stratified_split_is_not_grouped_by_class(self):
        X = np.arange(200).reshape(-1, 1)
        y = np.repeat([0, 1], 100)
        _, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        # Every consecutive CV fold should retain both target classes.
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        for _, val_idx in cv.split(np.empty((len(y_train), 1)), y_train):
            self.assertEqual(set(y_train[val_idx]), {0, 1})

    def test_synthetic_default_rate_is_realistic(self):
        _, y = generate_synthetic_data(20_000)
        self.assertGreater(y.mean(), 0.15)
        self.assertLess(y.mean(), 0.27)

    def test_invalid_numeric_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "annual_inc"):
            engineer_features({field: 1 for field in (
                "loan_amnt", "int_rate", "installment", "dti",
                "delinq_2yrs", "inq_last_6mths", "mths_since_last_delinq",
                "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
                "years_since_earliest_cr_line", "years_since_last_credit_pull",
            )})

    def test_real_data_zero_income_does_not_create_infinity(self):
        # Exercise the same boundary condition that previously corrupted the
        # scaler, using a tiny preprocessed CSV rather than the 374 MB source.
        import os
        import tempfile

        row = {
            "repay_fail": 0, "annual_inc": 0, "loan_amnt": 10000,
            "installment": 300, "int_rate": 12, "revol_util": 50,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "loans.csv")
            import pandas as pd
            pd.DataFrame([row]).to_csv(path, index=False)
            X, _ = load_real_data(path)
        numeric = X[:, :17].astype(float)
        self.assertFalse(np.isinf(numeric).any())


if __name__ == "__main__":
    unittest.main()
