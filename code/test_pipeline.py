import unittest

import pandas as pd
from pipeline import (
    add_hour_columns,
    create_features_and_labels,
    load_dataset,
    remove_duplicates,
    split_into_train_test,
)


class TestPipelineFunctions(unittest.TestCase):
    def test_load_dataset(self):
        # Test loading a valid dataset
        df = load_dataset("valid_path_to_csv.csv")
        self.assertIsInstance(df, pd.DataFrame)

        # Test loading a non-existent dataset
        with self.assertRaises(FileNotFoundError):
            load_dataset("invalid_path.csv")

    def test_remove_duplicates(self):
        data = {"col1": [1, 1, 2], "col2": [3, 3, 4]}
        df = pd.DataFrame(data)
        df_no_dup = remove_duplicates(df)
        self.assertEqual(len(df_no_dup), 2)

    def test_add_hour_columns(self):
        # Provide a test case for add_hour_columns
        pass

    def test_create_features_and_labels(self):
        # Provide a test case for create_features_and_labels
        pass

    def test_split_into_train_test(self):
        # Provide a test case for split_into_train_test
        pass


# More test cases for other functions...

if __name__ == "__main__":
    unittest.main()
