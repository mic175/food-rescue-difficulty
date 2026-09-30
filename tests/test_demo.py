"""Check label semantics and protect the demo's evaluation boundaries."""

import unittest

import numpy as np
import pandas as pd

from scripts.run_demo import binary_labels, generate_synthetic_rescues, split_data


class LabelTests(unittest.TestCase):
    def test_both_binary_tasks_keep_original_expert_labels(self):
        original = pd.Series([-1, 0, 1, -1])
        self.assertEqual(binary_labels(original, "easy").tolist(), [1, 0, 0, 1])
        self.assertEqual(binary_labels(original, "hard").tolist(), [0, 0, 1, 0])
        self.assertEqual(original.tolist(), [-1, 0, 1, -1])

    def test_unknown_labels_and_unknown_task_are_rejected(self):
        for labels in [pd.Series([-1, np.nan, 1]), pd.Series([-1, 2, 1])]:
            with self.assertRaises(ValueError):
                binary_labels(labels, "easy")
        with self.assertRaises(ValueError):
            binary_labels(pd.Series([-1, 0, 1]), "medium")


class SplitTests(unittest.TestCase):
    def test_no_rescue_can_cross_evaluation_boundaries(self):
        data = generate_synthetic_rescues()
        splits = split_data(data)
        ids = {name: set(frame["rescue_id"]) for name, frame in splits.items()}
        self.assertTrue(ids["train"].isdisjoint(ids["validation"]))
        self.assertTrue(ids["train"].isdisjoint(ids["test"]))
        self.assertTrue(ids["validation"].isdisjoint(ids["test"]))
        self.assertEqual(set.union(*ids.values()), set(data["rescue_id"]))
        self.assertEqual({name: len(frame) for name, frame in splits.items()},
                         {"train": 600, "validation": 200, "test": 200})
        for frame in splits.values():
            for task in ["easy", "hard"]:
                self.assertEqual(set(binary_labels(frame["label"], task)), {0, 1})

    def test_duplicate_rescue_ids_are_rejected(self):
        data = generate_synthetic_rescues()
        data.loc[1, "rescue_id"] = data.loc[0, "rescue_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            split_data(data)

    def test_split_is_reproducible(self):
        data = generate_synthetic_rescues()
        first, second = split_data(data), split_data(data)
        for name in first:
            self.assertEqual(first[name]["rescue_id"].tolist(), second[name]["rescue_id"].tolist())

    def test_small_or_incomplete_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            generate_synthetic_rescues(20)
        with self.assertRaisesRegex(ValueError, "missing"):
            split_data(generate_synthetic_rescues().drop(columns=["travel_km"]))
        data = generate_synthetic_rescues()
        with self.assertRaisesRegex(ValueError, "three"):
            split_data(data[data["label"] != 0])


if __name__ == "__main__":
    unittest.main()
