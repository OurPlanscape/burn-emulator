import unittest
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from burn_emulator.constants import DEFAULT_DTYPE
from burn_emulator.datasets import IgnitionDataset


def _load_init_args() -> dict:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return config["dataset"]["init_args"]


def _required_paths(init_args: dict) -> list[Path]:
    """All filesystem paths the dataset needs to actually build."""
    paths = [
        init_args["ignitions_path"],
        init_args["topo_path"],
        init_args["stats_path"],
        *init_args["fuels_paths"],
    ]
    if init_args.get("burn_paths"):
        paths.extend(init_args["burn_paths"])
    return [Path(p) for p in paths]


class TestIgnitionDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.init_args = _load_init_args()

        missing = [p for p in _required_paths(cls.init_args) if not p.exists()]
        if missing:
            raise unittest.SkipTest(
                "Skipping IgnitionDataset tests: missing data paths: "
                f"{[str(p) for p in missing]}"
            )

        cls.chip_size = cls.init_args["chip_size"]
        cls.n_burn_times = len(cls.init_args["burn_times"])
        cls.dataset = IgnitionDataset(**cls.init_args)

    def test_00_dataset_builds_and_is_nonempty(self):
        """Dataset constructs from the config and reports a sane, positive length."""
        self.assertGreater(len(self.dataset), 0)

    def test_01_len_matches_ignitions_times_burn_dirs(self):
        """__len__ should equal n_ignitions * n_burn_paths when burn_paths is set."""
        expected = len(self.dataset.ignitions) * len(self.dataset.burn_paths)
        self.assertEqual(len(self.dataset), expected)

    def test_02_getitem_returns_expected_shapes_and_dtypes(self):
        """A single sample has the right chip size, burn-time channel count, and dtypes."""
        arrX, arrY, mask = self.dataset[0]

        self.assertEqual(arrX.ndim, 3)
        self.assertEqual(tuple(arrX.shape[-2:]), (self.chip_size, self.chip_size))
        self.assertEqual(arrX.dtype, DEFAULT_DTYPE)

        self.assertEqual(tuple(arrY.shape[-2:]), (self.chip_size, self.chip_size))
        self.assertEqual(arrY.shape[0], self.n_burn_times)
        self.assertEqual(arrY.dtype, torch.bool)

        self.assertEqual(tuple(mask.shape[-2:]), (self.chip_size, self.chip_size))
        self.assertEqual(mask.dtype, torch.bool)

    def test_03_getitem_is_deterministic_without_jitter(self):
        """With jitter=None (per config), repeated reads of the same index match exactly."""
        arrX0, arrY0, mask0 = self.dataset[0]
        arrX1, arrY1, mask1 = self.dataset[0]

        torch.testing.assert_close(arrX0, arrX1)
        self.assertTrue(torch.equal(arrY0, arrY1))
        self.assertTrue(torch.equal(mask0, mask1))

    def test_04_circle_mask_excludes_outside_pixels(self):
        """When circle_mask=True (the default here), sampling mask must be False
        anywhere outside the inscribed circle."""
        self.assertTrue(self.dataset.circle_mask)
        _, _, mask = self.dataset[0]

        outside = ~self.dataset.cmask.bool()
        self.assertTrue(torch.all(~mask[:, outside]))

    def test_05_multiple_indices_do_not_error(self):
        """Sample a spread of indices, including near dataset boundaries, without error."""
        n = len(self.dataset)
        indices = sorted({0, 1, n // 2, n - 2, n - 1})
        for idx in indices:
            arrX, _, _ = self.dataset[idx]
            self.assertEqual(tuple(arrX.shape[-2:]), (self.chip_size, self.chip_size))

    def test_06_dataloader_batches_correctly(self):
        """Dataset is batchable via the default collate function (uniform chip shapes)."""
        loader = DataLoader(self.dataset, batch_size=4, shuffle=False)
        arrX, arrY, mask = next(iter(loader))

        self.assertEqual(arrX.shape[0], 4)
        self.assertEqual(tuple(arrX.shape[-2:]), (self.chip_size, self.chip_size))
        self.assertEqual(mask.shape[0], 4)


class TestIgnitionDatasetInferenceMode(unittest.TestCase):
    """Covers the burn_paths=None branch of __getitem__, used for inference
    (no ground-truth burn labels available)."""

    @classmethod
    def setUpClass(cls):
        init_args = dict(_load_init_args())
        init_args["burn_paths"] = None

        missing = [p for p in _required_paths(init_args) if not p.exists()]
        if missing:
            raise unittest.SkipTest(
                "Skipping inference-mode tests: missing data paths: "
                f"{[str(p) for p in missing]}"
            )

        cls.chip_size = init_args["chip_size"]
        cls.dataset = IgnitionDataset(**init_args)

    def test_00_len_matches_ignitions_only(self):
        """Without burn_paths, __len__ should equal n_ignitions (no burn-dir multiplier)."""
        self.assertEqual(len(self.dataset), len(self.dataset.ignitions))

    def test_01_getitem_returns_inference_tuple(self):
        """Inference-mode getitem returns arrX, mask, pad-diffs, window bounds, and indices."""
        arrX, mask, (ydiff, xdiff), (ymin, ymax, xmin, xmax), (sidx, bidx) = self.dataset[0]

        self.assertEqual(tuple(arrX.shape[-2:]), (self.chip_size, self.chip_size))
        self.assertEqual(tuple(mask.shape[-2:]), (self.chip_size, self.chip_size))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertIsInstance(sidx, int)
        self.assertIsInstance(bidx, int)
        self.assertGreaterEqual(ydiff, 0)
        self.assertGreaterEqual(xdiff, 0)


if __name__ == "__main__":
    unittest.main()