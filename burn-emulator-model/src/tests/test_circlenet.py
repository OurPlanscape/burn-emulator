import time
import unittest

import torch
from torchinfo import summary

from burn_emulator.constants import DEFAULT_DTYPE
from burn_emulator.models.circlenet import CircleNet

from tests.fixtures import (
    TEST_BATCH_SIZE,
    TEST_IMAGE_SIZE,
    TEST_IN_CHANS,
    TEST_OUT_CHANS,
)


class TestCircleNet(unittest.TestCase):
    """Unit tests for the CircleNet model: structure, forward pass, memory, timing."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.B = TEST_BATCH_SIZE
        cls.GRID = TEST_IMAGE_SIZE
        cls.C_in = TEST_IN_CHANS
        cls.N_OUTPUTS = TEST_OUT_CHANS

        cls.uncompiled_model = CircleNet(
            n_channels=cls.C_in,
            n_outputs=cls.N_OUTPUTS,
            bilinear=True,
        ).to(device=cls.device, dtype=DEFAULT_DTYPE)
        cls.uncompiled_model.eval()
        cls.model = torch.compile(cls.uncompiled_model)

        # Build a representative input tensor once, reused across tests.
        image = torch.zeros(
            cls.B, cls.C_in, cls.GRID, cls.GRID, device=cls.device, dtype=DEFAULT_DTYPE
        )
        image[:, 0] = (
            torch.rand(cls.B, cls.GRID, cls.GRID, device=cls.device) * 2 - 1
        ).to(DEFAULT_DTYPE) * 0.731  # flow_x
        image[:, 1] = (
            torch.rand(cls.B, cls.GRID, cls.GRID, device=cls.device) * 2 - 1
        ).to(DEFAULT_DTYPE) * 0.731  # flow_y
        image[:, 2:] = torch.rand(
            cls.B, cls.C_in - 2, cls.GRID, cls.GRID, device=cls.device
        ).to(DEFAULT_DTYPE)
        cls.image = image

    def test_00_parameter_breakdown(self):
        """Model builds and every named submodule reports a positive parameter count."""
        total = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total, 0)

        submodules = [
            ("conv0_0", self.model.conv0_0),
            ("conv1_0", self.model.conv1_0),
            ("conv2_0", self.model.conv2_0),
            ("conv3_0", self.model.conv3_0),
            ("conv4_0", self.model.conv4_0),
            ("conv0_1", self.model.conv0_1),
            ("conv1_1", self.model.conv1_1),
            ("conv2_1", self.model.conv2_1),
            ("conv3_1", self.model.conv3_1),
            ("conv0_2", self.model.conv0_2),
            ("conv1_2", self.model.conv1_2),
            ("conv2_2", self.model.conv2_2),
            ("conv0_3", self.model.conv0_3),
            ("conv1_3", self.model.conv1_3),
            ("conv0_4", self.model.conv0_4),
            ("outc", self.model.outc),
        ]

        print("\nParameter breakdown:")
        summed = 0
        for name, module in submodules:
            params = sum(p.numel() for p in module.parameters())
            summed += params
            print(f"  {name:<6} : {params:>10,}")
            self.assertGreater(params, 0, f"{name} has zero parameters")
        print(f"  {'Total':<6} : {total:>10,}")

    def test_01_torchinfo_summary_runs(self):
        """torchinfo.summary can trace the model at the expected input size without error."""
        self.uncompiled_model.to(torch.float32)
        info = summary(
            self.uncompiled_model,
            input_size=(self.B, self.C_in, self.GRID, self.GRID),
            device=self.device,
            col_names=["input_size", "output_size", "num_params"],
            depth=3,
            mode="eval",
            verbose=0,
        )
        self.uncompiled_model.to(DEFAULT_DTYPE)
        self.assertIsNotNone(info)

    def test_02_forward_pass_output_shape(self):
        """Forward pass produces the expected output shape."""
        with torch.no_grad():
            pred = self.model(self.image)
        self.assertEqual(pred.shape, (self.B, self.N_OUTPUTS, self.GRID, self.GRID))
        # Stash for downstream tests that don't want to recompute.
        self.__class__.pred = pred

    def test_03_output_is_finite(self):
        """Model output contains no NaNs or Infs, and sigmoid probs are in [0, 1]."""
        with torch.no_grad():
            pred = self.model(self.image)
        self.assertTrue(torch.isfinite(pred).all(), "Output contains NaN/Inf values")

        burn_prob = torch.sigmoid(pred)
        self.assertGreaterEqual(burn_prob.min().item(), 0.0)
        self.assertLessEqual(burn_prob.max().item(), 1.0)

    def test_04_forward_pass_timing(self):
        """Smoke-test forward pass timing over multiple runs (informational, not a strict assertion)."""
        with torch.no_grad():
            _ = self.model(self.image)  # warmup
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        n_runs = 5
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                pred = self.model(self.image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed = (t1 - t0) / n_runs
        print(
            f"\nForward pass : {elapsed * 1000:.1f} ms "
            f"(mean over {n_runs} runs for batch size = {self.B})"
        )
        self.assertEqual(pred.shape, (self.B, self.N_OUTPUTS, self.GRID, self.GRID))

    @unittest.skipUnless(torch.cuda.is_available(), "GPU memory check requires CUDA")
    def test_05_gpu_memory_within_budget(self):
        """Peak GPU memory usage stays under 90% of total device memory."""
        torch.cuda.reset_peak_memory_stats(self.device)
        with torch.no_grad():
            _ = self.model(self.image)
        torch.cuda.synchronize()

        peak = torch.cuda.max_memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        total_mem = torch.cuda.get_device_properties(self.device).total_memory / 1024**3

        print("\nMemory (GiB):")
        print(f"  peak      : {peak:.2f}")
        print(f"  reserved  : {reserved:.2f}")
        print(f"  total     : {total_mem:.2f}")
        print(f"  headroom  : {total_mem - peak:.2f}")

        self.assertLess(
            peak,
            0.9 * total_mem,
            f"Peak memory {peak:.2f} GiB exceeds 90% of device total {total_mem:.2f} GiB",
        )


if __name__ == "__main__":
    unittest.main()