import time
import unittest

import torch
from torchinfo import summary


from burn_emulator.constants import (
    DEFAULT_DTYPE,
    TEST_BATCH_SIZE,
    TEST_IMAGE_SIZE,
    TEST_IN_CHANS,
)
from burn_emulator.models.vit import PixelViT  # adjust import path as needed


class TestPixelViT(unittest.TestCase):
    """Unit tests for the PixelViT model: structure, forward pass, memory, timing."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.B = TEST_BATCH_SIZE
        cls.GRID = TEST_IMAGE_SIZE
        cls.C_in = TEST_IN_CHANS
        cls.N_OUTPUTS = TEST_OUT_CHANS

        cls.model = PixelViT(
            img_size=cls.GRID,
            in_chans=cls.C_in,
            radius=64,
            embed_dim=64,
            depth=16,
            num_heads=8,
            num_classes=cls.N_OUTPUTS,
            drop_path_rate=0.1,
        ).to(device=cls.device)
        cls.model.eval()

        # Build a representative input tensor once, reused across tests.
        cls.model.to(DEFAULT_DTYPE)
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
        """Model builds and every named submodule/param reports a positive count."""
        total = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total, 0)

        print("\nParameter breakdown:")
        for name, module in [
            ("pixel_embed", self.model.pixel_embed),
            ("blocks", self.model.blocks),
            ("norm", self.model.norm),
            ("head", self.model.head),
        ]:
            params = sum(p.numel() for p in module.parameters())
            print(f"  {name:<12} : {params:>10,}")
            self.assertGreater(params, 0, f"{name} has zero parameters")

        named_extras = {
            "cls_token": self.model.cls_token.numel(),
            "pos_embed": self.model.pos_embed.numel(),
        }
        for name, params in named_extras.items():
            print(f"  {name:<12} : {params:>10,}")
            self.assertGreater(params, 0, f"{name} has zero elements")

        print(f"  {'Total':<12} : {total:>10,}")
        print(
            f"  (tokens kept in circle: {self.model.num_pixels:,} / "
            f"{self.GRID * self.GRID:,})"
        )

        self.assertGreater(self.model.num_pixels, 0)
        self.assertLessEqual(self.model.num_pixels, self.GRID * self.GRID)

    def test_01_torchinfo_summary_runs(self):
        """torchinfo.summary can trace the model at the expected input size without error."""
        self.model.to(torch.float32)
        info = summary(
            self.model,
            input_size=(self.B, self.C_in, self.GRID, self.GRID),
            device=self.device,
            col_names=["input_size", "output_size", "num_params"],
            depth=3,
            mode="eval",
            verbose=0,
        )
        self.model.to(DEFAULT_DTYPE)
        self.assertIsNotNone(info)

    def test_02_forward_pass_output_shape(self):
        """Forward pass produces the expected output shape."""
        with torch.no_grad():
            pred = self.model(self.image)
        self.assertEqual(pred.shape, (self.B, self.N_OUTPUTS, self.GRID, self.GRID))

    def test_03_output_is_finite_and_softmax_valid(self):
        """Model output contains no NaNs/Infs, and softmax class probs are in [0, 1]."""
        with torch.no_grad():
            pred = self.model(self.image)
        self.assertTrue(torch.isfinite(pred).all(), "Output contains NaN/Inf values")

        class_probs = torch.softmax(pred, dim=1)
        print(
            f"\nClass prob range (sampled pixels only) : "
            f"[{class_probs.min():.3f}, {class_probs.max():.3f}]"
        )
        self.assertGreaterEqual(class_probs.min().item(), 0.0)
        self.assertLessEqual(class_probs.max().item(), 1.0)

        # Softmax over the class dim should sum to 1 at every pixel.
        sums = class_probs.sum(dim=1)
        torch.testing.assert_close(
            sums, torch.ones_like(sums), rtol=1e-4, atol=1e-4
        )

    @unittest.skip(
        "Disabled: relies on model.mask, which was commented out in the source script. "
        "Enable once mask export from PixelViT is confirmed stable."
    )
    def test_04_unsampled_pixels_are_zero(self):
        """Predictions outside the sampled circular mask should be exactly 0."""
        with torch.no_grad():
            pred = self.model(self.image)
        unsampled = ~self.model.mask  # (GRID, GRID)
        self.assertTrue(
            torch.all(pred[:, :, unsampled] == 0),
            "Found nonzero predictions outside the sampled circle",
        )

    def test_05_forward_pass_timing(self):
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
    def test_06_gpu_memory_within_budget(self):
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