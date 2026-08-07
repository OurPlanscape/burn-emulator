import time
import unittest

import torch
from torchinfo import summary

from burn_emulator.constants import DEFAULT_DTYPE
from burn_emulator.models.gnn import RadialGNN

from tests.fixtures import (
    TEST_BATCH_SIZE,
    TEST_IMAGE_SIZE,
    TEST_IN_CHANS,
    TEST_OUT_CHANS,
)




class TestRadialGNN(unittest.TestCase):
    """Unit tests for the RadialGNN model: graph structure, forward pass, memory, timing."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.B = TEST_BATCH_SIZE
        cls.GRID = TEST_IMAGE_SIZE
        cls.C_in = TEST_IN_CHANS
        cls.N_OUTPUTS = 3  # RadialGNN output channels are fixed at 1

        # cls.model = RadialGNN(
        #     img_channels=cls.C_in,
        #     hidden_channels=64,
        #     out_channels=cls.N_OUTPUTS,
        #     num_layers=(64, 32, 16),
        #     grid_size=cls.GRID,
        #     refine_ch=64,
        #     ring_scales=(64, 32, 16),
        #     grid_outward=(True, False, False),
        #     src_ratio=(None, 0.5, 0.25),
        #     n_dst=(None, 4, 4),
        #     n_neighbors=(None, 2, 2),
        #     lateral_edge_dropout=0.3,
        #     outward_edge_dropout=0.0,
        #     use_slope_gate=(True, False, False),
        #     train_batch_size=cls.B,
        # ).to(device=cls.device)
        cls.uncompiled_model = RadialGNN(
            img_channels=cls.C_in,
            hidden_channels=64,
            out_channels=cls.N_OUTPUTS,
            num_layers=(64,),
            grid_size=cls.GRID,
            refine_ch=64,
            ring_scales=(64,),
            grid_outward=(True,),
            src_ratio=(None,),
            n_dst=(None,),
            n_neighbors=(None,),
            lateral_edge_dropout=0.3,
            outward_edge_dropout=0.0,
            use_slope_gate=(True,),
            train_batch_size=cls.B,
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

    def test_00_branch_edge_structure(self):
        """Each branch reports a valid edge structure: same-ring edges, and outward
        edges that actually point from inner to outer rings (no inward edges)."""
        self.assertGreater(len(self.model.branches), 0)

        print()
        for i, branch in enumerate(self.model.branches):
            e_same = branch.same_edge_index.shape[1]
            e_out = branch.out_edge_index.shape[1]
            src_r = branch.ring_id[branch.out_edge_index[0]]
            dst_r = branch.ring_id[branch.out_edge_index[1]]
            inward = (dst_r < src_r).sum().item()
            print(
                f"  Branch {i} rings={branch.num_rings:2d} : "
                f"same={e_same}, outward={e_out}, inward={inward} <- 0"
            )

            self.assertGreaterEqual(e_same, 0)
            self.assertGreaterEqual(e_out, 0)
            self.assertEqual(
                inward, 0, f"Branch {i} has {inward} inward edges; expected 0"
            )

    def test_01_parameter_breakdown(self):
        """Model builds and every named submodule reports a positive parameter count."""
        total = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total, 0)

        print("\nParameter breakdown:")
        for i, branch in enumerate(self.model.branches):
            branch_params = sum(p.numel() for p in branch.parameters())
            print(f"  Branch {i} (rings={branch.num_rings:2d}) : {branch_params:>10,}")
            self.assertGreater(branch_params, 0, f"Branch {i} has zero parameters")

        scale_proj_params = sum(p.numel() for p in self.model.scale_proj.parameters())
        decoder_params = sum(p.numel() for p in self.model.decoder.parameters())
        print(f"  scale_proj   : {scale_proj_params:>10,}")
        print(f"  PixelDecoder : {decoder_params:>10,}")
        print(f"  Total        : {total:>10,}")

        self.assertGreater(scale_proj_params, 0)
        self.assertGreater(decoder_params, 0)

    def test_02_torchinfo_summary_runs(self):
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

    def test_03_forward_pass_output_shape(self):
        """Forward pass produces the expected output shape."""
        with torch.no_grad():
            pred = self.model(self.image)
        self.assertEqual(pred.shape, (self.B, self.N_OUTPUTS, self.GRID, self.GRID))

    def test_04_output_is_finite(self):
        """Model output contains no NaNs or Infs, and sigmoid probs are in [0, 1]."""
        with torch.no_grad():
            pred = self.model(self.image)
        self.assertTrue(torch.isfinite(pred).all(), "Output contains NaN/Inf values")

        burn_prob = torch.sigmoid(pred)
        self.assertGreaterEqual(burn_prob.min().item(), 0.0)
        self.assertLessEqual(burn_prob.max().item(), 1.0)

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