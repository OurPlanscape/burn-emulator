import math
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from numpy.typing import NDArray
from torch import Tensor
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

from burn_emulator.models.utils import circular_components


class CircleLayerBase(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | list[int] | tuple[int, int] = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, (list, tuple)):
            if kernel_size[0] != kernel_size[1]:
                raise NotImplementedError("Kernel_size h must be equal to w")
            kernel_size = kernel_size[0]
        if kernel_size % 2 != 1:
            print(f"Kernel_size must be even, {kernel_size} was given")
            raise NotImplementedError("Kernel_size must be even")
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel: tuple[int, int] = _pair(kernel_size)
        self.kernel_size: int = kernel_size
        self.padding_size: int = padding

        self.padding: tuple[int, int] = _pair(padding)
        self.stride: tuple[int, int] = _pair(stride)
        self.dilation: tuple[int, int] = _pair(dilation)
        self.groups: int = groups
        self.in_channel_group: int = self.in_channels // groups
        if bias:
            self.bias: Parameter | None = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
            self.bias = None

    def init_weights(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    def to_0_1(self, x: float, grid_x: int) -> tuple[float, float]:
        if grid_x < 0:
            x = -x
        pos: float = x
        x = x - np.floor(x)
        return x, pos

    def bilinear_interpolation(self, px: float, py: float) -> tuple[float, float, float, float]:
        return (1 - px) * (py), (px) * (py), (1 - px) * (1 - py), (px) * (1 - py)

    def coordinate_to_index(self, x: int, y: int, center: int) -> int:
        return x + center + self.kernel_size * (-y + center)

    def append_a_weight(
        self,
        angle: float,
        grid_x: int,
        grid_y: int,
        center: int,
        select_x_indexes: list[list[int]],
        weights: list[tuple[float, float, float, float]],
        dist_to_center: float,
    ) -> None:
        radius: float = np.floor(dist_to_center)

        x, posx = self.to_0_1(radius * np.cos(angle), grid_x)
        y, posy = self.to_0_1(radius * np.sin(angle), grid_y)
        w: tuple[float, float, float, float] = self.bilinear_interpolation(x, y)

        if grid_x > 0:
            tl_x: int = grid_x - 1
        else:
            tl_x = grid_x
        if grid_y < 0:
            tl_y: int = grid_y + 1
        else:
            tl_y = grid_y
        select_x_indexes.append(
            [
                self.coordinate_to_index(tl_x, tl_y, center),
                self.coordinate_to_index(tl_x + 1, tl_y, center),
                self.coordinate_to_index(tl_x, tl_y - 1, center),
                self.coordinate_to_index(tl_x + 1, tl_y - 1, center),
            ]
        )
        weights.append(w)

    @abstractmethod
    def init_bilinear_weights(self) -> tuple[NDArray[np.float64], list[list[int]]]:
        pass

    def get_w_transform_matrix(
        self,
        alpha: NDArray[np.float64] | None = None,
        select_x_indexes: list[list[int]] | None = None,
    ) -> Tensor:
        if alpha is None or select_x_indexes is None:
            alpha, select_x_indexes = self.init_bilinear_weights()
        w_transform_matrix: list[list[float]] = []
        alpha_index: int = 0
        for i in range(len(select_x_indexes)):
            cur_row: list[float] = [0 for _ in range(self.kernel_size * self.kernel_size)]
            if len(select_x_indexes[i]) == 1:
                cur_row[select_x_indexes[i][0]] = 1
            else:
                for index, j in enumerate(select_x_indexes[i]):
                    cur_row[j] = alpha[alpha_index, index]
                alpha_index += 1
            w_transform_matrix.append(cur_row)
        return torch.tensor(w_transform_matrix, dtype=torch.float)

    def print_w_transform_matrix(self) -> None:
        print(self.w_transform_matrix)


class CircleConv3x3(CircleLayerBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        if self.kernel_size != 3 and self.kernel_size != 1:
            print(f"Kernel_size must be 1 or 3, {kernel_size} was given")
            raise NotImplementedError("Kernel_size must be 1 or 3")
        self.weight = Parameter(
            torch.empty(out_channels, self.in_channel_group, self.kernel_size, self.kernel_size)
        )
        self.init_weights()
        if self.kernel_size != 1:
            w_transform_matrix: Tensor = self.get_w_transform_matrix()
            self.register_buffer("w_transform_matrix", w_transform_matrix)

            # this saves 3ms per forward pass for B=32 inference
            if not self.training:
                self.weight = self.weight.view(-1, self.kernel_size * self.kernel_size)
                self.weight = self.weight.matmul(self.w_transform_matrix)

    def forward(self, x: Tensor) -> Tensor:
        w_size: torch.Size = self.weight.shape
        w: Tensor = self.weight
        if self.kernel_size != 1 and self.training:
            w = w.view(-1, self.kernel_size * self.kernel_size)
            w = w.matmul(self.w_transform_matrix)
        w = w.view(w_size[0], w_size[1], self.kernel_size, self.kernel_size)
        return nn.functional.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, groups=self.groups
        )

    def init_bilinear_weights(self) -> tuple[NDArray[np.float64], list[list[int]]]:
        select_x_indexes: list[list[int]] = []
        weights: list[tuple[float, float, float, float]] = []
        center: int = self.kernel_size // 2
        for grid_y in range(center, -(center + 1), -1):
            for grid_x in range(-center, center + 1):
                if grid_y == 0 or grid_x == 0:
                    select_x_indexes.append([self.coordinate_to_index(grid_x, grid_y, center)])
                    continue
                dist_to_center: float = np.sqrt(np.power(grid_x, 2) + np.power(grid_y, 2))
                angle: float = np.arctan(np.abs(grid_y / grid_x))
                self.append_a_weight(
                    angle, grid_x, grid_y, center, select_x_indexes, weights, dist_to_center
                )

        return np.array(weights), select_x_indexes


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, num_channels: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(cond_dim * 4, 16)
        self.num_channels = num_channels
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_channels * 2),
        )
        init.zeros_(self.net[-1].weight)
        init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        gamma_beta: Tensor = self.net(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = (1.0 + gamma).view(-1, self.num_channels, 1, 1)
        beta = beta.view(-1, self.num_channels, 1, 1)
        return gamma * x + beta


class DoubleCircleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.cond_dim = cond_dim

        self.conv1 = CircleConv3x3(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = CircleConv3x3(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        if cond_dim is not None:
            self.film1 = FiLM(cond_dim, mid_channels)
            self.film2 = FiLM(cond_dim, out_channels)

    def forward(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        if self.cond_dim is not None:
            assert cond is not None, "cond_dim was set but no cond tensor was passed"
            x = self.film1(x, cond)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.cond_dim is not None:
            x = self.film2(x, cond)
        x = self.relu2(x)
        return x


class CircleDown(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        out_size: int | tuple[int, int],
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d(out_size)
        self.conv = DoubleCircleConv(in_channels, out_channels, cond_dim=cond_dim)

    def forward(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        x = self.pool(x)
        return self.conv(x, cond)


class CircleUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        self.bilinear: bool = bilinear
        if not bilinear:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: Tensor, target_size: torch.Size) -> Tensor:
        if self.bilinear:
            return nn.functional.interpolate(
                x, size=target_size, mode="bilinear", align_corners=True
            )
        return self.up(x, output_size=target_size)


class AngleEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, angle_degrees: Tensor) -> Tensor:
        sin, cos = circular_components(angle_degrees)
        return torch.stack([sin, cos], dim=-1)


class CirclePP(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_outputs: int,
        image_size: int,
        bilinear: bool = True,
        cond_dim: int | None = 2,
    ) -> None:
        super().__init__()
        self.n_channels: int = n_channels
        self.n_outputs: int = n_outputs
        self.image_size: int = image_size
        self.bilinear: bool = bilinear
        self.cond_dim: int | None = cond_dim

        if cond_dim is not None:
            self.angle_encoder = AngleEncoder()

        factor: int = 2 if bilinear else 1
        nb_filter: list[int] = [64, 128, 256, 512, 1024 // factor]
        self.nb_filter: list[int] = nb_filter

        down_sizes: list[int] = [image_size // 2**depth for depth in range(1, 5)]

        self.conv0_0 = DoubleCircleConv(n_channels, nb_filter[0], cond_dim=cond_dim)
        self.conv1_0 = CircleDown(nb_filter[0], nb_filter[1], down_sizes[0], cond_dim=cond_dim)
        self.conv2_0 = CircleDown(nb_filter[1], nb_filter[2], down_sizes[1], cond_dim=cond_dim)
        self.conv3_0 = CircleDown(nb_filter[2], nb_filter[3], down_sizes[2], cond_dim=cond_dim)
        self.conv4_0 = CircleDown(nb_filter[3], nb_filter[4], down_sizes[3], cond_dim=cond_dim)

        self.up1_0 = CircleUpsample(nb_filter[1], nb_filter[0], bilinear)
        self.up2_0 = CircleUpsample(nb_filter[2], nb_filter[1], bilinear)
        self.up3_0 = CircleUpsample(nb_filter[3], nb_filter[2], bilinear)
        self.up4_0 = CircleUpsample(nb_filter[4], nb_filter[3], bilinear)

        uc0: int = nb_filter[1] if bilinear else nb_filter[0]
        uc1: int = nb_filter[2] if bilinear else nb_filter[1]
        uc2: int = nb_filter[3] if bilinear else nb_filter[2]
        uc3: int = nb_filter[4] if bilinear else nb_filter[3]

        self.conv0_1 = DoubleCircleConv(nb_filter[0] * 1 + uc0, nb_filter[0], cond_dim=cond_dim)
        self.conv1_1 = DoubleCircleConv(nb_filter[1] * 1 + uc1, nb_filter[1], cond_dim=cond_dim)
        self.conv2_1 = DoubleCircleConv(nb_filter[2] * 1 + uc2, nb_filter[2], cond_dim=cond_dim)
        self.conv3_1 = DoubleCircleConv(nb_filter[3] * 1 + uc3, nb_filter[3], cond_dim=cond_dim)

        self.conv0_2 = DoubleCircleConv(nb_filter[0] * 2 + uc0, nb_filter[0], cond_dim=cond_dim)
        self.conv1_2 = DoubleCircleConv(nb_filter[1] * 2 + uc1, nb_filter[1], cond_dim=cond_dim)
        self.conv2_2 = DoubleCircleConv(nb_filter[2] * 2 + uc2, nb_filter[2], cond_dim=cond_dim)

        self.conv0_3 = DoubleCircleConv(nb_filter[0] * 3 + uc0, nb_filter[0], cond_dim=cond_dim)
        self.conv1_3 = DoubleCircleConv(nb_filter[1] * 3 + uc1, nb_filter[1], cond_dim=cond_dim)

        self.conv0_4 = DoubleCircleConv(nb_filter[0] * 4 + uc0, nb_filter[0], cond_dim=cond_dim)

        self.outc = nn.Sequential(
            CircleConv3x3(nb_filter[0], nb_filter[0]), CircleConv3x3(nb_filter[0], n_outputs)
        )

    def forward(self, x: Tensor, angle_degrees: Tensor | None = None) -> Tensor:
        cond: Tensor | None = None
        if self.cond_dim is not None:
            assert angle_degrees is not None, "cond_dim is set but no angle_degrees was passed"
            cond = self.angle_encoder(angle_degrees)  # (B, 2) = [sin, cos]

        x0_0: Tensor = self.conv0_0(x, cond)
        x1_0: Tensor = self.conv1_0(x0_0, cond)
        x0_1: Tensor = self.conv0_1(
            torch.cat([x0_0, self.up1_0(x1_0, x0_0.shape[2:])], dim=1), cond
        )

        x2_0: Tensor = self.conv2_0(x1_0, cond)
        x1_1: Tensor = self.conv1_1(
            torch.cat([x1_0, self.up2_0(x2_0, x1_0.shape[2:])], dim=1), cond
        )
        x0_2: Tensor = self.conv0_2(
            torch.cat([x0_0, x0_1, self.up1_0(x1_1, x0_0.shape[2:])], dim=1), cond
        )

        x3_0: Tensor = self.conv3_0(x2_0, cond)
        x2_1: Tensor = self.conv2_1(
            torch.cat([x2_0, self.up3_0(x3_0, x2_0.shape[2:])], dim=1), cond
        )
        x1_2: Tensor = self.conv1_2(
            torch.cat([x1_0, x1_1, self.up2_0(x2_1, x1_0.shape[2:])], dim=1), cond
        )
        x0_3: Tensor = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, self.up1_0(x1_2, x0_0.shape[2:])], dim=1), cond
        )

        x4_0: Tensor = self.conv4_0(x3_0, cond)
        x3_1: Tensor = self.conv3_1(
            torch.cat([x3_0, self.up4_0(x4_0, x3_0.shape[2:])], dim=1), cond
        )
        x2_2: Tensor = self.conv2_2(
            torch.cat([x2_0, x2_1, self.up3_0(x3_1, x2_0.shape[2:])], dim=1), cond
        )
        x1_3: Tensor = self.conv1_3(
            torch.cat([x1_0, x1_1, x1_2, self.up2_0(x2_2, x1_0.shape[2:])], dim=1), cond
        )
        x0_4: Tensor = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_0(x1_3, x0_0.shape[2:])], dim=1), cond
        )

        logits: Tensor = self.outc(x0_4)
        return logits


class CirclePPSiamese(CirclePP):
    def _encode(
        self, x: Tensor, cond: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x0_0: Tensor = self.conv0_0(x, cond)
        x1_0: Tensor = self.conv1_0(x0_0, cond)
        x2_0: Tensor = self.conv2_0(x1_0, cond)
        x3_0: Tensor = self.conv3_0(x2_0, cond)
        x4_0: Tensor = self.conv4_0(x3_0, cond)
        return x0_0, x1_0, x2_0, x3_0, x4_0

    @staticmethod
    def _diff(baseline: Tensor, change: Tensor) -> Tensor:
        return change - baseline

    def forward(
        self,
        x: Tensor,
        angle_degrees: Tensor | None = None,
    ) -> Tensor:
        cond: Tensor | None = None
        if self.cond_dim is not None:
            assert angle_degrees is not None, "cond_dim is set but no angle_degrees was passed"
            cond = self.angle_encoder(angle_degrees)  # (B, 2) = [sin, cos]

        a0_0, a1_0, a2_0, a3_0, a4_0 = self._encode(x[:, 0], cond)
        b0_0, b1_0, b2_0, b3_0, b4_0 = self._encode(x[:, 1], cond)

        x0_0: Tensor = self._diff(a0_0, b0_0)
        x1_0: Tensor = self._diff(a1_0, b1_0)
        x2_0: Tensor = self._diff(a2_0, b2_0)
        x3_0: Tensor = self._diff(a3_0, b3_0)
        x4_0: Tensor = self._diff(a4_0, b4_0)

        x0_1: Tensor = self.conv0_1(
            torch.cat([x0_0, self.up1_0(x1_0, x0_0.shape[2:])], dim=1), cond
        )

        x1_1: Tensor = self.conv1_1(
            torch.cat([x1_0, self.up2_0(x2_0, x1_0.shape[2:])], dim=1), cond
        )
        x0_2: Tensor = self.conv0_2(
            torch.cat([x0_0, x0_1, self.up1_0(x1_1, x0_0.shape[2:])], dim=1), cond
        )

        x2_1: Tensor = self.conv2_1(
            torch.cat([x2_0, self.up3_0(x3_0, x2_0.shape[2:])], dim=1), cond
        )
        x1_2: Tensor = self.conv1_2(
            torch.cat([x1_0, x1_1, self.up2_0(x2_1, x1_0.shape[2:])], dim=1), cond
        )
        x0_3: Tensor = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, self.up1_0(x1_2, x0_0.shape[2:])], dim=1), cond
        )

        x3_1: Tensor = self.conv3_1(
            torch.cat([x3_0, self.up4_0(x4_0, x3_0.shape[2:])], dim=1), cond
        )
        x2_2: Tensor = self.conv2_2(
            torch.cat([x2_0, x2_1, self.up3_0(x3_1, x2_0.shape[2:])], dim=1), cond
        )
        x1_3: Tensor = self.conv1_3(
            torch.cat([x1_0, x1_1, x1_2, self.up2_0(x2_2, x1_0.shape[2:])], dim=1), cond
        )
        x0_4: Tensor = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_0(x1_3, x0_0.shape[2:])], dim=1), cond
        )

        logits: Tensor = self.outc(x0_4)
        return logits
