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

from burn_emulator.models.unet import OutConv


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
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            if kernel_size[0] != kernel_size[1]:
                raise NotImplementedError("Kernel_size h must be equal to w")
            kernel_size = kernel_size[0]
        if kernel_size % 2 != 1:
            print("Kernel_size must be even, %d was given" % kernel_size)
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
            print("Kernel_size must be 1 or 3, %d was given" % kernel_size)
            raise NotImplementedError("Kernel_size must be 1 or 3")
        self.weight = Parameter(
            torch.empty(out_channels, self.in_channel_group, self.kernel_size, self.kernel_size)
        )
        self.init_weights()
        if self.kernel_size != 1:
            w_transform_matrix: Tensor = self.get_w_transform_matrix()
            self.register_buffer("w_transform_matrix", w_transform_matrix)

    def forward(self, x: Tensor) -> Tensor:
        w_size: torch.Size = self.weight.shape
        w: Tensor = self.weight
        if self.kernel_size != 1:
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


class DoubleCircleConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: int | None = None
    ) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            CircleConv3x3(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            CircleConv3x3(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)


class CircleDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleCircleConv(in_channels, out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.maxpool_conv(x)


class CircleUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        self.bilinear: bool = bilinear
        if bilinear:
            self.conv = DoubleCircleConv(in_channels, out_channels, in_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleCircleConv(in_channels, out_channels)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        target_size: torch.Size = x2.shape[2:]
        if self.bilinear:
            x1 = nn.functional.interpolate(
                x1, size=target_size, mode="bilinear", align_corners=True
            )
        else:
            x1 = self.up(x1, output_size=target_size)

        x: Tensor = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CircleNet(nn.Module):
    def __init__(self, n_channels: int, n_outputs: int, bilinear: bool = True) -> None:
        super().__init__()
        self.n_channels: int = n_channels
        self.n_outputs: int = n_outputs
        self.bilinear: bool = bilinear

        self.inc = DoubleCircleConv(n_channels, 64)
        self.down1 = CircleDown(64, 128)
        self.down2 = CircleDown(128, 256)
        self.down3 = CircleDown(256, 512)
        factor: int = 2 if bilinear else 1
        self.down4 = CircleDown(512, 1024 // factor)
        self.up1 = CircleUp(1024, 512 // factor, bilinear)
        self.up2 = CircleUp(512, 256 // factor, bilinear)
        self.up3 = CircleUp(256, 128 // factor, bilinear)
        self.up4 = CircleUp(128, 64, bilinear)
        self.outc = OutConv(64, n_outputs)

    def forward(self, x: Tensor) -> Tensor:
        x1: Tensor = self.inc(x)
        x2: Tensor = self.down1(x1)
        x3: Tensor = self.down2(x2)
        x4: Tensor = self.down3(x3)
        x5: Tensor = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits: Tensor = self.outc(x)
        return logits
