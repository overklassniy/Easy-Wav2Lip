from typing import Any

import torch
from torch import nn

from utils.logger import get_logger

logger = get_logger()


class Conv2d(nn.Module):
    """A convolutional block with Batch Normalization and ReLU activation.

    Optionally adds a residual connection.
    """

    def __init__(
            self,
            cin: int,
            cout: int,
            kernel_size: int,
            stride: int,
            padding: int,
            residual: bool = False,
            *args: Any,
            **kwargs: Any
    ) -> None:
        """
        Args:
            cin (int): Number of input channels.
            cout (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride for convolution.
            padding (int): Padding size.
            residual (bool, optional): If True, adds the input to the output. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        # Create a sequential block with convolution and batch normalization.
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()
        self.residual = residual
        logger.debug(f"Initialized Conv2d with residual={residual}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the convolutional block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after convolution, optional residual addition, and activation.
        """
        out = self.conv_block(x)
        if self.residual:
            out += x  # Residual connection.
        return self.act(out)


class nonorm_Conv2d(nn.Module):
    """A convolutional block without normalization using LeakyReLU activation."""

    def __init__(
            self,
            cin: int,
            cout: int,
            kernel_size: int,
            stride: int,
            padding: int,
            residual: bool = False,
            *args: Any,
            **kwargs: Any
    ) -> None:
        """
        Args:
            cin (int): Number of input channels.
            cout (int): Number of output channels.
            kernel_size (int): Kernel size.
            stride (int): Stride for convolution.
            padding (int): Padding size.
            residual (bool, optional): Whether to add residual connection. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding)
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.residual = residual
        logger.debug("Initialized nonorm_Conv2d.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activated output tensor.
        """
        out = self.conv_block(x)
        return self.act(out)


class Conv2dTranspose(nn.Module):
    """A transposed convolutional block with Batch Normalization and ReLU activation."""

    def __init__(
            self,
            cin: int,
            cout: int,
            kernel_size: int,
            stride: int,
            padding: int,
            output_padding: int = 0,
            *args: Any,
            **kwargs: Any
    ) -> None:
        """
        Args:
            cin (int): Number of input channels.
            cout (int): Number of output channels.
            kernel_size (int): Kernel size.
            stride (int): Stride for transposed convolution.
            padding (int): Padding size.
            output_padding (int, optional): Additional size added to one side of output shape. Defaults to 0.
        """
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()
        logger.debug("Initialized Conv2dTranspose.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the transposed convolutional block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activated output tensor.
        """
        out = self.conv_block(x)
        return self.act(out)
