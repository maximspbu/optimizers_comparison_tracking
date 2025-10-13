from typing import Optional
import torch
from .config import DEVICE
from torch import nn


class SimpleRegressionModel(nn.Module):
    def __init__(
        self,
        input_shape: int = 8,
        hidden_units: int = 64,
        output_shape: int = 1,
        activation_function: nn.Module = nn.ReLU(),
        normalization: nn.Module | None = None,
        dropout: nn.Module | None = None,
        device: torch.device = DEVICE,
    ) -> None:
        """SimpleRegressionModel initializer

        Args:
            input_shape (int): Number of units in input layer
            hidden_units (int): Number of units in hidden layers
            output_shape (int): Number of units in output layer
            activation_function (Optional[torch.nn.Module], optional): Activation function. Defaults to None.
            normalization (Optional[torch.nn.Module], optional): Use batch or layer normalization. Defaults to None.
            dropout (Optional[torch.nn.Module], optional): Dropout layer. Defaults to None.
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(
                in_features=input_shape,
                out_features=hidden_units,
                device=device,
            ),
            nn.Identity() if normalization is None else normalization,
            nn.Identity() if activation_function is None else activation_function,
            nn.Identity() if dropout is None else dropout,
            nn.Linear(
                in_features=hidden_units,
                out_features=hidden_units,
                device=device,
            ),
            nn.Identity() if activation_function is None else activation_function,
            nn.Linear(
                in_features=hidden_units,
                out_features=output_shape,
                device=device,
            ),
            # nn.Flatten(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Method for forward pass.

        Args:
            X (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Result of computations.
        """
        return self.block(X)


class SimpleClassificationModel(nn.Module):
    def __init__(
        self,
        input_shape: int = 8,
        hidden_units: int = 16,
        output_shape: int = 2,
        activation_function: Optional[torch.nn.Module] = None,
        normalization: Optional[torch.nn.Module] = None,
        dropout: Optional[torch.nn.Module] = None,
        device: torch.device = DEVICE,
    ) -> None:
        """SimpleClassificationModel initializer

        Args:
            input_shape (int): Number of units in input layer
            hidden_units (int): Number of units in hidden layers
            output_shape (int): Number of units in output layer
            activation_function (Optional[torch.nn.Module], optional): Activation function. Defaults to None.
            normalization (Optional[torch.nn.Module], optional): Use batch or layer normalization. Defaults to None.
            dropout (Optional[torch.nn.Module], optional): Dropout layer. Defaults to None.
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(
                in_features=input_shape,
                out_features=hidden_units,
                device=device,
            ),
            nn.Identity() if normalization is None else normalization,
            nn.Identity() if activation_function is None else activation_function,
            nn.Identity() if dropout is None else dropout,
            nn.Linear(
                in_features=hidden_units,
                out_features=output_shape,
                device=device,
            ),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Method for forward pass.

        Args:
            X (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Result of computations.
        """
        return self.block(X)
