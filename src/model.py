import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Generator, Union


class Branch(nn.Module):
    """
    Represents a branch in the BranchyResNet architecture.

    Each branch consists of an adaptive average pooling layer followed by a fully connected layer.
    """

    def __init__(self, input_size: int, num_classes: int):
        """
        Initialize a Branch.

        Args:
            input_size (int): The number of input features.
            num_classes (int): The number of output classes.
        """
        super(Branch, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Branch.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after average pooling and fully connected layer.
        """
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class BranchyResNet(nn.Module):
    """
    BranchyResNet: A modified ResNet architecture with early exit branches.

    This model adds three additional branches to a standard ResNet, allowing for
    early classification at different depths of the network.
    """

    def __init__(
        self, num_classes: int, base_model: str = "resnet18", yield_mode: bool = False
    ):
        """
        Initialize a BranchyResNet.

        Args:
            num_classes (int): The number of output classes.
            base_model (str): The base ResNet model to use ('resnet18', 'resnet34', or 'resnet50').
            yield_mode (bool): If True, the forward pass yields outputs one by one.
        """
        super(BranchyResNet, self).__init__()
        self.yield_mode = yield_mode
        self.num_classes = num_classes

        # Initialize base model
        if base_model == "resnet18":
            self.base_model = models.resnet18(weights=None)
        elif base_model == "resnet34":
            self.base_model = models.resnet34(weights=None)
        elif base_model == "resnet50":
            self.base_model = models.resnet50(weights=None)
        elif base_model == "resnet101":
            self.base_model = models.resnet101(weights=None)
        elif base_model == "resnet152":
            self.base_model = models.resnet152(weights=None)
        else:
            raise ValueError(f"Unsupported base model: {base_model}")
        # Get the layers from the base model
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    self.base_model.conv1,
                    self.base_model.bn1,
                    self.base_model.relu,
                    self.base_model.maxpool,
                    self.base_model.layer1,
                ),
                self.base_model.layer2,
                self.base_model.layer3,
            ]
        )

        # Final layers
        self.final_layer = self.base_model.layer4
        self.avgpool = self.base_model.avgpool
        self.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

        # Define branches
        branch_sizes = self._get_branch_sizes()
        self.branches = nn.ModuleList(
            [Branch(size, num_classes) for size in branch_sizes]
        )

    def _get_branch_sizes(self) -> List[int]:
        """
        Get the sizes for each branch based on the output channels of each layer.

        Returns:
            List[int]: A list of branch sizes.
        """
        return [
            self.base_model.layer1[-1].conv2.out_channels,
            self.base_model.layer2[-1].conv2.out_channels,
            self.base_model.layer3[-1].conv2.out_channels,
        ]

    def _forward_yield(self, x: torch.Tensor) -> Generator[torch.Tensor, None, None]:
        """
        Forward pass in yield mode, yielding outputs one by one.

        Args:
            x (torch.Tensor): Input tensor.

        Yields:
            torch.Tensor: Output tensors from each branch and the final classification.
        """
        for layer, branch in zip(self.layers, self.branches):
            x = layer(x)
            yield branch(x)

        # Final classification
        x = self.final_layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        yield x

    def _forward_full(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass in full mode, returning all outputs concatenated.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Concatenated outputs from all branches and final classification.
        """
        outputs = torch.empty((len(self.branches) + 1, x.size(0), self.num_classes), device=x.device)
        for i, (layer, branch) in enumerate(zip(self.layers, self.branches)):
            x = layer(x)
            outputs[i] = branch(x)

        # Final classification
        x = self.final_layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        outputs[-1] = x

        return outputs

    def forward(
        self, x: torch.Tensor
    ) -> Union[Generator[torch.Tensor, None, None], torch.Tensor]:
        """
        Forward pass of the BranchyResNet.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Union[Generator[torch.Tensor, None, None], torch.Tensor]:
                If in yield mode, returns a generator yielding outputs one by one.
                Otherwise, returns a tensor with all outputs concatenated.
        """
        if self.yield_mode:
            return self._forward_yield(x)
        else:
            return self._forward_full(x)
