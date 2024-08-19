import torch
import torch.nn as nn
import torchvision.models as models

class Branch(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Branch, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class BranchyResNet(nn.Module):
    def __init__(self, num_classes, base_model='resnet18', yield_mode=False):
        super(BranchyResNet, self).__init__()
        self.yield_mode = yield_mode
        
        # Initialize base model
        if base_model == 'resnet18':
            self.base_model = models.resnet18(weights=None)
        elif base_model == 'resnet34':
            self.base_model = models.resnet34(weights=None)
        elif base_model == 'resnet50':
            self.base_model = models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported base model: {base_model}")
        
        # Get the layers from the base model
        self.layers = nn.ModuleList([
            nn.Sequential(
                self.base_model.conv1,
                self.base_model.bn1,
                self.base_model.relu,
                self.base_model.maxpool,
                self.base_model.layer1
            ),
            self.base_model.layer2,
            self.base_model.layer3
        ])
        
        # Final layers
        self.final_layer = self.base_model.layer4
        self.avgpool = self.base_model.avgpool
        self.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        
        # Define branches
        branch_sizes = self._get_branch_sizes()
        self.branches = nn.ModuleList([Branch(size, num_classes) for size in branch_sizes])
    
    def _get_branch_sizes(self):
        return [
            self.base_model.layer1[-1].conv2.out_channels,
            self.base_model.layer2[-1].conv2.out_channels,
            self.base_model.layer3[-1].conv2.out_channels
        ]
    
    def _forward_yield(self, x):
        for layer, branch in zip(self.layers, self.branches):
            x = layer(x)
            yield branch(x)
        
        # Final classification
        x = self.final_layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        yield x

    def _forward_full(self, x):
        outputs = []
        for layer, branch in zip(self.layers, self.branches):
            x = layer(x)
            outputs.append(branch(x))
        
        # Final classification
        x = self.final_layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        outputs.append(x)
        
        return torch.stack(outputs)

    def forward(self, x):
        if self.yield_mode:
            return self._forward_yield(x)
        else:
            return self._forward_full(x)