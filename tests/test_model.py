import torch
import pytest
from src.model import Branch, BranchyResNet

def test_branch():
    input_size = 512
    num_classes = 10
    batch_size = 32
    
    branch = Branch(input_size, num_classes)
    x = torch.randn(batch_size, input_size, 1, 1)
    output = branch(x)
    
    assert output.shape == (batch_size, num_classes)

def test_branchy_resnet_yield_mode():
    batch_size = 32
    num_classes = 10
    input_channels = 3
    input_size = 224
    
    model = BranchyResNet(num_classes, base_model='resnet18', yield_mode=True)
    x = torch.randn(batch_size, input_channels, input_size, input_size)
    
    outputs = model(x)
    outs = []
    
    for output in outputs:
        outs.append(output)
        assert output.shape == (batch_size, num_classes)
        
    assert len(outs) == 4  # 3 branches + 1 main output
    
def test_branchy_resnet_concat_mode():
    batch_size = 32
    num_classes = 10
    input_channels = 3
    input_size = 224
    
    model = BranchyResNet(num_classes, base_model='resnet18', yield_mode=False)
    x = torch.randn(batch_size, input_channels, input_size, input_size)
    
    output = model(x)
    
    assert output.shape == (4, batch_size, num_classes)

def test_branchy_resnet_invalid_base_model():
    with pytest.raises(ValueError):
        BranchyResNet(10, base_model='invalid_model')

def test_branchy_resnet_feature_sizes():
    model = BranchyResNet(10, base_model='resnet18')
    for i, branch in enumerate(model.branches):
        assert branch.fc.in_features == 64 * 2**i, f"Expected {64 * 2**i} input features for branch {i}, but got {branch.fc.in_features}"

def test_branchy_resnet_branch_count():
    num_classes = 10
    model = BranchyResNet(num_classes, base_model='resnet18')
    
    # Check the number of branches
    assert len(model.branches) == 3, f"Expected 3 branches, but got {len(model.branches)}"

    # Check the output shape in yield mode
    model.yield_mode = True
    x = torch.randn(1, 3, 224, 224)
    outputs = list(model(x))
    assert len(outputs) == 4, f"Expected 4 outputs (3 branches + final), but got {len(outputs)}"

    # Check the output shape in full mode
    model.yield_mode = False
    output = model(x)
    assert output.shape == (4, 1, num_classes), f"Expected shape (4, 1, {num_classes}), but got {output.shape}"

if __name__ == '__main__':
    pytest.main([__file__])