import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from typing import List, Dict, Tuple
from src.model import BranchyResNet


def discretize(value, steps):
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value)

    if torch.any(value < 0) or torch.any(value > 1):
        raise ValueError("All values must be between 0 and 1")

    if steps < 2:
        raise ValueError("Number of steps must be at least 2")

    step_size = 1 / (steps - 1)
    discretized_value = torch.round(value / step_size) * step_size

    # Ensure all values are within [0, 1]
    discretized_value = torch.clamp(discretized_value, 0, 1)

    # Round to the same number of decimal places as step_size
    decimals = len(str(step_size).split(".")[-1])
    return torch.round(discretized_value * 10**decimals) / (10**decimals)


def build_path_count_dict(model, dataloader):
    """Builds a dictionary of paths and their counts for each head in the model.

    Returns:
        Lists[Dict]: A list of dictionaries, where each dictionary corresponds to a head in the model.
    """
    count_path_dicts = [{} for i in range(len(model.branches))]
    model.yield_mode = False

    for x, y in dataloader:
        max_preds = torch.max(
            model(x), dim=-1
        )  # Return max prob of shape [Heads, Batch]
        max_preds = discretize(max_preds.values, 10)
        for head in range(max_preds.shape[0]):
            keys = torch.cat((max_preds[: head + 1].T, y.unsqueeze(1)), dim=1)
            for key in keys:
                key = tuple([round(x, 5) for x in key.tolist()])
                count_path_dicts[head][key] = count_path_dicts[head].get(key, 0) + 1
    return count_path_dicts


def build_probs_dict(count_path_dicts):
    """Builds a dictionary of paths and their probabilities for each head in the model.

    Returns:
        Lists[Dict]: A list of dictionaries, where each dictionary corresponds to a head in the model.
    """
    probs_dicts = []
    for count_path_dict in count_path_dicts:
        probs_dict = {}
        for key, value in count_path_dict.items():
            probs_dict[key] = value / sum(count_path_dict.values())
        probs_dicts.append(probs_dict)
    return probs_dicts


def estimate_conditional_probabilities(
    model: BranchyResNet,
    dataloader: DataLoader,
    num_classes: int,
    discretization_steps: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> List[Dict[Tuple, List[float]]]:
    """
    Estimate conditional probabilities for each branch of the BranchyResNet model.

    Args:
        model (BranchyResNet): The BranchyResNet model.
        dataloader (DataLoader): DataLoader for the dataset.
        num_classes (int): Number of classes in the dataset.
        discretization_steps (int): Number of steps for discretization.
        device (str): Device to run the model on.

    Returns:
        List[Dict[Tuple, List[float]]]: Estimated conditional probabilities for each branch.
    """
    model.eval()
    model.to(device)

    # Initialize counters for each branch
    branch_counters = [
        defaultdict(lambda: [0] * num_classes) for _ in range(len(model.branches) + 1)
    ]

    # Count occurrences
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            model.yield_mode = False
            branch_outputs = model(inputs)

            for branch_idx, branch_output in enumerate(branch_outputs):
                probs = torch.softmax(branch_output, dim=1)

                for i, label in enumerate(labels):
                    # Create a tuple of discretized probabilities for previous branches
                    prev_probs = tuple(
                        discretize(
                            torch.softmax(prev_output[i], dim=0).max().item(),
                            discretization_steps,
                        )
                        for prev_output in branch_outputs[:branch_idx]
                    )

                    # Discretize the current branch's probability
                    current_prob = discretize(
                        probs[i].max().item(), discretization_steps
                    )

                    # Update the counter
                    key = prev_probs + (current_prob,)
                    branch_counters[branch_idx][key][label.item()] += 1

    # Calculate conditional probabilities
    conditional_probs = []
    for branch_counter in branch_counters:
        branch_probs = {}
        for key, counts in branch_counter.items():
            total = sum(counts)
            if total > 0:
                branch_probs[key] = [count / total for count in counts]
            else:
                branch_probs[key] = [
                    1 / num_classes
                ] * num_classes  # Uniform distribution if no data
        conditional_probs.append(branch_probs)

    return conditional_probs
