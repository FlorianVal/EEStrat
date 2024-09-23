import torch


def discretize(value, steps):
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value)
    if torch.any(value < 0) or torch.any(value > 1):
        min_val, max_val = value.min().item(), value.max().item()
        raise ValueError(f"All values must be between 0 and 1, but found values in range [{min_val}, {max_val}]")

    if steps < 2:
        raise ValueError(f"Number of steps must be at least 2, but got {steps}")

    step_size = 1 / (steps - 1)
    discretized_value = torch.round(value / step_size) * step_size

    # Ensure all values are within [0, 1]
    discretized_value = torch.clamp(discretized_value, 0, 1)

    # Round to the same number of decimal places as step_size
    decimals = len(str(step_size).split(".")[-1])
    return torch.round(discretized_value * 10**decimals) / (10**decimals)


def build_path_count_dict(model, dataloader, device, discretization_steps=10):
    """Builds a dictionary of paths and their counts for each head in the model.

    Returns:
        Dict[str, Dict]: A dictionary of dictionaries, where each inner dictionary corresponds to a head in the model.
    """
    count_path_dicts = {f"head_{i}": {} for i in range(len(model.branches))}
    model.yield_mode = False

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        softmaxed_outputs = torch.nn.functional.softmax(outputs, dim=-1)
        max_preds, pred_indices = torch.max(softmaxed_outputs, dim=-1)
        max_preds = discretize(max_preds, discretization_steps)
        for head in range(max_preds.shape[0] - 1):
            keys = []
            for i in range(head + 1):
                keys.append(((max_preds[head][i].item(), 5), pred_indices[head][i].item()))
            keys.append(y[head].item())
            key = tuple(keys)
            count_path_dicts[f"head_{head}"][key] = count_path_dicts[f"head_{head}"].get(key, 0) + 1
    return count_path_dicts


def build_probs_dict(count_path_dicts):
    """Builds a dictionary of paths and their probabilities for each head in the model.

    Returns:
        Lists[Dict]: A list of dictionaries, where each dictionary corresponds to a head in the model.
    """
    probs_dicts = {}
    for _, count_path_dict in count_path_dicts.items():
        total_count = sum(count_path_dict.values())
        for key, value in count_path_dict.items():
            probs_dicts[key] = value / total_count
    return probs_dicts


def compute_continuation_cost(M, K, c, P, P_set, X_set, A):
    """
    Compute the continuation costs for a multi-head model.

    Args:
        M (int): Number of heads in the model
        K (int): Number of classes
        c (List[float]): Array of computational costs for each head
        P (Dict): Conditional probabilities computed earlier
        P_set (List[Tuple]): Set of all possible discretized belief states
        X_set (List[float]): Set of all possible discretized outputs
        A (torch.Tensor): Cost matrix where A[j,k] is the cost of predicting class k when true class is j

    Returns:
        Dict: Table of continuation costs
    """
    
    V = {(h, p): float("inf") for h in range(M) for p in P_set}

    def f(p):
        return torch.min(torch.sum(A * torch.tensor(p).unsqueeze(1), dim=0)).item()

    def update_belief(p, xi, P):
        p_next = []
        denominator = sum(P.get((xi, p, j), 0) * p[j] for j in range(K))
        for k in range(K):
            p_next.append(
                P.get((xi, p, k), 0) * p[k] / denominator if denominator != 0 else 0
            )
        return tuple(p_next)

    for h in range(M - 1, -1, -1):
        for p in P_set:
            if h == M - 1:
                V[(h, p)] = f(p)
            else:
                cost = c[h + 1]
                for xi in X_set:
                    p_next = update_belief(p, xi, P)
                    P_xi_given_p = sum(P.get((xi, p, k), 0) * p[k] for k in range(K))
                    cost += P_xi_given_p * min(
                        f(p_next), V.get((h + 1, p_next), float("inf"))
                    )
                V[(h, p)] = cost

    return V
