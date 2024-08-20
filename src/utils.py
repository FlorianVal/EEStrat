def discretize(value, steps):
    if not 0 <= value <= 1:
        raise ValueError("Value must be between 0 and 1")
    if steps < 2:
        raise ValueError("Number of steps must be at least 2")
    
    step_size = 1 / (steps - 1)
    discretized_value = round(value / step_size) * step_size
    assert 0 <= discretized_value <= 1
    return round(discretized_value, len(str(step_size)))