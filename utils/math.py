def mean_std(values: list) -> tuple[float, float]:
    mean = sum(values) / len(values)
    std = sum([(mean - v) ** 2 for v in values]) / len(values)
    return mean, std ** 0.5
