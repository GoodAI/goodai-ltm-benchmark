def mean_std(values: list) -> tuple[float, float]:
    mean = sum(values) / len(values)
    var = sum([(mean - v) ** 2 for v in values]) / len(values)
    return mean, var ** 0.5
