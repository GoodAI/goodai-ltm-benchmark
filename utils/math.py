def mean(values: list) -> float:
    return sum(values) / len(values)


def std(values: list, ref_mean: float = None) -> float:
    if ref_mean is None:
        ref_mean = mean(values)
    var = sum([(ref_mean - v) ** 2 for v in values]) / len(values)
    return var ** 0.5


def mean_std(values: list) -> tuple[float, float]:
    avg = mean(values)
    return avg, std(values, ref_mean=avg)
