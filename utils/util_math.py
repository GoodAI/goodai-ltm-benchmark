from math import sqrt


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


def get_dimensions(n):
    divisors = []
    for currentDiv in range(n):
        if n % float(currentDiv + 1) == 0:
            divisors.append(currentDiv + 1)
    # print divisors this is to ensure that we're choosing well
    hIndex = min(range(len(divisors)), key=lambda i: abs(divisors[i] - sqrt(n)))
    if divisors[hIndex] * divisors[hIndex] == n:
        return divisors[hIndex], divisors[hIndex]
    else:
        wIndex = hIndex + 1
        return divisors[hIndex], divisors[wIndex]