from datetime import timedelta


def create_time_jump(mins_low, mins_high, random_gen):
    mins_skipped = random_gen.randint(mins_low, mins_high)
    return timedelta(minutes=mins_skipped)
