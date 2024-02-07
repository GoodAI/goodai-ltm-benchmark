from datetime import timedelta
from random import randint


def create_time_jump(mins_low, mins_high):
    mins_skipped = randint(mins_low, mins_high)
    return timedelta(minutes=mins_skipped)
