#!/bin/env python3
from __future__ import annotations

from math import inf
from sys import argv

import numpy as np
import pandas as pd
from haversine import haversine

from check import north_pole, weight_limit

# type aliases
lat = int
lon = int
location = tuple[lat, lon]
tripid = int
giftid = int

SOLUTION_COLUMNS = ["GiftId", "TripId"]


def export(solution: pd.DataFrame, path: str):
    with open(path, "w") as f:
        f.write(solution.to_csv(index=False))


def one_gift_per_trip(gifts: pd.DataFrame):
    return pd.DataFrame(
        columns=SOLUTION_COLUMNS,
        data=zip(gifts.index, gifts.index - 1)
    )


def nearest_neighbor(gifts: pd.DataFrame) -> pd.DataFrame:
    """nearest neighbor with full sleigh"""
    def _get_next_gift(loc: location) -> giftid:
        nearest_gift = (0, inf)
        for idx, gift in gifts.iterrows():
            distance = haversine(loc, (gift.Latitude, gift.Longitude))
            if distance < nearest_gift[1]:
                nearest_gift = (idx, distance)
        if nearest_gift[1] == inf:
            raise Exception("No nearest gift found")
        return nearest_gift[0]

    trip_id = 0
    weight = 0
    loc: location = north_pole
    trips: list[tuple[giftid, tripid]] = []
    gifts = gifts.copy()

    for _ in range(len(gifts)):
        gift_id = _get_next_gift(loc)
        gift = gifts.loc[gift_id]
        if (weight + gift.Weight) > weight_limit:
            trip_id += 1
            weight = 0
        weight += gift.Weight
        trips.append((gift_id, trip_id))
        loc = (gift.Latitude, gift.Longitude)
        gifts.drop(index=gift_id, inplace=True)

    return pd.DataFrame(columns=SOLUTION_COLUMNS, data=trips)


def pilot_method():
    raise NotImplementedError


def beam_search():
    raise NotImplementedError


if __name__ == '__main__' and '__file__' in globals():
    export_path = argv[1]

    gifts = pd.read_csv("./data/gifts.csv", index_col="GiftId")

    # only use a few gifts (takes forever otherwise)
    gifts = gifts[:1000]

    # solution = one_gift_per_trip(gifts)
    solution = nearest_neighbor(gifts)

    export(solution, export_path)
