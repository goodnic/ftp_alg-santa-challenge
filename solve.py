#!/bin/env python3
import copy
from math import inf
from sys import argv
from typing import Callable

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from haversine import haversine_vector  # type: ignore

import check

# type aliases
lat = np.int64
lon = np.int64
location = tuple[lat, lon]
tripid = int
giftid = np.int64

SOLUTION_COLUMNS = ["GiftId", "TripId"]
NORTH_POLE: location = check.north_pole
WEIGHT_LIMIT = check.weight_limit


def export(solution: pd.DataFrame, path: str):
    with open(path, "w") as f:
        f.write(solution.to_csv(index=False))


def weighted_reindeer_weariness(
    gifts: pd.DataFrame, trips: list[tuple[giftid, tripid]]
) -> float:
    solution = pd.DataFrame(columns=SOLUTION_COLUMNS, data=trips)
    df = pd.merge(solution, gifts.reset_index(), how="left")
    df["Position"] = list(zip(df["Latitude"], df["Longitude"]))
    return check.weighted_reindeer_weariness(df)


class State:
    def __init__(self, gifts):
        self.trip_id: int = 0
        self.weight: float = 0
        self.loc: location = NORTH_POLE
        self.trips: list[tuple[giftid, tripid]] = []
        self.gifts: pd.DataFrame = gifts.copy()


#
# Heuristics
#


def nearest_neighbor_heuristic(state: State, n: int = 0) -> None:
    """
    nearest neighbor with full sleigh

    finds nth nearest gift and updates state accordingly
    """

    def _get_next_gift(loc: location, n: int) -> giftid:
        distances = haversine_vector(
            [loc] * len(state.gifts),
            list(zip(state.gifts.Latitude, state.gifts.Longitude)),
        )
        for _ in range(n):
            distances[distances.argmin()] = inf
        return state.gifts.index[distances.argmin()]

    gift_id = _get_next_gift(state.loc, n)
    gift = state.gifts.loc[gift_id]
    if (state.weight + gift.Weight) > WEIGHT_LIMIT:
        state.trip_id += 1
        state.weight = 0
    state.weight += gift.Weight
    state.trips.append((gift_id, state.trip_id))
    state.loc = (gift.Latitude, gift.Longitude)
    state.gifts.drop(index=gift_id, inplace=True)


def nearest_longitude_bins(gifts):
    """
    return bins of gifts, that are exactly below the weight limit

    it is assumed, the state is allready ordered by longitude
    """
    def assemble_bin_from_ids(ids: list):
        trip_bin = gifts.loc[ids]

        return trip_bin

    bins = list()
    next_trip_bin_ids = list()
    next_trip_weight = 0
    for gift in gifts.iterrows():
        gift_weight = gift[1][2]

        # if weight is reached, start a new trip
        if next_trip_weight + gift_weight > WEIGHT_LIMIT:
            bins.append(assemble_bin_from_ids(next_trip_bin_ids))
            next_trip_bin_ids = list()
            next_trip_weight = 0

        # add the current gift id to the trip
        next_trip_bin_ids.append(gift[0])
        next_trip_weight += gift_weight

    # add last bin aswell
    bins.append(assemble_bin_from_ids(next_trip_bin_ids))

    return bins




#
# Meta-Heuristics
#


def beam_search(
    gifts: pd.DataFrame,
    heuristic: Callable[[State, int], None],
    beam_width: int = 2,
) -> State:
    """beam search with browse depth of 1"""

    state = State(gifts)
    for _ in range(len(state.gifts)):
        heuristic(state)
    base_state = state
    base_wrw = weighted_reindeer_weariness(gifts, base_state.trips)

    state = State(gifts)

    for _ in range(len(state.gifts)):
        sub_states = [copy.deepcopy(state) for _ in range(beam_width)]
        [heuristic(sub_states[n], n) for n in range(beam_width)]
        fork_states = copy.deepcopy(sub_states)

        # n=0 case is already done
        sub_states[0] = base_state

        for sub_state in sub_states[1:]:
            for _ in range(len(sub_state.gifts)):
                heuristic(sub_state)

        wrws = [base_wrw] + [
            weighted_reindeer_weariness(gifts, s.trips) for s in sub_states[1:]
        ]
        idx = np.argmin(wrws)
        state = fork_states[idx]
        base_state = sub_states[idx]
        base_wrw = wrws[idx]

    return state


def simple(gifts: pd.DataFrame, heuristic: Callable[[State], None]) -> State:
    state = State(gifts)
    for _ in range(len(state.gifts)):
        heuristic(state)
    return state

def vertical_lines(gifts: pd.DataFrame) -> State:
    # this basicaly delivers gift in sweep from west to east. 
    # It caries as many gifts as possible

    # order gifts by longitude (smalles first)
    gifts_sorted = gifts.sort_values(by='Longitude')

    # create the bins for the trips
    bins = nearest_longitude_bins(gifts_sorted)

    # sort each bin by longitude and assemble the trip (top to bottom)
    state = State(gifts)
    trip_id = 0
    for trip_bin in bins:
        trip_bin_sorted = trip_bin.sort_values(by='Latitude', ascending=False)

        # add the trips to the state
        for indx in trip_bin_sorted.index[:]:
            state.trips.append((indx, trip_id))
        # state.trips.append(

        trip_id += 1

    return state

    

def main():
    export_path = None
    try:
        export_path = argv[1]
    except IndexError:
        print("Not exporting to csv (no path supplied)")

    gifts = pd.read_csv("./data/gifts.csv", index_col="GiftId")

    # only use a few gifts (takes forever otherwise)
    gifts = gifts[:1000]

    state = simple(gifts, nearest_neighbor_heuristic)
    print("nearest neighbor:")
    print(weighted_reindeer_weariness(gifts, state.trips))

    state = beam_search(gifts, nearest_neighbor_heuristic, beam_width=2)
    print("beam search with nearest neighbor:")
    print(weighted_reindeer_weariness(gifts, state.trips))

    state = vertical_lines(gifts)
    print("distribute in vertical lines:")
    print(weighted_reindeer_weariness(gifts, state.trips))

    if export_path is not None:
        export(pd.DataFrame(columns=SOLUTION_COLUMNS, data=state.trips), export_path)

    return 0


if __name__ == "__main__" and "__file__" in globals():
    exit(main())
