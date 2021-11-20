#!/bin/env python3
import pandas as pd
from sys import argv


def export(solution: pd.DataFrame, path: str):
    with open(path, "w") as f:
        f.write(solution.sort_values(by="TripId").to_csv(index=False))


def one_gift_per_trip(gifts: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(data={
        "GiftId": gifts.GiftId,
        "TripId": gifts.GiftId - 1
    })


if __name__ == "__main__":
    export_path = argv[1]

    gifts = pd.read_csv("./data/gifts.csv")

    solution = one_gift_per_trip(gifts)

    export(solution, export_path)
