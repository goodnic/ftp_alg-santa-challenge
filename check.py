#!/bin/env python3
import random
from sys import argv

import folium  # type: ignore
import pandas as pd  # type: ignore
# I couldn't see an improvement in using cHaversine
from haversine import haversine  # type: ignore

# globals
north_pole = (90, 0)
weight_limit = 1000
sleigh_weight = 10

# Returns the distance of one trip
# Input: stops (list of stops such as [[latStopA,longStopA],[latStopB,longStopB],...])
# Input: weights (list of weights such as [weightGiftA,weightGiftB,...])
def weighted_trip_length(stops, weights):
    tuples = [tuple(x) for x in stops.values]
    # adding the last trip back to north pole, with just the sleigh weight
    tuples.append(north_pole)
    weights.append(sleigh_weight)

    dist = 0.0
    prev_stop = north_pole
    prev_weight = sum(weights)
    for location, weight in zip(tuples, weights):
        dist = dist + haversine(location, prev_stop) * prev_weight
        prev_stop = location
        prev_weight = prev_weight - weight
    return dist


# Returns the distance of all trips
# Input: all_trips (Pandas DataFrame)
def weighted_reindeer_weariness(all_trips):
    uniq_trips = all_trips.TripId.unique()

    dist = 0.0
    for t in uniq_trips:
        this_trip = all_trips[all_trips.TripId == t]
        dist = dist + weighted_trip_length(this_trip[['Latitude', 'Longitude']], this_trip.Weight.tolist())

    return dist


# Checks if one trip is over the weight limit
def check_for_overweight(all_trips):
    if any(all_trips.groupby('TripId').Weight.sum() > weight_limit):
        raise Exception("One of the sleighs over weight limit!")


def get_route_map(df, points_color='blue', include_home=False):
    m = folium.Map(location=[df.iloc[0]['Latitude'], df.iloc[0]['Longitude']], zoom_start=3)

    last_index = df.shape[0] - 1
    previous_point = None

    i = 0
    for index, row in df.iterrows():
        current_point = (row['Latitude'], row['Longitude'])

        if i == 0:
            color = 'green'
            if include_home:
                folium.PolyLine([[90, 0], current_point], color="green", weight=2, opacity=0.3).add_to(m)
        elif i == last_index:
            color = 'red'
        else:
            color = points_color

        tooltip = f"Tour-Point: {str(i)} Index: {str(index)}<br>Id: {row['GiftId']} Weight: {'{:.2f}'.format(row['Weight'])} <br>Lat: {'{:.2f}'.format(row['Latitude'])} Long: {'{:.2f}'.format(row['Latitude'])}"

        folium.CircleMarker(location=current_point, radius=5, color=color, fill=True,
                            tooltip=tooltip, fill_color=color).add_to(m)

        if previous_point:
            folium.PolyLine([previous_point, current_point], color="blue", weight=2, opacity=0.3).add_to(m)

        previous_point = current_point
        i += 1

    if include_home:
        folium.PolyLine([[90, 0], previous_point], color="darkred", weight=2, opacity=0.3).add_to(m)

    return m


class MapVisualizer:
    map = None

    def __init__(self):
        self._init_map()

    def add_route(self, path):
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        folium.PolyLine(path, color=color, weight=1).add_to(self.map)

    def _init_map(self):
        self.map = folium.Map(location=[40.52, 34.34], zoom_start=1)

    def save_map(self, save_path):
        self.map.save(save_path)


if __name__ == "__main__":
    submission = pd.read_csv(argv[1])
    map_export_path = None
    try:
        map_export_path = argv[2]
    except IndexError:
        print("Not exporting the map (no path supplied)")

    gifts = pd.read_csv('./data/gifts.csv')
    df = pd.merge(submission, gifts, how='left')
    df['Position'] = list(zip(df['Latitude'],df['Longitude']))

    # Weighted Reindeer Weariness
    wrw = weighted_reindeer_weariness(df)
    print('Weighted Reindeer Weariness: {:e}'.format(wrw))

    # No error here means all trips are legit!
    check_for_overweight(df)

    if map_export_path is None:
        exit(0)

    visualizer = MapVisualizer()
    # showing all trips from and to the pole can make the map messy
    show_poles = False
    i = 0
    for path in df.groupby(by='TripId')['Position'].apply(list):
        # Mark Beginning and End
        folium.CircleMarker(location=path[0], radius=5, color='green', weight=1, fill=True, tooltip=f"Begin {i}")\
              .add_to(visualizer.map)
        folium.CircleMarker(location=path[-1], radius=5, color='red', weight=1, fill=True, tooltip=f"End {i}")\
              .add_to(visualizer.map)

        if show_poles:
            path.insert(0,(90,0))
            path.append((90,0))

        i += 1

        visualizer.add_route(path)

    # Save map (better performance when viewing)
    visualizer.save_map(map_export_path)
