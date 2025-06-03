import pandas as pd
import googlemaps
from tqdm import tqdm

# Load the data
df = pd.read_csv("response.csv")

# Extract address column
df["address"] = df[
    "현재 거주 중인 지역(동까지 자세히 입력 부탁드립니다)\n(예시: 서울시 송파구 잠실동)"
]

# Basketball club locations
club_locations = ["서울특별시 성동구", "서울특별시 서초구", "서울특별시 중구"]

# Google Maps client
gmaps = googlemaps.Client(key="AIzaSyBnhLTCUvdBDNCwmrZKz4ttfUOGkPbqL80")


# Function to find closest club
def find_closest_club(origin):
    min_time = float("inf")
    best_club = None
    for club in club_locations:
        result = gmaps.distance_matrix(origin, club, mode="transit")
        element = result["rows"][0]["elements"][0]
        if element["status"] == "OK":
            travel_time = element["duration"]["value"]
            if travel_time < min_time:
                min_time = travel_time
                best_club = club
    return best_club, round(min_time / 60, 1)


# Progress bar setup
closest_clubs = []
travel_times = []

for addr in tqdm(df["address"]):
    club, time = find_closest_club(addr)
    closest_clubs.append(club)
    travel_times.append(time)

# Add columns at the end
df["travel_time_min"] = travel_times
df["closest_club"] = closest_clubs  # rightmost column

# Save results
df.to_csv("residence_to_closest_club.csv", index=False)
