# map_plot.py
from index import gu_score_list

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Convert list of tuples to DataFrame
score_df = pd.DataFrame(gu_score_list, columns=["name", "score"])

# Load Seoul districts GeoJSON
gdf = gpd.read_file("seoul_districts.geojson")

# Merge geodata and scores
merged = gdf.merge(score_df, on="name")

# Plot
plt.figure(figsize=(12, 12))
ax = plt.gca()
cmap = sns.color_palette("viridis", as_cmap=True)
merged.plot(
    column="score", cmap=cmap, linewidth=0.8, edgecolor="black", legend=True, ax=ax
)

plt.title("Seoul District Scores", fontsize=16)
ax.set_axis_off()
plt.show()
