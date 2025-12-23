import os
import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, LineString
from tqdm import tqdm
import pandas as pd


place = "Yanta District,xi'an, China"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

depot_lon, depot_lat = 108.91314, 34.20758
offset_distance = 100
max_dist_threshold = 200


G = ox.graph_from_place(place, network_type='drive')
G_proj = ox.project_graph(G)


tags = {"building": ["residential", "office"], "shop": True}
gdf = ox.features_from_place(place, tags)
gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
gdf["geometry"] = gdf["geometry"].centroid
gdf = gdf.set_geometry("geometry")


gdf = gdf.to_crs(epsg=3857)
coords = np.array([[p.x, p.y] for p in gdf.geometry])
db = DBSCAN(eps=300, min_samples=1).fit(coords)
gdf["cluster"] = db.labels_

clustered = gdf.dissolve(by="cluster", as_index=False)
clustered["geometry"] = clustered.centroid
clustered = clustered.set_geometry("geometry")
clustered = clustered.to_crs(epsg=4326)
demand_dict = gdf["cluster"].value_counts().to_dict()
clustered["demand"] = clustered["cluster"].map(demand_dict)


clustered_proj = clustered.to_crs(epsg=3857)
edges_proj = ox.graph_to_gdfs(G, nodes=False, edges=True).to_crs(epsg=3857)

adjusted_points, adjusted_demands = [], []

for i, pt in tqdm(enumerate(clustered_proj.geometry), total=len(clustered_proj)):
    demand = clustered_proj.iloc[i]["demand"]
    min_dist = float("inf")
    best_proj, best_seg = None, None

    for geom in edges_proj['geometry']:
        proj = geom.interpolate(geom.project(pt))
        dist = pt.distance(proj)
        if dist < min_dist:
            min_dist, best_proj, best_seg = dist, proj, geom

    if min_dist > max_dist_threshold or best_proj is None:
        continue

    coords = list(best_seg.coords)
    seg_idx = np.argmin([LineString([coords[i], coords[i + 1]]).distance(best_proj)
                         for i in range(len(coords) - 1)])
    p1, p2 = coords[seg_idx], coords[seg_idx + 1]
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = np.hypot(dx, dy)
    if length == 0:
        offset_pt = best_proj
    else:
        nx_, ny_ = -dy / length, dx / length
        offset_pt = Point(best_proj.x + nx_ * offset_distance, best_proj.y + ny_ * offset_distance)

    adjusted_points.append(offset_pt)
    adjusted_demands.append(demand)


adjusted_gdf = gpd.GeoDataFrame({
    "geometry": adjusted_points,
    "demand_raw": adjusted_demands
}, crs="EPSG:3857").to_crs(epsg=4326)


depot_geom = gpd.GeoSeries([Point(depot_lon, depot_lat)], crs="EPSG:4326")
depot_df = gpd.GeoDataFrame({"geometry": depot_geom, "demand_raw": [0]})
adjusted_gdf = pd.concat([depot_df, adjusted_gdf], ignore_index=True)


adjusted_gdf["demand"] = adjusted_gdf["demand_raw"].apply(lambda x: int(np.ceil(np.log1p(x) * 3)))
adjusted_gdf["lon"] = adjusted_gdf.geometry.x
adjusted_gdf["lat"] = adjusted_gdf.geometry.y

np.random.seed(42)


demand = adjusted_gdf["demand"].values


h = np.random.rand(len(demand))


reward = (0.5 + h) * demand
reward = np.round(reward).astype(int)
reward = np.maximum(reward, 1 * (demand > 0))  


adjusted_gdf["reward"] = reward


adjusted_gdf[["lon", "lat", "demand", "reward"]].to_csv(f"{output_dir}/demand_reward_info_adjusted.csv", index=False)

G_undirected = G_proj.to_undirected()
largest_cc = max(nx.connected_components(G_undirected), key=len)
G_connected = G_undirected.subgraph(largest_cc).copy()


adjusted_gdf_proj = adjusted_gdf.to_crs(G_connected.graph["crs"])
valid_points = []
node_ids = []

for pt in adjusted_gdf_proj.geometry:
    try:
        node = ox.distance.nearest_nodes(G_connected, pt.x, pt.y)
        node_ids.append(node)
        valid_points.append(pt)
    except:
        continue

adjusted_gdf_valid = gpd.GeoDataFrame(adjusted_gdf.loc[:len(valid_points)-1].copy(),
                                       geometry=valid_points,
                                       crs=adjusted_gdf_proj.crs)
with open(f"{output_dir}/profits_and_rewards.txt", "w") as f:
    for i, row in adjusted_gdf_valid.iterrows():
        demand = row["demand"]
        reward = row["reward"]
        f.write(f"{i} {demand} {reward}\n")


n = len(node_ids)
distance_matrix = np.zeros((n, n))

for i in tqdm(range(n)):
    for j in range(i + 1, n):
        try:
            d = nx.shortest_path_length(G_connected, node_ids[i], node_ids[j], weight="length")
            distance_matrix[i][j] = d / 15000
            distance_matrix[j][i] = d / 15000
        except nx.NetworkXNoPath:
            distance_matrix[i][j] = distance_matrix[j][i] = np.inf

np.savetxt(f"{output_dir}/distance_matrix_clean.csv", distance_matrix, delimiter=",")
adjusted_gdf_valid.to_crs("EPSG:4326")[["lon", "lat", "demand"]].to_csv(f"{output_dir}/demand_info_valid.csv", index=False)
np.savetxt(f"{output_dir}/distance_matrix.txt", distance_matrix, fmt="%.3f")

fig, ax = ox.plot_graph(
    G_proj,
    show=False,
    close=False,
    node_size=0,
    edge_color="#bbbbbb",
    bgcolor="white",
    edge_linewidth=0.5
)

adjusted_gdf_valid.iloc[1:].plot(ax=ax, color="red", markersize=5)  
adjusted_gdf_valid.iloc[[0]].plot(ax=ax, color="green", marker="^", markersize=20)  # depot
x0, y0 = adjusted_gdf_valid.iloc[0].geometry.x, adjusted_gdf_valid.iloc[0].geometry.y


for i, row in adjusted_gdf_valid.iterrows():
    x, y = row.geometry.x, row.geometry.y
    demand = row["demand"]
    reward = row["reward"]

    
    ax.text(x + 5, y + 5, str(i), fontsize=6, color="blue", ha="left", va="bottom")       


#plt.title("Delivery Points with Demand and Depot")
plt.savefig(f"{output_dir}/map_clean.eps", bbox_inches="tight", dpi=600)

