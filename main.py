import os
import json
import random
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import transform, nearest_points
import shapely.vectorized as sv
from k_means_constrained import KMeansConstrained
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse.csgraph import minimum_spanning_tree
import rasterio
from rasterio.features import geometry_mask
from sklearn.manifold import MDS
import networkx as nx
from itertools import combinations

app = Flask(__name__)
CORS(app)

# Define paths to shapefiles (unchanged)
shapefile_paths = {
    "combined_constraints": "Constraints/Combined Constraints.shp",
    "infrastructure_boundary": "Constraints/Infrastructure Boundary.shp",
    "project_boundary": "Constraints/Project Boundary.shp"
}

# **New**: paths for persisted JSON
INFRA_GEOJSON_PATH       = "infrastructure.geojson"
VALID_AREAS_JSON_PATH    = "valid_areas.json"
CLUSTER_DATA_JSON_PATH   = "cluster_data.json"

def drop_z(geom):
    if geom.has_z:
        return transform(lambda x, y, *_: (x, y), geom)  # Drops the Z-coordinate
    return geom

@app.route('/receive_dicts', methods=['POST'])
def receive_dicts():
    data = request.get_json()
    dict1 = data.get('dict1')
    dict2 = data.get('dict2')
    if not dict1 or not dict2:
        return jsonify({'error': 'Both dict1 and dict2 are required'}), 400

    # Build GeoDataFrames
    gdfprojectboundary = gpd.GeoDataFrame.from_features(dict1['Project Boundary']['features'])
    gdfinfrastructure   = gpd.GeoDataFrame.from_features(dict1['Infrastructure']['features'])

    # **Persist infra** to GeoJSON
    gdfinfrastructure.to_file(INFRA_GEOJSON_PATH, driver="GeoJSON")

    # Compute valid_area exactly as before
    project_boundary = gdfprojectboundary.union_all()
    infra_union      = gdfinfrastructure.union_all()
    valid_area       = project_boundary.difference(infra_union)
    for key in dict2.keys():
        feature_collection = {"type": "FeatureCollection", "features": [dict2[key]]}
        constrants = gpd.GeoDataFrame.from_features(feature_collection)
        valid_area = valid_area.difference(constrants.union_all())

    # Build the same validgeojsondict
    validgeojsondict = {}
    for idx, geom in enumerate(valid_area.geoms, start=1):
        valid_gdf = gpd.GeoDataFrame(geometry=[geom], crs=gdfprojectboundary.crs)
        valid_gdf.geometry = valid_gdf.geometry.apply(drop_z)
        validgeojsondict[idx] = valid_gdf.to_json()

    # **Persist valid areas** to JSON
    with open(VALID_AREAS_JSON_PATH, "w") as f:
        json.dump(validgeojsondict, f)

    return jsonify({'message': 'Dictionaries received successfully!'})

@app.route('/getgeojson', methods=['GET'])
def get_geojson():
    if os.path.exists(VALID_AREAS_JSON_PATH):
        with open(VALID_AREAS_JSON_PATH) as f:
            raw = json.load(f)
        # convert JSON keys back to int so API matches previous
        geojsonw = {int(k): v for k, v in raw.items()}
        return jsonify(geojsonw)
    return jsonify({'error': 'No GeoJSON available'}), 404

def distribute_points(selected, total_points, divisibility):
    # **Reload valid areas** from JSON
    with open(VALID_AREAS_JSON_PATH) as f:
        raw = json.load(f)
    geojsonw = {int(k): v for k, v in raw.items()}

    areas = []
    for i in selected:
        geojson_dict = json.loads(geojsonw[int(i)])
        gdf_from_str = gpd.GeoDataFrame.from_features(geojson_dict["features"])
        areas.append(gdf_from_str.area.iloc[0])
    total_area   = sum(areas)
    points       = [round((area / total_area) * total_points) for area in areas]
    adjusted     = [round(p / divisibility) * divisibility for p in points]
    diff         = total_points - sum(adjusted)
    if diff > 0:
        adjusted[0] += diff
    return {f"{i}": adjusted[i] for i in range(len(adjusted))}

def random_points_in_polygon(polygon, num_points, buffer, dist):
    minx, miny, maxx, maxy = polygon.bounds
    polygon2 = polygon.buffer(-buffer)
    xs = np.arange(minx, maxx, dist)
    ys = np.arange(miny, maxy, dist)
    X, Y = np.meshgrid(xs, ys)
    Xf, Yf = X.ravel(), Y.ravel()
    mask = sv.contains(polygon2, Xf, Yf)
    valid = np.column_stack((Xf[mask], Yf[mask]))
    if len(valid) < num_points:
        return random_points_in_polygon(polygon, num_points, buffer * 0.95, dist * 0.9)
    if len(valid) > num_points:
        idxs = np.random.choice(valid.shape[0], num_points, replace=False)
        valid = valid[idxs]
    return [Point(x, y) for x, y in valid]

def wireingcluster(points, serieslength, capacity):
    df = pd.DataFrame(points)
    supercluster, superdictinfo = {}, {}
    for grp in df['group'].unique():
        dfg = df[df['group'] == grp]
        kmeans = KMeansConstrained(
            n_clusters=int(len(dfg) / serieslength),
            size_min=serieslength,
            size_max=serieslength,
            random_state=42
        )
        labels = kmeans.fit_predict(dfg[['x', 'y']])
        dfg['clusterlabel'] = labels
        df.loc[df['group'] == grp, 'clusterlabel'] = labels

        centroids = kmeans.cluster_centers_
        kmeansc = KMeansConstrained(
            n_clusters=int(len(set(labels)) / capacity),
            size_min=1,
            size_max=capacity,
            random_state=40
        )
        labelsc    = kmeansc.fit_predict(centroids)
        centroidsc = kmeansc.cluster_centers_
        supercluster[grp] = {int(i): centroidsc[i].tolist() for i in range(len(centroidsc))}

        super_labels = np.array([labelsc[label] for label in labels])
        dfg['superclusterlabel'] = super_labels
        df.loc[df['group'] == grp, 'superclusterlabel'] = super_labels

        dictinfo = {}
        for sl in set(super_labels):
            sl = int(sl)
            dictinfo[sl] = {}
            for cl in set(labels[super_labels == sl]):
                cl = int(cl)
                pts = dfg[dfg['clusterlabel'] == cl][['x', 'y']].values
                stacked = np.vstack((pts, centroidsc[sl]))
                dictinfo[sl][cl] = stacked.tolist()
        superdictinfo[grp] = dictinfo

    return df.to_dict(orient='records'), supercluster, superdictinfo

@app.route('/submitselection', methods=['POST'])
def submit_selection():
    data          = request.get_json()
    selected      = data.get('selected', [])
    numPoints     = data.get('numPoints', 500)
    divisibility  = data.get('divisibility', 5)
    capacity      = data.get('capacity', 4)
    distance      = data.get('distance', 100) * 0.0001
    offset        = data.get('offset', 50) * 0.0001

    pointinfo = distribute_points(selected, numPoints, divisibility * capacity)

    # **Reload valid areas** for generating random points
    with open(VALID_AREAS_JSON_PATH) as f:
        raw = json.load(f)
    geojsonw = {int(k): v for k, v in raw.items()}

    points = []
    for key, cnt in pointinfo.items():
        idx = int(key) + 1
        geojson_dict = json.loads(geojsonw[idx])
        gdf_str = gpd.GeoDataFrame.from_features(geojson_dict["features"])
        rand_pts = random_points_in_polygon(
            gdf_str.geometry.iloc[0], cnt, offset, distance
        )
        for p in rand_pts:
            points.append({"x": p.x, "y": p.y, "group": key})
    #print(points)
    pointinfo, supercluster, superdictinfo = wireingcluster(points, divisibility, capacity)

    # **Persist clustering** results
    with open(CLUSTER_DATA_JSON_PATH, "w") as f:
        json.dump({
            "pointinfo": pointinfo,
            "supercluster": supercluster,
            "superdictinfo": superdictinfo
        }, f)

    return jsonify({
        "message": "Selection received!",
        "selected_count": len(selected),
        "received_keys": selected,
        "points": pointinfo,
        "supercluster": supercluster,
    }), 200

@app.route('/updatepointsinfo', methods=['POST'])
def updatepointsinfo():
    data         = request.get_json()
    points       = data.get('points', [])
    numPoints    = data.get('numPoints', 500)
    divisibility = data.get('divisibility', 5)
    capacity     = data.get('capacity', 4)
    distance     = data.get('distance', 100) * 0.0001
    #print(points)
    pointinfo, supercluster, superdictinfo = wireingcluster(points, divisibility, capacity)
    with open(CLUSTER_DATA_JSON_PATH, "w") as f:
        json.dump({
            "pointinfo": pointinfo,
            "supercluster": supercluster,
            "superdictinfo": superdictinfo
    }, f)
    return jsonify({
        "message": "Selection received!",
        "points": pointinfo,
        "supercluster": supercluster,
    }), 200

def process_infra(infra_gdf, distance_threshold, all_points_array):
    points = [Point(x, y) for x, y in all_points_array]
    roots_indices = []
    filtered_nearest = []

    for idx, pt in enumerate(points):
        infra_gdf['distance'] = infra_gdf.geometry.distance(pt)
        closest = infra_gdf.geometry.iloc[infra_gdf['distance'].idxmin()]
        nearest_pt = nearest_points(closest, pt)[0]
        if pt.distance(nearest_pt) < distance_threshold:
            roots_indices.append(idx)
            filtered_nearest.append(nearest_pt)

    coords = np.array(all_points_array)
    roots_set = set(roots_indices)
    clusters = {r: [] for r in roots_indices}
    for i in range(len(points)):
        if i not in roots_set:
            d = np.linalg.norm(coords[i] - coords[roots_indices], axis=1)
            r = roots_indices[d.argmin()]
            clusters[r].append(i)
    for r in clusters:
        clusters[r].append(r)

    mst_edges = {}
    for r, members in clusters.items():
        if len(members) < 2:
            mst_edges[r] = []
        else:
            G = nx.Graph()
            for i, j in combinations(members, 2):
                w = np.linalg.norm(coords[i] - coords[j])
                G.add_edge(i, j, weight=w)
            mst_edges[r] = list(nx.minimum_spanning_tree(G).edges(data=True))

    root_to_infra = {r: filtered_nearest[k] for k, r in enumerate(roots_indices)}
    all_paths = []
    for r, edges in mst_edges.items():
        if not edges:
            continue
        Gc = nx.Graph()
        for u, v, _ in edges:
            Gc.add_edge(u, v)
        infra_coord = (root_to_infra[r].x, root_to_infra[r].y)
        for t in clusters[r]:
            if t == r:
                continue
            path = nx.shortest_path(Gc, r, t)
            all_paths.append([list(infra_coord)] + [list(all_points_array[n]) for n in path])
    return all_paths

@app.route('/getroadinfo', methods=['POST'])
def getroadinfo():
    # reload infra & clusters
    if not os.path.exists(INFRA_GEOJSON_PATH) or not os.path.exists(CLUSTER_DATA_JSON_PATH):
        return jsonify({'error': 'Required data not available'}), 404
    infra_gdf = gpd.read_file(INFRA_GEOJSON_PATH)
    with open(CLUSTER_DATA_JSON_PATH) as f:
        cluster_data = json.load(f)
    superdictinfo = cluster_data["superdictinfo"]

    all_points = []
    for outer in superdictinfo.values():
        for inner in outer.values():
            for arr in inner.values():
                all_points.append(arr)
    all_points_array = np.vstack(all_points)
    dataroad = process_infra(infra_gdf, 0.012, all_points_array)
    return jsonify({
        "message": "Selection received!",
        "route": dataroad
    })

def compute_tsp_routes(superdictinfo):
    all_pathwire = []
    for grp in superdictinfo.values():
        for sub in grp.values():
            for arr in sub.values():
                pts = np.asarray(arr)
                n   = pts.shape[0]
                D   = cdist(pts, pts)
                visited = np.zeros(n, bool)
                route   = [n-1]
                visited[n-1] = True
                curr  = n-1
                for _ in range(n-1):
                    d = D[curr]
                    d[visited] = np.inf
                    nxt = d.argmin()
                    route.append(nxt)
                    visited[nxt] = True
                    curr = nxt
                all_pathwire.append(pts[route].tolist())
    return all_pathwire

@app.route('/getwireinfo', methods=['POST'])
def getwireinfo():
    if not os.path.exists(CLUSTER_DATA_JSON_PATH):
        return jsonify({'error': 'Cluster data not available'}), 404
    with open(CLUSTER_DATA_JSON_PATH) as f:
        cluster_data = json.load(f)
    superdictinfo   = cluster_data["superdictinfo"]
    dataconnection = compute_tsp_routes(superdictinfo)
    return jsonify({
        "message": "Selection received!",
        "route": dataconnection
    })

@app.route("/")
def home():
    return "Solar Farm Site Selection API"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug = True)
