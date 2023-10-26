# Data wrangling
import numpy as np
import pandas as pd  

# Data viz
from gtda.plotting import plot_point_cloud

# TDA magic
from gtda.mapper import (
    CubicalCover,
    make_mapper_pipeline,
    Projection,
    plot_static_mapper_graph,
    plot_interactive_mapper_graph
)

# ML tools
from sklearn import datasets
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA

def get_graph(X, filter_func, n_intervals=10, overlap_frac=0.5, min_samples=5, eps=5):
    cover = CubicalCover(n_intervals=n_intervals, overlap_frac=overlap_frac)
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    
    # Configure parallelism of clustering step
    n_jobs = 1

    # Initialise pipeline
    pipe = make_mapper_pipeline(
        filter_func=filter_func,
        cover=cover,
        clusterer=clusterer,
        verbose=False,
        n_jobs=n_jobs
    )

    graph = pipe.fit_transform(X)
    fig = plot_static_mapper_graph(pipe, X)
    fig.show(config={'scrollZoom': True})

    g = graph.to_networkx()
    node2pts = {i: list(v) for i, v in enumerate(graph.vs["node_elements"])}
    return g, node2pts
    