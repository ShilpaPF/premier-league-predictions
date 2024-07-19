# premier-league-predictions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
# Create a directory to save charts
charts_dir = "premier_league_charts"
os.makedirs(charts_dir, exist_ok=True)
