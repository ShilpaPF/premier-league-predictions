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
# Step 1: Load and preprocess data
file_path = "C:/Users/rejot/OneDrive - University of Hertfordshire/PROJECT/Premier_League_Data.csv"
df = pd.read_csv(file_path)
# Select the top 20 teams based on the number of matches played
top_teams = df['team'].value_counts().head(20).index.tolist()
df = df[df['team'].isin(top_teams)]
