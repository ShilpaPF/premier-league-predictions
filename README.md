# premier-league-predictions
Project report for Premier League Football Predictions
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Step 1: Load and preprocess data
file_path = "C:/Users/rejot/OneDrive - University of Hertfordshire/PROJECT/Premier_League_Data.csv"
df = pd.read_csv(file_path)

print("Columns in the dataset:", df.columns.tolist())
print("\nFirst few rows of the dataset:")
print(df[['date', 'team', 'opponent', 'result', 'gf', 'ga', 'poss', 'sh', 'sot', 'pk', 'pkatt']].head())

