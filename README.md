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
# Step 2: Analyze repeated matches and penalties
def analyze_matches(data):
    # Count matches between each pair of teams
    match_counts = data.groupby(['team', 'opponent']).size().reset_index(name='match_count')
    repeated_matches = match_counts[match_counts['match_count'] > 1]
    
    # Analyze penalties
    penalties = data[data['pk'] > 0]
    
    print("\nTeams that played against each other more than once:")
    print(repeated_matches)
    
    print("\nMatches with penalties:")
    print(penalties[['date', 'team', 'opponent', 'pk', 'pkatt']])

    # Check if there's any injury-related information
    if 'injuries' in data.columns:
        injuries = data[data['injuries'].notna()]
        print("\nMatches with reported injuries:")
        print(injuries[['date', 'team', 'opponent', 'injuries']])
    else:
        print("\nNo specific injury information found in the dataset.")

analyze_matches(df)


