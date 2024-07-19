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
# Step 2: Exploratory Data Analysis (EDA)
def perform_eda(data):
    print("Dataset Overview:")
    print(data.info())
    print("\nDescriptive Statistics:")
    print(data.describe())

    print("\nMissing Values:")
    print(data.isnull().sum())

    # Goals scored vs. Possession (Scatter Plot)
    team_avg = data.groupby('team').agg({'poss': 'mean', 'gf': 'mean'}).reset_index()
    team_avg = team_avg.sort_values('gf', ascending=False)
    
    # Split teams into two groups
    top_half = team_avg.iloc[:10]
    bottom_half = team_avg.iloc[10:]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    fig.suptitle('Average Goals Scored vs. Average Possession by Team', fontsize=16)
    
    # Plot top half
    ax1.scatter(top_half['poss'], top_half['gf'], s=100)
    for i, txt in enumerate(top_half['team']):
        ax1.annotate(txt, (top_half['poss'].iloc[i], top_half['gf'].iloc[i]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=10, 
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    ax1.set_xlabel('Average Possession (%)')
    ax1.set_ylabel('Average Goals Scored')
    ax1.set_title('Top 10 Teams')
# Plot bottom half
    ax2.scatter(bottom_half['poss'], bottom_half['gf'], s=100)
    for i, txt in enumerate(bottom_half['team']):
        ax2.annotate(txt, (bottom_half['poss'].iloc[i], bottom_half['gf'].iloc[i]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=10, 
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    ax2.set_xlabel('Average Possession (%)')
    ax2.set_ylabel('Average Goals Scored')
    ax2.set_title('Bottom 10 Teams')
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'goals_vs_possession.png'))
    plt.show()

    # Distribution of shots on target (Box Plot)
    plt.figure(figsize=(16, 10))
    sns.boxplot(x='team', y='sot', data=data)
    plt.title('Distribution of Shots on Target by Team')
    plt.xlabel('Team')
    plt.ylabel('Shots on Target')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'shots_on_target_distribution.png'))
    plt.show()
