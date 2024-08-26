# Premier League Football Predictions

## Dataset

This project uses the Premier League dataset, which contains detailed information about football matches in the English Premier League. The dataset includes various statistics for each match, such as goals scored, expected goals, possession, shots on target, and more.. 

### Dataset Source
- Available on Kaggle: [Premier League Dataset],(https://www.kaggle.com/datasets/tanmaynema/premier-league-match-data-2019-2023)

### Data Privacy
No personal data is used. All data is publicly available match statistics.

### Key Features
The dataset includes 29 features.

## Purpose

This project aims to analyze Premier League football data to predict match outcomes and provide insights into team performance. It uses machine learning techniques, specifically a Random Forest classifier, to identify key factors influencing match results and visualizes various aspects of team and player statistics.

## Features

- Data preprocessing and feature engineering.
- Exploratory Data Analysis (EDA) with visualizations.
- Outlier removal using the IQR method.
- Random Forest model for match outcome prediction.
- Feature importance analysis.

## Key Functions

### remove_outliers(df, columns)

Removes outliers using the Interquartile Range (IQR) method.

**Parameters:**
- df (pandas.DataFrame): The input dataframe.
- columns (list): List of column names to check.

**Returns:**
- pandas.DataFrame: Dataframe with outliers removed.

## Output

The script generates visualizations and prints summaries, including:
- Distribution of goals scored.
- Home vs Away performance comparisons.
- Feature importance for predictions.
- Model accuracy and classification report.

## License

This project is open-source under the [MIT License](LICENSE).

## Acknowledgements

- Dataset provided by Kaggle.
- Inspired by football analytics communities.
