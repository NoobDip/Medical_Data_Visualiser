import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = df['weight']/((df['height']/100))**2
df.loc[df['overweight'] <= 25, 'overweight'] = int(0)
df.loc[df['overweight'] > 25, 'overweight'] = int(1)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = df['cholesterol'].replace([1, 2, 3], [0, 1, 1])
df['gluc'] = df['gluc'].replace([1, 2, 3], [0, 1, 1])

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    cat_columns = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=cat_columns)
  
    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    #df_cat = None
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # Draw the catplot with 'sns.catplot()'
    g = sns.catplot(
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar',
        data=df_cat,
        height=5,
        aspect=1.2
    )                                                 

    # Get the figure for the output
    fig = g.fig
    fig.savefig('catplot.png')
    return fig
  
# Draw Heat Map
def draw_heat_map():
  
    # Clean the data
    df_heat = df[df['ap_lo']<=df['ap_hi']]
    df_heat = df_heat[df_heat['weight'] <= df['weight'].quantile(0.975)]
    df_heat = df_heat[df_heat['height'] <= df['height'].quantile(0.975)]
    df_heat = df_heat[df_heat['weight'] >= df['weight'].quantile(0.025)]
    df_heat = df_heat[df_heat['height'] >= df['height'].quantile(0.025)]
  
    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, annot=True, fmt='.1f', mask=mask, vmax=.8, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    fig.savefig('heatmap.png')
    return fig
