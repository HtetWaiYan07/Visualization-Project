import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import interact
import ipywidgets as widgets
from sklearn.decomposition import PCA

# Load the dataset
file_path = '/Users/corydangus/Downloads/winequality-red.csv'
wine_data = pd.read_csv(file_path)

# Add a unique identifier for each wine row
wine_data['Wine ID'] = wine_data.index + 1

# 1. Correlation Heatmap (Interactive)
def plot_correlation():
    fig_corr = px.imshow(
        wine_data.corr(),
        text_auto=True,
        color_continuous_scale='Viridis',
        title="Correlation Heatmap of Wine Features",
        labels=dict(color="Correlation Coefficient"),
    )
    fig_corr.show()

plot_correlation()

# 2. Alcohol Content Influence on Quality
def plot_alcohol_quality():
    fig_alcohol = px.box(
        wine_data,
        x='quality',
        y='alcohol',
        color='quality',
        title="Alcohol Content vs Wine Quality",
        labels={"quality": "Wine Quality", "alcohol": "Alcohol Content"},
    )
    fig_alcohol.show()

plot_alcohol_quality()

# 3. Interaction of Acidity Levels with Quality (2D Scatter Plot)
def plot_acidity_quality_2d():
    fig_acidity = px.scatter(
        wine_data,
        x='fixed acidity',
        y='volatile acidity',
        color='quality',
        color_continuous_scale='Viridis',
        title="Interaction of Acidity Levels and Wine Quality (2D)",
        labels={"fixed acidity": "Fixed Acidity", "volatile acidity": "Volatile Acidity", "quality": "Quality"},
    )
    fig_acidity.update_layout(
        xaxis=dict(tickmode='linear', dtick=1),
        yaxis=dict(tickmode='linear', dtick=1)
    )
    fig_acidity.show()

plot_acidity_quality_2d()

# 4. Clusters of Wines Based on Chemical Composition (PCA)
def plot_pca_clusters(selected_quality):
    try:
        if selected_quality == 'All':
            filtered_data = wine_data
        else:
            filtered_data = wine_data[wine_data['quality'] == int(selected_quality)]

        if filtered_data.empty:
            print(f"No data available for quality {selected_quality}.")
            return

        features = filtered_data.drop(['quality', 'Wine ID'], axis=1)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        filtered_data['PCA1'] = pca_result[:, 0]
        filtered_data['PCA2'] = pca_result[:, 1]

        fig_pca = px.scatter(
            filtered_data,
            x='PCA1',
            y='PCA2',
            color='quality',
            title=f"PCA Clustering of Wines (Quality: {selected_quality})",
            labels={"PCA1": "Principal Component 1", "PCA2": "Principal Component 2", "quality": "Wine Quality"},
            color_continuous_scale='Viridis',
        )
        fig_pca.show()
    except Exception as e:
        print(f"An error occurred: {e}")

# Dropdown widgets for PCA
dimensions_options = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'alcohol']
pca_quality_options = ['All'] + sorted(wine_data['quality'].unique().astype(str).tolist())
interact(
    plot_pca_clusters,
    selected_quality=widgets.Dropdown(options=pca_quality_options, description='Quality:')
)

# 5. Outliers in Features (Interactive Parallel Coordinate Plot)
def plot_parallel(selected_quality, selected_dimensions):
    try:
        # Filter data based on selected quality
        if selected_quality == 'All':
            filtered_data = wine_data
        else:
            filtered_data = wine_data[wine_data['quality'] == int(selected_quality)]

        # Check for empty data
        if filtered_data.empty:
            print(f"No data available for quality {selected_quality}.")
            return

        # Optimize performance by selecting only required columns
        filtered_data = filtered_data[['quality'] + list(selected_dimensions)]

        # Create Parallel Coordinate Plot
        fig_parallel = px.parallel_coordinates(
            filtered_data,
            dimensions=selected_dimensions,
            color='quality',
            color_continuous_scale='Viridis',
            title=f"Parallel Coordinate Plot of Wine Features (Quality: {selected_quality})",
            labels={"quality": "Wine Quality"},
        )
        fig_parallel.update_layout(
            title_text=f"Parallel Coordinate Plot (Quality: {selected_quality})",
            title_x=0.5,
        )
        fig_parallel.show()
    except Exception as e:
        print(f"An error occurred: {e}")

# Dropdown widgets for Parallel Coordinates
def update_parallel_plot(selected_quality, selected_dimensions):
    plot_parallel(selected_quality, selected_dimensions)

# Create interactive widgets for quality and dimensions
parallel_quality_options = ['All'] + sorted(wine_data['quality'].unique().astype(str).tolist())
dimensions_options = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'alcohol']

interact(
    update_parallel_plot,
    selected_quality=widgets.Dropdown(
        options=parallel_quality_options,
        value='All',
        description='Quality:',
        style={'description_width': 'initial'}
    ),
    selected_dimensions=widgets.SelectMultiple(
        options=dimensions_options,
        value=tuple(dimensions_options),
        description='Dimensions:',
        style={'description_width': 'initial'}
    )
)