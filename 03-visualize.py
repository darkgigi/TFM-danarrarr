import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import sqlite3
import numpy as np
import networkx as nx

dir = os.path.dirname(__file__)
genres_db_path = 'dataset/genres.db'
metadata_db_path = 'dataset/metadata.db'
tags_db_path = 'dataset/tags.db'

def load_data():
  conn = sqlite3.connect(genres_db_path)
  genres_df = pd.read_sql_query("SELECT * FROM genres", conn)
  conn.close()

  conn = sqlite3.connect(tags_db_path)
  tags_df = pd.read_sql_query("SELECT * FROM tags", conn)
  conn.close()

  conn = sqlite3.connect(metadata_db_path)
  metadata_df = pd.read_sql_query("SELECT * FROM metadata", conn)
  conn.close()

  return genres_df, tags_df, metadata_df

def visualize_density_genres(genres_df):
  '''Visualiza la distribución relativa de géneros en el dataset.'''
  genres_df = genres_df.drop(columns=['clip_id', 'mp3_path'])
  genres_density = genres_df.sum(axis=0)
  
  # Normalización: convertir a porcentaje
  total_fragments = genres_density.sum()
  genres_density = (genres_density / total_fragments) * 100
  
  genres_density = genres_density.sort_values(ascending=False)
  
  plt.figure(figsize=(10, 6))
  sns.barplot(x=genres_density.values, y=genres_density.index, hue=genres_density.index, palette='viridis', legend=False)
  plt.title("Densidad de géneros en el dataset (Porcentaje)")
  plt.xlabel("Porcentaje de fragmentos (%)")
  plt.ylabel("Género")
  plt.show()

def visualize_density_tags(tags_df):
  '''Visualiza la distribución relativa de tags en el dataset.'''
  tags_df = tags_df.drop(columns=['clip_id', 'mp3_path'])
  tags_density = tags_df.sum(axis=0)
  
  # Normalización: convertir a porcentaje
  total_fragments = tags_density.sum()
  tags_density = (tags_density / total_fragments) * 100
  
  tags_density = tags_density.sort_values(ascending=False)
  
  plt.figure(figsize=(10, 6))
  sns.barplot(x=tags_density.values, y=tags_density.index, hue=tags_density.index, palette='viridis', legend=False)
  plt.title("Densidad de tags en el dataset (Porcentaje)")
  plt.xlabel("Porcentaje de fragmentos (%)")
  plt.ylabel("Tag")
  plt.show()

def visualize_stacked_bar(tags_df, top_n=10):
  '''Muestra un gráfico de barras apiladas con la distribución de los top N tags.'''
  tags_df = tags_df.drop(columns=['clip_id', 'mp3_path'])
  
  # Seleccionamos los top_n tags más comunes
  top_tags = tags_df.sum().sort_values(ascending=False).head(top_n).index
  filtered_df = tags_df[top_tags]
  
  # Normalizar por total de fragmentos
  tag_distribution = filtered_df.sum() / filtered_df.shape[0] * 100

  # Graficar
  tag_distribution.plot(kind="bar", stacked=True, colormap="viridis", figsize=(10, 6))
  plt.title("Distribución de los tags más frecuentes (%)")
  plt.xlabel("Género")
  plt.ylabel("Porcentaje de fragmentos con este tag")
  plt.xticks(rotation=45)
  plt.show()


def visualize_tag_correlation(tags_df, top_n=30):
  '''Muestra una matriz de co-ocurrencia entre los tags más frecuentes.'''
  tags_df = tags_df.drop(columns=['clip_id', 'mp3_path'])
  
  # Seleccionamos los top_n tags más comunes
  top_tags = tags_df.sum().sort_values(ascending=False).head(top_n).index
  filtered_df = tags_df[top_tags]
  
  # Calculamos la correlación de co-ocurrencia
  correlation_matrix = filtered_df.T.dot(filtered_df)

  plt.figure(figsize=(12, 8))
  sns.heatmap(correlation_matrix, cmap="viridis", annot=False)
  plt.title("Co-ocurrencia de tags en el dataset")
  plt.xlabel("Tags")
  plt.ylabel("Tags")
  plt.show()


def visualize_tag_network(tags_df, top_n=30):
  '''Genera una red de relaciones entre los tags más comunes.'''
  tags_df = tags_df.drop(columns=['clip_id', 'mp3_path'])
  
  # Seleccionamos los top_n tags más frecuentes
  top_tags = tags_df.sum().sort_values(ascending=False).head(top_n).index
  filtered_df = tags_df[top_tags]
  
  # Construimos la red
  G = nx.Graph()
  
  WEIGHT_THRESHOLD = 50  # Ajusta este valor según el tamaño de tu dataset

  for tag1 in top_tags:
      for tag2 in top_tags:
          if tag1 != tag2:
              weight = (filtered_df[tag1] & filtered_df[tag2]).sum()
              if weight > WEIGHT_THRESHOLD:
                  G.add_edge(tag1, tag2, weight=weight)

  # Dibujar la red
  plt.figure(figsize=(10, 8))
  pos = nx.spring_layout(G, seed=42)
  edges = G.edges(data=True)
  nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=2000, font_size=10)
  plt.title("Red de relaciones entre tags")
  plt.show()



if __name__ == "__main__":
  genres_df, tags_df, metadata_df = load_data()
  visualize_density_genres(genres_df)
  visualize_density_tags(tags_df)
  visualize_stacked_bar(tags_df)
  visualize_tag_correlation(tags_df)
  visualize_tag_network(tags_df)