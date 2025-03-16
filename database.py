import pandas as pd
import sqlite3
import os
import re
from rich import print
from rich.progress import track
from rich.table import Table

dir = os.path.dirname(__file__)
genres_path = os.path.join(dir, 'dataset/genres.csv')
metadata_path = os.path.join(dir, 'dataset/annotations_final.csv')
metadata_db_path = os.path.join(dir, 'dataset/metadata.db')
tags_db_path = os.path.join(dir, 'dataset/tags.db')
genres_db_path = os.path.join(dir, 'dataset/genres.db')

def _extract_song_id(path):
  """Extrae el identificador de la canción sin los tiempos del fragmento."""
  return re.sub(r"-\d+-\d+\.mp3$", "", path)

def _assign_missing_genres(genres_df):
  """Asigna a los fragmentos sin género los géneros mayoritarios de su canción."""
  print("\n[bold cyan]▶ Asignando géneros a fragmentos sin género...[/bold cyan]")

  genres_df["song_id"] = genres_df["mp3_path"].apply(_extract_song_id)
  genre_columns = [col for col in genres_df.columns if col not in ["clip_id", "mp3_path", "song_id"]]
  genre_majority = genres_df.groupby("song_id")[genre_columns].max()

  # **No tenemos en cuenta canciones donde todos los fragmentos tienen no tienen género**
  valid_songs = genre_majority[genre_majority.sum(axis=1) > 0]
  missing_genres_mask = genres_df[genre_columns].sum(axis=1) == 0
  missing_genres_df = genres_df[missing_genres_mask]

  for index, row in track(missing_genres_df.iterrows(), total=len(missing_genres_df), description="Asignando géneros..."):
    song_id = row["song_id"]
    if song_id in valid_songs.index: 
      genres_df.loc[index, genre_columns] = valid_songs.loc[song_id]

  genres_df.drop(columns=["song_id"], inplace=True)

  return genres_df

def _print_table(title, df):
    """Muestra un DataFrame como una tabla en la terminal con rich."""
    table = Table(title=f"[bold]{title}[/bold]", show_lines=True)
    
    for col in df.columns[:6]:  # Solo mostramos las primeras 6 columnas para no hacer la tabla muy grande
      table.add_column(col, style="cyan", overflow="fold")
    table.add_column("...", style="cyan", overflow="fold")

    for _, row in df.head(5).iterrows():  # Solo mostramos 5 filas
      table.add_row(*[str(row[col]) for col in df.columns[:6]])

    print(table)


def load_data():
  if not os.path.exists(genres_path):
    print(f"[bold red]❌ Error:[/bold red] El archivo {genres_path} no existe.")
    return
  print("\n[bold green]📂 Cargando datos...[/bold green]")
  df = pd.read_csv(genres_path, delimiter='\t')
  genres = [
    "clip_id", "hard rock", "clasical", "classical", "classic", 
    "soft rock", "jazz", "folk", "ambient", "new age", 
    "electronic", "opera", "operatic", "country", "funky", "funk",
    "irish", "arabic", "celtic", "eastern", "middle eastern", "oriental",
    "spanish", "jazzy", "orchestra", "orchestral",
    "electro", "reggae", "tribal", "electronica", "heavy metal", "disco", "industrial", 
    "pop", "punk", "blues", "indian", "india", "rock", "dance", "techno", "house", "rap", 
    "metal", "hip hop", "trance", "baroque", "drone", "male opera", "female opera", "soprano",
    "mp3_path"
  ]
  tags = [
    "clip_id", "singer", "duet", "plucking", "world", "bongos", "harpsichord", 
    "female singing", "sitar", "chorus", "male vocal", "vocals", "clarinet", 
    "heavy", "silence", "beats", "men", "woodwind", "chimes", "foreign", 
    "horns", "female", "eerie", "spacey", "guitar", "quiet",
    "banjo", "solo", "violins", "female voice", "wind", "happy", "synth", 
    "trumpet", "percussion", "drum", "airy", "voice", "repetitive", 
    "birds", "space", "strings", "bass", "harpsicord", "medieval", "male voice", "girl", "keyboard", 
    "acoustic", "loud", "string", "drums", "chanting",
    "organ", "talking", "choral", "weird", "fast",
    "acoustic guitar", "electric guitar", "classical guitar", "violin",
    "male singer", "man singing", "dark", "horn", 
    "lol", "low", "instrumental", "chant", "strange", "synthesizer", "modern", 
    "bells", "man", "deep", "fast beat", "hard", "harp", "jungle", "lute", "female vocal", 
    "oboe", "mellow", "viola", "light", "echo", "piano", "male vocals", 
    "old", "flutes", "sad", "sax", "slow", "male", "scary", "woman", 
    "woman singing", "piano solo", "guitars", "singing", "cello", "calm", 
    "female vocals", "voices", "different", "clapping", "monks", "flute", 
    "beat", "upbeat", "soft", "noise", "choir", "female singer", "quick", "water", 
    "women", "fiddle", "mp3_path", "english", "electric", "no voice", "no strings", "no piano", "no voices",
    "no beat", "no singing", "not classical", "not rock", "no guitar", "no vocal", "no vocals", "no flute",
    "no singer", "no drums", "not opera", "not english"
  ]

  genres_df = pd.DataFrame(df[genres])
  genres_df = genres_df.astype({col: 'bool' for col in genres_df.columns if col not in ['clip_id', 'mp3_path']})
  tags_df = pd.DataFrame(df[tags])
  tags_df = tags_df.astype({col: 'bool' for col in tags_df.columns if col not in ['clip_id', 'mp3_path']})
  
  print(f"\n✅ [bold cyan]Géneros cargados:[/bold cyan] {genres_df.shape[1] - 2}")
  print(f"✅ [bold cyan]Tags cargados:[/bold cyan] {tags_df.shape[1] - 2}")

  # Convertimos los datos a boolean y fusionamos los campos parecidos con OR
  genre_merges = {
    'classical': ['clasical', 'classic', 'baroque', 'orchestra', 'orchestral'],
    'funk': ['funky'],
    'jazz': ['jazzy'],
    'arabic': ['middle eastern'],
    'indian': ['india'],
    'electronic': ['electronica', 'electro', 'techno', 'house', 'trance', 'disco', 'industrial', 'dance'],
    'folk': ['irish', 'celtic', 'spanish', 'tribal'],
    'ambient': ['new age', 'drone'],
    'opera': ['operatic', 'male opera', 'female opera', 'soprano'],
    'rock': ['hard rock', 'soft rock'],
    'metal': ['heavy metal'],
    'hip hop': ['rap']
  }
  
  for main_genre, similar_genres in genre_merges.items():
    genres_df[main_genre] = genres_df[main_genre] | genres_df[similar_genres].any(axis=1)
    genres_df = genres_df.drop(columns=similar_genres)

  _print_table("\nGéneros", genres_df.head())

  no_genre_count = genres_df[genres_df.drop(columns=['clip_id', 'mp3_path']).sum(axis=1) == 0].shape[0]
  print(f"\n[bold yellow]⚠️  {no_genre_count} fragmentos sin género serán procesados.[/bold yellow]")

  tag_merges = {
    'male vocals': ['male vocal', 'male voice', 'men', 'male', 'male singer', 'man singing'],
    'female vocals': ['female vocal', 'female voice', 'woman singing', 'female singer', 'girl', 'female singing', 'female', 'women', 'woman'],
    'vocals': ['voice', 'voices', 'singer', 'chanting', 'singing'],
    'guitar': ['guitars'],
    'calm': ['quiet', 'soft', 'slow', 'mellow', 'light'],
    'dark': ['eerie', 'scary'],
    'happy': ['upbeat'],
    'fast beat': ['fast', 'quick'],
    'strings': ['violins', 'fiddle', 'violin', 'viola', 'cello', 'string', 'harp', 'bass'],
    'flute': ['flutes'],
    'woodwind': ['clarinet', 'oboe'],
    'piano': ['piano solo', 'harpsicord', 'harpsichord', 'keyboard'],
    'drum': ['drums'],
    'choral': ['chorus', 'monks', 'choir'],
    'synthesizer': ['synth'], 
  }

  for main_tag, similar_tags in tag_merges.items():
    tags_df[main_tag] = tags_df[main_tag] | tags_df[similar_tags].any(axis=1)
    tags_df = tags_df.drop(columns=similar_tags)
  
  _print_table("\nTags",tags_df.head())

  no_tags_count = tags_df[tags_df.drop(columns=['clip_id','mp3_path']).sum(axis=1) == 0].shape[0]
  print(f"\n[bold yellow]⚠️  {no_tags_count} fragmentos sin tags serán procesados.[/bold yellow]")

  print(f"\n✅ [bold cyan]Géneros fusionados:[/bold cyan] {genres_df.shape[1] - 2}")
  print(f"✅ [bold cyan]Tags fusionados:[/bold cyan] {tags_df.shape[1] - 2}")

  _assign_missing_genres(genres_df)

  # Cuantos fragmentos hay sin género
  no_genre = genres_df[genres_df.drop(columns=['clip_id', 'mp3_path']).sum(axis=1) == 0]
  print(f"\n[bold yellow]⚠️  {no_genre.shape[0]} fragmentos sin género después de asignar géneros.[/bold yellow]")
  _print_table("\nFragmentos sin género", no_genre.head())

  # Borramos los fragmentos sin género
  genres_df = genres_df[genres_df.drop(columns=['clip_id', 'mp3_path']).sum(axis=1) > 0]
  print(f"\n✅ [bold cyan]Canciones con géneros asignados:[/bold cyan] {genres_df.shape[0]}")

  # Borramos las canciones sin género del dataframe de tags
  tags_df = tags_df[tags_df['clip_id'].isin(genres_df['clip_id'])]
  print(f"✅ [bold cyan]Canciones con tags asignados:[/bold cyan] {tags_df.shape[0]}")


  if os.path.exists(genres_db_path):
    os.remove(genres_db_path)
  if os.path.exists(tags_db_path):
    os.remove(tags_db_path)

  conn = sqlite3.connect(genres_db_path)
  genres_df.to_sql('genres', conn, index=False)
  conn.close()

  conn = sqlite3.connect(tags_db_path)
  tags_df.to_sql('tags', conn, index=False)
  conn.close()



if __name__ == "__main__":
  load_data()
