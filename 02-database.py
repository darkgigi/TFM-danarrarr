import pandas as pd
import sqlite3
import os
import re
from rich import print
from rich.progress import track
from rich.table import Table
import librosa
from joblib import Parallel, delayed
import soundfile as sf
import numpy as np

dir = os.path.dirname(__file__)
genres_path = os.path.join(dir, 'dataset/genres.csv')
metadata_path = os.path.join(dir, 'dataset/annotations_final.csv')
metadata_db_path = os.path.join(dir, 'dataset/metadata.db')
tags_db_path = os.path.join(dir, 'dataset/tags.db')
genres_db_path = os.path.join(dir, 'dataset/genres.db')
features_db_path = os.path.join(dir, 'dataset/features.db')

def _extract_song_id(path):
	"""Extrae el identificador de la canción sin los tiempos del fragmento."""
	return re.sub(r"-\d+-\d+\.mp3$", "", path)

def _assign_missing_genres(genres_df):
	"""Asigna a los fragmentos sin género los géneros mayoritarios de su canción si al
	menos el 50% de los fragmentos tienen género asignado."""

	print("\n[bold cyan]➡️  Asignando géneros a fragmentos sin género...[/bold cyan]")

	genres_df["song_id"] = genres_df["mp3_path"].apply(_extract_song_id)
	genre_columns = [col for col in genres_df.columns if col not in ["clip_id", "mp3_path", "song_id"]]

	total_fragments = genres_df.groupby("song_id").size()
	fragments_with_genre = genres_df[genre_columns].groupby(genres_df["song_id"]).sum().gt(0).sum(axis=1)

	valid_songs = fragments_with_genre / total_fragments >= 0.5

	genre_majority = genres_df.groupby("song_id")[genre_columns].max()
	# **No tenemos en cuenta canciones donde todos los fragmentos tienen no tienen género**
	genre_majority = genre_majority.loc[valid_songs[valid_songs].index]

	missing_genres_mask = genres_df[genre_columns].sum(axis=1) == 0
	missing_genres_df = genres_df[missing_genres_mask]

	for index, row in track(missing_genres_df.iterrows(), total=len(missing_genres_df), description="Asignando géneros..."):
			song_id = row["song_id"]
			if song_id in genre_majority.index:
					genres_df.loc[index, genre_columns] = genre_majority.loc[song_id]

	genres_df.drop(columns=["song_id"], inplace=True)

	return genres_df


def _extract_all_audio_features(genres_df, base_dir, n_jobs=-1):
	"""Extrae características de audio de todos los archivos en paralelo, solo si son válidos."""
	mp3_paths = genres_df[["clip_id", "mp3_path"]].values.tolist()

	valid_audio_paths = Parallel(n_jobs=n_jobs)(
		delayed(check_audio_file)(os.path.join(base_dir, mp3_path), mp3_path) 
		for clip_id, mp3_path in track(mp3_paths, total=len(mp3_paths), description="Verificando audios...\n",)
	)

	# Filtra los archivos válidos
	valid_mp3_paths = [mp3_paths[i] for i in range(len(valid_audio_paths)) if valid_audio_paths[i]]

	results = Parallel(n_jobs=n_jobs)(
		delayed(_extract_audio_features)(clip_id, mp3_path, base_dir) 
		for clip_id, mp3_path in track(valid_mp3_paths, total=len(valid_mp3_paths), description="Extrayendo características de audio...\n")
	)

	return pd.DataFrame([res for res in results if res is not None])


def _extract_audio_features(clip_id, mp3_path, base_dir):
	"""Extrae características de un solo archivo de audio."""
	try:
		path = os.path.join(base_dir, mp3_path)
		audio, sr = librosa.load(path, sr=22050, mono=True)

		stft = librosa.stft(audio, n_fft=2048, hop_length=512, window='hann')
		stft_magnitude = np.abs(stft) ** 2
		stft_mean = stft_magnitude.mean(axis=1)

		tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
		tempo_val = np.array([tempo], dtype=np.float32)

		mfcc = librosa.feature.mfcc(S=librosa.power_to_db(stft_magnitude), sr=sr, n_mfcc=13)
		mfcc_mean = mfcc.mean(axis=1)

		chroma = librosa.feature.chroma_stft(S=stft_magnitude, sr=sr)
		chroma_mean = chroma.mean(axis=1)

		return {
			"clip_id": clip_id,
			"mp3_path": mp3_path,
			"tempo": tempo_val,
			"stft_mean": stft_mean,
			"mfcc_mean": mfcc_mean,
			"chroma_mean": chroma_mean
		}
	except Exception as e:
		print(f"❌ [bold red]Error: No se pudieron extraer características de {mp3_path}: {e}[/bold red]")
		return None

def check_audio_file(path, mp3_path):
	"""Verifica si el archivo de audio es válido."""
	try:
		with sf.SoundFile(path) as f:
			return True
	except Exception as e:
		print(f"❌ [bold red]Error: El archivo {mp3_path} no es válido: {e}[/bold red]")
		return False
	
def _delete_fragments_without_audio_features(genres_df, tags_df, metadata_df, features_df):
		"""Elimina las canciones que tienen algún fragmento sin características de audio."""
		genres_df["song_id"] = genres_df["mp3_path"].apply(_extract_song_id)

		no_audio_features_songs = genres_df[~genres_df["clip_id"].isin(features_df["clip_id"])]

		songs_without_features = no_audio_features_songs["song_id"].unique()

		print(f"[bold red]❌ Hay {len(no_audio_features_songs)} fragmentos sin características de audio.[/bold red]")
		print(f"[bold red]❌ Se eliminarán {len(songs_without_features)} canciones con fragmentos sin características de audio.[/bold red]")

		genres_df = genres_df[~genres_df["song_id"].isin(songs_without_features)].copy()
		genres_df.drop(columns=["song_id"], inplace=True)
		tags_df = tags_df[tags_df["clip_id"].isin(genres_df["clip_id"])]
		metadata_df = metadata_df[metadata_df["clip_id"].isin(genres_df["clip_id"])]
		features_df = features_df[features_df["clip_id"].isin(genres_df["clip_id"])]

		return genres_df, tags_df, metadata_df, features_df

	

def _print_table(title, df):
	"""Muestra un DataFrame como una tabla en la terminal con rich."""
	table = Table(title=f"[bold]{title}[/bold]", show_lines=True)

	for col in df.columns[:6]:  # Solo mostramos las primeras 6 columnas para no hacer la tabla muy grande
		table.add_column(col, style="cyan", overflow="fold")
	table.add_column("...", style="cyan", overflow="fold")

	for _, row in df.head(5).iterrows():  # Solo mostramos 5 filas
		table.add_row(*[str(row[col]) for col in df.columns[:6]])

	print(table)


def load_data(n_jobs=-1):
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
		"no beat", "no singing", "no guitar", "no vocal", "no vocals", "no flute",
		"no singer", "no drums", "not english"
	]

	genres_df = pd.DataFrame(df[genres])
	genres_df = genres_df.astype({col: 'bool' for col in genres_df.columns if col not in ['clip_id', 'mp3_path']})
	tags_df = pd.DataFrame(df[tags])
	tags_df = tags_df.astype({col: 'bool' for col in tags_df.columns if col not in ['clip_id', 'mp3_path']})

	print(f"\n✅ [bold cyan]Fragmentos cargados:[/bold cyan] {genres_df.shape[0]}")
	print(f"✅ [bold cyan]Géneros cargados:[/bold cyan] {genres_df.shape[1] - 2}")
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
		'male vocals': ['male vocal', 'male voice', 'men', 'male', 'male singer', 'man singing', 'man'],
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
		'no vocals': ['no voice', 'no singer', 'no vocal', 'no voices'],
		'beats': ['beat'],
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
	tags = tags_df[tags_df.drop(columns=['clip_id', 'mp3_path']).sum(axis=1) > 0]
	print(f"✅ [bold cyan]Canciones con tags asignados:[/bold cyan] {tags.shape[0]}")


	metadata_df = pd.read_csv(metadata_path, delimiter='\t')
	metadata_df = metadata_df[metadata_df['clip_id'].isin(genres_df['clip_id'])]

	print(f"✅ [bold cyan]Cargados metadatos de {metadata_df.shape[0]} canciones.[/bold cyan]")
	_print_table("\nMetadatos", metadata_df.head())

	dataset_path = os.path.join(dir, "dataset/")
	features_df = _extract_all_audio_features(genres_df, dataset_path, n_jobs=n_jobs)
	print(f"✅ [bold cyan]Extraídas características de audio de {features_df.shape[0]} fragmentos.[/bold cyan]")
	_print_table("\nCaracterísticas de audio", features_df.head())

	# Eliminamos las canciones que tienen algún fragmento sin características de audio
	genres_df, tags_df, metadata_df, features_df = _delete_fragments_without_audio_features(genres_df, tags_df, metadata_df, features_df)

	if os.path.exists(genres_db_path):
		os.remove(genres_db_path)
	if os.path.exists(tags_db_path):
		os.remove(tags_db_path)
	if os.path.exists(metadata_db_path):
		os.remove(metadata_db_path)
	if os.path.exists(features_db_path):
		os.remove(features_db_path)

	conn = sqlite3.connect(genres_db_path)
	genres_df.to_sql('genres', conn, index=False)
	conn.close()

	conn = sqlite3.connect(tags_db_path)
	tags_df.to_sql('tags', conn, index=False)
	conn.close()

	conn = sqlite3.connect(metadata_db_path)
	metadata_df.to_sql('metadata', conn, index=False)
	conn.close()

	conn = sqlite3.connect(features_db_path)
	features_df.to_sql('features', conn, index=False)
	conn.close()


if __name__ == "__main__":
	load_data(n_jobs=6)
