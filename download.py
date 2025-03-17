import os
import zipfile
import requests
import sys
from rich.progress import track
from rich import print

dataset_urls = [
  "https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.001",
  "https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.002",
  "https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.003",
]

output_dir = "dataset"
zip_parts = [os.path.join(output_dir, f"mp3.zip.{i+1:03d}") for i in range(len(dataset_urls))]
final_zip = os.path.join(output_dir, "mp3.zip")

def dataset_already_extracted():
  """ Verifica si la carpeta dataset ya tiene archivos o subcarpetas. """
  if os.path.exists(output_dir):
    subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]

    if subdirs:
      print("✅ [bold cyan]Dataset ya extraído, omitiendo descarga y extracción.[/bold cyan]")
      return True
  return False

def download_files():
  """ Descarga solo las partes que faltan. """
  os.makedirs(output_dir, exist_ok=True)

  for i, url in enumerate(dataset_urls):
    part_file_path = zip_parts[i]

    if os.path.exists(part_file_path):
      print(f"✅ [bold cyan]{part_file_path} ya existe, omitiendo descarga.[/bold cyan]")
      continue

    print(f"⬇️ [bold mediumorchid] Descargando {part_file_path}...[/bold mediumorchid]")
    try:
      response = requests.get(url, stream=True)
      response.raise_for_status()
      total_size = int(response.headers.get('content-length', 0))
      block_size = 8192
      t = track(total=total_size, unit='iB', unit_scale=True, desc=part_file_path, ascii=True)
      with open(part_file_path, 'wb') as part_file:
        for chunk in response.iter_content(chunk_size=block_size):
          t.update(len(chunk))
          part_file.write(chunk)
      t.close()
      print(f"✅ [bold cyan]Descarga de {part_file_path} completada.[/bold cyan]")
    except requests.exceptions.RequestException as e:
      print(f"❌ [bold red]Error al descargar {part_file_path}: {e}[/bold red]")
      return False

def merge_parts():
  """ Une las partes en un solo archivo ZIP y borra los fragmentos. """
  if os.path.exists(final_zip):
    print("✅ [bold cyan]Archivo ZIP ya unido, omitiendo.[/bold cyan]")
    return

  print("🔗 Uniendo archivos en un solo ZIP...")
  with open(final_zip, "wb") as f_out:
    for part in zip_parts:
      with open(part, "rb") as f_in:
        f_out.write(f_in.read())

  print(f"✅ [bold cyan]Archivos unidos en {final_zip}[/bold cyan]")

  cleanup_zip_parts()

def extract_files():
  """ Extrae el ZIP unido y borra el archivo ZIP. """
  if dataset_already_extracted():
    return

  if not os.path.exists(final_zip):
    print("❌ [bold red]No se encontró el archivo ZIP completo. ¿Se fusionaron correctamente las partes?[/bold red]")
    return

  print("📂 [bold green]Extrayendo archivos ZIP...[/bold green]")
  try:
    with zipfile.ZipFile(final_zip, 'r') as zip_ref:
      zip_ref.extractall(output_dir)
    print("✅ [bold cyan]Extracción completada.[/bold cyan]")

    cleanup_final_zip()

  except zipfile.BadZipFile:
      print("❌ [bold red]Error: El archivo ZIP parece estar corrupto.[/bold red]")

def cleanup_zip_parts():
  """ Borra los fragmentos ZIP después de unirlos. """
  print("🗑️  [bold brightblack]Borrando partes ZIP...[/bold brightblack]")
  
  for zip_file in zip_parts:
    if os.path.exists(zip_file):
      os.remove(zip_file)
      print(f"✅ [bold cyan]Eliminado: {zip_file}[/bold cyan]")

def cleanup_final_zip():
  """ Borra el archivo ZIP final después de extraerlo. """
  if os.path.exists(final_zip):
    os.remove(final_zip)
    print(f"✅ [bold cyan]Eliminado: {final_zip}[/bold cyan]")

def reset_dataset():
  """ Elimina el dataset completo de forma segura. """
  if os.path.exists(output_dir):
    print("🗑️  [bold brightblack]Borrando dataset completo...[/bold brightblack]")

    # Cambiamos permisos si es necesario
    for root, dirs, files in os.walk(output_dir, topdown=False):
      for name in files:
        file_path = os.path.join(root, name)
        try:
          os.chmod(file_path, 0o777) 
          os.remove(file_path)
        except Exception as e:
          print(f"❌ [bold red]No se pudo eliminar {file_path}: {e}[/bold red]")

      for name in dirs:
        dir_path = os.path.join(root, name)
        try:
          os.rmdir(dir_path)
        except Exception as e:
          print(f"❌ [bold red]No se pudo eliminar {dir_path}: {e}[/bold red]")

    try:
      os.rmdir(output_dir)
      print("✅ [bold cyan]Dataset eliminado.[/bold cyan]")
    except Exception as e:
      print(f"❌ [bold red]No se pudo eliminar la carpeta {output_dir}: {e}[/bold red]")

def download_metadata_and_genres():
  """ Descarga los metadatos y géneros de la música. """
  metadata_url = "https://mirg.city.ac.uk/datasets/magnatagatune/clip_info_final.csv"
  genres_url = "https://mirg.city.ac.uk/datasets/magnatagatune/annotations_final.csv"

  metadata_path = os.path.join(output_dir, "annotations_final.csv")
  genres_path = os.path.join(output_dir, "genres.csv")

  if os.path.exists(metadata_path) and os.path.exists(genres_path):
    print("✅ [bold cyan]Metadatos y géneros ya descargados.[/bold cyan]")
    return

  print("⬇️  [bold mediumorchid]Descargando metadatos y géneros...[/bold mediumorchid]")
  response = requests.get(metadata_url)
  with open(metadata_path, 'wb') as metadata_file:
    metadata_file.write(response.content)

  response = requests.get(genres_url)
  with open(genres_path, 'wb') as genres_file:
    genres_file.write(response.content)

  print("✅ [bold cyan]Descarga de metadatos y géneros completada.[/bold cyan]")

if __name__ == "__main__":
  if "--reset" in sys.argv:
    reset_dataset()

  if not dataset_already_extracted():
    download_files()
    merge_parts()
    extract_files()
    download_metadata_and_genres()
