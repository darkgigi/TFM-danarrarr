import os
import zipfile
import requests
import sys
from tqdm import tqdm

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
      print("✅ Dataset ya extraído, omitiendo descarga y extracción.")
      return True
  return False

def download_files():
  """ Descarga solo las partes que faltan. """
  os.makedirs(output_dir, exist_ok=True)

  for i, url in enumerate(dataset_urls):
    part_file_path = zip_parts[i]

    if os.path.exists(part_file_path):
      print(f"✅ {part_file_path} ya existe, omitiendo descarga.")
      continue

    print(f"⬇️  Descargando {part_file_path}...")
    try:
      response = requests.get(url, stream=True)
      response.raise_for_status()
      total_size = int(response.headers.get('content-length', 0))
      block_size = 8192
      t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=part_file_path, ascii=True)
      with open(part_file_path, 'wb') as part_file:
        for chunk in response.iter_content(chunk_size=block_size):
          t.update(len(chunk))
          part_file.write(chunk)
      t.close()
      print(f"✅ Descarga de {part_file_path} completada.")
    except requests.exceptions.RequestException as e:
      print(f"❌ Error al descargar {part_file_path}: {e}")
      return False

def merge_parts():
  """ Une las partes en un solo archivo ZIP y borra los fragmentos. """
  if os.path.exists(final_zip):
    print("✅ Archivo ZIP ya unido, omitiendo.")
    return

  print("🔗 Uniendo archivos en un solo ZIP...")
  with open(final_zip, "wb") as f_out:
    for part in zip_parts:
      with open(part, "rb") as f_in:
        f_out.write(f_in.read())

  print("✅ Archivos unidos en", final_zip)

  # 🔥 BORRAR PARTES DESPUÉS DE UNIRLAS
  cleanup_zip_parts()

def extract_files():
  """ Extrae el ZIP unido y borra el archivo ZIP. """
  if dataset_already_extracted():
    return

  if not os.path.exists(final_zip):
    print("❌ No se encontró el archivo ZIP completo. ¿Se fusionaron correctamente las partes?")
    return

  print("📂 Extrayendo archivos ZIP...")
  try:
    with zipfile.ZipFile(final_zip, 'r') as zip_ref:
      zip_ref.extractall(output_dir)
    print("✅ Extracción completada.")

    # 🔥 BORRAR ZIP DESPUÉS DE EXTRAER
    cleanup_final_zip()

  except zipfile.BadZipFile:
      print("❌ Error: El archivo ZIP parece estar corrupto.")

def cleanup_zip_parts():
  """ Borra los fragmentos ZIP después de unirlos. """
  print("🗑️  Borrando partes ZIP...")
  
  for zip_file in zip_parts:
    if os.path.exists(zip_file):
      os.remove(zip_file)
      print(f"✅ Eliminado: {zip_file}")

def cleanup_final_zip():
  """ Borra el archivo ZIP final después de extraerlo. """
  if os.path.exists(final_zip):
    os.remove(final_zip)
    print(f"✅ Eliminado: {final_zip}")

def reset_dataset():
  """ Elimina el dataset completo de forma segura. """
  if os.path.exists(output_dir):
    print("🗑️  Borrando dataset completo...")

    # Cambiamos permisos si es necesario
    for root, dirs, files in os.walk(output_dir, topdown=False):
      for name in files:
        file_path = os.path.join(root, name)
        try:
          os.chmod(file_path, 0o777) 
          os.remove(file_path)
        except Exception as e:
          print(f"❌ No se pudo eliminar {file_path}: {e}")

      for name in dirs:
        dir_path = os.path.join(root, name)
        try:
          os.rmdir(dir_path)
        except Exception as e:
          print(f"❌ No se pudo eliminar {dir_path}: {e}")

    try:
      os.rmdir(output_dir)
      print("✅ Dataset eliminado.")
    except Exception as e:
      print(f"❌ No se pudo eliminar la carpeta {output_dir}: {e}")

def download_metadata_and_genres():
  """ Descarga los metadatos y géneros de la música. """
  metadata_url = "https://mirg.city.ac.uk/datasets/magnatagatune/clip_info_final.csv"
  genres_url = "https://mirg.city.ac.uk/datasets/magnatagatune/annotations_final.csv"

  metadata_path = os.path.join(output_dir, "annotations_final.csv")
  genres_path = os.path.join(output_dir, "genres.csv")

  if os.path.exists(metadata_path) and os.path.exists(genres_path):
    print("✅ Metadatos y géneros ya descargados.")
    return

  print("⬇️  Descargando metadatos y géneros...")
  response = requests.get(metadata_url)
  with open(metadata_path, 'wb') as metadata_file:
    metadata_file.write(response.content)

  response = requests.get(genres_url)
  with open(genres_path, 'wb') as genres_file:
    genres_file.write(response.content)

  print("✅ Descarga de metadatos y géneros completada.")

if __name__ == "__main__":
  if "--reset" in sys.argv:
    reset_dataset()

  if not dataset_already_extracted():
    download_files()
    merge_parts()
    extract_files()
    download_metadata_and_genres()
