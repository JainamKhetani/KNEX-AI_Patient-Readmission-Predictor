# download_data.py
import urllib.request
import zipfile
import os

print("📥 Downloading dataset from UCI ML Repository...")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
dest = "data/raw/dataset_diabetes.zip"

os.makedirs("data/raw", exist_ok=True)

# Download with progress
def show_progress(count, block_size, total_size):
    pct = count * block_size * 100 / total_size
    print(f"\r  Progress: {min(pct, 100):.1f}%", end="", flush=True)

urllib.request.urlretrieve(url, dest, reporthook=show_progress)
print("\n✅ Download complete!")

# Unzip
print("📦 Extracting files...")
with zipfile.ZipFile(dest, "r") as z:
    z.extractall("data/raw/")
    print(f"  Extracted: {z.namelist()}")

# Check what we got
for f in os.listdir("data/raw"):
    size = os.path.getsize(f"data/raw/{f}") / 1024
    print(f"  📄 {f}  ({size:.0f} KB)")

print("\n✅ Dataset ready in data/raw/")