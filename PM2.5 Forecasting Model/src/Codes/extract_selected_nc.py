from pathlib import Path
import shutil
import pandas as pd

# ======================
# PATH SETUP (CLEAN + TA-PROOF)
# ======================
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"                # where all .nc files already are
OUT_DIR = DATA_DIR / "selected_raw"       # where selected files go

CSV_FILES = [
    DATA_DIR / "processed" / "train_files.csv",
    DATA_DIR / "processed" / "val_files.csv",
    DATA_DIR / "processed" / "test_files.csv",
]

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================
# COLLECT FILENAMES
# ======================
def collect_filenames(csv_paths):
    names = set()

    for csv in csv_paths:
        if not csv.exists():
            raise FileNotFoundError(f"Missing CSV file: {csv}")

        df = pd.read_csv(csv)

        print(f"\nLoaded: {csv}")
        print("Columns:", df.columns.tolist())

        # Try to detect correct column
        if "File Path" in df.columns:
            col = "File Path"
        elif "file" in df.columns:
            col = "file"
        elif "filename" in df.columns:
            col = "filename"
        else:
            raise ValueError(f"Could not find file column in {csv}")

        for p in df[col].astype(str):
            names.add(Path(p).name)

    return names

# ======================
# COPY FILES
# ======================
def copy_selected_files(source_dir, filenames, out_dir):
    matches = []

    for fname in filenames:
        src = source_dir / fname

        if src.exists():
            matches.append(src)
        else:
            print(f"⚠️ Missing file: {fname}")

    print(f"\nRequested files: {len(filenames)}")
    print(f"Found in raw/: {len(matches)}")

    if not matches:
        raise RuntimeError("No matching .nc files found in data/raw!")

    for src in matches:
        dst = out_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

    print(f"\nCopied {len(matches)} files to:")
    print(out_dir)

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    print("Using raw data directory:", RAW_DIR)

    filenames = collect_filenames(CSV_FILES)
    print("Unique filenames collected:", len(filenames))

    copy_selected_files(RAW_DIR, filenames, OUT_DIR)