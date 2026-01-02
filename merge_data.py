import pandas as pd
import glob
import os
import argparse
from datetime import datetime

# --- Argument Parser ---
parser = argparse.ArgumentParser()
parser.add_argument('--epic', type=str, required=True, help='The EPIC symbol (folder name)')
args = parser.parse_args()

TARGET_FOLDER = args.epic

print(f"🔄 Start merge proces voor map: {TARGET_FOLDER}")

if not os.path.exists(TARGET_FOLDER):
    print(f"⚠️ Map {TARGET_FOLDER} bestaat niet. Niets te mergen.")
    exit()

# 1️⃣ Zoek alle CSV-bestanden IN de specifieke map
# We zoeken naar outputs_*.csv (de losse) en outputs_merged_*.csv (de oude merged)
search_pattern = os.path.join(TARGET_FOLDER, "outputs_*.csv")
csv_files = sorted(glob.glob(search_pattern))

if len(csv_files) > 0:
    print(f"Gevonden CSV-bestanden: {len(csv_files)}")

    df_list = []
    for f in csv_files:
        try:
            file_mtime = os.path.getmtime(f)
            df = pd.read_csv(f)
            df["source_file"] = f
            df["file_mtime"] = file_mtime
            df_list.append(df)
        except Exception as e:
            print(f"⚠️ Fout bij inlezen {f}: {e}")

    if not df_list:
        print("❌ Geen geldige data ingelezen.")
        exit()

    # 3️⃣ Combineer
    merged_df = pd.concat(df_list, ignore_index=True)

    # 4️⃣ Converteren + Onthubbelen
    if 'time' in merged_df.columns:
        merged_df['time'] = pd.to_datetime(merged_df['time'])
        
        # Sorteren: Tijd oplopend, maar bij dubbele tijd: nieuwste bestand eerst
        merged_df = merged_df.sort_values(by=["time", "file_mtime"], ascending=[True, False])
        
        # Dubbele tijden verwijderen (behoud de eerste = die uit het nieuwste bestand)
        merged_df = merged_df.drop_duplicates(subset=["time"], keep="first")
        
        # Definitief sorteren op tijd
        merged_df = merged_df.sort_values(by="time").reset_index(drop=True)
    else:
        print("⚠️ Kolom 'time' mist.")

    # 5️⃣ Oude CSV's verwijderen UIT de map
    for f in csv_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"⚠️ Kon {f} niet verwijderen: {e}")

    # 6️⃣ Nieuw bestand opslaan IN de map
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_filename = os.path.join(TARGET_FOLDER, f"outputs_merged_{timestamp}.csv")

    merged_df = merged_df.drop(columns=["source_file", "file_mtime"], errors="ignore")
    merged_df.to_csv(merged_filename, index=False)
    print(f"✅ Nieuwe merged CSV: {merged_filename}")

else:
    print(f"Geen CSV-bestanden gevonden in {TARGET_FOLDER}.")
