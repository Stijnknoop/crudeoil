import pandas as pd
import glob
import os
from datetime import datetime

# 1️⃣ Zoek alle CSV-bestanden (inclusief eerdere merged files)
csv_files = sorted(glob.glob("outputs_*.csv"))

if len(csv_files) > 0:
    print(f"Gevonden CSV-bestanden: {csv_files[:5]}{'...' if len(csv_files) > 5 else ''}")

    df_list = []
    for f in csv_files:
        try:
            # 2️⃣ Haal bestandstijd op (mtime)
            file_mtime = os.path.getmtime(f)

            df = pd.read_csv(f)

            # Voeg metadata toe om later nieuwste te kiezen
            df["source_file"] = f
            df["file_mtime"] = file_mtime

            df_list.append(df)
            print(f"Ingelezen: {f} (shape={df.shape})")
        except Exception as e:
            print(f"⚠️ Fout bij inlezen {f}: {e}")

    if not df_list:
        print("❌ Geen geldige CSV-bestanden ingelezen — stoppen.")
        exit()

    # 3️⃣ Combineer alle data
    merged_df = pd.concat(df_list, ignore_index=True)

    # 4️⃣ Converteren + sorteren
    if 'time' in merged_df.columns:

        # Convert to datetime
        try:
            merged_df['time'] = pd.to_datetime(merged_df['time'])
        except:
            print("⚠️ Kon 'time' niet converteren naar datetime.")

        # 4b️⃣ Sorteren zodat nieuwste bestand bovenaan staat
        merged_df = merged_df.sort_values(
            by=["time", "file_mtime"],
            ascending=[True, False]  # oudste time → nieuwste file
        )

        # 4c️⃣ Per 'time' alleen de nieuwste behouden
        merged_df = merged_df.drop_duplicates(subset=["time"], keep="first")

        # Sorteren op tijd
        merged_df = merged_df.sort_values(by="time").reset_index(drop=True)

        print("Ontdubbeld: nieuwste bestand wint bij dubbele 'time'.")

    else:
        print("⚠️ Kolom 'time' niet gevonden — geen ontdubbeling of sortering uitgevoerd.")

    # 5️⃣ Oude CSV's verwijderen
    for f in csv_files:
        try:
            os.remove(f)
            print(f"Verwijderd: {f}")
        except Exception as e:
            print(f"⚠️ Fout bij verwijderen {f}: {e}")

    # 6️⃣ Nieuw bestand opslaan
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_filename = f"outputs_merged_{timestamp}.csv"

    # Extra kolommen weghalen
    merged_df = merged_df.drop(columns=["source_file", "file_mtime"], errors="ignore")

    merged_df.to_csv(merged_filename, index=False)
    print(f"✅ Nieuwe samengevoegde CSV opgeslagen als: {merged_filename}")

else:
    print("Geen CSV-bestanden gevonden — geen actie ondernomen.")
