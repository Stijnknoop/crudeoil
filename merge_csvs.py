import pandas as pd
import glob
import os
from datetime import datetime

# 1️⃣ Zoek alle CSV-bestanden (inclusief eerdere merged files)
csv_files = sorted(glob.glob("outputs_*.csv"))

if len(csv_files) > 0:
    print(f"Gevonden CSV-bestanden: {csv_files[:5]}{'...' if len(csv_files) > 5 else ''}")

    # 2️⃣ Lees alle CSV’s in een lijst
    df_list = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
            print(f"Ingelezen: {f} (shape={df.shape})")
        except Exception as e:
            print(f"⚠️ Fout bij inlezen {f}: {e}")

    if not df_list:
        print("❌ Geen geldige CSV-bestanden ingelezen — stoppen.")
        exit()

    # 3️⃣ Combineer alle data
    merged_df = pd.concat(df_list, ignore_index=True)

    # 4️⃣ Ontdubbel en sorteer op kolom 'time'
    if 'time' in merged_df.columns:
        # probeer 'time' te converteren naar datetime
        try:
            merged_df['time'] = pd.to_datetime(merged_df['time'])
        except Exception:
            print("⚠️ Kon 'time' niet converteren naar datetime; sorteer tekstueel.")
        merged_df = merged_df.drop_duplicates(subset=['time'])
        merged_df = merged_df.sort_values(by='time', ascending=True).reset_index(drop=True)
        print("Ontdubbeld en gesorteerd op 'time' (ascending).")
    else:
        print("⚠️ Kolom 'time' niet gevonden — geen ontdubbeling of sortering uitgevoerd.")

    # 5️⃣ Verwijder ALLE oude CSV-bestanden
    for f in csv_files:
        try:
            os.remove(f)
            print(f"Verwijderd: {f}")
        except Exception as e:
            print(f"⚠️ Fout bij verwijderen {f}: {e}")

    # 6️⃣ Maak timestamp voor nieuw bestand
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_filename = f"outputs_merged_{timestamp}.csv"

    # 7️⃣ Sla nieuwe merged CSV op
    merged_df.to_csv(merged_filename, index=False)
    print(f"✅ Nieuwe samengevoegde CSV opgeslagen als: {merged_filename}")

else:
    print("Geen CSV-bestanden gevonden — geen actie ondernomen.")
