import pandas as pd
import glob
import os
from datetime import datetime

# Stap 1: Verwijder eerst oude merged-bestanden
old_merged_files = glob.glob("outputs_merged_*.csv")
for f in old_merged_files:
    os.remove(f)
    print(f"Verwijderd oud merged bestand: {f}")

# Stap 2: Zoek alle overige CSV-bestanden
csv_files = sorted(glob.glob("outputs_*.csv"))

if len(csv_files) > 0:
    print(f"Gevonden CSV-bestanden: {csv_files[:5]}{'...' if len(csv_files) > 5 else ''}")

    # Lees alles in
    df_list = [pd.read_csv(f) for f in csv_files]
    merged_df = pd.concat(df_list, ignore_index=True)

    # Ontdubbel + sorteer op 'time'
    if 'time' in merged_df.columns:
        merged_df = merged_df.drop_duplicates(subset=['time'])
        merged_df = merged_df.sort_values(by='time', ascending=True)
        print("Ontdubbeld en gesorteerd op kolom 'time' (ascending).")
    else:
        print("⚠️ Waarschuwing: kolom 'time' niet gevonden — geen ontdubbeling of sortering uitgevoerd.")

    # Nieuw bestand met timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_filename = f"outputs_merged_{timestamp}.csv"

    # Sla nieuwe merge op
    merged_df.to_csv(merged_filename, index=False)
    print(f"Samengevoegde CSV opgeslagen als: {merged_filename}")

    # Verwijder alle originele CSV’s (die net gemerged zijn)
    for f in csv_files:
        if f != merged_filename:
            os.remove(f)
            print(f"Verwijderd: {f}")

else:
    print("Geen CSV-bestanden gevonden — geen actie ondernomen.")
