import pandas as pd
import glob
import os
from datetime import datetime

# Zoek ALLE CSV-bestanden (inclusief eerdere merges)
csv_files = sorted(glob.glob("outputs_*.csv"))

# Alleen doorgaan als er minstens één CSV is
if len(csv_files) > 0:
    print(f"Gevonden CSV-bestanden: {csv_files[:5]}{'...' if len(csv_files) > 5 else ''}")

    # Lees alle CSV-bestanden in
    df_list = [pd.read_csv(f) for f in csv_files]

    # Voeg alles samen
    merged_df = pd.concat(df_list, ignore_index=True)

    # Ontdubbel op kolom 'time'
    if 'time' in merged_df.columns:
        merged_df = merged_df.drop_duplicates(subset=['time'])
        # Sorteer op kolom 'time'
        merged_df = merged_df.sort_values(by='time', ascending=True)
        print("Ontdubbeld en gesorteerd op kolom 'time' (ascending).")
    else:
        print("⚠️ Waarschuwing: kolom 'time' niet gevonden — geen ontdubbeling of sortering uitgevoerd.")

    # Maak timestamp voor de bestandsnaam
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_filename = f"outputs_merged_{timestamp}.csv"

    # Sla op
    merged_df.to_csv(merged_filename, index=False)
    print(f"Samengevoegde CSV opgeslagen als: {merged_filename}")

    # Verwijder ALLE oude CSV-bestanden (inclusief vorige merges)
    for f in csv_files:
        if f != merged_filename:
            os.remove(f)
            print(f"Verwijderd: {f}")

else:
    print("Geen CSV-bestanden gevonden — geen actie ondernomen.")
