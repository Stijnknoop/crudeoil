import pandas as pd
import glob
import os
from datetime import datetime

# Zoek alle CSV-bestanden in de repo (pas eventueel pad aan)
csv_files = sorted(glob.glob("outputs_*.csv"))

# Alleen doorgaan als er meerdere CSV's zijn
if len(csv_files) > 1:
    print(f"Gevonden CSV-bestanden: {csv_files[:5]}{'...' if len(csv_files) > 5 else ''}")

    # Lees ALLE CSV-bestanden in
    df_list = [pd.read_csv(f) for f in csv_files]

    # Voeg alles samen
    merged_df = pd.concat(df_list, ignore_index=True)

    # Ontdubbel op kolom 'time'
    if 'time' in merged_df.columns:
        merged_df = merged_df.drop_duplicates(subset=['time'])
        print("Ontdubbeld op kolom 'time'.")
    else:
        print("⚠️ Waarschuwing: kolom 'time' niet gevonden — geen ontdubbeling uitgevoerd.")

    # Maak timestamp voor de bestandsnaam
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_filename = f"outputs_merged_{timestamp}.csv"

    # Sla op
    merged_df.to_csv(merged_filename, index=False)
    print(f"Samengevoegde CSV opgeslagen als: {merged_filename}")

    # Verwijder ALLE oude CSV-bestanden (behalve de nieuwe merged)
    for f in csv_files:
        if f != merged_filename:
            os.remove(f)
            print(f"Verwijderd: {f}")

else:
    print("Er is slechts één of geen CSV-bestand — geen samenvoeging nodig.")
