# Crude Oil Trading Bot & Analysis Pipeline

Dit project bevat een geautomatiseerde pipeline voor het ophalen van financiële data (Crude Oil) via de Capital.com API, en een Machine Learning model (Random Forest) dat handelsstrategieën simuleert en evalueert.

## 📂 Projectstructuur

- **`fetch_data.py`**: Script om actuele OHLC-data (Open, High, Low, Close) op te halen van Capital.com en lokaal (of in een repo) op te slaan.
- **`analysis.py`**: Het hoofdscript dat data inlaadt, features berekent, een ML-model traint op historische data en de huidige handelsdag simuleert.
- **`OIL_CRUDE/`**: Map waarin ruwe data en logs worden opgeslagen.
- **`trading_logs.csv`**: Logboek van alle gesimuleerde trades.


**DAILY REPORT LOGICA**

    graph TD
        A[Start Daily Report] --> B{Lees trading_logs.csv};
        B -- Bestaat --> C[Haal lijst verwerkte dagen op];
        B -- Bestaat niet --> D[Start lege lijst];
        C & D --> E[Vergelijk met nieuwe Data];
        E --> F{Zijn er nieuwe dagen?};
        F -- Nee --> G[Stop Script];
        F -- Ja --> H[Start Loop per Nieuwe Dag];
    
        subgraph "Rolling Window Proces"
            H --> I[Selecteer vorige 40 dagen];
            I --> J[Train Random Forest (75% data)];
            J --> K[Valideer op recente data (25%)];
            K --> L{Check Correlatie Score};
            L -- Goed Model --> M[Lagere Drempel (Aggressiever)];
            L -- Slecht Model --> N[Hogere Drempel (Defensief)];
        end
    
        M & N --> O[Simuleer Huidige Dag];
        
        subgraph "Trade Simulatie"
            O --> P{Signaal > Drempel?};
            P -- Ja --> Q[Open Trade];
            Q --> R{Check SL / TP / Tijd};
            R -- Exit --> S[Sla resultaat op in geheugen];
        end
    
        S --> H;
        P -- Nee --> O;
        
        H -- Alle dagen klaar --> T[Update trading_logs.csv];
        T --> U[Einde];


<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/53d7a4ce-5b5a-409a-89b1-b30b21ea2f36" />
