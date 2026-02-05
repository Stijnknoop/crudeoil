# Context Sniper: Adaptive Crude Oil Trading Bot

Dit project bevat een geavanceerde, zelf-lerende trading pipeline voor Crude Oil. In plaats van prijzen te voorspellen met traditionele ML-modellen, gebruikt dit systeem een **"Context Sniper"** benadering.

Het algoritme analyseert continu het marktsentiment (Context) en berekent **on-the-fly** welke marktomstandigheden historisch gezien winstgevend zijn. Via een **Rolling Window Walk-Forward Analysis** past de bot zich elke dag aan aan de veranderende markt.

![Equity Curve](https://github.com/user-attachments/assets/53d7a4ce-5b5a-409a-89b1-b30b21ea2f36)

## 🧠 De Strategie: "Context Sniper"

De kernfilosofie is: *"Handel niet op wat je denkt dat de prijs gaat doen, maar handel op situaties die statistisch bewezen winst opleveren."*

Het model kijkt naar 4 variabelen (De Context):
1.  **Trend van Gisteren:** Was de vorige dag Groen of Rood?
2.  **Intraday Positie:** Waar zijn we nu t.o.v. de dag-range? (Low, Mid, High).
3.  **Momentum (RSI):** Is de markt Overbought of Oversold?
4.  **Tijd:** Welk uur van de dag is het?

### Hoe het werkt (Walk-Forward):
1.  **Training:** Voor elke handelsdag kijkt de bot naar de **afgelopen 40 dagen**.
2.  **Kalibratie:** Hij berekent de gemiddelde ROI voor elke mogelijke combinatie van bovenstaande variabelen.
3.  **Selectie:** Alleen combinaties met een bewezen "Edge" (bv. >0.25% gemiddelde winst) komen op de **Whitelist** voor die specifieke dag.
4.  **Executie:** Als de huidige marktmatcht met de Whitelist, wordt er gekocht.

## 📂 Projectstructuur

- **`main.py`** (of jouw scriptnaam): Het hoofdscript. Haalt data op, draait de rolling window simulatie en logt trades.
- **`equity_plot.py`**: Visualiseert de resultaten en berekent de compounding curve.
- **`OIL_CRUDE/`**: Map waarin ruwe data wordt opgeslagen.
- **`OIL_CRUDE/Trading_details/trading_logs.csv`**: Het logboek met alle trades (Entry, Exit, P&L, Reden).

## ⚙️ Features & Money Management

* **Fixed Fractional Compounding:** Het kapitaal wordt opgedeeld in **10 slots**. Winst wordt direct herinvesteerd, waardoor de positiegrootte automatisch meegroeit met het account.
* **Leverage:** Ondersteuning voor hefboom (standaard 5x) in de simulatie.
* **Ruis Filter:** Trades worden alleen genomen als de dag-range significant genoeg is (>0.08% van prijs).
* **Time-out:** Alle posities worden geforceerd gesloten om 22:00 (geen overnight risico).
* **Cooldown:** Na een aankoop wacht het systeem 10 minuten om spam-trades te voorkomen.

## 🔄 Workflow Diagram

Hieronder de logica van het `main.py` script:

```mermaid
graph TD
    A[Start Script] --> B{Lees trading_logs.csv};
    B -- Bestaat --> C[Bepaal laatste verwerkte datum];
    B -- Bestaat niet --> D[Start bij dag 40];
    C & D --> E[Haal nieuwe Data op van GitHub];
    
    E --> F{Zijn er nieuwe dagen?};
    F -- Nee --> G[Stop Script];
    F -- Ja --> H[Start Rolling Window Loop];
    
    subgraph "Dagelijkse Her-Kalibratie"
        H --> I[Definieer Window: Dag T-40 t/m T-1];
        I --> J[Bereken ROI per Markt-Conditie];
        J --> K{Is ROI > Drempel? (bv 0.25%)};
        K -- Ja --> L[Voeg toe aan 'Daily Rulebook'];
        K -- Nee --> M[Negeer Conditie];
    end
    
    L --> N[Simuleer Huidige Dag (T)];
    
    subgraph "Intraday Executie"
        N --> O[Check Huidige Context (RSI, Positie, Tijd)];
        O --> P{Staat Context in Rulebook?};
        P -- Ja --> Q{Zijn er Slots vrij?};
        Q -- Ja --> R[KOOP (10% van Equity * Leverage)];
        
        R --> S[Manage Positie];
        S --> T{Exit Signaal?};
        T -- Target Hit --> U[Winst Pakken];
        T -- 22:00 Uur --> V[Force Close (Time-out)];
    end
    
    U & V --> W[Sla Trade op in Lijst];
    W --> H;
    
    H -- Alle dagen klaar --> X[Update trading_logs.csv];
    X --> Y[Einde];
