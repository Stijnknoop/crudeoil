import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

def generate_performance_plot():
    log_path = "Trading_details/trading_logs.csv"
    if not os.path.exists(log_path):
        print("Geen logboek gevonden.")
        return

    df = pd.read_csv(log_path)
    # Alleen dagen met trades meenemen
    df = df[df['exit_reason'] != 'No Trade'].copy()
    
    if df.empty:
        print("Nog geen trades om te plotten.")
        return

    # Zet entry_time om naar echte datums
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df = df.sort_values('entry_time')

    # Hefboom van 5 toepassen
    leverage = 5
    df['leverage_return'] = df['return'] * leverage
    
    # Bereken Compound Equity
    df['equity_compound'] = (1 + df['leverage_return']).cumprod()
    
    # Voor de lijn grafiek voegen we een 'startpunt' toe (1 dag voor de eerste trade)
    start_date = df['entry_time'].min() - pd.Timedelta(days=1)
    
    # Plot instellen
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- 1. De Compound Lijn (Links) ---
    ax1.set_ylabel('Portfolio Waarde (Start = 1.0)', color='darkgreen', fontweight='bold')
    # We plotten de lijn over de datums
    ax1.plot(df['entry_time'], df['equity_compound'], color='darkgreen', linewidth=2.5, label='Compound Equity (5x)', zorder=3)
    ax1.tick_params(axis='y', labelcolor='darkgreen')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # --- 2. De Daily Profit/Loss Bars (Rechts) ---
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Daily Return (%)', color='gray', fontweight='bold')
    
    # Kleuren: Groen voor winst, Rood voor verlies
    colors = ['#a1d99b' if r > 0 else '#fb9a99' for r in df['leverage_return']]
    
    # Bars plotten met datum op de x-as
    # width=0.8 zorgt dat de bars mooi zichtbaar zijn tussen de datums
    ax2.bar(df['entry_time'], df['leverage_return'] * 100, color=colors, alpha=0.6, label='Daily P/L %', zorder=2, width=0.6)
    
    # Nul-lijn
    ax2.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor='gray')

    # --- X-AS FORMATTERING (DATUMS) ---
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate() # Draait de datums schuin voor leesbaarheid

    # Titels en Legenda
    plt.title(f'Portfolio Performance: Compound Curve vs Daily Returns (5x Leverage)\nUpdate: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}', fontsize=12)
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # Opslaan
    plt.tight_layout()
    plt.savefig("Trading_details/equity_curve.png")
    print("Nieuwe grafiek met datums op de x-as gegenereerd.")

if __name__ == "__main__":
    generate_performance_plot()
