import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_performance_plot():
    log_path = "Trading_details/trading_logs.csv"
    if not os.path.exists(log_path):
        print("Geen logboek gevonden.")
        return

    df = pd.read_csv(log_path)
    # Alleen dagen met trades meenemen voor de winst-berekening
    df = df[df['exit_reason'] != 'No Trade'].copy()
    
    if df.empty:
        print("Nog geen trades om te plotten.")
        return

    # Hefboom van 5 toepassen op de returns
    leverage = 5
    df['leverage_return'] = df['return'] * leverage
    
    # Bereken Compound Equity
    df['equity_compound'] = (1 + df['leverage_return']).cumprod()
    # Voeg een startpunt toe op 1.0
    equity_curve = [1.0] + df['equity_compound'].tolist()
    
    # Plot instellen
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- 1. De Compound Lijn (Links) ---
    ax1.set_xlabel('Aantal Trades')
    ax1.set_ylabel('Portfolio Waarde (Start = 1.0)', color='darkgreen')
    ax1.plot(equity_curve, color='darkgreen', linewidth=2, label='Compound Equity (5x)')
    ax1.tick_params(axis='y', labelcolor='darkgreen')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- 2. De Daily Profit/Loss Bars (Rechts) ---
    ax2 = ax1.twinx()  # Maak een tweede y-as aan
    ax2.set_ylabel('Daily Return (%)', color='gray')
    
    # Kleuren bepalen: Groen voor winst, Rood voor verlies
    colors = ['green' if r > 0 else 'red' for r in df['leverage_return']]
    
    # Bars plotten (we beginnen bij index 1 omdat index 0 het startpunt 1.0 is)
    bars = ax2.bar(range(1, len(df) + 1), df['leverage_return'] * 100, 
                   color=colors, alpha=0.3, label='Daily P/L %')
    
    # Nul-lijn voor de bars
    ax2.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor='gray')

    # Titels en Legenda
    plt.title(f'Portfolio Performance: Compound Curve vs Daily Returns (5x Leverage)\nUpdate: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}')
    
    # Gecombineerde legenda
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # Opslaan
    plt.tight_layout()
    plt.savefig("Trading_details/equity_curve.png")
    print("Nieuwe grafiek met bars gegenereerd.")

if __name__ == "__main__":
    generate_performance_plot()
