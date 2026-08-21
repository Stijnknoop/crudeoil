# 📊 MANTRA: Layered Z-Score Session Report (2026-08-20)

* **Strategy Architecture:** `MULTI-SLOT GRID WITH VEILIGE ZONE BREAK-EVEN STOP`
* **Filters:** Expected Win (`>=0.15%`) | Dwell Block (`10m`) | Cluster Exit (`30m`) | BE Trigger (`|Z|=0.5`)

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 6
* **Batch Win Rate:** 50.00%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **-0.6993%**

### 📜 Session Transaction Ledger
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL Trade Combination | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 14:01 | 14:12 | `SHORT` | 7683.10 | 7665.20 | `LONG` | 4459.36 | 4463.69 | **0.1650%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 15:01 | 15:31 | `LONG` | 7679.40 | 7684.20 | `SHORT` | 4483.76 | 4470.93 | **0.1743%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 16:02 | 16:32 | `LONG` | 7681.00 | 7683.10 | `SHORT` | 4487.47 | 4494.98 | **-0.0700%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 16:33 | 17:03 | `LONG` | 7685.50 | 7698.10 | `SHORT` | 4496.86 | 4533.31 | **-0.3233%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 16:35 | 17:03 | `LONG` | 7681.50 | 7698.10 | `SHORT` | 4497.72 | 4533.31 | **-0.2876%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 17:26 | 17:56 | `LONG` | 7681.80 | 7676.80 | `SHORT` | 4529.96 | 4521.41 | **0.0618%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
