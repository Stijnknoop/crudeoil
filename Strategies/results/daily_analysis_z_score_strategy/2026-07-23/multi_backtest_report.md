# 📊 MANTRA: Layered Z-Score Session Report (2026-07-23)

* **Strategy Architecture:** `MULTI-SLOT GRID WITH VEILIGE ZONE BREAK-EVEN STOP`
* **Filters:** Expected Win (`>=0.15%`) | Dwell Block (`10m`) | Cluster Exit (`30m`) | BE Trigger (`|Z|=0.5`)

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 11
* **Batch Win Rate:** 81.82%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **2.1200%**

### 📜 Session Transaction Ledger
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL Trade Combination | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 04:02 | 04:32 | `SHORT` | 7502.20 | 7495.60 | `LONG` | 4116.06 | 4115.53 | **0.0375%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 04:04 | 04:32 | `SHORT` | 7503.20 | 7495.60 | `LONG` | 4114.12 | 4115.53 | **0.0678%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 05:22 | 05:52 | `LONG` | 7487.40 | 7490.50 | `SHORT` | 4132.64 | 4125.97 | **0.1014%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 09:42 | 10:12 | `SHORT` | 7462.00 | 7474.40 | `LONG` | 4094.59 | 4094.67 | **-0.0821%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 10:25 | 10:55 | `SHORT` | 7472.40 | 7477.60 | `LONG` | 4090.44 | 4093.30 | **0.0002%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 15:07 | 15:27 | `SHORT` | 7424.70 | 7413.40 | `LONG` | 4049.54 | 4054.41 | **0.1362%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 2** | 15:08 | 15:27 | `SHORT` | 7423.00 | 7413.40 | `LONG` | 4046.73 | 4054.41 | **0.1596%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 3** | 15:08 | 15:27 | `SHORT` | 7423.00 | 7413.40 | `LONG` | 4046.73 | 4054.41 | **0.1596%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 4** | 15:08 | 15:27 | `SHORT` | 7423.00 | 7413.40 | `LONG` | 4046.73 | 4054.41 | **0.1596%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 15:44 | 15:59 | `SHORT` | 7443.70 | 7420.30 | `LONG` | 4058.07 | 4043.16 | **-0.0265%** | `BREAK_EVEN_PROTECTION_EXIT` |
| **Slot 1** | 16:03 | 16:25 | `SHORT` | 7424.60 | 7421.70 | `LONG` | 4045.18 | 4054.51 | **0.1349%** | `MEAN_REVERSION_CONVERGENCE` |
