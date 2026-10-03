# 📊 MANTRA: Layered Z-Score Session Report (2026-10-02)

* **Strategy Architecture:** `MULTI-SLOT GRID WITH VEILIGE ZONE BREAK-EVEN STOP`
* **Filters:** Expected Win (`>=0.15%`) | Dwell Block (`10m`) | Cluster Exit (`30m`) | BE Trigger (`|Z|=0.5`)

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 12
* **Batch Win Rate:** 58.33%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **0.1792%**

### 📜 Session Transaction Ledger
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL Trade Combination | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 05:03 | 05:33 | `LONG` | 7685.70 | 7684.10 | `SHORT` | 4172.06 | 4174.09 | **-0.0347%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 05:04 | 05:33 | `LONG` | 7686.20 | 7684.10 | `SHORT` | 4172.79 | 4174.09 | **-0.0292%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 06:02 | 06:32 | `LONG` | 7688.90 | 7688.00 | `SHORT` | 4184.33 | 4189.27 | **-0.0649%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 14:30 | 15:00 | `LONG` | 7722.10 | 7739.30 | `SHORT` | 4213.79 | 4218.45 | **0.0561%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 2** | 14:31 | 15:00 | `LONG` | 7729.60 | 7739.30 | `SHORT` | 4218.13 | 4218.45 | **0.0590%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 3** | 14:31 | 15:00 | `LONG` | 7729.60 | 7739.30 | `SHORT` | 4218.13 | 4218.45 | **0.0590%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 4** | 14:31 | 15:00 | `LONG` | 7729.60 | 7739.30 | `SHORT` | 4218.13 | 4218.45 | **0.0590%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 15:01 | 15:09 | `LONG` | 7743.40 | 7739.10 | `SHORT` | 4222.73 | 4205.36 | **0.1779%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 15:22 | 15:43 | `SHORT` | 7735.70 | 7740.10 | `LONG` | 4187.14 | 4186.93 | **-0.0309%** | `BREAK_EVEN_PROTECTION_EXIT` |
| **Slot 1** | 15:44 | 16:14 | `SHORT` | 7743.40 | 7743.70 | `LONG` | 4186.81 | 4188.68 | **0.0204%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 16:25 | 16:55 | `SHORT` | 7750.40 | 7743.50 | `LONG` | 4178.48 | 4157.19 | **-0.2102%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 16:56 | 17:26 | `SHORT` | 7743.40 | 7708.40 | `LONG` | 4159.64 | 4141.71 | **0.0105%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
