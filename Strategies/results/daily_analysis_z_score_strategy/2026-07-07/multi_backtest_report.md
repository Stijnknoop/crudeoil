# 📊 MANTRA: Layered Z-Score Session Report (2026-07-07)

* **Strategy Architecture:** `PURE MATHEMATICAL MULTI-SLOT GRID WITH 15M Z-SMA FILTER`
* **Configured Slot Thresholds:** Slot 1 (`1.5`), Slot 2 (`2.0`), Slot 3 (`2.5`), Slot 4 (`3.0`)
* **Filters:** Z-Score Smoothing (`15m SMA`) | Max Z-Stop (`3.5`) | Freeze (`120m`)
* **Operational Windows:** Entries `04:00 - 20:00` | Forced Hard EOD Close `22:00`

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 4
* **Batch Win Rate:** 25.00%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **-1.9928%**

### 📜 Session Transaction Ledger (Slot Decomposition)
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL Trade Combination | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 07:48 | 08:33 | `SHORT` | 7519.20 | 7521.00 | `LONG` | 4121.52 | 4129.17 | **0.0808%** | `MEAN_REVERSION_CONVERGENCE` |
| **Slot 1** | 13:31 | 16:31 | `LONG` | 7524.00 | 7493.00 | `SHORT` | 4139.59 | 4157.24 | **-0.4192%** | `MAX_HOLDING_TIME_EXCEEDED` |
| **Slot 1** | 18:05 | 21:05 | `SHORT` | 7504.60 | 7499.70 | `LONG` | 4141.89 | 4119.59 | **-0.2366%** | `MAX_HOLDING_TIME_EXCEEDED` |
| **Slot 2** | 18:20 | 21:05 | `SHORT` | 7510.40 | 7499.70 | `LONG` | 4143.91 | 4119.59 | **-0.2222%** | `MAX_HOLDING_TIME_EXCEEDED` |
