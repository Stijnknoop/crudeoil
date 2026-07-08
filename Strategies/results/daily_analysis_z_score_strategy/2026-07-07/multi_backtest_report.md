# 📊 MANTRA: Layered Z-Score Session Report (2026-07-07)

* **Strategy Architecture:** `PURE MATHEMATICAL MULTI-SLOT GRID WITH TIMED CLUSTER RISK CONTROLS`
* **Configured Slot Thresholds:** Slot 1 (`2.0`), Slot 2 (`2.5`), Slot 3 (`3.0`), Slot 4 (`3.5`)
* **Filters:** Min Expected Win (`0.2%`) | Max Entry Dwell (`15m`) | Max Cluster Hold (`30m`)
* **Operational Windows:** Entries `04:00 - 20:00` | Forced Hard EOD Close `22:00`

### 🔍 Session Statistical Highlights
* **Max Z-Score Dwell Time (|Z| >= 2.0):** `68 minutes`
* **Peak Session Expected Win Value:** `0.4336%`

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 3
* **Batch Win Rate:** 66.67%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **0.4265%**

### 📜 Session Transaction Ledger
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL Trade Combination | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 14:28 | 14:58 | `LONG` | 7533.80 | 7530.40 | `SHORT` | 4168.41 | 4171.40 | **-0.0584%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 15:36 | 16:06 | `LONG` | 7525.00 | 7516.50 | `SHORT` | 4178.35 | 4155.41 | **0.2180%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
| **Slot 1** | 18:45 | 19:15 | `SHORT` | 7517.10 | 7515.30 | `LONG` | 4142.37 | 4142.29 | **0.0110%** | `CRITICAL_DWELL_TIME_EXCEEDED` |
