# 📊 MANTRA: Layered Z-Score Session Report (2026-07-07)

* **Strategy Architecture:** `PURE MATHEMATICAL MULTI-SLOT GRID WITH RISK CONTROLS`
* **Configured Slot Thresholds:** Slot 1 (`1.5`), Slot 2 (`2.0`), Slot 3 (`2.5`), Slot 4 (`3.0`)
* **Risk Configuration:** Max Z-Stop (`3.5`) | Freeze (`120m`) | Max Hold (`180m`)
* **Operational Windows:** Entries `04:00 - 20:00` | Forced Hard EOD Close `22:00`

### 📈 Session Key Performance Metrics
* **Total Scaled Batches Executed:** 4
* **Batch Win Rate:** 0.00%
* **Pure Combination Trade Yield (Rauw Totaal):** -0.7856%
* **Net Portfolio Session Yield (1x Base Portfolio):** -0.1964%
* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **-1.9639%**
* **Average Yield per Executed Slot (1x Base Portfolio):** -0.0491%
* **Average Yield per Executed Slot (10x Leveraged Portfolio):** -0.4910%

### 📜 Session Transaction Ledger (Slot Decomposition)
| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | PnL US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL GOLD | PnL Trade Combination | Cash PnL (1x) | Cash PnL (10x Leverage) | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slot 1** | 07:41 | 07:42 | `SHORT` | 7521.40 | 7522.30 | -0.0120% | `LONG` | 4118.78 | 4117.99 | -0.0192% | **-0.0156%** | -0.0039% | **-0.0389%** | `HARD_Z_SCORE_STOP` |
| **Slot 1** | 13:19 | 16:19 | `LONG` | 7527.30 | 7501.60 | -0.3414% | `SHORT` | 4140.93 | 4158.99 | -0.4361% | **-0.3888%** | -0.0972% | **-0.9719%** | `MAX_HOLDING_TIME_EXCEEDED` |
| **Slot 1** | 17:52 | 20:52 | `SHORT` | 7499.10 | 7504.70 | -0.0747% | `LONG` | 4143.53 | 4127.71 | -0.3818% | **-0.2282%** | -0.0571% | **-0.5706%** | `MAX_HOLDING_TIME_EXCEEDED` |
| **Slot 2** | 18:13 | 20:52 | `SHORT` | 7508.50 | 7504.70 | 0.0506% | `LONG` | 4142.48 | 4127.71 | -0.3565% | **-0.1530%** | -0.0382% | **-0.3824%** | `MAX_HOLDING_TIME_EXCEEDED` |
