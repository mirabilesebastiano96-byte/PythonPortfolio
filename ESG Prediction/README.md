# Healthcare Stock Analysis with ESG S-Scores

Analysis of the relationship between healthcare stock returns, ESG (Environmental, Social, Governance) S-Scores, and Fama-French risk factors.

## Overview

This project investigates whether ESG S-Scores have a significant influence on the risk-adjusted returns of selected healthcare stocks, and how they interact with traditional market factors (Mkt-RF, SMB, HML) in explaining stock performance.

**Stocks analyzed:** GILD, PODD, BSX, ISRG, PFE
**Period:** February 2015 – March 2025 (monthly data)

## Project structure

1. **Data collection** – historical closing prices downloaded via `yfinance`
2. **Returns calculation** – monthly percentage returns for each stock
3. **ESG & factor data** – ESG S-Scores and Fama-French factors (Mkt-RF, SMB, HML, RF) loaded from an external Excel file (`datiesg.xlsx`)
4. **Correlation analysis** – heatmap of correlations between financial and ESG variables
5. **Regression analysis** – individual OLS models (`statsmodels`) for each stock's excess return:
6. Stock-RF ~ Stock_S_Score + Mkt_RF + SMB + HML

7. ## Results

OLS regressions show that market factors (especially Mkt-RF) are consistently significant across all stocks, while ESG S-Scores generally show weak/non-significant coefficients within this sample — suggesting ESG scores alone have limited explanatory power on excess returns once market risk factors are controlled for.

## Requirements

```bash
pandas
numpy
yfinance
matplotlib
seaborn
statsmodels
openpyxl   # for reading the Excel file
```

## Usage

1. Clone the repository
2. Place `datiesg.xlsx` (ESG S-Scores + Fama-French factors) in the project root
3. Run the notebook — it will automatically download stock prices via `yfinance`

## Data sources

- Stock prices: [Yahoo Finance](https://finance.yahoo.com/) via `yfinance`
- ESG S-Scores and Fama-French factors: external file `datiesg.xlsx` (source not specified — add citation if needed)
