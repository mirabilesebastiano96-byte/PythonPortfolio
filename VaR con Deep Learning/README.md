# Stima del Value-at-Risk tramite Deep Learning

Confronto tra architetture neurali (LSTM, CNN 1D) per la stima diretta del Value-at-Risk (VaR) tramite quantile regression.

## Obiettivo

Stimare direttamente i quantili della distribuzione dei rendimenti (VaR al 95% e al 99%) tramite reti neurali addestrate con pinball loss (quantile loss), confrontando due architetture:

- **LSTM** — cattura dipendenze sequenziali nella serie storica dei rendimenti
- **CNN 1D** — estrae pattern locali tramite convoluzioni sulla finestra temporale

I modelli vengono validati tramite backtesting standard (Kupiec test, Christoffersen test) e confrontati sulla pinball loss del test set.

## Dataset

- **Asset**: ESGU (iShares ESG Aware MSCI USA ETF)
- **Fonte dati**: Yahoo Finance (via `yfinance`)
- **Periodo**: dal 2010-01-01 a oggi (2.434 osservazioni giornaliere di prezzo)
- **Target**: rendimento log cumulato a 5 giorni (orizzonte settimanale)
- **Finestra di input**: 20 giorni di rendimenti storici
- **Campioni dopo windowing**: 2.409 (X: 2409×20, y: 2409)
- **Split temporale (no shuffle)**: train 1.686 / val 361 / test 362

## Struttura del progetto

```
.
├── notebook.ipynb          # Pipeline completa: dati, modelli, backtesting
└── README.md
```

## Modelli

**LSTM**: 2 layer LSTM (32 → 16 unità) + Dense(8, relu) + Dense(1), un modello indipendente per ciascun quantile (0.01, 0.05), ottimizzatore Adam, early stopping su val_loss (patience=10).

**CNN 1D**: 2 layer Conv1D (32 → 16 filtri, kernel=3) + GlobalAveragePooling1D + Dense(8, relu) + Dense(1), stessa procedura di training della LSTM.

Entrambe le architetture sono addestrate con **pinball loss** (quantile loss) per stimare direttamente i quantili 0.01 e 0.05 della distribuzione dei rendimenti a 5 giorni.

## Risultati

### VaR medio stimato sul test set

| Modello | VaR 99% medio | VaR 95% medio |
|---|---|---|
| LSTM | 6.45% | 3.24% |
| CNN  | 6.21% | 3.28% |

### Backtesting — Test di Kupiec (unconditional coverage)

| Modello | Livello | Violazioni osservate | Attese | Violation rate | p-value | Esito |
|---|---|---|---|---|---|---|
| LSTM | 99% | 3 | 3.62 | 0.83% | 0.736 | ✅ superato |
| LSTM | 95% | 14 | 18.10 | 3.87% | 0.304 | ✅ superato |
| CNN | 99% | 3 | 3.62 | 0.83% | 0.736 | ✅ superato |
| CNN | 95% | 15 | 18.10 | 4.14% | 0.442 | ✅ superato |

Entrambi i modelli producono un tasso di violazione coerente con quello atteso ai due livelli di confidenza.

### Backtesting — Test di Christoffersen (independence)

| Modello | Livello | LR_ind | p-value | Esito |
|---|---|---|---|---|
| LSTM | 99% | 17.14 | 3.5e-05 | ❌ fallito |
| LSTM | 95% | 38.74 | 4.8e-10 | ❌ fallito |
| CNN | 99% | 17.14 | 3.5e-05 | ❌ fallito |
| CNN | 95% | 53.40 | 2.7e-13 | ❌ fallito |

Entrambi i modelli **falliscono** il test di indipendenza con p-value molto bassi: le violazioni tendono a raggrupparsi nel tempo invece di essere distribuite casualmente — un limite critico per un modello VaR, perché segnala che il rischio non viene aggiornato abbastanza rapidamente in periodi di stress.

### Pinball Loss sul test set

| Quantile | LSTM | CNN | Modello migliore |
|---|---|---|---|
| 0.01 (VaR 99%) | 0.001047 | 0.000967 | CNN (leggero vantaggio) |
| 0.05 (VaR 95%) | 0.002594 | 0.002619 | LSTM (leggero vantaggio) |

Le differenze tra i due modelli sono minime: performance sostanzialmente comparabili in termini di pinball loss.

## Conclusioni

- **Accuratezza aggregata (Kupiec)**: entrambi i modelli stimano correttamente il tasso di violazione atteso, a entrambi i livelli di confidenza.
- **Indipendenza (Christoffersen)**: entrambi i modelli falliscono, indicando clustering delle violazioni — il limite più importante emerso dall'analisi.
- **Pinball loss**: performance pressoché equivalenti tra LSTM e CNN, con vantaggi marginali e opposti ai due livelli di confidenza.

**Sviluppi futuri possibili**:
- modelli ibridi (es. GARCH-LSTM) per catturare meglio la dinamica temporale della volatilità
- multi-task learning per prevedere più quantili contemporaneamente
- feature aggiuntive (volatilità realizzata, indicatori macro)
- ottimizzazione degli iperparametri più sistematica

## Requisiti

```
pandas
numpy
matplotlib
seaborn
scipy
statsmodels
tensorflow
yfinance
```

Installazione:
```bash
pip install -r requirements.txt
```

## Utilizzo

```bash
# 1. Scarica i dati
python download_data.py

# 2. Esegui il notebook con la pipeline completa
jupyter notebook notebook.ipynb
```

## Note metodologiche

- Il VaR è stimato come opposto del quantile previsto (convenzione: perdita espressa come valore positivo)
- Vengono addestrati modelli separati per ciascun quantile (approccio quantile regression, non distribuzionale)
- Nessun modello econometrico (GARCH o simili) è incluso nel confronto: l'obiettivo è valutare le architetture neurali tra loro, non DL vs econometria classica

