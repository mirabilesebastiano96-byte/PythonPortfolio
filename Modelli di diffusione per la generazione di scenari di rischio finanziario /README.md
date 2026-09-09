# Diffusion Models per la generazione di scenari di rischio finanziario

Confronto tra un **modello di diffusione (DDPM)** e due benchmark — **GARCH(1,1)-t** e **QuantGAN** — per la generazione di rendimenti giornalieri dell'indice **S&P 500** e la stima del **Value-at-Risk (VaR)** e dell'**Expected Shortfall (CVaR)**.

Progetto sviluppato nell'ambito della tesi magistrale in *Statistica e Data Science* (LM-82), area *Informatica e Finanza* — Università degli Studi di Palermo.

---

## Obiettivo

Un diffusion model impara la distribuzione dei rendimenti a partire da **dati reali** e genera scenari sintetici. Il progetto verifica due ipotesi:

1. **Realismo** — le serie generate riproducono i *fatti stilizzati* dei mercati (code grasse, volatility clustering, assenza di autocorrelazione nei rendimenti, effetto leva).
2. **Utilità per il risk management** — le stime di VaR/CVaR a 1 giorno ottenute dal modello superano il *backtesting* out-of-sample, al pari o meglio dei benchmark.

---

## Modelli a confronto

| Modello | Tipo | Ruolo |
|---|---|---|
| **GARCH(1,1)-t** | econometrico parametrico | benchmark classico |
| **QuantGAN** | deep, avversariale (TCN) | benchmark deep |
| **Diffusion (DDPM)** | deep, score-based | modello proposto |

---

## Dati

- **Fonte:** `yfinance` — prezzi *adjusted close* dell'indice S&P 500 (`^GSPC`), dal 1990 a oggi.
- **Trasformazione:** rendimenti logaritmici `r_t = ln(P_t / P_{t-1})`.
- **Split temporale:** train fino al 2018, test dal 2019 in poi (nessun leakage; media e deviazione standard stimate solo sul train).
- Per i mercati energetici basta cambiare ticker: `CL=F` (petrolio), `NG=F` (gas naturale).

---

## Pipeline

```
prezzi S&P500 (yfinance)
   └─ log-rendimenti
        └─ split temporale (train ≤2018 | test 2019→)
             └─ standardizzazione (μ, σ solo dal train)
                  └─ finestre scorrevoli (L = 64 giorni)
                       ├─ GARCH(1,1)-t   → volatilità condizionale → VaR/CVaR
                       ├─ QuantGAN (TCN) → forma delle code + scala EWMA → VaR/CVaR
                       └─ Diffusion DDPM → forma delle code + scala EWMA → VaR/CVaR
                             └─ valutazione:
                                  ├─ fatti stilizzati (curtosi, ACF, QQ-plot)
                                  └─ backtesting VaR/CVaR (Kupiec, Christoffersen, CC)
```

Per rendere confrontabili i tre VaR, i modelli generativi forniscono la **forma delle code** (quantili empirici delle innovazioni standardizzate) mentre la **scala condizionale** è data dalla volatilità **EWMA (RiskMetrics)**; il GARCH usa la propria volatilità condizionale. Tutte le stime di VaR risultano così condizionali e comparabili.

---

## Risultati (esecuzione preliminare)

Backtesting del **VaR 99% a 1 giorno** sul periodo di test (2019→), livello di significatività 5% sui p-value. Configurazione *illustrativa* a poche epoche (`EPOCHS_GAN = 40`, `EPOCHS_DIFF = 60`); su ~19 violazioni attese.

| Modello | Violazioni | Attese | Freq. % | Kupiec p | Christoffersen p | CC p | Esito |
|---|---|---|---|---|---|---|---|
| GARCH(1,1)-t | 0 | 19.3 | — | — | — | — | VaR troppo estremo|
| QuantGAN | 38 | 19.3 | 1.97 | 0.0002 | 0.218 | 0.0004 | VaR troppo stretto |
| Diffusion | 31 | 19.3 | 1.61 | 0.014 | 0.100 | 0.013 | migliore dei deep; passa l'indipendenza |

A questo livello di addestramento entrambi i modelli deep **sottostimano il rischio** (più violazioni delle attese: code apprese ancora troppo sottili). Il **Diffusion** risulta però più calibrato del QuantGAN: meno violazioni e unico a superare il test di indipendenza di Christoffersen (le violazioni non si concentrano a grappoli). Il QuantGAN, tipicamente instabile nell'addestramento avversariale, produce un VaR più stretto.

I risultati sopra sono a poche epoche e servono a validare la pipeline end-to-end. 
---

## Contenuto del repository

```
.
├── Diffusion_VaR_SP500.ipynb   # notebook completo (download → modelli → backtesting)
└── README.md
```

---

## Come eseguire

Il notebook è pensato per **Google Colab**.

1. Apri `Diffusion_VaR_SP500.ipynb` in Colab.
2. `Runtime ▸ Change runtime type ▸ GPU`.
3. `Runtime ▸ Run all`.

La prima cella installa `yfinance` e `arch`; `torch` è già presente su Colab.

### Esecuzione in locale

```bash
pip install yfinance arch statsmodels torch scipy matplotlib pandas numpy
jupyter notebook Diffusion_VaR_SP500.ipynb
```

---

## Iperparametri principali

Tutti raccolti nella cella di setup del notebook:

| Parametro | Default | Descrizione |
|---|---|---|
| `TICKER` | `^GSPC` | asset analizzato |
| `SPLIT_DATE` | `2018-12-31` | confine train/test |
| `L` | `64` | lunghezza finestra (giorni) |
| `ALPHA` | `0.01` | livello VaR/CVaR (99%) |
| `LAMBDA_EWMA` | `0.94` | fattore di decadimento EWMA (scala dei VaR generativi) |
| `EPOCHS_GAN` | `40` | epoche QuantGAN (≈200 per risultati definitivi) |
| `EPOCHS_DIFF` | `60` | epoche Diffusion (≈300 per risultati definitivi) |

---

## Riproducibilità e calibrazione

Per passare dai risultati preliminari a quelli definitivi:

- eseguire su **GPU** con `EPOCHS_GAN ≈ 200` ed `EPOCHS_DIFF ≈ 300` (code apprese più realistiche → meno violazioni);
- ridurre `LAMBDA_EWMA` verso ~0.90 per una scala di volatilità più reattiva;
- ripetere con **seed diversi** per stimare la variabilità dei risultati.

> **Nota tecnica.** Il modello GARCH va stimato su rendimenti in **scala percentuale** (`× 100`) e riportato indietro: su rendimenti in decimali l'ottimizzatore di `arch` converge male e produce un VaR piatto e ingiustificatamente estremo.

---

## Come leggere i risultati

**Fatti stilizzati** — il modello migliore è quello le cui statistiche (curtosi, asimmetria, ACF dei rendimenti²) sono più vicine a quelle della serie reale.

**Backtesting** (soglia 5% sui p-value):

- `Kupiec p > 0.05` → il numero di violazioni è coerente con il 99%;
- `Christoffersen p > 0.05` → le violazioni sono indipendenti (non a grappoli);
- `CC p > 0.05` → copertura condizionale corretta.

---

## Estensioni

- Generazione **condizionale** end-to-end col diffusion (senza scala EWMA).
- Multi-asset con matrice di correlazione.
- Mercati energetici (`CL=F`, `NG=F`).
- QuantGAN con trasformazione di Lambert-W e WGAN-GP.

---

## Riferimenti

- Ho, Jain, Abbeel (2020), *Denoising Diffusion Probabilistic Models*.
- Song et al. (2021), *Score-Based Generative Modeling through Stochastic Differential Equations*.
- Wiese et al. (2020), *Quant GANs: Deep Generation of Financial Time Series*.
- Kupiec (1995); Christoffersen (1998) — backtesting del VaR.

