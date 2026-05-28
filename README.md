# 📊 Stock Market Analysis & Portfolio Optimization (S&P 500)

Questo progetto analizza sei azioni appartenenti all’indice **S&P 500** attraverso tecniche di analisi dei dati finanziari, statistica descrittiva, modellazione predittiva e ottimizzazione di portafoglio.

L’obiettivo è confrontare performance, rischio e correlazioni tra titoli di settori differenti e costruire portafogli efficienti.

---

## Asset analizzati

Il dataset include 6 titoli appartenenti a 3 settori:

### Tecnologia
- NVIDIA (NVDA)
- AMD (Advanced Micro Devices)

### Servizi di pagamento
- Visa (V)
- Mastercard (MA)

### Food & Beverage
- McDonald’s (MCD)
- Starbucks (SBUX)

---

## Fonte dati

I dati storici dei prezzi sono stati scaricati da **Yahoo Finance** e preprocessati utilizzando Python.

È stato mantenuto principalmente il prezzo **Adjusted Close** per ciascun titolo.

---

## Obiettivi del progetto

- Analisi dell’andamento dei prezzi azionari
- Calcolo di rendimenti semplici, logaritmici e cumulati
- Analisi statistica descrittiva (media, varianza, skewness, kurtosis)
- Studio delle correlazioni tra asset
- Modelli previsivi (ARIMA)
- Strategie di trading (moving averages)
- Modelli di rischio (CAPM e Fama-French)
- Ottimizzazione di portafoglio (media-varianza)

---

# 1. Analisi esplorativa dei prezzi

Le serie temporali mostrano comportamenti eterogenei tra i settori:

- **NVIDIA**: crescita esponenziale, trainata dal boom AI
- **AMD**: crescita elevata ma più volatile
- **Visa & Mastercard**: crescita stabile e correlata
- **McDonald’s & Starbucks**: andamento più stabile e conservativo

---

# 2. Rendimenti

## Rendimenti annui medi

- NVDA: 0.7336  
- AMD: 0.4532  
- MA: 0.1983  
- V: 0.1848  
- MCD: 0.1228  
- SBUX: 0.0994  

👉 Il settore tecnologico domina in termini di performance.

---

## Rendimenti cumulati e logaritmici

- Forte crescita di NVDA negli ultimi anni
- Settore tech altamente dominante
- Settore consumer più stabile ma meno redditizio

---

# 3. Statistica descrittiva

Sono state analizzate:

- media
- varianza
- deviazione standard
- skewness
- curtosi

### Risultati principali

- Tutti i titoli hanno media leggermente positiva
- **AMD**: maggiore volatilità
- **MCD & V**: più stabili
- Distribuzioni approssimativamente normali ma con outliers

---

# 4. Correlazioni tra asset

- Visa ↔ Mastercard: correlazione ~0.89 (molto forte)
- AMD ↔ NVIDIA: correlazione ~0.58
- MCD ↔ SBUX: correlazione ~0.57

👉 Le correlazioni sono più forti tra aziende dello stesso settore.

---

# 5. Modello ARIMA (previsioni)

È stato implementato un modello **ARIMA con auto-arima selection**:

- training / validation / test split
- test di stazionarietà (ADF)
- log-transform + differenziazione
- rolling forecasting

### Risultati

- buona capacità predittiva
- errori (MSE, MAE, RMSE) contenuti
- NVIDIA e Visa tra i migliori risultati previsivi

---

# 6. Strategie di trading

Sono state confrontate:

- Moving Average semplice (20 vs 120)
- Exponential Moving Average (EMA 12 vs 26)
- Buy & Hold

### Risultati

- EMA generalmente più performante
- NVIDIA e AMD mostrano i guadagni più elevati
- Visa e McDonald’s più stabili ma meno redditizi

---

# 7. CAPM (Capital Asset Pricing Model)

È stato stimato il beta dei titoli rispetto all’S&P500:

- NVDA: 1.69
- AMD: 1.61
- MA: 1.18
- V: 1.07
- MCD: 0.70
- SBUX: 1.00

👉 I titoli tech risultano più rischiosi del mercato.

---

## Rendimenti attesi (CAPM)

- NVDA: 19.35%
- AMD: 18.48%
- MA: 13.70%
- V: 12.39%
- SBUX: 11.56%
- MCD: 8.25%

---

# 8. Fama-French 3 factors

Sono stati stimati:

- SMB (size factor)
- HML (value factor)
- market risk factor

👉 Risultati:
- modelli debolmente esplicativi per alcuni titoli
- miglior adattamento per aziende più stabili
- settore tech meno spiegato dai fattori tradizionali

---

# 9. Portfolio optimization

Sono stati costruiti portafogli con:

- Minimum Variance Portfolio (MVP)
- Maximum Sharpe Ratio Portfolio

## Risultati principali

### MVP (simulativo)
- Return: ~19%
- Volatility: ~20%
- Sharpe: ~0.97

### Max Sharpe
- Return: ~41%
- Volatility: ~30%
- Sharpe: ~1.35

---

## Confronto con Equal Weighted Portfolio

- Equal Weight:
  - Return: ~31.8%
  - Volatility: ~26.1%
  - Sharpe: 1.22

👉 Sorprendentemente competitivo rispetto agli ottimizzati.

---

# Conclusioni

- Il settore tecnologico domina per rendimento e rischio
- Le correlazioni sono forti intra-settore
- ARIMA fornisce buone previsioni ma non perfette
- EMA supera spesso strategie classiche
- CAPM spiega solo parzialmente i rendimenti
- Equal weighting resta una baseline sorprendentemente robusta

---

# Tech stack

- Python
- pandas, numpy
- matplotlib, seaborn
- statsmodels
- yfinance
- scikit-learn
- arch / finance libs (se presenti)

