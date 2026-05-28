#!/usr/bin/env python
# coding: utf-8

# # PROGETTO BUSINESS INTELLIGENCE GIUDITTA ADEZIO

# **1. IMPORT LIBRERIE E CARICAMENTO SERIE STORICHE**

# In[793]:


import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import yfinance as yf
import scipy.stats as stats
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import itertools
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore")
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from math import sqrt
from pylab import mpl, plt
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import statsmodels.api as sm
from matplotlib.gridspec import GridSpec
import statsmodels.formula.api as smf
import scipy.optimize as sco


# In[351]:


start = datetime.date(2014,5,31)
end = datetime.date(2024,5,31)


# In[354]:


# funzione per scaricare i dati
def get(tickers, start, end):
    def data(ticker):
        return yf.download(ticker, start, end)
    datas = map(data, tickers)
    return pd.concat(datas, keys = tickers, names = ['Ticker','Date'])


# In[352]:


tickers = ['V', 'MA', 'NVDA', 'AMD', 'MCD', 'SBUX']


# In[355]:


# scarico i dati
data = get(tickers, start, end)
data[:5]


# In[356]:


# scarico s&p500
sp500 = yf.download('^GSPC', start, end)


# In[357]:


# seleziono solo adjusted close
adj_close = data[['Adj Close']].reset_index()
adj_close


# In[666]:


# trasformo per comodità in una tabella pivot
daily_adj_close = adj_close.pivot(index = 'Date', columns = 'Ticker', values = 'Adj Close')
daily_adj_close[:5]


# **GRAFICO ANDAMENTO PREZZI**

# In[359]:


daily_adj_close.plot(figsize=(12,8), color = ['purple', 'magenta', 'blue', 'lawngreen', 'yellow', 'red'])
plt.grid()


# In[13]:


# punto numero 1
data[:5]


# # 2. STATISTICHE DESCRITTIVE

# **CALCOLO E RAPPRESENTAZIONE RITORNI ANNUI, CUMULATI, SEMPLICI E LOGARITMICI**

# In[15]:


# per selezionare prima e ultima data per il calcolo dei ritorni annui
selected_dates = adj_close.iloc[[0,-1]]
selected_dates


# In[16]:


ritorni_annui = (adj_close.iloc[-1]/adj_close.iloc[0])**(1/10) - 1
ritorni_annui


# In[667]:


ritorni_lordi = (daily_adj_close/daily_adj_close.shift(1)).dropna()
ritorni_cumulati = np.cumprod(daily_adj_close/daily_adj_close.shift(1)).dropna()
ritorni_cumulati[:5]


# In[18]:


# rappresento ritorni cumulati
ritorni_cumulati.plot(figsize = (12,8), color = ['purple', 'magenta', 'blue', 'lawngreen', 'yellow', 'red'])
plt.grid()


# In[668]:


# b
ritorni_semplici = daily_adj_close.pct_change().dropna()
ritorni_semplici[:5]


# In[20]:


(daily_adj_close/daily_adj_close.shift() -1).dropna()


# In[21]:


ritorni_logaritmici = np.log((daily_adj_close/daily_adj_close.shift(1))).dropna() 
ritorni_logaritmici


# In[22]:


# rappresento ritorni semplici
ritorni_semplici.plot(figsize = (12,8), color = ['purple', 'magenta', 'blue', 'lawngreen', 'yellow', 'red'])


# In[673]:


# li rappresento anche singolarmente in modo datle da vedere bene l'andamento dei singoli titoli
fig, axs = plt.subplots(3, 2, figsize=(12, 12))

# Itera sui nomi delle colonne
for i, column in enumerate(ritorni_semplici.columns):
    # Calcola le coordinate del subplot in base all'indice i
    row = i // 2  # Riga (da 0 a 2)
    col = i % 2   # Colonna (0 o 1)

    # Plot del grafico per la colonna corrente
    ritorni_semplici[column].plot(ax=axs[row, col], title=column, color='purple')
    axs[row, col].set_ylabel('Value')
    axs[row, col].set_xlabel('Date')
    axs[row, col].grid()

# Imposta il layout in modo che i grafici non si sovrappongano
plt.tight_layout()
plt.show()


# In[24]:


# rappresento ritorni logaritmici
ritorni_logaritmici.plot(figsize = (12,8), color = ['purple', 'magenta', 'blue', 'lawngreen', 'yellow', 'red'])


# In[675]:


fig, axs = plt.subplots(3, 2, figsize=(12, 12))

for i, column in enumerate(ritorni_logaritmici.columns):
    # Calcola le coordinate del subplot in base all'indice i
    row = i // 2  # Riga (da 0 a 2)
    col = i % 2   # Colonna (0 o 1)

    # Plot del grafico per la colonna corrente
    ritorni_logaritmici[column].plot(ax=axs[row, col], title=column, color='lawngreen')
    axs[row, col].set_ylabel('Value')
    axs[row, col].set_xlabel('Date')
    axs[row, col].grid()

# Imposta il layout in modo che i grafici non si sovrappongano
plt.tight_layout()
plt.show()  


# In[25]:


# d
ritorni_semplici.hist(bins = 100, sharex = True, figsize = (12,8), color = 'purple');


# In[26]:


# e
def diagnostic_plots(data, title):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.histplot(data.dropna(), kde = True, ax = axes[0], color = 'purple', bins = 50)
    axes[0].set_title(f'{title} - Istogramma e Kernel density')
    
    sns.boxplot(y = data.dropna(), ax = axes[1], color = 'lawngreen')
    axes[1].set_title(f'{title} - Boxplot')
    
    stats.probplot(data.dropna(), dist = "norm", plot = axes[2])
    axes[2].set_title(f'{title} - QQ Plot')
    
    plt.tight_layout()
    plt.show()

for column in ritorni_semplici.columns:
    diagnostic_plots(ritorni_semplici[column], column)


# **CALCOLO STATISTICHE DESCRITTIVE GENERALI, MATRICE DI VARIANZA E COVARIANZA E CORRELAZIONI**

# In[27]:


# f
statistiche_descrittive = pd.DataFrame(columns = ['AMD', 'MA', 'MCD', 'NVDA', 'SBUX', 'V'])

statistiche_descrittive.loc['media']      = ritorni_semplici.apply(lambda x: np.mean(x))
statistiche_descrittive.loc['varianza']   = ritorni_semplici.apply(lambda x: np.var(x))
statistiche_descrittive.loc['std']        = ritorni_semplici.apply(lambda x: np.std(x))
statistiche_descrittive.loc['curtosi']    = ritorni_semplici.apply(lambda x: stats.kurtosis(x))
statistiche_descrittive.loc['asimmetria'] = ritorni_semplici.apply(lambda x: stats.skew(x))


statistiche_descrittive


# In[28]:


# g
varcov = ritorni_semplici.cov()
varcov


# In[29]:


corr = ritorni_semplici.corr()
corr


# In[685]:


# eatmap per visualizzare graficamente matrice var-cov e matrice corr
fig, axes = plt.subplots(ncols=2, figsize=(20, 8))

sns.heatmap(varcov, annot = True, cmap = 'viridis', ax = axes[0], linewidth = 1, linecolor= 'white')
axes[0].set_title('Matrice di varianza covarianza')

sns.heatmap(corr, annot = True, cmap = 'viridis', ax=axes[1], linewidth = 1, linecolor= 'white')
axes[1].set_title('Matrice di correlazione')

plt.tight_layout()
plt.show()


# In[686]:


# h
ritorni_semplici['NVDA'].rolling(250).corr(ritorni_semplici['AMD']).plot(figsize = (12,6), color = 'lawngreen', title = 'Correlazione nel tempo tra i titoli NVDA e AMD')
plt.grid()


# In[688]:


ritorni_semplici['MA'].rolling(250).corr(ritorni_semplici['V']).plot(figsize = (12,6), color = 'purple', title = 'Correlazione nel tempo tra i titoli MA e V')
plt.grid()


# In[689]:


ritorni_semplici['MCD'].rolling(250).corr(ritorni_semplici['SBUX']).plot(figsize = (12,6), color = 'blue', title = 'Correlazione nel tempo tra i titoli MCD e SBUX')
plt.grid()


# In[692]:


plt.figure(figsize=(12, 8))
sns.regplot(x = 'NVDA', y = 'AMD', data = ritorni_semplici, marker = '*', color = 'lawngreen',  scatter_kws={'alpha': 0.5})
plt.title('Scatterplot NVDA AMD') 
plt.grid()


# In[693]:


plt.figure(figsize=(12, 8))
sns.regplot(x = 'MA', y = 'V', data = ritorni_semplici, marker = '*', color = 'purple',  scatter_kws={'alpha': 0.5})
plt.title('Scatterplot MA V')
plt.grid()


# In[694]:


plt.figure(figsize=(12, 8))
sns.regplot(x = 'MCD', y = 'SBUX', data = ritorni_semplici, marker = '*', color = 'blue',  scatter_kws={'alpha': 0.5})
plt.title('Scatterplot MCD SBUX')
plt.grid()


# # 3. PREVISIONE

# In[37]:


data3 = adj_close.resample('M').last()
data3


# **ARIMA**

# In[206]:


# divisione training test e validation
start = adj_close.index.min()
val_start = start + pd.DateOffset(months=80)
test_start = val_start + pd.DateOffset(months=30)

training = adj_close[adj_close.index < val_start]
validation = adj_close[(adj_close.index >= val_start) & (adj_close.index < test_start)]
test = adj_close[adj_close.index >= test_start]


# In[207]:


# test di stazionarietà
def adf_test(series):
    result = adfuller(series)
    return {'Test Statistic': result[0], 
            'p-value': result[1], 
            'Lags Used': result[2], 
            'Observations Used': result[3], 
            'Critical Value 1%': result[4]['1%'],
            'Critical Value 5%': result[4]['5%'],
            'Critical Value 10%': result[4]['10%'],
            'IC best': result[5]}

adf_results = {column: adf_test(training[column]) for column in training.columns}
adf_results_df = pd.DataFrame(adf_results)
adf_results_df


# In[696]:


# log e dfferenziazione della serie per renderla stazionaria
log_series = np.log(training).dropna()
diff_series = log_series.diff().dropna()
log_validation = np.log(validation).dropna()
log_test = np.log(test).dropna()

# testiamola
adf_results = {column: adf_test(diff_series[column]) for column in diff_series.columns}
adf_results_df = pd.DataFrame(adf_results)
adf_results_df


# In[209]:


# le serie sono diventate stazionarie, grafichiamole per verificare ulteriormente
diff_series.plot(figsize = (12,8), color = ['purple', 'magenta', 'blue', 'lawngreen', 'yellow', 'red'])


# In[210]:


# rappresentiamo le serie singolarmente per vedere bene 
for column in diff_series.columns:
    plt.figure()
    diff_series[column].plot(figsize = (12,8), title = column, color = 'purple')
    plt.ylabel('Value')
    plt.xlabel('Date')
    plt.grid()
    plt.show() 


# In[211]:


# decomponiamo le serie con la scomposizione classica
for column in diff_series.columns:
    decompose_result = seasonal_decompose(diff_series[column], period=12)
    plt.rcParams['figure.figsize'] = (12,8)
    decompose_result.plot()
    plt.show()


# In[212]:


# 3
def arima_grid_search(dataframe, s):
    p = d = q = range(2)
    param_combinations = list(itertools.product(p, d, q))
    lowest_aic, pdq, pdqs = None, None, None
    total_iterations = 0
    for order in param_combinations:
        for (p, q, d) in param_combinations:
            seasonal_order = (p, q, d, s)
            total_iterations += 1
            try:
                model = SARIMAX(dataframe, order = order,
                               seasonal_order= seasonal_order,
                               enforce_stationarity=False,
                               enforce_invertibility=False,
                               disp = False)
                model_result = model.fit(maxiter=700, disp=False)

                if not lowest_aic or model_result.aic < lowest_aic:
                    lowest_aic = model_result.aic
                    pdq, pdqs = order, seasonal_order

            except Exception as ex:
                continue

    return lowest_aic, pdq, pdqs


# In[64]:


results = {}
for column in log_series.columns:
    lowest_aic, order, seasonal_order = arima_grid_search(log_series[column], 12)
    results[column] = {'lowest_aic': lowest_aic, 'order': order, 'seasonal_order': seasonal_order}

results_data = pd.DataFrame(results)
results_data # arima grid search spesso non arriva a convergenza


# In[213]:


# uso la funzione auto-arima per ottenere risultati migliori in tempi computazionali più rapidi
for column in log_series.columns:
    serie_storica = log_series[column]
    modello = auto_arima(serie_storica, seasonal = True, m = 12, suppress_warnings = True)
    results[column] = {'order': modello.order, 'seasonal_order': modello.seasonal_order}

results_data3 = pd.DataFrame(results)
results_data3


# In[214]:


# limito gli ordini dell'autoarima a 2
for column in log_series.columns:
    serie_storica = log_series[column]
    modello = auto_arima(serie_storica, seasonal = True, m = 12, max_p = 2, max_q = 2, max_d = 2, suppress_warnings = True)
    results[column] = {'order': modello.order, 'seasonal_order': modello.seasonal_order}

results_data2 = pd.DataFrame(results)
results_data2


# In[215]:


validation_forecast = pd.DataFrame(index = range(len(log_validation)), columns = range(6))
validation_forecast.columns = ['AMD', 'MA', 'MCD', 'NVDA', 'SBUX', 'V']
train_and_val = np.log(adj_close[adj_close.index < test_start]) # unisco training e validation

metriche = pd.DataFrame(index = range(6), columns = ['MSE', 'MAE', 'MAPE', 'RMSE'])
metriche.index=['AMD', 'MA', 'MCD', 'NVDA', 'SBUX', 'V']
for col in log_series.columns:
    order = results_data2[col]['order']
    seasonal_order = results_data2[col]['seasonal_order']
    
    model = SARIMAX(log_series[col], order = order,
                    seasonal_order = seasonal_order,
                    enforce_stationarity = False,
                    enforce_invertibility = False,
                    disp = False)
    model_results = model.fit(maxiter = 200, disp = False)
    # stampiamo i summary per tutti i modelli
    print(f"Risultati del modello per {col}:")
    print(model_results.summary())
    
    column = train_and_val.columns.get_loc(col)
    for i in range(len(log_validation)):
        output = model_results.forecast(steps=1)
        yhat   = output.values[0]
        validation_forecast.iloc[i, column] = yhat
        
        model            = SARIMAX(train_and_val.iloc[0:i+len(log_series),column], order = order,
                                           seasonal_order = seasonal_order,
                                           enforce_stationarity = False,
                                           enforce_invertibility = False,
                                           disp = False)
        model_results = model.fit(maxiter = 200)
     
    # calcoliamo le metriche
    metriche.loc[col, 'MSE'] = mean_squared_error(log_validation.loc[:, col], validation_forecast.loc[:, col])
    metriche.loc[col, 'MAE'] = mean_absolute_error(log_validation.loc[:, col], validation_forecast.loc[:, col])
    metriche.loc[col, 'MAPE'] = mean_absolute_percentage_error(log_validation.loc[:, col], validation_forecast.loc[:, col])
    metriche.loc[col, 'RMSE'] = sqrt(mean_squared_error(log_validation.loc[:, col], validation_forecast.loc[:, col]))
    print("\n--------------------------------\n")
metriche 


# In[216]:


# stampo il modello finale per tutti i tioli
for column in train_and_val.columns:
    serie_storica = log_series[column]
    modello = auto_arima(serie_storica, seasonal = True, m = 12, max_p = 2, max_q = 2, max_d = 2, suppress_warnings = True)
    results[column] = {'order': modello.order, 'seasonal_order': modello.seasonal_order}

results_data_final = pd.DataFrame(results)
results_data_final


# In[217]:


# ristimo il modello su training e validation con i parametri finali trovati
for col in train_and_val.columns:
    order = results_data_final[col]['order']
    seasonal_order = results_data_final[col]['seasonal_order']
    
    model = SARIMAX(train_and_val[col], order = order,
                    seasonal_order = seasonal_order,
                    enforce_stationarity = False,
                    enforce_invertibility = False,
                    disp = False)
    model_results = model.fit(maxiter = 200, disp = False)
    print(f"Risultati del modello per {col}:")
    print(model_results.summary())
    # stampo i grafici di diagnostica per tutti i modelli 
    print(f"Grafico diagnostico del modello per {col}:")
    print(model_results.plot_diagnostics())


# In[218]:


# facciamo previsioni per il test
test_forecast = pd.DataFrame(index = range(len(log_test)), columns = range(6))
test_forecast.columns = ['AMD', 'MA', 'MCD', 'NVDA', 'SBUX', 'V']
log_data = np.log(adj_close)

lower_ic = test_forecast.copy()
upper_ic = test_forecast.copy()

metriche = pd.DataFrame(index = range(6), columns = ['MSE', 'MAE', 'MAPE', 'RMSE'])
metriche.index=['AMD', 'MA', 'MCD', 'NVDA', 'SBUX', 'V']
for col in log_data.columns:
    order = results_data_final[col]['order']
    seasonal_order = results_data_final[col]['seasonal_order']
    
    model = SARIMAX(log_data[col], order = order,
                    seasonal_order = seasonal_order,
                    enforce_stationarity = False,
                    enforce_invertibility = False,
                    disp = False, return_conf_int = True)
    model_results = model.fit(maxiter = 200, disp = False)
    
    column = log_data.columns.get_loc(col)
    for i in range(len(log_test)):
        output = model_results.get_forecast(steps=1)
        yhat = output.predicted_mean.iloc[0] # salvo il valore previsto
        test_forecast.iloc[i, column] = yhat
        
        prediction_ci = output.conf_int()    # salvo intervallo di confidenza con lower e upper bound
        lower_bound = prediction_ci.iloc[0, 0]
        upper_bound = prediction_ci.iloc[0, 1]
        
        lower_ic.iloc[i, column] = lower_bound
        upper_ic.iloc[i, column] = upper_bound
        
        
        model            = SARIMAX(log_data.iloc[0:i+len(train_and_val),column], order = order,
                                           seasonal_order = seasonal_order,
                                           enforce_stationarity = False,
                                           enforce_invertibility = False,
                                           disp = False, return_conf_int = True)
        model_results = model.fit(maxiter = 200)
     
    # calcolo metriche finali
    metriche.loc[col, 'MSE'] = mean_squared_error(log_test.loc[:, col], test_forecast.loc[:, col])
    metriche.loc[col, 'MAE'] = mean_absolute_error(log_test.loc[:, col], test_forecast.loc[:, col])
    metriche.loc[col, 'MAPE'] = mean_absolute_percentage_error(log_test.loc[:, col], test_forecast.loc[:, col])
    metriche.loc[col, 'RMSE'] = sqrt(mean_squared_error(log_test.loc[:, col], test_forecast.loc[:, col]))
metriche 


# In[236]:


lower_ic.index = log_test.index
upper_ic.index = log_test.index
log_test[col]


# In[324]:


# grafico prendendo in considerazione solo validation e test con intervallo 
# di confidenza per le previsioni stimato prima con l'ARIMA
val_and_test = np.log(adj_close[adj_close.index >= val_start])

fn_cols = 2
n_rows = (len(val_and_test.columns) + n_cols - 1) // n_cols
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16, n_rows * 4.3))
axes = axes.flatten()

for i, col in enumerate(val_and_test.columns):
    ax = axes[i]
    ax.plot(val_and_test.index, val_and_test[col], label = 'Actual', color = 'blue')

    ax.fill_between(lower_ic.index, lower_ic[col], upper_ic[col], color = 'red', alpha = 0.2, label = 'Intervallo di Confidenza')
   
    ax.plot(lower_ic.index, lower_ic[col], linestyle='--', color='red', label = 'Lower Bound ic')
    ax.plot(lower_ic.index, upper_ic[col], linestyle='--', color='red', label = 'Upper Bound ic')
    
    ax.set_title(col)
    ax.set_xlabel('Tempo')
    ax.set_ylabel('Prezzo')
    ax.grid()
    ax.legend()

plt.tight_layout()
plt.show()


# # 4. STRATEGIE DI TRADING E BACKTESTING

# In[183]:


VIX = yf.download('^VIX', start, end)
VIX['VIX_lag1'] = VIX['Adj Close'].shift(1)
VIX


# In[191]:


for stock in tickers:
    stock_df = pd.DataFrame()
    
    stock_df[f'{stock}_lag1'] = adj_close[stock].shift(1)
    stock_df[f'{stock}_lag2'] = adj_close[stock].shift(2)
    stock_df[f'{stock}_lag3'] = adj_close[stock].shift(3)
    stock_df['VIX_lag1'] = VIX['VIX_lag1']

    stock_df.dropna(inplace = True)

    globals()[f'{stock}_lags'] = stock_df


# **MEDIE MOBILI**

# In[369]:


# salvo il volume in un dataset a parte
volume = data[['Volume']].reset_index()
daily_volume = volume.pivot(index = 'Date', columns = 'Ticker', values = 'Volume')
daily_volume[:3]


# In[390]:


datasets = {}

n_cols = 2 
n_rows = (len(daily_adj_close.columns) + n_cols - 1) // n_cols 

fig = plt.figure(figsize=(16, n_rows * 12))
gs = GridSpec(n_rows * 3, n_cols, height_ratios=[1.5, 0.8, 0.8] * n_rows)  

for i, col in enumerate(daily_adj_close.columns):
    sma20_col = daily_adj_close[col].rolling(window=20).mean()           # media mobile a 20 giorni
    sma120_col = daily_adj_close[col].rolling(window=120).mean()         # media mobile a 120 giorni
    ewm12_col = daily_adj_close[col].ewm(span=12, adjust=False).mean()   # media mobile esponenziale con span 12
    ewm26_col = daily_adj_close[col].ewm(span=26, adjust=False).mean()   # media mobile esponenziale con span 26
    
    temp_df = daily_adj_close[[col]].copy()
    temp_df[f'SMA20_{col}'] = sma20_col
    temp_df[f'SMA120_{col}'] = sma120_col
    temp_df[f'EWM12_{col}'] = ewm12_col
    temp_df[f'EWM26_{col}'] = ewm26_col
    temp_df['Volume'] = daily_volume[col]
    
    temp_df[f'Invested SMA_{col}'] = temp_df[f'SMA20_{col}'] > temp_df[f'SMA120_{col}']
    temp_df[f'Invested SMA_{col}'] = temp_df[f'Invested SMA_{col}'].astype(int)
    temp_df[f'RitorniLordi_{col}'] = daily_adj_close[col] / daily_adj_close[col].shift(1)
    temp_df['Buy_and_hold'] = np.cumprod(temp_df[f'RitorniLordi_{col}'])
    
    # Filtro temp_df per includere solo le righe dove Invested SMA è 1
    temp_df_invested = temp_df[temp_df[f'Invested SMA_{col}'] == 1].copy()
    temp_df_invested[f'Ritorni SMA_{col}'] = np.cumprod(temp_df_invested[f'RitorniLordi_{col}'].fillna(1))
    
    temp_df[f'Ritorni SMA_{col}'] = temp_df_invested[f'Ritorni SMA_{col}']
    
    # EWM Strategy
    temp_df['Change'] = temp_df[f'RitorniLordi_{col}'].fillna(1)
    temp_df['EWM12'] = temp_df[f'EWM12_{col}']
    temp_df['EWM26'] = temp_df[f'EWM26_{col}']
    temp_df['Invested_EWM'] = (temp_df['EWM12'] > temp_df['EWM26']).astype(int)
    ewm = temp_df[temp_df['Invested_EWM'] == 1].copy()
    ewm['Return'] = np.cumprod(ewm['Change'])
    
    datasets[col] = temp_df
    
    # Plotting
    row = i // n_cols  
    col_pos = i % n_cols 
    
    ax1 = fig.add_subplot(gs[row * 3, col_pos])
    ax2 = fig.add_subplot(gs[row * 3 + 1, col_pos])
    ax3 = fig.add_subplot(gs[row * 3 + 2, col_pos])
    
    ax1.plot(temp_df.index, temp_df[col], label='Price', color='red', alpha=0.9)
    ax1.plot(temp_df.index, temp_df[f'SMA20_{col}'], label='SMA20', color='yellow')
    ax1.plot(temp_df.index, temp_df[f'SMA120_{col}'], label='SMA120', color='lawngreen')
    ax1.plot(temp_df.index, temp_df[f'EWM12_{col}'], label='EWM12', color='violet')
    ax1.plot(temp_df.index, temp_df[f'EWM26_{col}'], label='EWM26', color='blue')
    ax1.set_title(col)
    ax1.grid(True)
    ax1.legend()
    
    ax2.bar(temp_df.index, temp_df['Volume'], label='Volume', color='orange')
    ax2.grid(True)
    ax2.legend()
    
    # plotto i ritoni della buy and hold della SMA e dell'EWM
    ax3.plot(temp_df.index, temp_df['Buy_and_hold'], label='Buy and hold', color='red')
    ax3.plot(temp_df.index, temp_df[f'Ritorni SMA_{col}'], label='SMA', color='lawngreen')
    ax3.plot(ewm.index, ewm['Return'], label='EWM', color='blue')
    ax3.set_xlabel('Date (Year - month)')
    ax3.grid(True)
    ax3.legend()
    
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
plt.tight_layout()
plt.show()


# In[391]:


# dataset calcolati prima
for col, df in datasets.items():
    print(f"\nDataset for {col}:\n")
    print(df.head())  # Stampa le prime righe del dataset per ogni azione
    print("\n")


# # 5. CAPM

# **CALCOLO DEI BETA**

# In[393]:


# aggiungo al mio solito dataset il mercato(s&p500) e il risk free
tickers2 = ['V', 'MA', 'NVDA', 'AMD', 'MCD', 'SBUX', '^GSPC', '^IRX']
data_capm = get(tickers2, start, end)


# In[637]:


adj_close_capm = data_capm[['Adj Close']].reset_index()
daily_adjclose_capm = adj_close_capm.pivot(index = 'Date', columns = 'Ticker', values = 'Adj Close').pct_change().dropna()
daily_adjclose_capm = daily_adjclose_capm.rename(columns = {'^GSPC': 'MARKET', '^IRX': 'RF'})
daily_adjclose_capm[:3]


# In[638]:


daily_adjclose_capm.cov()


# In[410]:


daily_adjclose_capm.corr()


# In[429]:


# calcolo rf e rm per capm
Rf_annual = 0.00417 # Moltiplicato per 12 per ottenere il tasso annuo
Rm_annual = daily_adjclose_capm['MARKET'].mean() * 12
print(Rf_annual)
print(Rm_annual)


# In[703]:


market_var = daily_adjclose_capm.MARKET.var()

#funzione per il calcolo dei beta
def calculate_beta(stock_returns, market_returns):
    cov_matrix = np.cov(stock_returns, market_returns)
    cov_with_market = cov_matrix[0, 1]
    beta = cov_with_market / market_var
    return beta

betas = {}
for col in daily_adjclose_capm.columns:
    if col not in ['MARKET', 'RF']: # Escludo il mercato e il riskfree
        betas[col] = calculate_beta(daily_adjclose_capm[col], daily_adjclose_capm['MARKET'])
        print(f"Beta for {col}: {betas[col]:.4}")


# **RENDIMENTI ATTESI**

# In[434]:


expected_returns = {}
for stock, beta in betas.items():
    expected_return = Rf_annual + beta * (Rm_annual - Rf_annual)
    expected_returns[stock] = expected_return

for stock, beta in betas.items():
    print(f"Expected annual return for {stock}: {expected_returns[stock]:.4%}")


# **STIMA CAPM**

# In[436]:


# Calcolare il rendimento privo di rischio giornaliero
daily_adjclose_capm['RF'] = (1 + daily_adjclose_capm['RF']/100)**(1/252) - 1

# Calcolare i rendimenti in eccesso (rendimenti dell'asset meno il rendimento privo di rischio)
excess_returns = daily_adjclose_capm.subtract(daily_adjclose_capm['RF'], axis=0)

# Definire il rendimento del mercato in eccesso
excess_returns['MARKET'] = daily_adjclose_capm['MARKET'] - daily_adjclose_capm['RF']

# Definire il DataFrame per il modello CAPM
capm_data = excess_returns.copy()

# Funzione per stimare il modello CAPM per un singolo asset
def estimate_capm(asset):
    y = capm_data[asset]
    X = sm.add_constant(capm_data['MARKET'])
    model = sm.OLS(y, X).fit()
    return model

# Stima del modello CAPM per ciascun asset
capm_models = {}
for asset in daily_adjclose_capm.columns:
    if asset not in ['MARKET', 'RF']:
        capm_models[asset] = estimate_capm(asset)

# Stampare i riassunti del modello
for asset, model in capm_models.items():
    print(f"CAPM Model Summary for {asset}:")
    print(model.summary())

# Calcolare i beta e i rendimenti attesi utilizzando il modello CAPM
betas = {asset: model.params['MARKET'] for asset, model in capm_models.items()}

Rf_annual = 0.00417  # Tasso privo di rischio annuale
Rm_annual = daily_adjclose_capm['MARKET'].mean() * 252  # Rendimenti medi annuali del mercato

expected_returns = {asset: Rf_annual + beta * (Rm_annual - Rf_annual) for asset, beta in betas.items()}

# Stampare i beta e i rendimenti attesi annuali per ciascun titolo
for asset, beta in betas.items():
    print(f"Beta for {asset}: {beta:.3f}")
    print(f"Expected annual return for {asset}: {expected_returns[asset]:.4%}")


# **FAMA-FRENCH**

# In[504]:


# importo tre fattori fama-french
factors = pd.read_csv("C:\\Users\\giudi\\OneDrive\\Desktop\\uni\\business intelligence\\F-F_Research_Data_Factors.CSV", skiprows = 3)
factors.head()
factors.tail()


# In[505]:


stringa = ' Annual Factors: January-December '
indices = factors.iloc[:,0] == stringa
start_of_annual = factors[indices].index[0]
factors = factors[factors.index < start_of_annual]


# In[506]:


factors.columns = ['data', 'mkt', 'smb', 'hml', 'rf']
factors.head()


# In[507]:


factors['data'] = pd.to_datetime(factors['data'], format='%Y%m').dt.strftime("%Y-%m")


# In[508]:


start_date = '2014-05-31'
end_date = '2024-05-31'
factors = factors.set_index('data')
factors = factors[start_date:end_date]
factors.head()


# In[509]:


# Ci assicuriamo che i fattori Fama-French siano in un formato numerico e 
# correttamente scalati per essere utilizzati in ulteriori analisi 
factors = factors.apply(pd.to_numeric, errors='coerce').div(100)


# In[635]:


daily_adjclose_capm2 = daily_adjclose_capm.resample('M').last()
daily_adjclose_capm


# In[636]:


if not isinstance(daily_adjclose_capm.index, pd.DatetimeIndex):
    daily_adjclose_capm2.index = pd.to_datetime(daily_adjclose_capm2.index)

if not isinstance(factors.index, pd.DatetimeIndex):
    factors.index = pd.to_datetime(factors.index)
    
daily_adjclose_capm2.index = daily_adjclose_capm2.index.to_period('M').to_timestamp()
factors.index = factors.index.to_period('M').to_timestamp()

# Unisco daily_adjclose_capm con factor_df
ff_data = daily_adjclose_capm2.join(factors, how='inner')

# Calcolo i rendimenti in eccesso per ciascuna azione
for col in daily_adjclose_capm2.columns:
    if col not in ['MARKET', 'RF']:  
        ff_data[f'excess_rtn_{col}'] = ff_data[col] - ff_data['RF']

ff_data


# In[516]:


azioni = ['AMD', 'MA', 'MCD', 'NVDA', 'SBUX', 'V']
results = {}

for action in azioni:
    # per ogni azione considera eccesso di ritorno dell'azione, il mercato e i fattori Fama-French
    data = ff_data[['excess_rtn_' + action, 'mkt', 'smb', 'hml', 'rf']].dropna()

    # modello di regressione lineare
    X = data[['mkt', 'smb', 'hml']]
    X = sm.add_constant(X)  # intercetta
    y = data['excess_rtn_' + action]

    model = sm.OLS(y, X)
    result = model.fit()

    results[action] = result

    # Stampo summary del modello per ogni azione 
    print(f"Regression results for {action}:")
    print(result.summary())
    print("\n")


# # 6. COSTRUZIONE DI PORTAFOGLIO

# In[749]:


# prendo in considerazione solo i primi 108 mesi
start_108 = ritorni_semplici.index.min()
end_108 = start + pd.DateOffset(months=108)

ritorni_semplici_108 = ritorni_semplici[ritorni_semplici.index < end_108]
ritorni_semplici_108.tail()


# In[750]:


n_port = 10 ** 5 # 100000 portafogli
n_days = 252     # andiamo a vedere quali sono i portafogli a un anno
n_assets = len(tickers)
np.random.seed(42)

# stima pesi casuali per 100000 portafogli e 6 asset normalizzati per farli sommare a 1 in ciascun portafoglio
weights = np.random.random(size=(n_port, n_assets))
weights /= np.sum(weights, axis=1)[:, np.newaxis]
weights 


# In[751]:


avg_returns = ritorni_semplici_108.mean() * n_days
cov_mat = ritorni_semplici_108.cov() * n_days
portf_rtns = np.dot(weights, avg_returns)

portf_vol = []
for i in range(0, len(weights)):
    portf_vol.append(np.sqrt(np.dot(weights[i].T,
                                    np.dot(cov_mat, weights[i]))))
portf_vol = np.array(portf_vol)
portf_sharpe_ratio = portf_rtns / portf_vol


# In[752]:


portf_results_df = pd.DataFrame({'returns': portf_rtns,
                                 'volatility': portf_vol,
                                 'sharpe_ratio': portf_sharpe_ratio})


# In[753]:


n_points = 100
portf_vol_ef = []
indices_to_skip = []

portf_rtns_ef = np.linspace(portf_results_df.returns.min(),
                            portf_results_df.returns.max(),
                           n_points)
portf_rtns_ef = np.round(portf_rtns_ef, 2)
portf_rtns = np.round(portf_rtns, 2)

for point_index in range(n_points):
    if portf_rtns_ef[point_index] not in portf_rtns:
        indices_to_skip.append(point_index)
        continue
    matched_ind = np.where(portf_rtns == portf_rtns_ef[point_index])
    portf_vol_ef.append(np.min(portf_vol[matched_ind]))

portf_rtns_ef = np.delete(portf_rtns_ef, indices_to_skip)


# In[754]:


# rappresento graficamente i titoli e la frontiera efficiente

MARKS = ['o', 'X', 'v', '*', 'p', 'h']

fig, ax = plt.subplots()
portf_results_df.plot(kind='scatter', x='volatility',
                      y='returns', c='sharpe_ratio',
                      cmap='viridis', edgecolors='black',
                      ax=ax)
ax.set(xlabel='Volatility',
       ylabel='Expected Returns',
       title='Efficient Frontier')
ax.plot(portf_vol_ef, portf_rtns_ef, 'b--', color = 'red', linewidth = 2.5)
for asset_index in range(n_assets):
    ax.scatter(x=np.sqrt(cov_mat.iloc[asset_index, asset_index]),
                y=avg_returns[asset_index],
                marker=MARKS[asset_index],
                s=150,
                color='black',
                label= tickers[asset_index])
ax.legend()

plt.tight_layout()
plt.show()


# In[755]:


max_sharpe_ind = np.argmax(portf_results_df.sharpe_ratio)
max_sharpe_portf = portf_results_df.loc[max_sharpe_ind]

min_vol_ind = np.argmin(portf_results_df.volatility)
min_vol_portf = portf_results_df.loc[min_vol_ind]


# In[756]:


print('Maximum Sharpe Ratio portfolio')
print('Performance')
for index, value in max_sharpe_portf.items():
    print(f'{index}: {100 * value:.2f}% ', end="", flush=True)
print('\nWeights')
for x, y in zip(tickers, weights[np.argmax(portf_results_df.sharpe_ratio)]):
    print(f'{x}: {100*y:.2f}% ', end="", flush=True)


# In[757]:


print('Minimum Volatility portfolio ----')
print('Performance')
for index, value in min_vol_portf.items():
    print(f'{index}: {100 * value:.2f}% ', end="", flush=True)
print('\nWeights')
for x, y in zip(tickers, weights[np.argmin(portf_results_df.volatility)]):
    print(f'{x}: {100*y:.2f}% ', end="", flush=True)


# In[736]:


fig, ax = plt.subplots()

# Frontiera efficiente
ax.plot(portf_vol_ef, portf_rtns_ef, 'b--', color='red', linewidth=2.5)

portf_results_df.plot(kind='scatter', x='volatility',
                      y='returns', c='sharpe_ratio',
                      cmap='viridis', edgecolors='black',
                      ax=ax)

# Max Sharpe Ratio
ax.scatter(x=max_sharpe_portf.volatility,
           y=max_sharpe_portf.returns,
           c='black', marker='*', edgecolors='black',
           s=200, label='Max Sharpe Ratio', zorder=5)

# Minimum Volatility
ax.scatter(x=min_vol_portf.volatility,
           y=min_vol_portf.returns,
           c='black', marker='P', edgecolors='black',
           s=200, label='Minimum Volatility', zorder=5)

ax.set(xlabel='Volatility', ylabel='Expected Returns',
       title='Efficient Frontier')
ax.legend()

plt.tight_layout()
plt.show()


# In[737]:


def get_portf_rtn(w, avg_rtns):
    return np.sum(avg_rtns * w)

def get_portf_vol(w, avg_rtns, cov_mat):
    return np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))


# In[738]:


rtns_range = np.linspace(-0.22, 0.32, 200)
def get_efficient_frontier(avg_rtns, cov_mat, rtns_range):

    efficient_portfolios = []

    n_assets = len(avg_returns)
    args = (avg_returns, cov_mat)
    bounds = tuple((0,1) for asset in range(n_assets))
    initial_guess = n_assets * [1. / n_assets, ]

    for ret in rtns_range:
        constraints = ({'type': 'eq',
                        'fun': lambda x: get_portf_rtn(x, avg_rtns) - ret},
                       {'type': 'eq',
                        'fun': lambda x: np.sum(x) - 1})
        efficient_portfolio = sco.minimize(get_portf_vol, initial_guess,
                                           args=args, method='SLSQP',
                                           constraints=constraints,
                                           bounds=bounds)
        efficient_portfolios.append(efficient_portfolio)

    return efficient_portfolios

efficient_portfolios = get_efficient_frontier(avg_returns, cov_mat, rtns_range)
vols_range = [x['fun'] for x in efficient_portfolios]


# In[739]:


min_vol_ind1 = np.argmin(vols_range)
min_vol_portf_rtn1 = rtns_range[min_vol_ind1]
min_vol_portf_vol1 = efficient_portfolios[min_vol_ind1]['fun']

min_vol_portf1 = {'Return': min_vol_portf_rtn1,
                 'Volatility': min_vol_portf_vol1,
                 'Sharpe Ratio': (min_vol_portf_rtn1 /
                                  min_vol_portf_vol1)}

min_vol_portf1


# In[740]:


print('Minimum Volatility portfolio ----')
print('Performance')

for index, value in min_vol_portf1.items():
    print(f'{index}: {100 * value:.2f}% ', end="", flush=True)

print('\nWeights')
for x, y in zip(tickers, efficient_portfolios[min_vol_ind1]['x']):
    print(f'{x}: {100*y:.2f}% ', end="", flush=True)


# In[579]:


def neg_sharpe_ratio(w, avg_rtns, cov_mat, rf_rate):
    portf_returns = np.sum(avg_rtns * w)
    portf_volatility = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
    portf_sharpe_ratio = (portf_returns - rf_rate) / portf_volatility
    return -portf_sharpe_ratio


# In[741]:


n_assets = len(avg_returns)
RF_RATE = 0

args = (avg_returns, cov_mat, RF_RATE)
constraints = ({'type': 'eq',
                'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0,1) for asset in range(n_assets))
initial_guess = n_assets * [1. / n_assets]

max_sharpe_portf = sco.minimize(neg_sharpe_ratio,
                                x0=initial_guess,
                                args=args,
                                method='SLSQP',
                                bounds=bounds,
                                constraints=constraints)


# In[742]:


max_sharpe_portf_w = max_sharpe_portf['x']
max_sharpe_portf = {'Return': get_portf_rtn(max_sharpe_portf_w,
                                            avg_returns),
                    'Volatility': get_portf_vol(max_sharpe_portf_w,
                                                avg_returns,
                                                cov_mat),
                    'Sharpe Ratio': -max_sharpe_portf['fun']}
max_sharpe_portf


# In[582]:


print('Maximum Sharpe Ratio portfolio ----')
print('Performance')

for index, value in max_sharpe_portf.items():
    print(f'{index}: {100 * value:.2f}% ', end="", flush=True)

print('\nWeights')
for x, y in zip(tickers, max_sharpe_portf_w):
    print(f'{x}: {100*y:.2f}% ', end="", flush=True)


# **USANDO RENDIMENTI ATTESI**

# In[604]:


d = {'AMD': 0.184800, 'MA': 0.136985, 'MCD': 0.082483, 'NVDA': 0.193545, 'SBUX': 0.115597,
     'V': 0.123911}
means_exp = pd.Series(data=d, index=['AMD', 'MA', 'MCD', 'NVDA', 'SBUX', 'V'])
means_exp


# In[607]:


portf_rtns2 = np.dot(weights, means_exp)

portf_vol2 = []
for i in range(0, len(weights)):
    portf_vol2.append(np.sqrt(np.dot(weights[i].T,
                                    np.dot(cov_mat, weights[i]))))
portf_vol2 = np.array(portf_vol)
portf_sharpe_ratio2 = portf_rtns2 / portf_vol2


# In[608]:


portf_results_df2 = pd.DataFrame({'returns': portf_rtns2,
                                 'volatility': portf_vol2,
                                 'sharpe_ratio': portf_sharpe_ratio2})


# In[609]:


n_points2 = 100
portf_vol_ef2 = []
indices_to_skip2 = []

portf_rtns_ef2 = np.linspace(portf_results_df2.returns.min(),
                            portf_results_df2.returns.max(),
                           n_points2)
portf_rtns_ef2 = np.round(portf_rtns_ef2, 2)
portf_rtns2 = np.round(portf_rtns2, 2)

for point_index in range(n_points2):
    if portf_rtns_ef2[point_index] not in portf_rtns2:
        indices_to_skip2.append(point_index)
        continue
    matched_ind = np.where(portf_rtns2 == portf_rtns_ef2[point_index])
    portf_vol_ef2.append(np.min(portf_vol2[matched_ind]))

portf_rtns_ef2 = np.delete(portf_rtns_ef2, indices_to_skip2)


# In[711]:


MARKS = ['o', 'X', 'v', '*', 'p', 'h']

fig, ax = plt.subplots()
portf_results_df2.plot(kind='scatter', x='volatility',
                      y='returns', c='sharpe_ratio',
                      cmap='viridis', edgecolors='black',
                      ax=ax)
ax.set(xlabel='Volatility',
       ylabel='Expected Returns',
       title='Efficient Frontier')
ax.plot(portf_vol_ef2, portf_rtns_ef2, 'b--', color = 'red', linewidth = 2.5)
for asset_index in range(n_assets):
    ax.scatter(x=np.sqrt(cov_mat.iloc[asset_index, asset_index]),
                y=means_exp[asset_index],
                marker=MARKS[asset_index],
                s=150,
                color='black',
                label= tickers[asset_index])
ax.legend()

plt.tight_layout()
plt.show()


# In[762]:


max_sharpe_ind2 = np.argmax(portf_results_df2.sharpe_ratio)
max_sharpe_portf2 = portf_results_df2.loc[max_sharpe_ind2]

min_vol_ind2 = np.argmin(portf_results_df2.volatility)
min_vol_portf2 = portf_results_df2.loc[min_vol_ind2]


# In[763]:


print('Maximum Sharpe Ratio portfolio')
print('Performance')
for index, value in max_sharpe_portf2.items():
    print(f'{index}: {100 * value:.2f}% ', end="", flush=True)
print('\nWeights')
for x, y in zip(tickers, weights[np.argmax(portf_results_df2.sharpe_ratio)]):
    print(f'{x}: {100*y:.2f}% ', end="", flush=True)


# In[764]:


print('Minimum Volatility portfolio ----')
print('Performance')
for index, value in min_vol_portf2.items():
    print(f'{index}: {100 * value:.2f}% ', end="", flush=True)
print('\nWeights')
for x, y in zip(tickers, weights[np.argmin(portf_results_df2.volatility)]):
    print(f'{x}: {100*y:.2f}% ', end="", flush=True)


# In[766]:


fig, ax = plt.subplots()

# Frontiera efficiente
ax.plot(portf_vol_ef2, portf_rtns_ef2, 'b--', color='red', linewidth=2.5)

portf_results_df2.plot(kind='scatter', x='volatility',
                      y='returns', c='sharpe_ratio',
                      cmap='viridis', edgecolors='black',
                      ax=ax)

# Max Sharpe Ratio
ax.scatter(x=max_sharpe_portf2["volatility"],
           y=max_sharpe_portf2["returns"],
           c='black', marker='*', edgecolors='black',
           s=200, label='Max Sharpe Ratio', zorder=5)

# Minimum Volatility
ax.scatter(x=min_vol_portf2["volatility"],
           y=min_vol_portf2["returns"],
           c='black', marker='P', edgecolors='black',
           s=200, label='Minimum Volatility', zorder=5)

ax.set(xlabel='Volatility', ylabel='Expected Returns',
       title='Efficient Frontier')
ax.legend()

plt.tight_layout()
plt.show()


# In[767]:


efficient_portfolios2 = get_efficient_frontier(means_exp, cov_mat, rtns_range)
vols_range2 = [x['fun'] for x in efficient_portfolios2]


# In[768]:


min_vol_ind3 = np.argmin(vols_range2)
min_vol_portf_rtn3 = rtns_range[min_vol_ind3]
min_vol_portf_vol3 = efficient_portfolios2[min_vol_ind3]['fun']

min_vol_portf3 = {'Return': min_vol_portf_rtn3,
                 'Volatility': min_vol_portf_vol3,
                 'Sharpe Ratio': (min_vol_portf_rtn3 /
                                  min_vol_portf_vol3)}

min_vol_portf3


# In[769]:


print('Minimum Volatility portfolio ----')
print('Performance')

for index, value in min_vol_portf3.items():
    print(f'{index}: {100 * value:.2f}% ', end="", flush=True)

print('\nWeights')
for x, y in zip(tickers, efficient_portfolios2[min_vol_ind3]['x']):
    print(f'{x}: {100*y:.2f}% ', end="", flush=True)


# In[621]:


RF_RATE = 0

args2 = (means_exp, cov_mat, RF_RATE)
constraints2 = ({'type': 'eq',
                'fun': lambda x: np.sum(x) - 1})
bounds2 = tuple((0,1) for asset in range(n_assets))
initial_guess2 = n_assets * [1. / n_assets]

max_sharpe_portf2 = sco.minimize(neg_sharpe_ratio,
                                x0=initial_guess2,
                                args=args2,
                                method='SLSQP',
                                bounds=bounds2,
                                constraints=constraints2)


# In[623]:


max_sharpe_portf_w2 = max_sharpe_portf2['x']
max_sharpe_portf2 = {'Return': get_portf_rtn(max_sharpe_portf_w2,
                                            means_exp),
                    'Volatility': get_portf_vol(max_sharpe_portf_w2,
                                                means_exp,
                                                cov_mat),
                    'Sharpe Ratio': -max_sharpe_portf2['fun']}
max_sharpe_portf2


# In[624]:


print('Maximum Sharpe Ratio portfolio ----')
print('Performance')

for index, value in max_sharpe_portf2.items():
    print(f'{index}: {100 * value:.2f}% ', end="", flush=True)

print('\nWeights')
for x, y in zip(tickers, max_sharpe_portf_w2):
    print(f'{x}: {100*y:.2f}% ', end="", flush=True)


# **CALCOLO DEI BETA**

# In[640]:


daily_adjclose_capm_108 = daily_adjclose_capm[daily_adjclose_capm.index < end_108]
daily_adjclose_capm_108.tail()


# Beta del portafoglio a varianza minima calcolato con metodo simulativo

# In[772]:


market_returns = daily_adjclose_capm_108['MARKET']
min_vol_weights = weights[min_vol_ind]

# Calcolo dei rendimenti storici del portafoglio a minima volatilità calcolato per simulazione 
min_vol_portf_returns_hist = ritorni_semplici_108[tickers].dot(min_vol_weights)

# Calcolo della covarianza tra i rendimenti del portafoglio a minima volatilità e i rendimenti del mercato
cov_matrix_min_vol = np.cov(min_vol_portf_returns_hist, market_returns)
market_var = market_returns.var()

# Calcolo del beta del portafoglio a minima volatilità
min_vol_beta = cov_matrix_min_vol[0, 1] / market_var

# Stampa dei risultati del portafoglio a minima volatilità
for index, value in min_vol_portf.items():
    print(f'{index}: {100 * value:.2f}% ', end="", flush=True)
print(f'Beta: {min_vol_beta:.3f}')


# Beta del portafoglio a varianza minima calcolato con metodo analitico

# In[777]:


min_vol_weights1 = weights[min_vol_ind1]

# Calcolo dei rendimenti storici del portafoglio a minima volatilità calcolato per simulazione 
min_vol_portf_returns_hist1 = ritorni_semplici_108[tickers].dot(min_vol_weights1)

# Calcolo della covarianza tra i rendimenti del portafoglio a minima volatilità e i rendimenti del mercato
cov_matrix_min_vol1 = np.cov(min_vol_portf_returns_hist1, market_returns)

# Calcolo del beta del portafoglio a minima volatilità
min_vol_beta1 = cov_matrix_min_vol1[0, 1] / market_var

# Stampa dei risultati del portafoglio a minima volatilità
for index, value in min_vol_portf1.items():
    print(f'{index}: {100 * value:.2f}% ', end="", flush=True)
print(f'Beta: {min_vol_beta1:.3f}')


# Beta del portafoglio a varianza minima calcolato con metodo simulativo con rendimenti calcolati con CAPM

# In[774]:


min_vol_weights2 = weights[min_vol_ind2]

# Calcolo dei rendimenti storici del portafoglio a minima volatilità
min_vol_portf_returns_hist2 = ritorni_semplici_108[tickers].dot(min_vol_weights2)

# Calcolo della covarianza tra i rendimenti del portafoglio a minima volatilità e i rendimenti del mercato
cov_matrix_min_vol2 = np.cov(min_vol_portf_returns_hist2, market_returns)

# Calcolo del beta del portafoglio a minima volatilità
min_vol_beta2 = cov_matrix_min_vol2[0, 1] / market_var

# Stampa dei risultati del portafoglio a minima volatilità
for index, value in min_vol_portf2.items():
    print(f'{index}: {100 * value:.2f}% ', end="", flush=True)
print(f'Beta: {min_vol_beta2:.3f}')


# Beta del portafoglio a varianza minima calcolato con metodo analitico con rendimenti calcolati con CAPM

# In[778]:


min_vol_weights3 = weights[min_vol_ind3]

# Calcolo dei rendimenti storici del portafoglio a minima volatilità
min_vol_portf_returns_hist3 = ritorni_semplici_108[tickers].dot(min_vol_weights3)

# Calcolo della covarianza tra i rendimenti del portafoglio a minima volatilità e i rendimenti del mercato
cov_matrix_min_vol3 = np.cov(min_vol_portf_returns_hist3, market_returns)

# Calcolo del beta del portafoglio a minima volatilità
min_vol_beta3 = cov_matrix_min_vol3[0, 1] / market_var

# Stampa dei risultati del portafoglio a minima volatilità
for index, value in min_vol_portf3.items():
    print(f'{index}: {100 * value:.2f}% ', end="", flush=True)
print(f'Beta: {min_vol_beta3:.3f}')


# **CONFRONTO PORTAFOGLIO OTTIMALE E EFFETTIVO**

# In[780]:


min_vol_portf


# In[789]:


equal_weights = np.array([1/n_assets] * n_assets)
equal_weighted_portf_returns = np.dot(equal_weights, avg_returns)

print('MVP metodo simulativo')
print(f'Annualized Return: {min_vol_portf["returns"] * 100:.2f}%')
print(f'Volatility: {min_vol_portf["volatility"] * 100:.2f}%')
print(f'Sharpe Ratio: {min_vol_portf["sharpe_ratio"] * 100:.2f}')
print(f'Beta: {min_vol_beta:.2f}')

print('\nEqual Weighted Portfolio')
print(f'Annualized Return: {equal_weighted_portf_returns * 100:.2f}%')

# Calcolo della volatilità del portafoglio equiponderato
equal_weighted_portf_vol = np.sqrt(np.dot(equal_weights.T, np.dot(cov_mat, equal_weights)))
print(f'Volatility: {equal_weighted_portf_vol * 100:.2f}%')

# Calcolo del rapporto di Sharpe del portafoglio equiponderato
equal_weighted_sharpe_ratio = equal_weighted_portf_returns / equal_weighted_portf_vol
print(f'Sharpe Ratio: {equal_weighted_sharpe_ratio:.2f}')

# Calcolo del beta del portafoglio equiponderato
equal_weighted_portf_returns_hist = ritorni_semplici_108[tickers].dot(equal_weights)
cov_matrix_equal_weighted = np.cov(equal_weighted_portf_returns_hist, market_returns)
equal_weighted_beta = cov_matrix_equal_weighted[0, 1] / market_returns.var()
print(f'Beta: {equal_weighted_beta:.2f}')


# In[790]:


print('MVP metodo analitico')
print(f'Annualized Return: {min_vol_portf1["Return"] * 100:.2f}%')
print(f'Volatility: {min_vol_portf1["Volatility"] * 100:.2f}%')
print(f'Sharpe Ratio: {min_vol_portf1["Sharpe Ratio"] * 100:.2f}')
print(f'Beta: {min_vol_beta1:.2f}')

print('\nEqual Weighted Portfolio')
print(f'Annualized Return: {equal_weighted_portf_returns * 100:.2f}%')
print(f'Volatility: {equal_weighted_portf_vol * 100:.2f}%')
print(f'Sharpe Ratio: {equal_weighted_sharpe_ratio:.2f}')
print(f'Beta: {equal_weighted_beta:.2f}')


# In[791]:


print('MVP metodo simulativo con rendimenti calcolati con CAPM')
print(f'Annualized Return: {min_vol_portf2["returns"] * 100:.2f}%')
print(f'Volatility: {min_vol_portf2["volatility"] * 100:.2f}%')
print(f'Sharpe Ratio: {min_vol_portf2["sharpe_ratio"] * 100:.2f}')
print(f'Beta: {min_vol_beta2:.2f}')

print('\nEqual Weighted Portfolio')
print(f'Annualized Return: {equal_weighted_portf_returns * 100:.2f}%')
print(f'Volatility: {equal_weighted_portf_vol * 100:.2f}%')
print(f'Sharpe Ratio: {equal_weighted_sharpe_ratio:.2f}')
print(f'Beta: {equal_weighted_beta:.2f}')


# In[792]:


print('MVP metodo analitico con rendimenti calcolati con CAPM')
print(f'Annualized Return: {min_vol_portf3["Return"] * 100:.2f}%')
print(f'Volatility: {min_vol_portf3["Volatility"] * 100:.2f}%')
print(f'Sharpe Ratio: {min_vol_portf3["Sharpe Ratio"] * 100:.2f}')
print(f'Beta: {min_vol_beta3:.2f}')

print('\nEqual Weighted Portfolio')
print(f'Annualized Return: {equal_weighted_portf_returns * 100:.2f}%')
print(f'Volatility: {equal_weighted_portf_vol * 100:.2f}%')
print(f'Sharpe Ratio: {equal_weighted_sharpe_ratio:.2f}')
print(f'Beta: {equal_weighted_beta:.2f}')

