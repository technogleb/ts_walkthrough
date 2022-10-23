---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
# CHECK IF RUNNING INSIDE GOOGLE.COLAB

import sys
import os

IN_COLAB = 'google.colab' in sys.modules
CLONED = 0
if IN_COLAB and not CLONED:
    # clone the repo
    !git clone https://github.com/technogleb/ts_walkthrough
    CLONED = 1
    # append to sys.path
    sys.path.append('/content/ts_walkthrough')
    # change working directory to project root
    os.chdir('/content/ts_walkthrough')
    # install requirements
    !pip install -r requirements.txt
```

```python
%matplotlib inline
%load_ext autoreload
%autoreload 2
```

```python
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import boxcox

from dataset import Dataset
import plotting
```

```python
%load_ext autoreload
%autoreload 2
```

```python
dataset = Dataset('./data/dataset')
```

```python
ts = dataset['alcohol_sales.csv']
ts.index = ts.index.map(pd.to_datetime)
```

```python
ts, ts_test = ts[:250], ts[250:]
```

```python
ts.plot()
```

#### Задание 1. Box-Jenkins approach
1. Убрать гетероскедастичность при помощи логарифмирования.
2. Продифференцировать разными способами и узнать нужный порядок дифференцирования d, D, анализируя
   получившийся ряд при помощи графика acf_plot и теста дикки-фуллера.
3. Найти (если модель разделима) или оценить (если не разделима) параметры (q, Q), (p, P) по графикам acf, pacf продифференцированного ряда.
4. Обучить модель SARIMAX и попробовать сделать предсказания исходного ряда.
5. Критерий хорошей модели (в идеале) - стационарные остатки.

```python
# <your_code_here>

# 1.
ts_log = np.log(ts)

# 2.
ts_flat = ts_log.diff().diff(12).dropna()
plot_acf(ts_flat);
plot_pacf(ts_flat);
d = 1
D = 1
# как мы видим модель четко разделима

# 3.
# Так как модель разделима, то там, где график корреляции не показывает резкого падения, параметры равны 0
# т.е. q, Q = 0
q = 0
Q = 0

# По графику частичной автокорреляции находим, что p = 2, P = 0
p = 2
P = 0

# 4. 
pdq = (2, 1, 0)
PDQ = (0, 1, 0, 12)

model = SARIMAX(ts_log, order=pdq, seasonal_order=PDQ)
res = model.fit()
preds = res.forecast(69)
# восстановим в изначальном масштабе
plotting.plot_ts(np.exp(ts_log), ts_test, np.exp(preds))

```

#### Задание 3. (используя auto-arima)
1. Установить аналог auto-arima из R, библиотеку pmdarima (pip install pmdarima)
2. Обучить модель, используя функцию auto_arima, передав в нее нужные параметры.

```python
# <your_code_here>
# наиболее удобная реализация AutoArima находится в модуле pmdarima
from pmdarima import auto_arima

# на вход подается ряд со всеми преобразованиями кроме дифференцирования, границы поиска параметров p, q, P, Q, 
# периодом сезонности m, параметрами дифференцирования d, D
model = auto_arima(
     ts_log, start_p=0, start_q=0,
     max_p=3, max_q=3, m=12,
     start_P=0, start_Q=0, seasonal=True,
     d=1, D=1, trace=True,
     error_action='ignore',
     suppress_warnings=True,
     stepwise=True
)

# прогноз можем получить методом predict, указав горизонт прогнозирования
preds = model.predict(len(ts_test))

plotting.plot_ts(np.exp(ts_log), ts_test, pd.Series(np.exp(preds), index=ts_test.index))
```

```python

```
