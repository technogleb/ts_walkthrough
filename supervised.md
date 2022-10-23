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
%load_ext autoreload
%autoreload 2
```

```python
import sys
from pathlib import Path

sys.path.append(str(Path().cwd().parent))
```

```python
import numpy as np
import pandas as pd
```

```python
import dataset
from plotting import plot_ts
```

### Возьмем из датасета ряд с минимальными температурами.

```python
dataset = dataset.Dataset('./data/dataset/')
```

```python
ts = dataset['daily-min-temperatures.csv']
```

```python
ts.head()
```

```python
ts.tail()
```

```python
plot_ts(ts)
```

## Поэтапное построение самого базового пайплайна для прогнозирования


### Задание 1.


Напишите функцию transform_to_matrix, которая бы принимала одномерный ряд и возвращала матрицу лагов

1. Функция должна принимать временной ряд (pd.Series) и количество лагов (оно же количество признаков). 

2. Функция должна проходить по ряду скользящим окном с шириной num_lags и брать лаги текущего окна в качестве признаков, а следующую точку в качестве ответа на данном обьекте.
3. Каждое перемещение окна на одну точку вперед сооветствует новому обьекту матрицы.
4. Функция должная вернуть pd.Dataframe - т.е. матрицу объекты признаки.

```python
def transform_to_matrix(ts, num_lags):
    """
    Transforms time series into lags matrix to allow
    applying supervised learning algorithms

    Parameters
    ------------
    ts
        Time series to transform

    Returns
    --------
    lags_matrix
        Dataframe with transformed values
    """
    # <your code here>
    ts_values = ts.values
    data = {}
    for i in range(num_lags + 1):
        data[f'lag_{num_lags - i}'] = np.roll(ts_values, -i)

    lags_matrix = pd.DataFrame(data)[:-num_lags]
    lags_matrix.index = ts.index[num_lags:]

    return lags_matrix
```

### Задание 2.


Получите матрицу обьекты-признаки в качестве num_lags выбрав период сезонности ряда (найти его можно либо по изначальному графику, либо по графику автокорреляции)

```python
from statsmodels.graphics.tsaplots import plot_acf

# <your_code_here>
num_lags = 365
plot_acf(ts, lags=num_lags)
matrix = transform_to_matrix(ts, 365)
```

```python
matrix
```

### Задание 3.


Добавьте в модель следующие дополнительные признаки

* Для всех лагов кроме целевого на каждой строчке матрицы расчитайте среднее и стандартное отклонение (не включайте текущий лаг на данном объекте, чтобы избежать лика информации из будущего).
  
Для всех обьектов добавьте в качестве признаков следующие datetime признаки (см пример в основном обзоре)
1. sin/cos проекции дня недели текущего объекта на единичный круг
2. sin/cos проекции месяца текущего объекта на единичный круг
3. sin/cos проекции дня текущего объекта на единичный круг

Подсказка (получить соответствующие значения можно из аттрибутов обьекта datetime)

```python
# <your code here>

import math

from pandas._libs.tslibs.timestamps import Timestamp

def get_month_sin(timestamp: Timestamp) -> float:
    theta = timestamp.month * (2*math.pi / 12)
    return math.sin(theta)

def get_month_cos(timestamp: Timestamp) -> float:
    theta = timestamp.month * (2*math.pi / 12)
    return math.cos(theta)


def get_day_sin(timestamp: Timestamp) -> float:
    theta = timestamp.day * (2*math.pi / timestamp.days_in_month)
    return math.sin(theta)


def get_day_cos(timestamp: Timestamp) -> float:
    theta = timestamp.day * (2*math.pi / timestamp.days_in_month)
    return math.cos(theta)


def get_dayofweek_sin(timestamp: Timestamp) -> float:
    theta = timestamp.dayofweek * (2*math.pi / 7)
    return math.sin(theta)


def get_dayofweek_cos(timestamp: Timestamp) -> float:
    theta = timestamp.dayofweek * (2*math.pi / 7)
    return math.cos(theta)


datetime_mappers = {
    'month_sin': get_month_sin,
    'month_cos': get_month_cos,
    'day_sin': get_day_sin,
    'day_cos': get_day_cos,
    'dayofweek_sin': get_dayofweek_sin,
    'dayofweek_cos': get_dayofweek_cos,
}
```

```python
for f_name, f in datetime_mappers.items():
    matrix[f_name] = matrix.index.map(f)
```

```python
matrix
```

### Задание 4.


Разделите на train/test в соотношении примерно 0.7 на 0.3, а также на X и y

Обратите внимание что тест всегда должен быть после трейна, чтобы избежать ликов из будущего, поэтому никакого stratified split

```python
# <your code here>
split_idx = int(len(matrix)*0.7)
train, test = matrix[:split_idx], matrix[split_idx:]

X_train, y_train = train.drop('lag_0', axis=1), train['lag_0']
X_test, y_test = test.drop('lag_0', axis=1), test['lag_0'] 
```

### Задание 5. 


Обучите модель линейной регрессии из sklearn (LinearRegression, Ridge, Lasso)`

```python
# <your code here>
from sklearn.linear_model import Lasso

lr = Lasso()

lr.fit(X_train, y_train)
```

### Задание 6.


Сделайте in-sample прогноз на test (по сути это просто вызов метода predict на данных из теста)

Нарисуйте рядом графики прогноза

```python
# <your_code_here>
y_pred = lr.predict(X_test)
```

```python
y_pred = pd.Series(y_pred, index=y_test.index)
```

```python
plot_ts(y_test, y_pred)
```

### Задание 7.


Сделайте out-of-sample прогноз на test.

Данный прогноз делается рекурсивным способом по следующему алгоритму:

Пусть есть временной ряд $Y_T$. 

1) Для следующей точки $Y_{T+1}$, в качестве лагов обьекта возьмите предыдущие $num\_lags$ точек.  
2) Добавьте дополнительные признаки к лагам  
3) Сделайте прогноз на полученном обьекте моделью линейной регрессии  
4) Добавьте прогноз к ряду $Y_T$  
5) Повторите пункт 1

И так далее пока не сделаeте прогноз до конца тестовой выборки.

```python
# <your_code_here>

from copy import deepcopy

history = ts[:train.index[-1]]
granularity = ts.index[1] - ts.index[0]

y_pred = pd.Series()
for _ in range(len(test)):
    lags = history[-num_lags:]
    lags = lags.to_frame().T
    lags.index = [history.index[-1] + granularity]  # время следующей точки
    lags.columns = matrix.columns.drop(
        ['lag_0'] + [f for f in matrix.columns if (f.endswith('sin') or f.endswith('cos'))]
    )

    for f_name, f in datetime_mappers.items():
        lags[f_name] = lags.index.map(f)
    
    pred = lr.predict(lags)[0]
    y_pred = pd.concat([y_pred, pd.Series(pred, index=lags.index)])
    history = pd.concat([history, pd.Series(pred, index=lags.index)])
```

```python
plot_ts(y_test, y_pred)
```

```python

```
