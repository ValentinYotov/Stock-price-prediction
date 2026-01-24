# План за Проект: Прогнозиране на Пазара на Акции
## От нула до професионална система

---

## 📋 Обща визия на проекта

**Цел:** Създаване на модел от нулата за прогнозиране на пазара на акции с използване на Hugging Face datasets и съвременни техники за time series forecasting.

**Технологичен стек:**
- Python 3.8+
- PyTorch / TensorFlow (препоръчително PyTorch за по-голяма контрол)
- Hugging Face Datasets & Transformers
- NumPy, Pandas
- Matplotlib, Plotly (за визуализация)
- scikit-learn (за предобработка и базови модели)

---

## 🏗️ Структура на проекта

```
stock-price-prediction/
│
├── data/
│   ├── raw/              # Сирови данни от Hugging Face
│   ├── processed/        # Предобработени данни
│   └── features/         # Извлечени features
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py     # Зареждане на данни от Hugging Face
│   │   ├── preprocessor.py  # Предобработка и очистване
│   │   └── feature_engineering.py  # Създаване на features
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py     # Базов клас за модели
│   │   ├── transformer_model.py  # Transformer архитектура от нулата
│   │   ├── lstm_model.py     # LSTM baseline
│   │   └── components/       # Отде Architectural components
│   │       ├── attention.py
│   │       ├── encoder.py
│   │       └── decoder.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py        # Training loop
│   │   ├── losses.py         # Loss functions
│   │   └── callbacks.py      # Callbacks (early stopping, checkpointing)
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py        # Evaluation metrics
│   │   └── visualizations.py # Графики и визуализации
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py         # Конфигурация
│       └── helpers.py        # Helper функции
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_development.ipynb
│   ├── 04_training.ipynb
│   └── 05_evaluation.ipynb
│
├── configs/
│   ├── default_config.yaml   # Конфигурационен файл
│   └── model_configs.yaml    # Конфигурации на модели
│
├── scripts/
│   ├── train.py              # Скрипт за обучение
│   ├── evaluate.py           # Скрипт за оценка
│   └── inference.py          # Скрипт за прогнозиране
│
├── models/                   # Записани модели
│   └── checkpoints/
│
├── results/                  # Резултати и графики
│   ├── plots/
│   └── metrics/
│
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

---

## 📅 Етап 1: Избор на Dataset и Проучване (1-2 дни)

### 1.1 Избор на Hugging Face Dataset
**Предложени datasets:**
- `TimeSeries/stock_prices` - ако съществува
- `t4tiana/store-sales-time-series-forecasting` - за time series forecasting
- `ykotseruba/stock-prices-daily` - дневни цени на акции
- `HUPD/stock-market` - пазарни данни
- Или комбиниране на няколко datasets

**Действия:**
- ✅ Проучване на наличните datasets в Hugging Face
- ✅ Избор на подходящ dataset или комбинация
- ✅ Анализ на структурата и качеството на данните
- ✅ Документиране на избора в notebook `01_data_exploration.ipynb`

### 1.2 Анализ на данните
- Размери на dataset (брой записи, период)
- Налични features (Open, High, Low, Close, Volume, etc.)
- Липсващи стойности и outliers
- Статистически анализ (дистрибуция, корелации)
- Визуализация на времеви редици

---

## 📊 Етап 2: Предобработка и Feature Engineering (2-3 дни)

### 2.1 Създаване на Data Pipeline (`src/data/loader.py`)
**Функционалности:**
```python
- load_hf_dataset(dataset_name, splits)
- save_raw_data()
- load_raw_data()
```

### 2.2 Предобработка (`src/data/preprocessor.py`)
**Задачи:**
- Обработка на липсващи стойности
  - Forward fill, backward fill
  - Interpolation
  - Drop rows с критични липси
- Обработка на outliers
  - IQR method
  - Z-score method
  - Clipping на екстремни стойности
- Normalization/Standardization
  - Min-Max scaling
  - Z-score normalization
  - Robust scaling
- Времеви features
  - Extract date components (day, month, year, day_of_week)
  - Cyclical encoding (sin/cos за ден и месец)
  - Lag features (предходни стойности)

### 2.3 Feature Engineering (`src/data/feature_engineering.py`)
**Технически индикатори:**
- Moving Averages (SMA, EMA, WMA)
- Momentum indicators (RSI, MACD, Stochastic)
- Volatility indicators (Bollinger Bands, ATR)
- Volume indicators (OBV, Volume SMA ratio)
- Price patterns (Candlestick patterns - опционално)

**Създаване на features:**
```python
- calculate_technical_indicators(df, windows=[5, 10, 20, 50])
- create_lag_features(df, lags=[1, 2, 3, 5, 10])
- create_rolling_statistics(df)
- encode_temporal_features(df)
```

### 2.4 Train/Validation/Test Split
- **Времеви split** (не случайно!) - важно за time series
- Train: 70% (най-стари данни)
- Validation: 15%
- Test: 15% (най-нови данни)
- Обеспечаване на последователност

---

## 🧠 Етап 3: Архитектура на модела (3-4 дни)

### 3.1 Базов клас (`src/models/base_model.py`)
Абстрактен клас с общи методи:
- `forward()`
- `predict()`
- `save_model()`
- `load_model()`

### 3.2 Transformer Model от нулата (`src/models/transformer_model.py`)

**Компоненти които трябва да създадем:**

#### 3.2.1 Positional Encoding (`src/models/components/positional_encoding.py`)
- Sinusoidal positional encoding
- Или learnable positional embeddings

#### 3.2.2 Multi-Head Attention (`src/models/components/attention.py`)
- Scaled Dot-Product Attention
- Multi-Head Attention mechanism
- Опционално: Self-attention и Cross-attention

#### 3.2.3 Encoder (`src/models/components/encoder.py`)
- Transformer Encoder Layer:
  - Multi-Head Self-Attention
  - Feed-Forward Network
  - Layer Normalization
  - Residual connections
- Stack of encoder layers

#### 3.2.4 Decoder (опционално) (`src/models/components/decoder.py`)
- Transformer Decoder Layer:
  - Masked Multi-Head Self-Attention
  - Multi-Head Cross-Attention
  - Feed-Forward Network
- Stack of decoder layers

#### 3.2.5 Пълна Transformer архитектура
```python
class StockTransformer(nn.Module):
    - Input embedding layer
    - Positional encoding
    - Encoder stack (6-8 слоя)
    - Output projection layer
    - Forecast head (може да предсказва няколко стъпки напред)
```

**Хиперпараметри за експериментиране:**
- `d_model`: 128, 256, 512 (embedding dimension)
- `n_heads`: 4, 8, 16 (брой attention heads)
- `n_layers`: 4, 6, 8 (брой encoder слоеве)
- `d_ff`: 512, 1024, 2048 (feed-forward dimension)
- `dropout`: 0.1, 0.2, 0.3
- `context_length`: 60, 90, 120 (дни история)
- `prediction_horizon`: 1, 5, 10 (дни напред)

### 3.3 Baseline модели

#### 3.3.1 LSTM Baseline (`src/models/lstm_model.py`)
- Vanilla LSTM или Bidirectional LSTM
- За сравнение с Transformer

#### 3.3.2 Linear Baseline
- Проста линейна регресия
- Служи като минимален baseline

---

## 🎯 Етап 4: Training Pipeline (3-4 дни)

### 4.1 Loss Functions (`src/training/losses.py`)
**Възможни loss функции:**
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Huber Loss (комбинация от MSE и MAE)
- Quantile Loss (за вероятностни прогнози)

### 4.2 Trainer Class (`src/training/trainer.py`)
**Функционалности:**
- Training loop с batched data
- Validation loop
- Gradient clipping (за стабилност)
- Learning rate scheduling
- Logging на метрики
- TensorBoard или Weights & Biases integration

**Методи:**
```python
- train_epoch()
- validate()
- save_checkpoint()
- load_checkpoint()
- train()  # Main training method
```

### 4.3 Callbacks (`src/training/callbacks.py`)
- Early Stopping (спиране при overfitting)
- Model Checkpointing (записване на най-добър модел)
- Learning Rate Scheduler
- Metric Logger

### 4.4 Конфигурация (`src/utils/config.py`, `configs/default_config.yaml`)
YAML файл с всички хиперпараметри:
```yaml
data:
  dataset_name: "ykotseruba/stock-prices-daily"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  
model:
  type: "transformer"
  d_model: 256
  n_heads: 8
  n_layers: 6
  dropout: 0.1
  context_length: 90
  prediction_horizon: 1
  
training:
  batch_size: 32
  learning_rate: 0.0001
  num_epochs: 100
  optimizer: "adam"
  scheduler: "cosine"
```

---

## 📈 Етап 5: Оценка и Визуализация (2-3 дни)

### 5.1 Метрики (`src/evaluation/metrics.py`)
**Регресионни метрики:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)
- Directional Accuracy (% правилни посоки)

### 5.2 Визуализации (`src/evaluation/visualizations.py`)
**Графики:**
- Прогнози vs Реални стойности (time series plot)
- Residual plots (за анализ на грешки)
- Error distribution
- Метрики по епохи (training curves)
- Feature importance (ако приложимо)
- Backtesting резултати

### 5.3 Сравнение с Baselines
- Таблица с метрики за всички модели
- Статистически тестове (опционално)
- Визуално сравнение

---

## 📝 Етап 6: Документация и Кодова Организация (2-3 дни)

### 6.1 README.md
**Секции:**
- Описание на проекта
- Инсталация и изисквания
- Структура на проекта
- Как да използваш модела
- Примери за употреба
- Резултати и метрики
- Автор и лиценз

### 6.2 Docstrings и Type Hints
- Документация на всички функции и класове
- Type hints за по-добра читаемост
- Коментари за сложни алгоритми

### 6.3 Requirements.txt
```txt
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
datasets>=2.14.0
transformers>=4.30.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
plotly>=5.14.0
yfinance>=0.2.0
pyyaml>=6.0
tqdm>=4.65.0
```

### 6.4 Git Repository
- Инициализация на git
- .gitignore файл
- Структурирани commit messages
- README и документация

---

## 🚀 Етап 7: Финален Тест и Полиране (1-2 дни)

### 7.1 Тестване на целия pipeline
- End-to-end тест: от данни до прогноза
- Проверка за bugs и edge cases
- Performance тестове

### 7.2 Код ревю
- Проверка за code quality
- Оптимизация (ако е необходимо)
- Рефакториране

### 7.3 Финален анализ
- Написание на заключение
- Анализ на ограниченията
- Предложения за надграждане

---

## 📋 Резюме на компонентите

### Backend код (src/): ~2000-3000 реда
- Data pipeline: ~400-500 реда
- Model архитектури: ~800-1000 реда
- Training pipeline: ~400-500 реда
- Evaluation: ~300-400 реда
- Utils и helpers: ~200-300 реда

### Notebooks: ~5 notebooks по 200-300 клетки всеки
- Експериментиране и анализ
- Визуализации
- Примерни употреби

### Скриптове: ~3 скрипта по 100-200 реда
- Автоматизиране на процеса
- Command-line интерфейс

### Конфигурация и документация: ~500-1000 реда
- YAML configs
- README
- Docstrings

**ОБЩО: ~4000-5000+ реда професионален код**

---

## ✅ Критерии за успех

1. ✅ Модел обучен от нулата (не просто предаване на готов)
2. ✅ Използване на Hugging Face datasets
3. ✅ Добре структуриран и модулен код
4. ✅ Сравнение с baseline модели
5. ✅ Детайлна документация
6. ✅ Репроизводими резултати
7. ✅ Визуализации и анализ

---

## 🎓 За курсова работа

**Препоръчителна структура на документ:**

1. **Въведение** - Цел, мотивация, обхват
2. **Проучване на литература** - Свързани работи, методи
3. **Dataset и Методология** - Описание на данните, предобработка
4. **Архитектура на модела** - Детайлно описание на Transformer
5. **Експерименти** - Хиперпараметри, training процес
6. **Резултати** - Метрики, сравнения, визуализации
7. **Заключение** - Научени неща, ограничения, бъдещи работи

---

**Готови ли сте да започнем? С какъв етап искате да почнем първо?**
