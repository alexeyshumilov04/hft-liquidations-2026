---

# HFT Liquidations 2026

Проект по построению ML-фильтра для сделок Binance на основе данных о ликвидациях с Binance и Bybit. Цель — максимизировать maker PnL за счёт отсеивания неблагоприятных сделок при сохранении минимального дневного оборота.

## Данные

Все таблицы — Parquet, `timestamp` в микросекундах UNIX epoch (UTC). Юниверс: `perp:btcusdt`, `perp:ethusdt`.

| Источник | Путь | Колонки |
|---|---|---|
| Binance trades | `data/binance_trades/perp_<sym>.parquet` | timestamp, ticker, side, price, amount |
| Binance booktickers | `data/binance_booktickers/perp_<sym>.parquet` | timestamp, ticker, bid_price, bid_amount, ask_price, ask_amount |
| Binance liquidations | `data/binance_liquidations/perp_<sym>.parquet` | timestamp, ticker, side, price, amount |
| Bybit liquidations | `data/bybit_liquidations/<sym>.parquet` | timestamp, ticker, side, price, amount |

**Side:** в trades — сторона тейкера; в liquidations — сторона ордера-ликвидации (`buy` ⇒ закрытие шорта ⇒ давление вверх).

**Задержка Bybit:** timestamp ликвидаций Bybit сдвигается на +200 мс перед сопоставлением с Binance.

## Задания

- **Task 1: EDA** — разведочный анализ: качество данных, распределения, паттерны времени, кросс-биржевое сравнение, микроструктура, режимы рынка.
- **Task 2: Classes** — поиск интерпретируемых рыночных классов, разделяющих сделки по markout и turnover.
- **Task 3: Validation** — оценка полезности прогноза: официальный score, walk-forward, purged/embargoed CV, статистические тесты, бутстрэп.
- **Task 4: Modelling** — построение фильтра: feature engineering (ликвидации, микроструктура), walk-forward обучение (W1–W5), интерпретация и анализ ошибок.
