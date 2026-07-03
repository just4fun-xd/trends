# Как подключить fetch_resilient в scripts/fetch_databento_futures.py

Меняется ТОЛЬКО функция `fetch_via_api` — тракт build_panels/
write_panels остаётся как есть. Замени тело fetch_via_api на вызов
устойчивого загрузчика.

## Было (роняется на обрыве любого потока, теряет всю выгрузку):

```python
def fetch_via_api(symbols, start, end, schema="ohlcv-1d"):
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError(...)
    import databento as db
    client = db.Historical(api_key)
    out = {}
    for sym in symbols:
        m1 = client.timeseries.get_range(...).to_df()  # <- падает тут
        m2 = client.timeseries.get_range(...).to_df()
        df = m1[[...]].copy()
        df["close_m2"] = m2["close"].reindex(df.index)
        df = df[~df.index.duplicated(keep="last")]
        out[sym] = df
    return out
```

## Стало:

```python
from pathlib import Path
from scripts.fetch_resilient import fetch_symbols_cached


def fetch_via_api(symbols, start, end, schema="ohlcv-1d",
                  cache_dir="data/panels_4h/futures/raw"):
    """Тянет M1/M2 из Databento устойчиво: per-symbol кэш + retry.

    Обрыв потока на середине корзины больше не теряет уже скачанное
    (оплаченное) — повторный запуск дособирает недостающее из кэша.
    Softs не на GLBX (KC/CC/SB) пропускаются без падения.
    """
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Нет DATABENTO_API_KEY. export DATABENTO_API_KEY=... "
            "или запусти с --demo."
        )
    try:
        import databento as db
    except ImportError as exc:
        raise RuntimeError(
            "databento не установлен: pip install databento"
        ) from exc

    client = db.Historical(api_key)
    return fetch_symbols_cached(
        client, symbols, start, end, schema,
        dataset=DATASET, cache_dir=Path(cache_dir), with_m2=True,
    )
```

## Что это даёт

1. Падение на CT/PA/PL больше не убивает уже скачанные CL..ZM —
   они в data/panels_4h/futures/raw/{sym}.parquet.
2. Перезапуск той же команды дособирает только недостающие символы;
   оплаченное не перекачивается.
3. KC/CC/SB (кофе/какао/сахар — они на ICE, не на CME/GLBX.MDP3)
   пропускаются, корзина становится 13 инструментов. Это ожидаемо.
4. Транзиентный "Response ended prematurely" ретраится с backoff.

## Softs с ICE (если нужны CC/KC/SB на H4)

Они не на GLBX.MDP3. Нужен ICE-датасет Databento (например
IFEU.IMPACT для ICE Europe, IFUS для softs) — это ОТДЕЛЬНЫЙ
fetch_via_api с другим DATASET и, возможно, другой symbology. Пока
корзина 13 (энергия+металлы+зерно) более чем достаточна для H4-
проверки края MR/trend; softs добавишь отдельным треком, если
конкретно они понадобятся.

## После правки — перезапусти ту же команду

```bash
python -m scripts.fetch_databento_futures --interval 4h \
    --symbols CL NG GC SI HG ZW ZC ZS ZL ZM CT PA PL \
    --start 2021-01-01 --end 2026-01-01 --out data/panels_4h/futures
```

(убрал KC CC SB из списка — они всё равно выпадут). Если прошлый
запуск успел что-то скачать до обрыва — при наличии кэша они
подхватятся мгновенно.
```
```
