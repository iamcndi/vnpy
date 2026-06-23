"""
A 股日线数据下载：akshare / 东方财富直连（优先） + baostock（回退）

用法：
    from datafeed import download_daily_data, data_end_date
    download_daily_data(lab, STOCK_LIST)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import datetime, timedelta

import baostock as bs
import polars as pl

from vnpy.alpha import AlphaLab
from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.object import BarData

DEFAULT_START = "2023-01-01"
VOLUME_LOT_SIZE = 100  # 东方财富成交量单位：手 → 股
REQUEST_INTERVAL = 1.0  # 限速，避免被东方财富断开连接

EXCHANGE_VT2BS = {
    Exchange.SSE: "sh",
    Exchange.SZSE: "sz",
    Exchange.BSE: "bj",
}

BAOSTOCK_FIELDS = "date,open,high,low,close,volume,amount"
EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"


def data_end_date() -> str:
    """数据下载截止日期（今日）"""
    return datetime.now().strftime("%Y-%m-%d")


def _to_ak_date(date_str: str) -> str:
    return date_str.replace("-", "")


def _next_date(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)
    return dt.strftime("%Y-%m-%d")


def _is_etf(symbol: str) -> bool:
    return symbol.startswith(("51", "15", "56", "58"))


def _eastmoney_secid(code: str) -> str:
    """东方财富 secid：沪市 1.xxxxxx，深市 0.xxxxxx"""
    if _is_etf(code):
        return f"1.{code}" if code.startswith(("51", "56", "58")) else f"0.{code}"
    return f"1.{code}" if code.startswith("6") else f"0.{code}"


def _klines_to_bars(
    klines: list[str],
    code: str,
    exchange: Exchange,
    gateway_name: str,
) -> list[BarData]:
    bars: list[BarData] = []
    for line in klines:
        parts = line.split(",")
        if len(parts) < 7:
            continue
        date_str, o_str, c_str, h_str, l_str, v_str, a_str = parts[:7]
        close = float(c_str)
        if close <= 0:
            continue
        bars.append(BarData(
            symbol=code,
            exchange=exchange,
            datetime=datetime.strptime(date_str, "%Y-%m-%d"),
            interval=Interval.DAILY,
            open_price=float(o_str),
            high_price=float(h_str),
            low_price=float(l_str),
            close_price=close,
            volume=float(v_str) * VOLUME_LOT_SIZE,
            turnover=float(a_str),
            gateway_name=gateway_name,
        ))
    return bars


def fetch_bars_eastmoney(
    code: str,
    exchange: Exchange,
    start_date: str,
    end_date: str,
) -> list[BarData]:
    """东方财富 K 线（curl_cffi / curl 直连，绕过系统代理）"""
    secid = _eastmoney_secid(code)
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101",
        "fqt": "1",
        "secid": secid,
        "beg": _to_ak_date(start_date),
        "end": _to_ak_date(end_date),
    }

    # 1) curl_cffi 模拟浏览器 TLS
    try:
        from curl_cffi import requests as cffi_requests

        resp = cffi_requests.get(
            EASTMONEY_KLINE_URL,
            params=params,
            impersonate="chrome",
            timeout=30,
            proxies={"http": None, "https": None},
        )
        payload = resp.json()
        klines = (payload.get("data") or {}).get("klines") or []
        bars = _klines_to_bars(klines, code, exchange, "EM")
        if bars:
            return bars
    except Exception:
        pass

    # 2) 系统 curl 回退
    if not shutil.which("curl"):
        raise RuntimeError("curl 不可用")

    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{EASTMONEY_KLINE_URL}?{query}"
    curl_cmd = [
        "curl", "-sS", "--noproxy", "*", "-m", "30",
        "-A", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "-H", "Referer: https://quote.eastmoney.com/",
        url,
    ]

    proc = subprocess.run(curl_cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        time.sleep(1.5)
        proc = subprocess.run(curl_cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"curl exit {proc.returncode}")

    payload = json.loads(proc.stdout)
    klines = (payload.get("data") or {}).get("klines") or []
    return _klines_to_bars(klines, code, exchange, "EM")


def _df_to_bars(
    df,
    code: str,
    exchange: Exchange,
    gateway_name: str,
    volume_scale: float = VOLUME_LOT_SIZE,
) -> list[BarData]:
    """将 akshare DataFrame 转为 BarData 列表"""
    if df is None or df.empty:
        return []

    bars: list[BarData] = []
    for _, row in df.iterrows():
        date_val = row["日期"]
        if hasattr(date_val, "strftime"):
            dt = datetime.combine(date_val, datetime.min.time())
        else:
            dt = datetime.strptime(str(date_val)[:10], "%Y-%m-%d")

        close = float(row["收盘"])
        if close <= 0:
            continue

        bars.append(BarData(
            symbol=code,
            exchange=exchange,
            datetime=dt,
            interval=Interval.DAILY,
            open_price=float(row["开盘"]),
            high_price=float(row["最高"]),
            low_price=float(row["最低"]),
            close_price=close,
            volume=float(row["成交量"]) * volume_scale,
            turnover=float(row["成交额"]),
            gateway_name=gateway_name,
        ))
    return bars


def fetch_bars_akshare(
    code: str,
    exchange: Exchange,
    start_date: str,
    end_date: str,
) -> list[BarData]:
    """从 akshare（东方财富）拉取前复权日线"""
    import os
    import akshare as ak

    ak_start = _to_ak_date(start_date)
    ak_end = _to_ak_date(end_date)

    # 东方财富接口常因系统代理配置失败，临时禁用代理环境变量
    proxy_keys = (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    )
    saved_proxy = {k: os.environ.pop(k, None) for k in proxy_keys}

    try:
        if _is_etf(code):
            df = ak.fund_etf_hist_em(
                symbol=code,
                period="daily",
                start_date=ak_start,
                end_date=ak_end,
                adjust="qfq",
            )
        else:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=ak_start,
                end_date=ak_end,
                adjust="qfq",
            )
    finally:
        for key, value in saved_proxy.items():
            if value is not None:
                os.environ[key] = value

    return _df_to_bars(df, code, exchange, "AK")


def fetch_bars_baostock(
    code: str,
    exchange: Exchange,
    start_date: str,
    end_date: str,
) -> list[BarData]:
    """从 baostock 拉取前复权日线（需已 login）"""
    bs_code = f"{EXCHANGE_VT2BS[exchange]}.{code}"
    rs = bs.query_history_k_data_plus(
        bs_code,
        BAOSTOCK_FIELDS,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2",
    )
    if rs.error_code != "0":
        raise RuntimeError(rs.error_msg)

    bars: list[BarData] = []
    while rs.next():
        date_str, o_str, h_str, l_str, c_str, v_str, a_str = rs.get_row_data()
        if not c_str or c_str == "":
            continue
        bars.append(BarData(
            symbol=code,
            exchange=exchange,
            datetime=datetime.strptime(date_str, "%Y-%m-%d"),
            interval=Interval.DAILY,
            open_price=float(o_str),
            high_price=float(h_str),
            low_price=float(l_str),
            close_price=float(c_str),
            volume=float(v_str or 0),
            turnover=float(a_str or 0),
            gateway_name="BS",
        ))
    return bars


def _resolve_start_date(
    parquet_path,
    backtest_start: str,
) -> tuple[str, str]:
    """返回 (start_date, action_label)"""
    if parquet_path.exists():
        df = pl.read_parquet(parquet_path)
        last_date = df["datetime"].max()
        if last_date:
            return _next_date(last_date.strftime("%Y-%m-%d")), "增量更新"
    return backtest_start, "全量下载"


def download_daily_data(
    lab: AlphaLab,
    stock_list: list[tuple[str, str, str]],
    backtest_start: str = DEFAULT_START,
) -> None:
    """下载/增量更新日线至最新（东方财富/akshare 优先，失败则回退 baostock）"""
    end_date = data_end_date()
    print(f"  目标截止日期: {end_date}")
    print("  数据源: 东方财富/akshare (优先) → baostock (回退)")

    bs_logged_in = False

    try:
        for code, exchange_str, name in stock_list:
            exchange = Exchange(exchange_str)
            vt_symbol = f"{code}.{exchange_str}"
            parquet_path = lab.daily_path / f"{vt_symbol}.parquet"

            start_date, action = _resolve_start_date(parquet_path, backtest_start)
            if start_date > end_date:
                if parquet_path.exists():
                    df = pl.read_parquet(parquet_path)
                    last = df["datetime"].max().strftime("%Y-%m-%d")
                    print(f"  ✓ {vt_symbol} ({name}) 已是最新 ({last})")
                continue

            print(f"  ↓ {vt_symbol} ({name}) {action}: {start_date} ~ {end_date}")

            bars: list[BarData] = []
            source = ""

            try:
                bars = fetch_bars_eastmoney(code, exchange, start_date, end_date)
                if bars:
                    source = "eastmoney"
            except Exception as exc:
                print(f"    东方财富直连失败: {exc}")

            if not bars:
                try:
                    bars = fetch_bars_akshare(code, exchange, start_date, end_date)
                    if bars:
                        source = "akshare"
                except Exception as exc:
                    print(f"    akshare 失败: {exc}")

            if not bars:
                if not bs_logged_in:
                    lg = bs.login()
                    if lg.error_code != "0":
                        print(f"    baostock 登录失败: {lg.error_msg}")
                        continue
                    bs_logged_in = True
                try:
                    bars = fetch_bars_baostock(code, exchange, start_date, end_date)
                    if bars:
                        source = "baostock"
                except Exception as exc:
                    print(f"    baostock 失败: {exc}")

            if bars:
                lab.save_bar_data(bars)
                latest = max(b.datetime for b in bars).strftime("%Y-%m-%d")
                print(f"    ✓ [{source}] 保存 {len(bars)} 条, 最新 {latest}")
            else:
                print("    - 无新数据")

            time.sleep(REQUEST_INTERVAL)

    finally:
        if bs_logged_in:
            bs.logout()
