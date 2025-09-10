import hmac, hashlib, time, io, re
from urllib.parse import urlencode
from datetime import timedelta, datetime
from typing import Dict, Any, List, Tuple

import requests
import numpy as np
import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta

# ---------------------- Config UI ----------------------
st.set_page_config(page_title="Crypto KPI Dashboard", page_icon="📈", layout="wide")
st.title("📈 Crypto KPI Dashboard – SOL/USDC")
st.caption("Legge i dati direttamente da Binance. Imposta le date, poi clicca **Calcola KPI**.")

PAIR_DEFAULT = "SOLUSDC"  # simbolo per API Binance (senza slash)
HUMAN_PAIR = "SOL/USDC"   # label umana
BASE_URL = "https://api.binance.com"  # Spot

EPS = 1e-12

# ---------------------- Secrets ----------------------
def get_keys():
    try:
        api_key = st.secrets["binance"]["api_key"]
        api_secret = st.secrets["binance"]["api_secret"]
        if not api_key or not api_secret:
            raise KeyError
        return api_key, api_secret
    except Exception:
        st.error("🔐 Chiavi Binance mancanti. Aggiungile in **Settings → Secrets**:\n[binance]\napi_key=...\napi_secret=...")
        st.stop()

# ---------------------- HTTP helpers ----------------------
def binance_signed_get(path: str, params: Dict[str, Any], api_key: str, api_secret: str):
    """
    GET firmata per endpoint privati Spot (HMAC SHA256).
    """
    params = {**params, "timestamp": int(time.time() * 1000), "recvWindow": 5000}
    q = urlencode(params, doseq=True)
    sig = hmac.new(api_secret.encode(), q.encode(), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": api_key}
    r = requests.get(f"{BASE_URL}{path}?{q}&signature={sig}", headers=headers, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Binance error {r.status_code}: {r.text}")
    return r.json()

def binance_public_get(path: str, params: Dict[str, Any] = None):
    r = requests.get(f"{BASE_URL}{path}", params=params or {}, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"Binance public error {r.status_code}: {r.text}")
    return r.json()

# ---------------------- Data fetchers ----------------------
@st.cache_data(show_spinner=True)
def fetch_trades(api_key: str, api_secret: str, symbol: str, start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
    """
    /api/v3/myTrades: trade eseguiti (fill) per il symbol SPOT.
    Supporta startTime/endTime, limit=1000. Se tanti, fai più finestre.
    """
    all_trades: List[Dict[str, Any]] = []
    params = {"symbol": symbol, "limit": 1000, "startTime": start_ms, "endTime": end_ms}
    # Nota: Binance a volte ignora endTime; splittiamo per sicurezza in finestre mensili
    start_dt = datetime.utcfromtimestamp(start_ms/1000)
    end_dt = datetime.utcfromtimestamp(end_ms/1000)

    cursor = start_dt
    while cursor < end_dt:
        window_end = min(cursor + relativedelta(months=1), end_dt)
        chunk = binance_signed_get("/api/v3/myTrades",
                                   {"symbol": symbol,
                                    "limit": 1000,
                                    "startTime": int(cursor.timestamp()*1000),
                                    "endTime": int(window_end.timestamp()*1000)},
                                   api_key, api_secret)
        all_trades.extend(chunk)
        # avanzamento cursore
        cursor = window_end + timedelta(milliseconds=1)
    # Ordina per time
    all_trades.sort(key=lambda x: x["time"])
    return all_trades

@st.cache_data(show_spinner=False)
def fetch_price(symbol: str) -> float:
    data = binance_public_get("/api/v3/ticker/price", {"symbol": symbol})
    return float(data["price"])

# ---------------------- Transforms ----------------------
def df_from_trades(trades: List[Dict[str, Any]], human_pair: str) -> pd.DataFrame:
    """
    Converte myTrades in un dataframe ordini normalizzato (BUY/SELL atomici).
    Considera fee se in SOL o USDC; altre fee (es. BNB) non sono convertite.
    """
    rows = []
    for t in trades:
        qty = float(t["qty"])
        price = float(t["price"])
        is_buyer = bool(t["isBuyer"])
        commission = float(t["commission"])
        commission_asset = t["commissionAsset"]

        # Stima “total” lato ordini: qty * price
        total = qty * price

        # Fee in USDC (se asset è USDC), oppure in SOL convertita a USDC col prezzo del trade
        fee_usdc = 0.0
        if commission > 0:
            if commission_asset == "USDC":
                fee_usdc = commission
            elif commission_asset == "SOL":
                fee_usdc = commission * price
            # Se BNB/altro: per semplicità ignoriamo o potremmo aggiungere conversione con un altro ticker

        # BUY = ingresso, SELL = uscita
        rows.append({
            "date": pd.to_datetime(t["time"], unit="ms"),
            "pair": human_pair,
            "type": "BUY" if is_buyer else "SELL",
            "qty": qty,
            "price": price,
            "total": total,
            "fee_usdc": fee_usdc,
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df

def fifo_trades_per_match(orders: pd.DataFrame, pair_label: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """
    BUY→SELL FIFO *completo* con split + costruzione inventario (posizioni aperte).
    Ritorna (trades_chiusi, meta, inventario_aperto).
    """
    df = orders[orders["pair"] == pair_label].copy()
    if df.empty:
        return [], {"start": None, "end": None, "totalMoved": 0.0}, []

    buy_q: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []

    total_moved = float(df["total"].abs().sum())
    start = df["date"].min()
    end = df["date"].max()

    for _, o in df.iterrows():
        if o["type"] == "BUY":
            buy_q.append(o.to_dict())
        elif o["type"] == "SELL":
            sell_qty = float(o["qty"])
            while sell_qty > EPS and buy_q:
                b = buy_q[0]
                take = float(min(b["qty"], sell_qty))
                pnl = (float(o["price"]) - float(b["price"])) * take

                # Fee: sottraiamo fee USDC su buy e sell proporzionale alla porzione
                fee_buy = float(b.get("fee_usdc", 0.0)) * (take / float(b["qty"])) if float(b["qty"]) > 0 else 0.0
                fee_sell = float(o.get("fee_usdc", 0.0)) * (take / float(o["qty"])) if float(o["qty"]) > 0 else 0.0
                pnl -= (fee_buy + fee_sell)

                size_usdc = take * float(b["price"])  # size valorizzata a prezzo di carico
                dur_min = (o["date"] - b["date"]).total_seconds() / 60.0
                trades.append({
                    "buy_date": b["date"],
                    "sell_date": o["date"],
                    "qty": take,
                    "buy_price": float(b["price"]),
                    "sell_price": float(o["price"]),
                    "size_usdc": size_usdc,
                    "pnl": pnl,
                    "dur_min": dur_min,
                })
                b["qty"] -= take
                sell_qty -= take
                if b["qty"] <= EPS:
                    buy_q.pop(0)

    # Inventario aperto = quello che resta nella coda BUY
    inventory = []
    for b in buy_q:
        if float(b["qty"]) > EPS:
            inventory.append({
                "date": b["date"],
                "qty": float(b["qty"]),
                "avg_cost": float(b["price"]),
                "fee_usdc": float(b.get("fee_usdc", 0.0)),
            })

    return trades, {"start": start, "end": end, "totalMoved": total_moved}, inventory

def compute_kpis(trades: List[Dict[str, Any]], meta: Dict[str, Any], residual_cfg: Dict[str, float] | None) -> Dict[str, Any]:
    total_pnl = float(sum(t["pnl"] for t in trades))
    trades_n = len(trades)
    best = max(trades, key=lambda t: t["pnl"]) if trades else None
    worst = min(trades, key=lambda t: t["pnl"]) if trades else None
    avg_size = float(np.mean([t["size_usdc"] for t in trades])) if trades else 0.0
    avg_pnl = float(np.mean([t["pnl"] for t in trades])) if trades else 0.0
    success_rate = float(100.0 * sum(1 for t in trades if t["pnl"] > 0) / trades_n) if trades_n else 0.0
    avg_dur = float(np.mean([t["dur_min"] for t in trades])) if trades else 0.0

    residual_block = {"qty": 0.0, "avgCost": 0.0, "targetPrice": 0.0, "cost": 0.0, "value": 0.0, "pnl": 0.0}
    if residual_cfg and residual_cfg.get("qty"):
        residual_block["qty"] = float(residual_cfg["qty"])
        residual_block["avgCost"] = float(residual_cfg.get("avgCost", 0))
        residual_block["targetPrice"] = float(residual_cfg.get("targetPrice", 0))
        residual_block["cost"] = residual_block["qty"] * residual_block["avgCost"]
        residual_block["value"] = residual_block["qty"] * residual_block["targetPrice"]
        residual_block["pnl"] = residual_block["value"] - residual_block["cost"]

    total_with_residual = total_pnl + residual_block["pnl"]

    start = meta.get("start")
    end = meta.get("end")
    months = 1
    if start and end:
        months = (end.year - start.year) * 12 + (end.month - start.month) + 1
    monthly_avg = total_with_residual / max(1, months)

    # Ultimi 10 giorni (sulla SELL date)
    last10 = []
    last10_pnl = 0.0
    if end:
        cutoff = end - timedelta(days=10)
        last10 = [t for t in trades if t["sell_date"] >= cutoff]
        last10_pnl = float(sum(t["pnl"] for t in last10))

    return {
        "period": {"start": start, "end": end},
        "pnl": {"total": round(total_pnl, 2),
                "totalWithResidual": round(total_with_residual, 2),
                "residual": residual_block},
        "bestTrade": best,
        "worstTrade": worst,
        "counts": {"trades": trades_n, "successRatePct": round(success_rate, 2)},
        "sizes": {"avgTradeUSDC": round(avg_size, 2), "totalUSDCMoved": round(meta.get("totalMoved", 0.0), 2)},
        "perTrade": {"avgPnl": round(avg_pnl, 2), "avgDurationMin": round(avg_dur, 2)},
        "last10d": {"pnl": round(last10_pnl, 2), "trades": len(last10)},
        "monthlyAvgGain": round(monthly_avg, 2),
        "trades": trades,
    }

def format_number(x: float) -> str:
    return f"{x:,.2f}".replace(",", "@").replace(".", ",").replace("@", ".")

def format_duration(mins: float | int) -> str:
    if mins is None:
        return "-"
    mins = float(mins)
    if mins < 60:
        return f"~{format_number(mins)} min"
    hours = mins / 60
    if hours < 24:
        return f"~{format_number(hours)} h"
    days = hours / 24
    return f"~{format_number(days)} giorni"

# ---------------------- UI ----------------------
with st.sidebar:
    st.header("⚙️ Input")
    # Finestra temporale
    default_start = datetime.utcnow() - relativedelta(months=4)
    default_end = datetime.utcnow()
    start_date = st.date_input("Inizio (UTC)", value=default_start.date())
    end_date = st.date_input("Fine (UTC)", value=default_end.date())
    st.caption("Suggerimento: usa l'intervallo del tuo periodo di trading.")

    st.subheader("Residuo manuale (opzionale)")
    res_qty = st.number_input("Qty residua", value=15.0, step=1.0)
    res_avg = st.number_input("Prezzo medio residuo", value=201.0, step=0.1)
    res_target = st.number_input("Prezzo target residuo", value=220.0, step=0.1)

    st.divider()
    go = st.button("Calcola KPI", type="primary")

if not go:
    st.info("⬅️ Imposta l'intervallo e premi **Calcola KPI**. Le chiavi sono lette da Secrets.")
    st.stop()

# ---------------------- Fetch + Compute ----------------------
api_key, api_secret = get_keys()

start_ms = int(datetime.combine(start_date, datetime.min.time()).timestamp()*1000)
end_ms = int(datetime.combine(end_date, datetime.max.time()).timestamp()*1000)

trades_raw = fetch_trades(api_key, api_secret, PAIR_DEFAULT, start_ms, end_ms)
if not trades_raw:
    st.warning("Nessun trade trovato nell'intervallo scelto.")
    st.stop()

orders = df_from_trades(trades_raw, HUMAN_PAIR)

# FIFO chiusi + inventario aperto
trades, meta, inventory = fifo_trades_per_match(orders, HUMAN_PAIR)

# KPI (chiusi) + residuo manuale
residual_cfg = {"qty": res_qty, "avgCost": res_avg, "targetPrice": res_target}
result = compute_kpis(trades, meta, residual_cfg)

# Live price
try:
    live_price = fetch_price(PAIR_DEFAULT)
except Exception as e:
    live_price = None
    st.warning("Prezzo live non disponibile.")

# ======== Header date ========
header1, header2, _ = st.columns([1,1,2])
with header1:
    start = result['period']['start']
    st.metric("Data inizio", start.strftime('%d/%m/%Y %H:%M') if start else '-')
with header2:
    end = result['period']['end']
    st.metric("Data fine", end.strftime('%d/%m/%Y %H:%M') if end else '-')

st.divider()

# ======== KPI principali ========
col_left, col_mid, col_right = st.columns(3)
with col_left:
    st.metric("PNL totale (con residuo)", f"{format_number(result['pnl']['totalWithResidual'])} USDC")
    st.metric("Numero di trade", result["counts"]["trades"])
    st.metric("Success rate", f"{format_number(result['counts']['successRatePct'])}%")

with col_mid:
    st.metric("Totale USDC mossi", f"{format_number(result['sizes']['totalUSDCMoved'])} USDC")
    st.metric("Guadagno mensile medio", f"{format_number(result['monthlyAvgGain'])} USDC")
    st.metric("PNL medio per trade", f"{format_number(result['perTrade']['avgPnl'])} USDC")
    st.metric("Valore residuo a target", f"{format_number(result['pnl']['residual']['value'])} USDC")

with col_right:
    st.metric("Size media per trade", f"{format_number(result['sizes']['avgTradeUSDC'])} USDC")
    st.metric("Durata media trade", format_duration(result['perTrade']['avgDurationMin']))
    st.metric("PNL totale (senza residuo)", f"{format_number(result['pnl']['total'])} USDC")

st.caption(
    f"Residuo manuale: {int(result['pnl']['residual']['qty'])} SOL @ {format_number(result['pnl']['residual']['avgCost'])} → "
    f"target {format_number(result['pnl']['residual']['targetPrice'])} | "
    f"PnL residuo: {format_number(result['pnl']['residual']['pnl'])} USDC"
)

st.divider()

# ======== Posizioni aperte (inventario) + PnL live ========
st.subheader("📌 Posizioni aperte (live)")
if inventory:
    inv_df = pd.DataFrame(inventory)
    if live_price:
        inv_df["price_live"] = live_price
        inv_df["value_live"] = inv_df["qty"] * inv_df["price_live"]
        inv_df["cost"] = inv_df["qty"] * inv_df["avg_cost"]
        inv_df["pnl_live"] = inv_df["value_live"] - inv_df["cost"] - inv_df["fee_usdc"]
        inv_df["pnl_live_pct"] = np.where(inv_df["cost"] > 0, inv_df["pnl_live"] / inv_df["cost"] * 100.0, 0.0)
    inv_df.rename(columns={
        "date":"Data buy",
        "qty":"Qty (SOL)",
        "avg_cost":"Prezzo medio",
        "price_live":"Prezzo live",
        "value_live":"Valore live (USDC)",
        "pnl_live":"PnL live (USDC)",
        "pnl_live_pct":"PnL live (%)"
    }, inplace=True)
    st.dataframe(inv_df, use_container_width=True)
    if live_price:
        st.info(f"Prezzo live {HUMAN_PAIR}: **{format_number(live_price)} USDC**")
else:
    st.write("Nessuna posizione aperta.")

st.divider()

# ======== Miglior / Peggior trade (chiusi) ========
b = result.get("bestTrade")
w = result.get("worstTrade")
c1, c2 = st.columns(2)
with c1:
    st.subheader("🥇 Miglior trade (chiusi)")
    if b:
        st.write(f"PnL: **{format_number(b['pnl'])} USDC**  |  Size: {format_number(b['size_usdc'])} USDC")
        st.write(f"Buy: {pd.to_datetime(b['buy_date']).strftime('%d/%m/%Y %H:%M')} → Sell: {pd.to_datetime(b['sell_date']).strftime('%d/%m/%Y %H:%M')}")
        st.write(f"Durata: {format_duration(b['dur_min'])}")
    else:
        st.write("-")
with c2:
    st.subheader("🥶 Peggior trade (chiusi)")
    if w:
        st.write(f"PnL: **{format_number(w['pnl'])} USDC**  |  Size: {format_number(w['size_usdc'])} USDC")
        st.write(f"Buy: {pd.to_datetime(w['buy_date']).strftime('%d/%m/%Y %H:%M')} → Sell: {pd.to_datetime(w['sell_date']).strftime('%d/%m/%Y %H:%M')}")
        st.write(f"Durata: {format_duration(w['dur_min'])}")
    else:
        st.write("-")

st.divider()

# ======== Ultimi 10 giorni (chiusi) ========
st.subheader("🗓️ Ultimi 10 giorni (chiusi)")
if result['period']['end']:
    cutoff = result['period']['end'] - timedelta(days=10)
    last10 = [t for t in result['trades'] if t['sell_date'] >= cutoff]
    df10 = pd.DataFrame(last10)
    if not df10.empty:
        df10 = df10[["buy_date","sell_date","qty","buy_price","sell_price","size_usdc","pnl","dur_min"]].copy()
        df10.rename(columns={
            "buy_date":"Buy",
            "sell_date":"Sell",
            "qty":"Qty (SOL)",
            "buy_price":"Prezzo Buy",
            "sell_price":"Prezzo Sell",
            "size_usdc":"Size (USDC)",
            "pnl":"PnL (USDC)",
            "dur_min":"Durata"
        }, inplace=True)
        df10["Durata"] = df10["Durata"].map(format_duration)
        st.dataframe(df10, use_container_width=True)
        st.info(f"PnL ultimi 10gg: **{format_number(sum(x['pnl'] for x in last10))} USDC** su {len(last10)} trade")
    else:
        st.write("Nessun trade chiuso negli ultimi 10 giorni.")
else:
    st.write("Periodo non disponibile.")
