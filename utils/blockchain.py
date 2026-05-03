"""
utils/blockchain.py — Etherscan REST API + Web3.py live blockchain helpers.
"""
import os
import logging
from datetime import datetime

import requests
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

ETHERSCAN_BASE = "https://api.etherscan.io/api"
DEFAULT_RPC    = "https://eth.mainnet.public.blastapi.io"

# ── API key resolution ─────────────────────────────────────────────────────
# Priority: ETHERSCAN_API_KEY secret → ETHERSCAN_API_KEY_DEFAULT env var → built-in fallback
_BUILTIN_FALLBACK = "41NENJ1WX2RNDWTFIFUJXSNECMQ8P9KB45"
ETHERSCAN_API_KEY: str = (
    os.environ.get("ETHERSCAN_API_KEY")
    or os.environ.get("ETHERSCAN_API_KEY_DEFAULT")
    or _BUILTIN_FALLBACK
)

# ── ERC-20 Transfer event signature ──────────────────────────────────────
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

# ── Known DeFi / protocol addresses (lower-case) ─────────────────────────
DEFI_PROTOCOLS: dict[str, str] = {
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d": "Uniswap V2 Router",
    "0xe592427a0aece92de3edee1f18e0157c05861564": "Uniswap V3 Router",
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45": "Uniswap Universal Router",
    "0xef1c6e67703c7bd7107eed8303fbe6ec2554bf6b": "Uniswap Universal Router 2",
    "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9": "Aave V2 Lending Pool",
    "0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2": "Aave V3 Pool",
    "0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b": "Compound Comptroller",
    "0x7be8076f4ea4a4ad08075c2508e481d6c946d12b": "OpenSea Wyvern V1",
    "0x7f268357a8c2552623316e2562d90e642bb538e5": "OpenSea Wyvern V2",
    "0x00000000006c3852cbef3e08e8df289169ede581": "OpenSea Seaport 1.1",
    "0x00000000000000adc04c56bf30ac9d3c0aaf14dc": "OpenSea Seaport 1.5",
    "0x1111111254fb6c44bac0bed2854e76f90643097d": "1inch V4",
    "0x1111111254eeb25477b68fb85ed929f73a960582": "1inch V5",
    "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f": "SushiSwap Router",
    "0xd51a44d3fae010294c616388b506acda1bfaae46": "Curve Tricrypto2",
    "0xa5407eae9ba41422680e2e00537571bcc53efbfd": "Curve sUSD Pool",
    "0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7": "Curve 3Pool",
    "0x881d40237659c251811cec9c364ef91dc08d300c": "MetaMask Swap Router",
    "0xba12222222228d8ba445958a75a0704d566bf2c8": "Balancer Vault",
    "0xdef1c0ded9bec7f1a1670819833240f027b25eff": "0x Exchange Proxy",
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "Wrapped ETH (WETH)",
    "0x6b175474e89094c44da98b954eedeac495271d0f": "DAI Stablecoin",
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "USD Coin (USDC)",
    "0xdac17f958d2ee523a2206206994597c13d831ec7": "Tether (USDT)",
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": "Wrapped Bitcoin (WBTC)",
    "0x514910771af9ca656af840dff83e8264ecf986ca": "Chainlink (LINK)",
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984": "Uniswap Token (UNI)",
    "0xc18360217d8f7ab5e7c516566761ea12ce7f9d72": "ENS Token",
    "0x57f1887a8bf19b14fc0df6fd9b2acc9af147ea85": "ENS Registrar",
    "0xae7ab96520de3a18e5e111b5eaab095312d7fe84": "Lido stETH",
    "0x5a98fcbea516cf06857215779fd812ca3bef1b32": "Lido (LDO)",
    "0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2": "MakerDAO (MKR)",
    "0x0bc529c00c6401aef6d220be8c6ea1667f6ad93e": "Yearn Finance (YFI)",
    "0x408e41876cccdc0f92210600ef50372656052a38": "Ren (REN)",
}

# ── Known event topics (topic0) ───────────────────────────────────────────
KNOWN_EVENTS: dict[str, str] = {
    TRANSFER_TOPIC: "Transfer",
    "0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925": "Approval",
    "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822": "Swap (Uniswap V2)",
    "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67": "Swap (Uniswap V3)",
    "0xe1fffcc4923d04b559f4d29a8bfc6cda04eb5b0d3c460751c2402c5c5cc9109c": "Deposit (WETH)",
    "0x7fcf532c15f0a6db0bd6d0e038bea71d30d808c7d98cb3bf7268a95bf5081b65": "Withdrawal (WETH)",
    "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef": "Transfer (ERC-20)",
}


# ── Web3.py helpers ───────────────────────────────────────────────────────

def get_web3():
    """Return a connected Web3 instance or None."""
    try:
        from web3 import Web3
        rpc_url = os.environ.get("WEB3_PROVIDER_URL", DEFAULT_RPC)
        w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 12}))
        if w3.is_connected():
            return w3
        logger.warning("Web3 connected but is_connected() returned False")
    except ImportError:
        logger.warning("web3 package not installed")
    except Exception as exc:
        logger.warning(f"Web3 connection error: {exc}")
    return None


@st.cache_data(ttl=120)
def fetch_event_logs_web3(address: str, blocks_back: int = 8000) -> pd.DataFrame:
    """
    Fetch ERC-20 Transfer event logs for `address` using Web3.py eth_getLogs.
    Looks back `blocks_back` blocks (≈ 1 day at 12s/block).
    Returns decoded DataFrame or empty DataFrame on failure.
    """
    w3 = get_web3()
    if w3 is None:
        return pd.DataFrame()

    try:
        latest    = w3.eth.block_number
        from_block = max(0, latest - blocks_back)
        addr_lower = address.lower()
        padded     = "0x" + "0" * 24 + addr_lower[2:]

        rows = []
        for direction, topics in [
            ("Received", [TRANSFER_TOPIC, None, padded]),
            ("Sent",     [TRANSFER_TOPIC, padded, None]),
        ]:
            try:
                logs = w3.eth.get_logs({
                    "fromBlock": from_block,
                    "toBlock":   "latest",
                    "topics":    topics,
                })
                for log in logs:
                    try:
                        from_addr = "0x" + log["topics"][1].hex()[-40:]
                        to_addr   = "0x" + log["topics"][2].hex()[-40:]
                        raw_data  = log["data"]
                        val_hex   = raw_data.hex() if isinstance(raw_data, bytes) else raw_data
                        value_int = int(val_hex, 16) if val_hex not in ("0x", "0x0", "") else 0
                        contract  = log["address"].lower()
                        tx_hash   = log["transactionHash"].hex()
                        blk_num   = log.get("blockNumber", 0)
                        token_label = DEFI_PROTOCOLS.get(contract, "Unknown Token")
                        rows.append({
                            "Direction": direction,
                            "From":      from_addr,
                            "To":        to_addr,
                            "Value (Wei)": value_int,
                            "Contract":  log["address"],
                            "Token":     token_label,
                            "Block":     blk_num,
                            "TX Hash":   tx_hash[:18] + "…",
                            "Full Hash": tx_hash,
                            "Event":     "Transfer",
                        })
                    except Exception:
                        pass
            except Exception as exc:
                logger.warning(f"get_logs ({direction}) failed: {exc}")

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values("Block", ascending=False)
        df["Etherscan"] = df["Full Hash"].apply(
            lambda h: f"https://etherscan.io/tx/{h}"
        )
        return df.drop(columns=["Full Hash"])
    except Exception as exc:
        logger.warning(f"fetch_event_logs_web3 error: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def get_latest_block_info() -> dict | None:
    """Return basic info about the latest Ethereum block."""
    w3 = get_web3()
    if w3 is None:
        return None
    try:
        blk = w3.eth.get_block("latest")
        return {
            "number":    blk["number"],
            "timestamp": datetime.fromtimestamp(blk["timestamp"]).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "tx_count":  len(blk["transactions"]),
            "gas_used":  blk["gasUsed"],
            "gas_limit": blk["gasLimit"],
            "utilisation": round(blk["gasUsed"] / blk["gasLimit"] * 100, 1),
        }
    except Exception as exc:
        logger.warning(f"get_latest_block_info error: {exc}")
        return None


# ── Etherscan REST helpers ────────────────────────────────────────────────

def _etherscan_get(params: dict) -> dict | None:
    key = ETHERSCAN_API_KEY
    if key:
        params["apikey"] = key
    try:
        r = requests.get(ETHERSCAN_BASE, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning(f"Etherscan error: {exc}")
        return None


@st.cache_data(ttl=300)
def fetch_eth_balance(address: str) -> float | None:
    data = _etherscan_get({"module": "account", "action": "balance",
                           "address": address, "tag": "latest"})
    if data and data.get("status") == "1":
        try:
            return int(data["result"]) / 1e18
        except Exception:
            return None
    return None


@st.cache_data(ttl=300)
def fetch_transactions(address: str, limit: int = 25) -> pd.DataFrame:
    data = _etherscan_get({
        "module": "account", "action": "txlist",
        "address": address, "startblock": 0, "endblock": 99999999,
        "page": 1, "offset": limit, "sort": "desc",
    })
    if data and data.get("status") == "1" and isinstance(data.get("result"), list):
        rows = []
        for tx in data["result"]:
            val_eth       = int(tx.get("value", 0)) / 1e18
            gas_gwei      = int(tx.get("gasPrice", 0)) / 1e9
            ts            = int(tx.get("timeStamp", 0))
            dt            = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "—"
            protocol      = DEFI_PROTOCOLS.get(tx.get("to", "").lower(), "")
            tx_hash       = tx.get("hash", "")
            rows.append({
                "Time":         dt,
                "Hash":         tx_hash[:18] + "…",
                "Full Hash":    tx_hash,
                "From":         tx.get("from", ""),
                "To":           tx.get("to", ""),
                "Value (ETH)":  round(val_eth, 6),
                "Gas (Gwei)":   round(gas_gwei, 2),
                "Status":       "✅" if tx.get("isError", "0") == "0" else "❌",
                "Protocol":     protocol if protocol else "—",
                "Block":        int(tx.get("blockNumber", 0)),
            })
        return pd.DataFrame(rows)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def fetch_token_transfers(address: str, limit: int = 20) -> pd.DataFrame:
    data = _etherscan_get({
        "module": "account", "action": "tokentx",
        "address": address, "page": 1, "offset": limit, "sort": "desc",
    })
    if data and data.get("status") == "1" and isinstance(data.get("result"), list):
        rows = []
        for tx in data["result"]:
            decimals = int(tx.get("tokenDecimal", 18) or 18)
            val      = int(tx.get("value", 0)) / (10 ** decimals)
            ts       = int(tx.get("timeStamp", 0))
            dt       = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "—"
            rows.append({
                "Time":     dt,
                "Token":    tx.get("tokenSymbol", "?"),
                "Name":     tx.get("tokenName", ""),
                "Value":    round(val, 6),
                "From":     tx.get("from", ""),
                "To":       tx.get("to", ""),
                "Contract": tx.get("contractAddress", ""),
                "Hash":     tx.get("hash", "")[:18] + "…",
            })
        return pd.DataFrame(rows)
    return pd.DataFrame()


@st.cache_data(ttl=600)
def check_is_contract(address: str) -> tuple[bool, str]:
    """Returns (is_contract, contract_name)."""
    data = _etherscan_get({"module": "proxy", "action": "eth_getCode",
                           "address": address, "tag": "latest"})
    is_contract = bool(data and len(data.get("result", "0x")) > 4)
    contract_name = ""
    if is_contract:
        src = _etherscan_get({"module": "contract", "action": "getsourcecode",
                               "address": address})
        if src and isinstance(src.get("result"), list) and src["result"]:
            contract_name = src["result"][0].get("ContractName", "")
    return is_contract, contract_name


@st.cache_data(ttl=300)
def fetch_tx_count(address: str) -> int | None:
    data = _etherscan_get({"module": "proxy", "action": "eth_getTransactionCount",
                           "address": address, "tag": "latest"})
    if data and "result" in data:
        try:
            return int(data["result"], 16)
        except Exception:
            return None
    return None


def detect_protocols_from_txs(tx_df: pd.DataFrame) -> list[str]:
    if tx_df.empty or "To" not in tx_df.columns:
        return []
    return sorted({DEFI_PROTOCOLS[a] for a in tx_df["To"].str.lower() if a in DEFI_PROTOCOLS})
