# AWS Setup — SKA Engine C

## Instance

| Parameter | Value |
|-----------|-------|
| Region | ap-northeast-1 (Tokyo) |
| Instance | m7i.xlarge (4 vCPU, 16 GB RAM) | 
| OS | Ubuntu 24.04 LTS x86_64 (ami-067bcf851477ebb78) | 
| Storage | 20 GB gp3 |


## Network

Outbound access required:

| Endpoint | Port | Protocol | Purpose |
|----------|------|----------|---------|
| `stream.binance.com` | 443 | WSS | Live tick stream (`@trade`) |
| `api.binance.com` | 443 | HTTPS | REST API (orders, balance) |


## Docker Environment

Same image as development. Required packages:

```
gcc 12.2+
g++ 12.2+
cmake 3.25+
python3 3.11+
make
```

## Build

```bash
cd source
cmake -B build
cmake --build build
```

Produces:
- `source/build/libska_bot.so` — combined C library (ska_engine + encoder + signal core)
- `source/build/test_ska_engine` — Phase 1 tests
- `source/build/test_phase2` — Phase 2 tests
- `source/build/test_phase3` — Phase 3 tests


## Verify

```bash
cd source
./build/test_ska_engine
./build/test_phase2
./build/test_phase3
python3 test/test_phase4.py
python3 test/validate.py
```

All tests must pass before running Phase 5.


## Binance API Keys

Required environment variables:

```bash
export BINANCE_API_KEY="..."
export BINANCE_SECRET_KEY="..."
```

Keys must have:
- Spot trading enabled
- IP whitelist set to the AWS instance elastic IP
- Ed25519 signing (used by `trading_bot_v3.py`)


## Monitoring (when ready)

| Component | URL | Purpose |
|-----------|-----|---------|
| Grafana | `https://grafana.<domain>` | Real-time dashboards — entropy chart, state panel, signal markers on price |
| QuestDB | `https://questdb.<domain>` | Time-series storage — tick, entropy, state, signal per tick |
| VS Code | `https://vscode.<domain>` | Remote development — code-server (browser-based IDE) |

Both run as Docker containers alongside the bot. Reverse proxy (Nginx/Caddy) routes subdomains to internal ports, handles HTTPS via Let's Encrypt. No code changes to the C engine — `ska_trading_bot.py` writes to QuestDB in addition to CSV.


## Security

- No inbound ports required for trading (bot initiates all connections)
- Grafana/QuestDB: bind to localhost or VPN only — not public
- API keys stored as environment variables, never in code
- Elastic IP recommended for Binance IP whitelist stability
