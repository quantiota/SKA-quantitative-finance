# AWS Setup — SKA Engine C

## Instance

| Parameter | Value |
|-----------|-------|
| Region | ap-northeast-1 (Tokyo) |
| Instance | m7i.xlarge (4 vCPU, 16 GB RAM) |
| OS | Ubuntu 24.04 LTS x86_64 (ami-067bcf851477ebb78) |
| Storage | 20 GB gp3 |


## Latency Test

Measures delay between Binance trade timestamp and local clock.

```bash
~/venv/bin/pip install websocket-client
```

```python
import websocket
import json
import time

count = 0
latencies = []

def on_message(ws, message):
    global count, latencies
    data = json.loads(message)
    trade_time = data['T']
    now = int(time.time() * 1000)
    latency = now - trade_time
    latencies.append(latency)
    count += 1
    if count <= 5 or count % 50 == 0:
        print(f'tick {count}: latency={latency}ms')
    if count >= 200:
        import statistics
        print(f'--- {count} ticks ---')
        print(f'min={min(latencies)}ms  median={statistics.median(latencies):.0f}ms  mean={statistics.mean(latencies):.0f}ms  max={max(latencies)}ms')
        ws.close()

def on_open(ws):
    print('Connected to Binance XRPUSDT@trade')

ws = websocket.WebSocketApp(
    'wss://stream.binance.com:9443/ws/xrpusdt@trade',
    on_message=on_message,
    on_open=on_open
)
ws.run_forever()
```

### Results (m7i.xlarge, ap-northeast-1)

| Metric | Value |
|--------|-------|
| Min | 2ms |
| Median | 3ms |
| Mean | 3ms |
| Max | 5ms |

Full pipeline latency: **~3ms** (network 3ms + C engine 0.01ms).


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

All services run as Docker containers managed by `docker-compose`. Reverse proxy (Nginx/Caddy) routes subdomains to internal ports, handles HTTPS via Let's Encrypt. No code changes to the C engine — `ska_trading_bot.py` writes to QuestDB in addition to CSV.




## Security

- No inbound ports required for trading (bot initiates all connections)
- Grafana/QuestDB: bind to localhost or VPN only — not public
- API keys stored as environment variables, never in code
- Elastic IP recommended for Binance IP whitelist stability
