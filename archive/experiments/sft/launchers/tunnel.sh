#!/bin/bash
# Auto-reconnecting tunnel bundle, both directions in one ssh:
#   -R 28471: server localhost:28471 -> Mac clash :7897   (server gets internet)
#   -L 18000: Mac localhost:18000   -> server vLLM :8000  (Mac reaches the model API)
# Survives Mac sleep: ssh exits within ~45s of wake and the loop reconnects.
# Start detached:  nohup bash src/sft/tunnel.sh > /tmp/tunnel.log 2>&1 & disown
# Stop:            pkill -f "tunnel.s[h]" && pkill -f "R 2847[1]"
while true; do
  caffeinate -is ssh -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes -o ConnectTimeout=10 -N \
    -R 28471:127.0.0.1:7897 -L 18000:127.0.0.1:8000 NewGNN
  echo "[tunnel] disconnected $(date), retry in 10s" >&2
  sleep 10
done
