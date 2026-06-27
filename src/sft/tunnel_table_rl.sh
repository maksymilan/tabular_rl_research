#!/bin/bash
# Auto-reconnecting tunnel for table_rl (amax, 2x RTX 4090D), both directions:
#   -R 28472: table_rl localhost:28472 -> Mac clash :7897   (table_rl gets internet)
#   -L 18001: Mac localhost:18001      -> table_rl vLLM :8000 (Mac reaches the model API)
# Distinct ports from the NewGNN tunnel (28471/18000) so both can run at once.
# Survives Mac sleep: ssh exits within ~45s of wake and the loop reconnects.
# Start detached:  nohup bash src/sft/tunnel_table_rl.sh > /tmp/tunnel_table_rl.log 2>&1 & disown
# Stop:            pkill -f "tunnel_table_r[l]" && pkill -f "R 2847[2]"
# On table_rl, export http(s)_proxy=http://127.0.0.1:28472 to use the egress.
while true; do
  caffeinate -is ssh -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes -o ConnectTimeout=10 -N \
    -R 28472:127.0.0.1:7897 -L 18001:127.0.0.1:8000 table_rl
  echo "[tunnel-table_rl] disconnected $(date), retry in 10s" >&2
  sleep 10
done
