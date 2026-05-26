#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# 1. Create next-generation folder (genetic evolution from latest completed tournament)
python experiments/evolve.py

# 2. Run tournament — processes only folders without results.json
python 76-RL_tournament.py 2>/dev/null

# 3. Parse results, output METRIC lines, commit results.json
python experiments/parse_results.py
