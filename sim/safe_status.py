"""Print only policy-safe fields from status.json (no coordinate readouts)."""
import json
import sys
from pathlib import Path

d = json.loads((Path(sys.argv[1]) / 'status.json').read_text())
print({k: d[k] for k in ['observation', 'steps', 'terminated',
                         'remaining_decisions', 'remaining_ticks'] if k in d})
