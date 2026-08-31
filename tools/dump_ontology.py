#!/usr/bin/env python3
"""Print a human-readable snapshot of the pentest governance ontology.

  python tools/dump_ontology.py > /mnt/c/Users/josh/Desktop/pentest-ontology-reference.txt

Code (tools/pentest_ontology.py) is authoritative. Regenerate this file after
any change to the ACTIONS registry or the declarative tables.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pentest_ontology import dump_ontology_text  # noqa: E402

if __name__ == "__main__":
    sys.stdout.write(dump_ontology_text())
