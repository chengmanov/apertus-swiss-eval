"""Download the pinned Code of Obligations (OR) corpus from Fedlex.

Fetches the consolidated OR HTML per language (data/manifest.json), verifies
SHA-256 against data/checksums.json on later runs, and writes to data/raw/.
A silently changed upstream consolidation fails loudly rather than corrupting
the corpus.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
RAW = ROOT / "data" / "raw"
CHECKSUMS = ROOT / "data" / "checksums.json"
HEADERS = {"User-Agent": "apertus-legal-eval/0.1 (research corpus; contact: hello@sysf.io)"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(langs: list[str]) -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    base = manifest["base_url"]
    RAW.mkdir(parents=True, exist_ok=True)
    checksums = json.loads(CHECKSUMS.read_text(encoding="utf-8")) if CHECKSUMS.exists() else {}

    failures = 0
    for lang, rel in manifest["languages"].items():
        if langs and lang not in langs:
            continue
        name = f"vvg_{lang}.html"
        dest = RAW / name
        if dest.exists() and name in checksums and sha256(dest.read_bytes()) == checksums[name]:
            print(f"  ok       {name}")
            continue
        try:
            resp = requests.get(base + rel, headers=HEADERS, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  FAIL     {name}: {exc}")
            failures += 1
            continue
        if b'id="art_' not in resp.content:
            print(f"  FAIL     {name}: no article anchors (upstream format changed?)")
            failures += 1
            continue
        digest = sha256(resp.content)
        if name in checksums and checksums[name] != digest:
            print(f"  CHANGED  {name}: differs from pinned checksum -- review and bump the corpus version")
            failures += 1
            continue
        dest.write_bytes(resp.content)
        checksums[name] = digest
        print(f"  fetched  {name} ({len(resp.content)/1024:.0f} KiB)")
        time.sleep(0.5)

    CHECKSUMS.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n{len(checksums)} files pinned, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
