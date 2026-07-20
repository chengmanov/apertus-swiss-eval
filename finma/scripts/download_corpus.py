"""Download the pinned FINMA circular corpus defined in data/manifest.json.

Writes PDFs to data/raw/<slug>_<lang>.pdf and records SHA-256 checksums in
data/checksums.json. On first run the checksum file is created; on later runs
every download is verified against it, so a silently changed upstream document
fails loudly instead of corrupting the corpus.
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

HEADERS = {"User-Agent": "apertus-finma-eval/0.1 (research corpus download; contact: hello@sysf.io)"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(langs: list[str]) -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    base = manifest["base_url"]
    RAW.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, str] = {}
    if CHECKSUMS.exists():
        checksums = json.loads(CHECKSUMS.read_text(encoding="utf-8"))

    failures = 0
    for circ in manifest["circulars"]:
        for lang, rel in circ["langs"].items():
            if langs and lang not in langs:
                continue
            name = f"{circ['slug']}_{lang}.pdf"
            dest = RAW / name
            if dest.exists() and name in checksums and sha256(dest.read_bytes()) == checksums[name]:
                print(f"  ok       {name}")
                continue
            url = base + rel
            try:
                resp = requests.get(url, headers=HEADERS, timeout=60)
                resp.raise_for_status()
            except requests.RequestException as exc:
                print(f"  FAIL     {name}: {exc}")
                failures += 1
                continue
            if not resp.content.startswith(b"%PDF"):
                print(f"  FAIL     {name}: response is not a PDF (upstream changed?)")
                failures += 1
                continue
            digest = sha256(resp.content)
            if name in checksums and checksums[name] != digest:
                print(f"  CHANGED  {name}: upstream document differs from pinned checksum -- "
                      f"review and bump the corpus version before accepting")
                failures += 1
                continue
            dest.write_bytes(resp.content)
            checksums[name] = digest
            print(f"  fetched  {name} ({len(resp.content)/1024:.0f} KiB)")
            time.sleep(0.5)  # be polite to finma.ch

    CHECKSUMS.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n{len(checksums)} files pinned, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
