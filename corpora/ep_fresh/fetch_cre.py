#!/usr/bin/env python
"""Fetch EP "Verbatim report of proceedings" (CRE) XML for a set of sitting dates.

The CRE site is bot-gated (plain curl returns HTTP 202 / 0 bytes), so we render
with headless Chrome which wraps the raw XML in an xml-viewer div. We try the
upload date and +/-1 day (uploads sometimes land the morning after the sitting).
A fetch is VALID only if it parses to >=1 LG="EN" intervention with text.

Writes one cre_<YYYY-MM-DD>.xml per successful date into --cache-dir, plus a
manifest (cre_index.json) of date -> {url, status, n_en_turns}. REPORTS any
date that fails all candidates rather than dropping silently.

Run on a SLURM cpu node (web egress). Chrome at /usr/bin/google-chrome.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cre_parse import parse_cre  # noqa: E402

CHROME = "/usr/bin/google-chrome"
URL_TMPL = "https://www.europarl.europa.eu/doceo/document/CRE-10-{date}_EN.xml"


def candidate_dates(yyyymmdd: str, before: int = 3, after: int = 6) -> list[str]:
    """Plenary sittings cluster in session WEEKS, not on the YouTube upload date,
    so probe a window of WEEKDAYS around the upload date (Mon-Thu are sitting days).
    Order by distance from upload date so the nearest valid CRE wins ties."""
    d = dt.datetime.strptime(yyyymmdd, "%Y%m%d").date()
    offs = sorted(range(-before, after + 1), key=abs)
    out = []
    for off in offs:
        cand = d + dt.timedelta(days=off)
        if cand.weekday() <= 4:  # Mon-Fri (CRE rarely on weekends)
            out.append(cand.strftime("%Y-%m-%d"))
    return out


def chrome_fetch(url: str, out: Path, budget_ms: int = 22000) -> bool:
    # NOTE: do NOT wrap in `ulimit -v` — Chrome reserves >16 GiB of virtual
    # address space and SIGTRAPs under the cap. Bounded by timeout instead.
    udd = Path(tempfile.mkdtemp(prefix="chrome_cre_"))
    try:
        with out.open("w") as f:
            subprocess.run(
                [CHROME, "--headless", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
                 f"--user-data-dir={udd}", "--dump-dom", f"--virtual-time-budget={budget_ms}", url],
                stdout=f, stderr=subprocess.DEVNULL, timeout=budget_ms / 1000 + 40, check=False,
            )
        return out.stat().st_size > 5000
    except Exception:  # noqa: BLE001
        return False
    finally:
        shutil.rmtree(udd, ignore_errors=True)


def validate(path: Path) -> int:
    """Return n EN turns with text; 0 if unparseable/empty/bot-gated."""
    try:
        return len(parse_cre(path, lang="EN"))
    except Exception:  # noqa: BLE001
        return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", required=True, help="YYYYMMDD upload dates.")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    index = {}
    idx_path = cache / "cre_index.json"
    if idx_path.exists():
        index = json.loads(idx_path.read_text())
    prog = cache / "fetch_progress.log"

    def plog(msg):
        with prog.open("a") as f:
            f.write(msg + "\n")
        print(msg, flush=True)

    # Sittings cluster in session weeks, so the YouTube upload date is only a
    # hint. Build the UNION of candidate CRE dates across all upload dates and
    # fetch every one that resolves to a valid CRE. The builder later assigns
    # each video to its best-matching CRE by floor-English yield.
    cre_dates: dict[str, list[str]] = {}  # cre_date -> [upload dates that suggested it]
    for yyyymmdd in sorted(set(args.dates)):
        for cd in candidate_dates(yyyymmdd):
            cre_dates.setdefault(cd, []).append(yyyymmdd)
    print(f"{len(set(args.dates))} upload dates -> {len(cre_dates)} candidate CRE dates", flush=True)

    ok_dates, empty_dates, fail_dates = [], [], []
    for cre_date in sorted(cre_dates):
        # cached & valid?
        if cre_date in index and index[cre_date].get("status") in ("ok", "empty"):
            st = index[cre_date]
            tag = f"{st['n_en_turns']} EN turns" if st["status"] == "ok" else "no EN turns"
            plog(f"[{cre_date}] cached: {tag}")
            (ok_dates if st["status"] == "ok" else empty_dates).append(cre_date)
            continue
        url = URL_TMPL.format(date=cre_date)
        tmp = cache / f"cre_{cre_date}.xml"
        n, fetched = 0, False
        for attempt in range(args.retries + 1):
            if chrome_fetch(url, tmp):
                fetched = True
                n = validate(tmp)
                if n > 0:
                    break
            time.sleep(2 + 3 * attempt)
        if n > 0:
            index[cre_date] = {"cre_date": cre_date, "url": url, "file": tmp.name,
                               "n_en_turns": n, "status": "ok"}
            ok_dates.append(cre_date)
            plog(f"[{cre_date}] OK: {n} EN turns -> {tmp.name}")
        elif fetched:
            # page returned but no EN interventions = not a sitting day / no EN floor speech
            index[cre_date] = {"cre_date": cre_date, "url": url, "n_en_turns": 0, "status": "empty"}
            empty_dates.append(cre_date)
            tmp.unlink(missing_ok=True)
            plog(f"[{cre_date}] empty (page ok, 0 EN turns)")
        else:
            index[cre_date] = {"cre_date": cre_date, "url": url, "n_en_turns": 0, "status": "fetch_failed"}
            fail_dates.append(cre_date)
            plog(f"[{cre_date}] FETCH FAILED")
        idx_path.write_text(json.dumps(index, indent=2))

    print("=" * 60, flush=True)
    print(f"valid CREs: {len(ok_dates)} | empty(no-EN): {len(empty_dates)} | fetch-failed: {len(fail_dates)}", flush=True)
    print(f"valid dates: {ok_dates}", flush=True)
    if fail_dates:
        print(f"FETCH-FAILED (need manual retry): {fail_dates}", flush=True)
    idx_path.write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
