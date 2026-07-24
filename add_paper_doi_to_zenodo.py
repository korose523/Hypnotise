#!/usr/bin/env python3
"""
add_paper_doi_to_zenodo.py
==========================
After the PLOS ONE manuscript is ACCEPTED, add its formal paper DOI to the
already-published Zenodo software record as an `isDocumentedBy` related
identifier. This links the archived code/results to the published article.

The Zenodo record (v1.0.0) is published/locked for *files*, but Zenodo still
permits **metadata edits** to a published deposition via the deposit API — the
edit updates the same record (no new version is created).

USAGE
-----
  # 1) set the token as an environment variable (NEVER hardcode / never commit)
  export ZENODO_TOKEN="<your_zenodo_token>"

  # 2) dry-run first to see the planned change
  python add_paper_doi_to_zenodo.py --paper-doi 10.1371/journal.pone.0123456 --dry-run

  # 3) apply it
  python add_paper_doi_to_zenodo.py --paper-doi 10.1371/journal.pone.0123456

  # optional overrides:
  #   --deposition-id 21531273            (default; this project's record)
  #   ZENODO_DEPOSITION_ID env var

If the PUT is rejected because the record is fully locked, the script prints
the `actions/newversion` fallback steps.

SECURITY: the token is read ONLY from the environment. This file contains no
secret and is safe to commit.
"""
import os
import re
import sys
import json
import argparse
import requests

BASE = "https://zenodo.org/api"


def normalize_type(identifier: str) -> str:
    """Zenodo returns identifier_type=None for stored entries; re-derive it."""
    s = (identifier or "").strip()
    if re.match(r"^10\.\d{4,9}/", s) or s.lower().startswith("10."):
        return "doi"
    return "url"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper-doi", required=True,
                    help="Accepted paper DOI, e.g. 10.1371/journal.pone.0123456")
    ap.add_argument("--deposition-id",
                    default=os.environ.get("ZENODO_DEPOSITION_ID", "21531273"),
                    help="Zenodo deposition id (default 21531273 = this project)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch + print the planned change, do not write")
    args = ap.parse_args()

    TOK = os.environ.get("ZENODO_TOKEN")
    if not TOK:
        sys.exit("ERROR: set the ZENODO_TOKEN environment variable first.")
    P = {"access_token": TOK}

    # 1) fetch current deposition
    r = requests.get(f"{BASE}/deposit/depositions/{args.deposition_id}",
                     params=P, timeout=30)
    if not r.ok:
        sys.exit(f"GET failed {r.status_code}: {r.text[:300]}")
    dep = r.json()
    md = dep.get("metadata", {})
    existing = md.get("related_identifiers", [])

    # 2) normalize existing entries (fix any missing identifier_type)
    norm = []
    for ri in existing:
        it = ri.get("identifier_type") or normalize_type(ri.get("identifier"))
        norm.append({
            "relation": ri.get("relation"),
            "identifier": ri.get("identifier"),
            "identifier_type": it,
        })

    # 3) build the new isDocumentedBy entry (skip if already present)
    new_entry = {
        "relation": "isDocumentedBy",
        "identifier": args.paper_doi.strip(),
        "identifier_type": normalize_type(args.paper_doi),
    }
    dup = any(x.get("relation") == new_entry["relation"] and
              x.get("identifier") == new_entry["identifier"] for x in norm)
    if dup:
        print("Already present; nothing to do.\n")
        print(json.dumps(norm, indent=2, ensure_ascii=False))
        return

    norm.append(new_entry)
    md["related_identifiers"] = norm

    if args.dry_run:
        print("DRY RUN — would PUT metadata.related_identifiers:\n")
        print(json.dumps(norm, indent=2, ensure_ascii=False))
        return

    # 4) write back
    r = requests.put(f"{BASE}/deposit/depositions/{args.deposition_id}",
                     params=P, json={"metadata": md}, timeout=60)
    if r.ok:
        print("OK — isDocumentedBy added.\n")
        print("Record:",
              dep.get("links", {}).get("record_html") or
              dep.get("links", {}).get("record"))
        print("Updated related_identifiers:")
        for x in r.json().get("metadata", {}).get("related_identifiers", []):
            print("  ", x.get("relation"), "|",
                  x.get("identifier_type"), "|", x.get("identifier"))
    else:
        print("PUT FAILED", r.status_code, r.text[:400])
        print("\nIf the record is locked, create a new version instead:")
        print(f'  POST {BASE}/deposit/depositions/{args.deposition_id}/actions/newversion?access_token=...')
        print("  then edit the new draft's metadata (add this isDocumentedBy entry) and publish.")


if __name__ == "__main__":
    main()
