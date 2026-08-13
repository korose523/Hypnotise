#!/usr/bin/env python3
"""
publish_zenodo_v1.0.12.py
=========================
Publish a NEW Zenodo version of the Hypnotise software archive containing the
v1.0.12 manuscript (21 terminology-consistency fixes) + code.

Flow (Zenodo deposit API, mirrors add_paper_doi_to_zenodo.py style):
  1) POST .../depositions/21920943/actions/newversion   -> new draft deposition
  2) DELETE each inherited file from the draft
  3) PUT the new tarball into the draft's bucket
  4) PUT metadata (version=v1.0.12 + changelog note)
  5) POST .../actions/publish   (only with --publish)

SECURITY: token is read ONLY from the ZENODO_TOKEN env var. Never hardcode.
"""
import os
import sys
import json
import argparse
import requests

BASE = "https://zenodo.org/api"
OLD_ID = "21920943"          # current deposition (latest published v1.0.11; v1.0.12 draft to be published this round)
TARBALL = r"E:\universal_bci_hypnosis\hypnotise-v1.0.12.tar.gz"
FNAME = "hypnotise-v1.0.12.tar.gz"
VERSION = "v1.0.12"
CHANGELOG = (
    "<p>v1.0.12: applied a systematic terminology-consistency and technical-expression review "
    "of the manuscript plus a full audit pass. Changes include: en/US spelling unification; "
    "within-/cross-dataset terminology unification; first-use definitions for WFSC, ZS and RF; "
    "eigendecomposition renamed to generalized eigenvalue decomposition; Table 14 reproducibility "
    "filename correction; dataset name normalized to SEED-IV (D10); and complete page numbers for "
    "AAAI references [12] (pp. 3490-3497) and [14] (pp. 2058-2065). Companion code at GitHub tag "
    "v1.0.12.</p>"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--publish", action="store_true",
                    help="Actually publish. Without it, the script stages the draft but does NOT publish.")
    args = ap.parse_args()

    TOK = os.environ.get("ZENODO_TOKEN")
    if not TOK:
        sys.exit("ERROR: set the ZENODO_TOKEN environment variable first.")

    P = {"access_token": TOK}
    H = {"Authorization": f"Bearer {TOK}"}

    def jget(url):
        r = requests.get(url, params=P, timeout=60); r.raise_for_status(); return r.json()
    def jpost(url, json=None):
        r = requests.post(url, params=P, json=json, timeout=60); r.raise_for_status(); return r.json()
    def jput(url, json=None):
        r = requests.put(url, params=P, json=json, timeout=60); r.raise_for_status(); return r.json()
    def jdel(url):
        r = requests.delete(url, params=P, timeout=60); r.raise_for_status(); return r.status_code

    # 1) new version
    print("[1] creating new version from deposition", OLD_ID)
    newdep = jpost(f"{BASE}/deposit/depositions/{OLD_ID}/actions/newversion")
    new_id = newdep["id"]
    print("    new draft deposition id:", new_id)

    # 2) refresh draft (files + bucket + metadata)
    newdep = jget(f"{BASE}/deposit/depositions/{new_id}")
    bucket = newdep["links"]["bucket"]
    files = newdep.get("files", [])
    print("[2] draft bucket:", bucket)
    print("    inherited files:", [f.get("key") for f in files])

    # 3) delete inherited files
    for f in files:
        url = f.get("links", {}).get("self") or f"{bucket}/{f.get('id')}"
        print("    deleting", f.get("key"), "->", jdel(url))

    # 4) upload new tarball
    print("[3] uploading", FNAME, "(%d bytes)" % os.path.getsize(TARBALL))
    with open(TARBALL, "rb") as fh:
        r = requests.put(f"{bucket}/{FNAME}", headers=H, data=fh, timeout=180)
    r.raise_for_status()
    print("    upload status:", r.status_code)

    # 5) update metadata
    md = newdep["metadata"]
    md["version"] = VERSION
    desc = md.get("description", "")
    if VERSION not in desc:
        md["description"] = desc + CHANGELOG
    print("[4] setting metadata version ->", VERSION)
    upd = jput(f"{BASE}/deposit/depositions/{new_id}", json={"metadata": md})
    print("    metadata updated; version now:", upd.get("metadata", {}).get("version"))

    draft_url = newdep.get("links", {}).get("html") or f"https://zenodo.org/deposit/{new_id}"
    print("    DRAFT:", draft_url)

    # 6) publish
    if not args.publish:
        print("\nDRY-STAGE complete (draft created, file uploaded, metadata set).")
        print("Re-run with --publish to publish the new version.")
        return

    print("[5] PUBLISHING...")
    pub = jpost(f"{BASE}/deposit/depositions/{new_id}/actions/publish")
    print("    PUBLISHED.")
    print("    DOI :", pub.get("doi"))
    print("    URL :", pub.get("links", {}).get("record_html") or pub.get("links", {}).get("record"))
    print("    concept DOI still resolves to latest: https://doi.org/10.5281/zenodo.21531272")


if __name__ == "__main__":
    main()
