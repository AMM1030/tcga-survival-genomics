"""
Fetch TCGA data from the cBioPortal REST API.
-----------------------------------------------
Pulls per-patient clinical attributes and mutation data for 4 cancer types,
saves to data/raw/ as CSVs.

cBioPortal REST API docs: https://docs.cbioportal.org/
The API is free, public, and requires no authentication.

Run:
    python src/fetch_tcga_data.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import pandas as pd
import requests
from tqdm import tqdm

BASE_URL = "https://www.cbioportal.org/api"
HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}

# Pan-Cancer Atlas 2018 — the standardised, harmonised TCGA dataset
STUDIES = {
    "HNSC": "hnsc_tcga_pan_can_atlas_2018",
    "LUAD": "luad_tcga_pan_can_atlas_2018",
    "BRCA": "brca_tcga_pan_can_atlas_2018",
    "COAD": "coadread_tcga_pan_can_atlas_2018",
}

# Well-known cancer driver genes across many tumor types.
# Source: OncoKB cancer gene list (top frequently-mutated drivers).
TOP_DRIVER_GENES = [
    "TP53", "KRAS", "PIK3CA", "EGFR", "BRAF", "APC", "PTEN", "MYC",
    "ARID1A", "ATM", "CDH1", "CDKN2A", "FBXW7", "GATA3", "KMT2C",
    "KMT2D", "MAP3K1", "NF1", "NOTCH1", "NRAS", "RB1", "SMAD4",
    "STK11", "VHL", "FAT1", "NSD1", "CASP8", "HRAS", "PIK3R1", "EP300",
]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get(path: str, **kwargs) -> dict:
    r = requests.get(f"{BASE_URL}{path}", headers=HEADERS, timeout=60, **kwargs)
    r.raise_for_status()
    return r.json()


def _post(path: str, json_body: dict, **kwargs) -> list:
    r = requests.post(f"{BASE_URL}{path}", headers=HEADERS, json=json_body,
                      timeout=60, **kwargs)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Clinical data
# ---------------------------------------------------------------------------

def fetch_patient_clinical(study_id: str) -> pd.DataFrame:
    """Fetch all PATIENT-level clinical attributes and pivot to wide form."""
    data = _get(f"/studies/{study_id}/clinical-data",
                params={"clinicalDataType": "PATIENT", "projection": "DETAILED"})
    df = pd.DataFrame(data)
    if df.empty:
        return df
    wide = (df.pivot_table(index="patientId",
                            columns="clinicalAttributeId",
                            values="value",
                            aggfunc="first")
              .reset_index())
    wide.columns.name = None
    return wide


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------

def fetch_gene_entrez_ids(gene_symbols: List[str]) -> List[int]:
    """Translate Hugo gene symbols to Entrez IDs (the API requires Entrez)."""
    data = _post("/genes/fetch", gene_symbols,
                 params={"geneIdType": "HUGO_GENE_SYMBOL"})
    return [g["entrezGeneId"] for g in data]


def fetch_mutations(study_id: str, entrez_ids: List[int]) -> pd.DataFrame:
    """Fetch mutations for given genes across all samples in the study."""
    profile_id = f"{study_id}_mutations"
    sample_list_id = f"{study_id}_all"
    payload = {
        "entrezGeneIds": entrez_ids,
        "sampleListId": sample_list_id,
    }
    data = _post(f"/molecular-profiles/{profile_id}/mutations/fetch",
                 payload, params={"projection": "DETAILED"})
    df = pd.DataFrame(data)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)

    # Translate gene symbols to entrez IDs once
    print(f"Resolving Entrez IDs for {len(TOP_DRIVER_GENES)} genes...")
    entrez_ids = fetch_gene_entrez_ids(TOP_DRIVER_GENES)
    print(f"  -> {len(entrez_ids)} genes resolved")

    summary = []
    for cancer, study_id in tqdm(STUDIES.items(), desc="Studies"):
        print(f"\n=== {cancer}  ({study_id}) ===")

        # Clinical
        try:
            clin = fetch_patient_clinical(study_id)
            clin["cancer_type"] = cancer
            clin.to_csv(out / f"{cancer}_clinical.csv", index=False)
            n_pts = len(clin)
            print(f"  Clinical : {n_pts:>4d} patients  ({clin.shape[1]} cols)")
        except Exception as e:
            print(f"  Clinical : ERROR {e}")
            n_pts = 0

        # Mutations
        try:
            muts = fetch_mutations(study_id, entrez_ids)
            muts["cancer_type"] = cancer
            muts.to_csv(out / f"{cancer}_mutations.csv", index=False)
            n_muts = len(muts)
            print(f"  Mutations: {n_muts:>4d} records")
        except Exception as e:
            print(f"  Mutations: ERROR {e}")
            n_muts = 0

        summary.append({"cancer": cancer, "n_patients": n_pts, "n_mutations": n_muts})
        time.sleep(1)  # be polite to the public API

    # Save summary
    pd.DataFrame(summary).to_csv(out / "_fetch_summary.csv", index=False)
    print("\n--- Done ---")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()