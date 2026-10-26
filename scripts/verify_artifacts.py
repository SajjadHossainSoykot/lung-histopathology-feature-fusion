#!/usr/bin/env python3
"""
Automated Artifact and Reproducibility Verification Script
Project: Stain-Robust Explainable Deep Feature Fusion for Lung Histopathology Classification
Author: S M Sajjad Hossain Soykot
Institution: Dept. of Information and Communication Technology (ICT), Islamic University, Bangladesh
"""

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

# Expected Cryptographic SHA-256 Checksums
EXPECTED_CHECKSUMS = {
    "splits/manifest_candidate.csv": "d931c9b11d03716c3c426f251bc75eaa2051002acca99d3ba66d59d964285dea",
    "splits/class_mapping.json": "4d14782118e3a438c13b209d7a594f2f9676d09cec25125b599d8c6d6c26a6a8",
    "metadata/lc25000_image_groups.csv": "2ff9e992f759f265dfc9aa2363de0dfcc8e6559884d8bb02a4c925214fdf7e23",
    "results/audit.json": "cd9aecfb8bfe64ca956b205e333ce9c9697dbfa88d4c46dc89c3ac088115fac0",
    "notebooks/00_Lung_Setup_and_Data_Checks.ipynb": "79cceafda7f48ce039d8b5d9827b29067aed5655e7ddb31dfc13457847b28dd0",
    "notebooks/H1_Lung_MobileNetV2_Baseline.ipynb": "ef3b600c720d4533740ff2d6370c947d0c309e5a01f877fc93be3b9251595c96",
    "notebooks/H2_Lung_EfficientNetB0_Baseline.ipynb": "c1779f0f776b0a8d1c9075b4c08d9fb5ed3d80052e1ac91b83e0d16e6b9446b9",
    "notebooks/H3_Lung_Deep_Feature_Fusion.ipynb": "6a0c35492b1474fbdc896571e77f3f5b4602c3a81d8d7e1d3976d63e15460efc",
    "notebooks/H4_Lung_Stain_Normalization_Augmentation_Robustness.ipynb": "884e159367422faeff9a5289736591c3ed3bf67135b5199e2328f8335204ec16",
    "notebooks/H_GradCAM_H3_Deep_Feature_Fusion.ipynb": "6208748dfdbdd9f8e250245d5fe349538e58c35e6ca9e98e6131a0488297e5b6",
    "notebooks/H_Final_Test_H3_Deep_Feature_Fusion.ipynb": "4d9844af57e242317832ead573cb12eff4d9330508c41039dfd96807a737d034",
}

EXPECTED_COUNTS = {
    "total_images": 15000,
    "train_images": 10394,
    "validation_images": 2273,
    "test_images": 2333,
    "classes": ["lung_aca", "lung_n", "lung_scc"],
    "total_groups": 740,
    "train_groups": 514,
    "validation_groups": 113,
    "test_groups": 113,
}


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checksums(repo_root: Path) -> bool:
    print("\n[1/4] Verifying Cryptographic SHA-256 Checksums...")
    all_passed = True
    for relative_path, expected_hash in EXPECTED_CHECKSUMS.items():
        file_path = repo_root / relative_path
        if not file_path.exists():
            print(f"  ❌ MISSING: {relative_path}")
            all_passed = False
            continue
        actual_hash = compute_sha256(file_path)
        if actual_hash == expected_hash:
            print(f"  ✅ MATCH: {relative_path}")
        else:
            print(f"  ❌ MISMATCH: {relative_path}")
            print(f"     Expected: {expected_hash}")
            print(f"     Actual:   {actual_hash}")
            all_passed = False
    return all_passed


def verify_partitions(repo_root: Path) -> bool:
    print("\n[2/4] Verifying Dataset Partitions and Split Integrity...")
    manifest_path = repo_root / "splits/manifest_candidate.csv"
    if not manifest_path.exists():
        print(f"  ❌ Manifest missing at {manifest_path}")
        return False

    split_counts = {"train": 0, "validation": 0, "test": 0}
    class_counts = {"lung_aca": 0, "lung_n": 0, "lung_scc": 0}
    total_rows = 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            split = row.get("split")
            label = row.get("class_name")
            if split in split_counts:
                split_counts[split] += 1
            if label in class_counts:
                class_counts[label] += 1

    passed = True
    if total_rows == EXPECTED_COUNTS["total_images"]:
        print(f"  ✅ Total manifest images: {total_rows} matches expected 15,000")
    else:
        print(f"  ❌ Total images mismatch: {total_rows} vs {EXPECTED_COUNTS['total_images']}")
        passed = False

    if (
        split_counts["train"] == EXPECTED_COUNTS["train_images"]
        and split_counts["validation"] == EXPECTED_COUNTS["validation_images"]
        and split_counts["test"] == EXPECTED_COUNTS["test_images"]
    ):
        print(f"  ✅ Split image distribution: {split_counts} verified")
    else:
        print(f"  ❌ Split distribution mismatch: {split_counts}")
        passed = False

    for cls_name in EXPECTED_COUNTS["classes"]:
        if class_counts[cls_name] == 5000:
            print(f"  ✅ Balanced class count for '{cls_name}': 5,000 images")
        else:
            print(f"  ❌ Class count imbalance: '{cls_name}' has {class_counts[cls_name]}")
            passed = False

    return passed


def verify_group_disjointness(repo_root: Path) -> bool:
    print("\n[3/4] Verifying Reconstructed Source-Group Independence (0 Leakage)...")
    manifest_path = repo_root / "splits/manifest_candidate.csv"
    if not manifest_path.exists():
        print(f"  ❌ Manifest missing at {manifest_path}")
        return False

    train_groups = set()
    val_groups = set()
    test_groups = set()

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            group_id = row["group_id"]
            if split == "train":
                train_groups.add(group_id)
            elif split == "validation":
                val_groups.add(group_id)
            elif split == "test":
                test_groups.add(group_id)

    total_groups = len(train_groups | val_groups | test_groups)
    tv_overlap = len(train_groups & val_groups)
    tt_overlap = len(train_groups & test_groups)
    vt_overlap = len(val_groups & test_groups)

    passed = True
    if total_groups == EXPECTED_COUNTS["total_groups"]:
        print(f"  ✅ Reconstructed source groups: {total_groups} unique groups verified")
    else:
        print(f"  ❌ Source group count mismatch: {total_groups} vs {EXPECTED_COUNTS['total_groups']}")
        passed = False

    if (
        len(train_groups) == EXPECTED_COUNTS["train_groups"]
        and len(val_groups) == EXPECTED_COUNTS["validation_groups"]
        and len(test_groups) == EXPECTED_COUNTS["test_groups"]
    ):
        print(f"  ✅ Group split counts: Train={len(train_groups)}, Val={len(val_groups)}, Test={len(test_groups)}")
    else:
        print(f"  ❌ Group split count mismatch: Train={len(train_groups)}, Val={len(val_groups)}, Test={len(test_groups)}")
        passed = False

    if tv_overlap == 0 and tt_overlap == 0 and vt_overlap == 0:
        print("  ✅ ZERO group overlap across all splits (Strict Group-Disjoint Partitions)")
    else:
        print(f"  ❌ DATA LEAKAGE DETECTED: Train/Val={tv_overlap}, Train/Test={tt_overlap}, Val/Test={vt_overlap}")
        passed = False

    return passed


def verify_test_metrics(repo_root: Path) -> bool:
    print("\n[4/4] Verifying Final Held-Out Test Evaluation Metrics...")
    metrics_path = repo_root / "results/H_Final_Test_H3_seed42/test_metrics.json"
    matrix_path = repo_root / "results/H_Final_Test_H3_seed42/test_confusion_matrix.csv"

    if not metrics_path.exists() or not matrix_path.exists():
        print("  ❌ Test metrics or confusion matrix file missing")
        return False

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    passed = True
    acc = metrics.get("accuracy", 0.0)
    macro_f1 = metrics.get("macro_f1", 0.0)
    n_images = metrics.get("n_images", 0)

    # Check accuracy ~ 96.2280%
    if abs(acc - 0.962280325760823) < 1e-5:
        print(f"  ✅ Test Accuracy verified: {acc * 100:.4f}%")
    else:
        print(f"  ❌ Test Accuracy mismatch: {acc}")
        passed = False

    # Check macro-f1 ~ 0.961961
    if abs(macro_f1 - 0.9619609902960032) < 1e-5:
        print(f"  ✅ Test Macro-F1 verified: {macro_f1:.6f}")
    else:
        print(f"  ❌ Test Macro-F1 mismatch: {macro_f1}")
        passed = False

    if n_images == 2333:
        print("  ✅ Test Sample Size: 2,333 images verified")
    else:
        print(f"  ❌ Test Sample Size mismatch: {n_images}")
        passed = False

    return passed


def main():
    repo_root = Path(__file__).resolve().parent.parent
    print("=" * 70)
    print(" LUNG HISTOPATHOLOGY FEATURE FUSION: ARTIFACT & REPRODUCIBILITY AUDIT ")
    print("=" * 70)
    print(f"Repository Root: {repo_root}")

    c_ok = verify_checksums(repo_root)
    p_ok = verify_partitions(repo_root)
    g_ok = verify_group_disjointness(repo_root)
    m_ok = verify_test_metrics(repo_root)

    print("\n" + "=" * 70)
    if c_ok and p_ok and g_ok and m_ok:
        print(" 🎉 ALL REPRODUCIBILITY CHECKS PASSED SUCCESSFULLY (STATUS: AUDIT PASS)")
        print("=" * 70)
        sys.exit(0)
    else:
        print(" ❌ AUDIT FAILED: ONE OR MORE INTEGRITY VERIFICATIONS FAILED")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
