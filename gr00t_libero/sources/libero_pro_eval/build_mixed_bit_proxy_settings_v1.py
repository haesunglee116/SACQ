#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_sensitivity(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["module_name"]] = {
                "module_name": row["module_name"],
                "layer": int(float(row["layer"])),
                "module_type": row["module_type"],
                "rank": int(float(row["rank"])),
                "err": float(row["dit_input_global_rel_rms_error"]),
            }
    return out


def module_param_count(module_name: str) -> int:
    # Qwen3 language model active linears in this project have regular shapes.
    if module_name.endswith(".self_attn.k_proj") or module_name.endswith(".self_attn.v_proj"):
        return 1024 * 2048
    if module_name.endswith(".self_attn.q_proj") or module_name.endswith(".self_attn.o_proj"):
        return 2048 * 2048
    if module_name.endswith(".mlp.gate_proj") or module_name.endswith(".mlp.up_proj"):
        return 6144 * 2048
    if module_name.endswith(".mlp.down_proj"):
        return 2048 * 6144
    raise ValueError(f"Unknown module shape for {module_name}")


def load_rows(sens2_csv: Path, sens3_csv: Path) -> list[dict[str, Any]]:
    s2 = load_sensitivity(sens2_csv)
    s3 = load_sensitivity(sens3_csv)
    rows = []
    for name, row2 in s2.items():
        row3 = s3[name]
        err2 = float(row2["err"])
        err3 = float(row3["err"])
        rows.append(
            {
                "module_name": name,
                "module_type": row2["module_type"],
                "layer": row2["layer"],
                "params": module_param_count(name),
                "err2": err2,
                "err3": err3,
                "gain23": max(0.0, err2 - err3),
            }
        )
    return rows


def allocate(rows: list[dict[str, Any]], *, target_avg: float, rule: str, top4: int = 0) -> dict[str, int]:
    if rule == "sens2":
        ordered = sorted(rows, key=lambda r: r["err2"], reverse=True)
    elif rule == "gain23":
        ordered = sorted(rows, key=lambda r: (r["gain23"] / max(1, r["params"]), r["gain23"]), reverse=True)
    else:
        raise ValueError(rule)

    total_params = sum(r["params"] for r in rows)
    extra_budget = int(round((target_avg - 2.0) * total_params))
    assigned = {r["module_name"]: 2 for r in rows}
    used_extra = 0

    # Budget-neutral top4: a 2bit->4bit promotion consumes 2 extra bits per parameter.
    for r in ordered[:top4]:
        cost = 2 * r["params"]
        if used_extra + cost <= extra_budget:
            assigned[r["module_name"]] = 4
            used_extra += cost

    for r in ordered:
        name = r["module_name"]
        if assigned[name] != 2:
            continue
        cost = r["params"]
        if used_extra + cost <= extra_budget:
            assigned[name] = 3
            used_extra += cost

    return assigned


def specs_from_assignment(assignment: dict[str, int]) -> str:
    # Quant compiler uses VLM_BITS=2 as base, so only non-2bit modules need forcing.
    return ";".join(f"{name}={bits}" for name, bits in assignment.items() if bits != 2)


def assignment_stats(rows: list[dict[str, Any]], assignment: dict[str, int]) -> dict[str, Any]:
    params_by_name = {r["module_name"]: r["params"] for r in rows}
    total = sum(params_by_name.values())
    weighted_bits = sum(params_by_name[name] * bits for name, bits in assignment.items())
    return {
        "avg_bits": weighted_bits / total,
        "num_2bit": sum(1 for b in assignment.values() if b == 2),
        "num_3bit": sum(1 for b in assignment.values() if b == 3),
        "num_4bit": sum(1 for b in assignment.values() if b == 4),
        "total_params": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sens2-csv", required=True)
    parser.add_argument("--sens3-csv", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    rows = load_rows(Path(args.sens2_csv), Path(args.sens3_csv))
    settings: list[dict[str, Any]] = []

    for bits in (2.0, 3.0):
        settings.append(
            {
                "name": f"uniform_{bits:g}",
                "kind": "uniform",
                "target_avg": bits,
                "env": {"VLM_BITS": f"{bits:g}", "DIT_BITS": "4", "VLM_QUANT_MODE": "uniform", "DIT_QUANT_MODE": "uniform", "VLM_ROTATION_MODE": "none"},
                "stats": {"avg_bits": bits},
            }
        )

    for bits in (2.5, 2.7, 2.9):
        settings.append(
            {
                "name": f"internal_avg_{bits:g}",
                "kind": "internal_avg",
                "target_avg": bits,
                "env": {"VLM_BITS": f"{bits:g}", "DIT_BITS": "4", "VLM_QUANT_MODE": "uniform", "DIT_QUANT_MODE": "uniform", "VLM_ROTATION_MODE": "none"},
                "stats": {"avg_bits": bits},
            }
        )

    for rule in ("sens2", "gain23"):
        for bits in (2.5, 2.7, 2.9):
            assignment = allocate(rows, target_avg=bits, rule=rule, top4=0)
            settings.append(
                {
                    "name": f"{rule}_{bits:g}",
                    "kind": "explicit",
                    "rule": rule,
                    "target_avg": bits,
                    "top4": 0,
                    "env": {
                        "VLM_BITS": "2",
                        "DIT_BITS": "4",
                        "VLM_QUANT_MODE": "uniform",
                        "DIT_QUANT_MODE": "uniform",
                        "VLM_ROTATION_MODE": "none",
                        "VLM_FORCE_BITS_SPECS": specs_from_assignment(assignment),
                    },
                    "stats": assignment_stats(rows, assignment),
                }
            )
        for top4 in (1, 3, 5):
            assignment = allocate(rows, target_avg=2.5, rule=rule, top4=top4)
            settings.append(
                {
                    "name": f"{rule}_2.5_top{top4}to4",
                    "kind": "explicit",
                    "rule": rule,
                    "target_avg": 2.5,
                    "top4": top4,
                    "env": {
                        "VLM_BITS": "2",
                        "DIT_BITS": "4",
                        "VLM_QUANT_MODE": "uniform",
                        "DIT_QUANT_MODE": "uniform",
                        "VLM_ROTATION_MODE": "none",
                        "VLM_FORCE_BITS_SPECS": specs_from_assignment(assignment),
                    },
                    "stats": assignment_stats(rows, assignment),
                }
            )

    payload = {"settings": settings}
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(out_path)
    for setting in settings:
        stats = setting.get("stats", {})
        print(setting["name"], stats)


if __name__ == "__main__":
    main()
