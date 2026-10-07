#!/usr/bin/env python3
"""아티팩트 단일 조회 창구 (single source of truth).

이 프로젝트의 반복 사고 원인은 "같은 코드북을 어디서 찾을지"를 스크립트마다
따로 구현해 규칙이 어긋난 것이었다. (예: 커버리지 점검은 A/B 폴더를 보는데
학습 로더는 A만 봐서, 점검은 통과하고 실행은 실패)

앞으로 모든 스크립트는 이 모듈만 쓴다:
    from artifact_registry import find_packed, check_alloc, KB
    p = find_packed("vlm", name, k, init="aware")            # 경로 or 예외
    miss = check_alloc(vlm_csv, dit_csv, suite, init="vanilla")  # 누락 목록

사용법(CLI):
    python artifact_registry.py check <vlm_csv> <dit_csv> <suite> [aware|vanilla]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJ = Path("@WORKSPACE@/libero_pro_eval")
QA = PROJ / "quantization_analysis"
ROT = QA / "vlm_rotgrid_2026-07-25"
FS2 = QA / "gr00t_dit_fs2_2026-07-23"

# 비트 표현 3종의 유일한 변환표 (k <-> kbit 문자열 <-> 비트값)
KB = {2048: "275", 1024: "250", 512: "225", 256: "200",
      128: "175", 64: "150", 32: "125", 16: "100", 8: "075"}
BITS = {k: (len(bin(k)) - 3) / 4 for k in KB}   # log2(k)/4


def _vlm_dirs(k: int, init: str) -> list[Path]:
    """VLM packed 코드북이 있을 수 있는 모든 위치 (우선순위 순)."""
    if init == "vanilla":
        return [ROT / f"vlm_packed_van_u{KB[k]}", ROT / "vlm_packed_vanilla_a230"]
    return [ROT / f"vlm_packed_u{KB[k]}", ROT / "vlm_packed_a230", ROT / "vlm_packed_a210"]


def _dit_dirs(k: int, suite: str, init: str) -> list[Path]:
    if init == "vanilla":
        return [FS2 / f"fits_packed_van_{KB[k]}_{suite}",
                FS2 / f"fits_packed_vanilla_a230_{suite}"]
    return [FS2 / f"fits_packed_u{KB[k]}_{suite}",
            FS2 / f"fits_packed_a230_{suite}", FS2 / f"fits_packed_a210_{suite}"]


def find_packed(kind: str, name: str, k: int, suite: str = "libero_spatial",
                init: str = "aware", verify: bool = True, exact: bool = False):
    """(kind, layer, k) packed 아티팩트 경로. 없으면 FileNotFoundError.

    verify=True 면 파일을 열어 코드북 크기가 실제로 k 인지 확인한다
    (같은 이름의 다른 k 파일을 잘못 집는 사고 방지)."""
    import torch
    dirs = _vlm_dirs(k, init) if kind == "vlm" else _dit_dirs(k, suite, init)
    if exact:
        # The first directory is the unique per-rate artifact set. Legacy
        # a230/a210 fallbacks are forbidden for paper-main final allocations.
        dirs = dirs[:1]
    fname = f"{name}.pt" if kind == "vlm" else f"{name}__ck.pt"
    for d in dirs:
        p = d / fname
        if not p.exists():
            continue
        if not verify:
            return p
        # Legacy files may also contain a multi-megabyte dense reconstruction.
        # Registry verification needs only the tiny packed header/codebook, so
        # mmap prevents an eager dense read and leaves callers in control.
        o = torch.load(p, map_location="cpu", mmap=True)
        if isinstance(o, dict) and "packed" in o and o["packed"]["codebook"].shape[0] == k:
            return p
    raise FileNotFoundError(
        f"packed 없음: kind={kind} name={name} k={k} suite={suite} init={init}\n"
        f"  탐색 위치: {[str(d) for d in dirs]}")


def check_alloc(vlm_csv, dit_csv, suites, init="aware"):
    """배분 CSV가 요구하는 모든 아티팩트가 있는지 검사. 누락 목록을 돌려준다."""
    miss = []
    for r in csv.DictReader(open(vlm_csv)):
        try:
            find_packed("vlm", r["layer"], int(r["k"]), init=init)
        except FileNotFoundError:
            miss.append(("vlm", "-", r["layer"], int(r["k"])))
    for su in ([suites] if isinstance(suites, str) else suites):
        for r in csv.DictReader(open(dit_csv)):
            try:
                find_packed("dit", r["layer"], int(r["k"]), suite=su, init=init)
            except FileNotFoundError:
                miss.append(("dit", su, r["layer"], int(r["k"])))
    return miss


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "check":
        vlm_csv, dit_csv, suite = sys.argv[2], sys.argv[3], sys.argv[4]
        init = sys.argv[5] if len(sys.argv) > 5 else "aware"
        suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"] if suite == "all" else suite
        m = check_alloc(vlm_csv, dit_csv, suites, init)
        print(f"누락 {len(m)}건")
        for kind, su, name, k in m[:20]:
            print(f"  {kind} {su} {name} k={k}")
        sys.exit(1 if m else 0)
    print(__doc__)
