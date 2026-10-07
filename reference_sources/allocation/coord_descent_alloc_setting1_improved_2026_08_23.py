#!/usr/bin/env python3
"""세팅1(Legacy-11 사전보호) 베이스에 알고리즘 개선 3종을 추가한 버전.
1) calibration 순환: 200개 풀(task당 5개)에서 라운드마다 새 40개를 뽑음(라운드 내에서는 고정 -
   공정 비교 유지, 라운드 사이에서만 바뀜 -> 같은 40개에 대한 과적합 방지).
2) refresh_every를 기존 100보다 촘촘하게(기본 50).
3) --rollback: 새로 잰 base_err이 최근 추세 대비 비정상적으로 튀면, 그 직전 라운드의 강등을
   되돌리고 refresh_every 절반으로 한 번만 재시도(무한루프 방지).
--resume-from: 이전 체크포인트 CSV에서 state를 복원해 이어서 시작(background QOS 선점 대비 -
   죽으면 마지막 0.1bit 체크포인트부터 재제출 가능)."""
from __future__ import annotations
import argparse
import csv
import gc
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, "/data/workspace/libero_pro_eval")
sys.path.insert(0, "/data/workspace/libero_pro_eval/Isaac-GR00T-N1.7")
from analyze_hqq_hadamard_mlp_outliers import quantize_weight_hqq  # noqa: E402
from artifact_registry import find_packed  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.eval.cosmos_quant_shim import quantize_gr00t_vlm27_dit4  # noqa: E402
from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402
from measure_dit_single_layer_downgrade_sensitivity import patched_env, replace_embedding  # noqa: E402
from measure_dit_scalar_vs_vector_2026_07_17 import load_batches  # noqa: E402
from measure_embedding_vision_quant_memory_v1 import quantize_visual_linears  # noqa: E402
from measure_gr00t_dit_fullstack_proxy_2026_07_21 import VLM_ENV  # noqa: E402
from measure_hotspot_protection_proxy import move_to_device, run_outputs  # noqa: E402
from proxy_fast_2026_07_24 import load_refs, local_ckpt  # noqa: E402

PROJ = Path("/data/workspace/libero_pro_eval")
QA = PROJ / "quantization_analysis"
ALLOC = QA / "alloc_floor_2026-08-06"
SUITE = "libero_spatial"
POOL_DIR = PROJ / "request_replay_runs/calib_rotation_pool_2026-08-23/val/requests"
POOL_SIZE = 200
ROUND_SAMPLE = 40

BIT_OF_K = {2048: 2.75, 1024: 2.50, 512: 2.25, 256: 2.00, 128: 1.75, 64: 1.50, 32: 1.25, 16: 1.00}
LADDER = sorted(BIT_OF_K, reverse=True)

HOT_SPEC_FILE = QA / "vlm_hotspot11_new_2026_08_18_specs.txt"
HOT = {}
for kv in open(HOT_SPEC_FILE).read().strip().split(";"):
    n, b = kv.split("=")
    HOT[n] = int(b)
assert len(HOT) == 11


def rotation_subset(round_idx: int, pool_size: int, k: int) -> list[int]:
    rng = random.Random(20260823_000 + round_idx)
    return rng.sample(range(pool_size), k)


def dense_from_packed_cached(kind, name, k, cache):
    key = (kind, name, k)
    if key in cache:
        return cache[key].float()
    p = find_packed(kind, name, k, suite=SUITE, init="aware", verify=False)
    o = torch.load(p, map_location="cpu")
    if isinstance(o, dict) and "dense" in o:
        W = o["dense"].float()
    else:
        pk = o["packed"]
        vec = pk["codebook"].float()[pk["indices"].long()]
        W = vec.reshape(pk["out_features"], -1, pk["group_size"]) * pk["scales"].float().unsqueeze(-1)
        W = W.reshape(pk["out_features"], pk["in_features"])
    cache[key] = W.to(torch.bfloat16)
    return W


class Replica:
    def __init__(self, device_id: int):
        self.device = torch.device(f"cuda:{device_id}")
        self.cache: dict = {}
        self.refs = load_refs("rotation_pool", limit=POOL_SIZE)
        assert self.refs is not None, "rotation_pool bf16 refs 캐시가 없음 - 먼저 빌드 필요"
        self.policy = Gr00tPolicy(EmbodimentTag.resolve("LIBERO_PANDA"), local_ckpt(SUITE),
                                  device=str(self.device), strict=True)
        self.policy.model.eval()
        for p in self.policy.model.parameters():
            p.requires_grad_(False)
        self.batches = load_batches(self.policy, POOL_DIR, POOL_SIZE)
        assert len(self.batches) == POOL_SIZE, f"풀 크기 불일치: {len(self.batches)} != {POOL_SIZE}"
        named = dict(self.policy.model.named_modules())
        orig_hot = {n: named[n].weight.detach().to(self.device, torch.float32).clone() for n in HOT}

        with patched_env(VLM_ENV):
            quantize_gr00t_vlm27_dit4(self.policy.model, group_size=64, device=str(self.device))
        replace_embedding(self.policy.model, 4, 64, True)
        quantize_visual_linears(self.policy.model, nbits=4, group_size=64, device=str(self.device))
        self.named_q = dict(self.policy.model.named_modules())

        for n, nb in HOT.items():
            Wr = quantize_weight_hqq(orig_hot[n], nbits=nb, group_size=64, device=str(self.device)).to(torch.float32)
            self.put(n, Wr)
        del orig_hot
        gc.collect(); torch.cuda.empty_cache()

    def put(self, name, W):
        m = self.named_q[name]
        b = getattr(m, "bias", None)
        lin = nn.Linear(W.shape[1], W.shape[0], bias=b is not None, device=self.device, dtype=torch.bfloat16)
        lin.weight.data.copy_(W.to(self.device, torch.bfloat16))
        if b is not None:
            lin.bias.data.copy_(b.detach().to(self.device, torch.bfloat16))
        parent_name, attr = name.rsplit(".", 1)
        setattr(self.named_q[parent_name], attr, lin)

    def measure(self, subset_idx):
        errs = []
        for i in subset_idx:
            rid, bb, ai = self.batches[i]
            o = run_outputs(self.policy, move_to_device(bb, self.device, torch.bfloat16),
                            move_to_device(ai, self.device, torch.bfloat16), seed=7000 + i)
            a = o["final_decoded_action"].detach().to("cpu", torch.float64)
            errs.append(((a - self.refs[rid]).norm() / self.refs[rid].norm().clamp_min(1e-9)).item())
        return sum(errs) / len(errs)

    def install_full_state(self, state, kind_of):
        for n, k in state.items():
            self.put(n, dense_from_packed_cached(kind_of(n), n, k, self.cache))

    def measure_shard_deltas(self, shard, state, kind_of, subset_idx):
        base_err = self.measure(subset_idx)
        out = {}
        for n in shard:
            idx = LADDER.index(state[n])
            if idx + 1 >= len(LADDER):
                continue
            k_next = LADDER[idx + 1]
            cur_w = dense_from_packed_cached(kind_of(n), n, state[n], self.cache)
            w_next = dense_from_packed_cached(kind_of(n), n, k_next, self.cache)
            self.put(n, w_next)
            err = self.measure(subset_idx)
            out[n] = err - base_err
            self.put(n, cur_w)
        return base_err, out


def run_worker(args) -> None:
    """--worker-mode: GPU 하나만 맡아 지정된 shard의 델타를 재고 JSON으로 결과를 써서 끝냄.
    매 라운드 별도 프로세스로 새로 뜨므로 모델을 매번 새로 빌드한다(GIL 경합 회피가 목적,
    복제본 재사용은 포기)."""
    state = {}
    for r in csv.DictReader(open(args.state_file)):
        state[r["layer"]] = int(r["k"])
    shard = [ln.strip() for ln in open(args.shard_file) if ln.strip()]
    subset = [int(x) for x in open(args.subset_file).read().split()]

    def kind_of(name):
        return "dit" if name.startswith("action_head") else "vlm"

    replica = Replica(args.device)
    replica.install_full_state(state, kind_of)
    base_err, deltas = replica.measure_shard_deltas(shard, state, kind_of, subset)
    with open(args.out_json, "w") as f:
        json.dump({"base_err": base_err, "deltas": deltas}, f)


def run_worker_persistent(args) -> None:
    """--worker-mode --persistent: GPU 하나를 맡아 모델을 한 번만 빌드하고, 이후 라운드마다
    work_dir에 새 round_{r}_state.csv/subset.txt가 나타나길 폴링하다가 나타나면 측정하고
    round_{r}_result_{device}.json을 (임시파일->rename으로) 원자적으로 써서 다음 라운드를 기다림.
    stop.flag가 생기면 종료."""
    shard = [ln.strip() for ln in open(args.shard_file) if ln.strip()]
    work_dir = Path(args.shard_file).parent

    def kind_of(name):
        return "dit" if name.startswith("action_head") else "vlm"

    replica = Replica(args.device)
    r = 0
    while True:
        if (work_dir / "stop.flag").exists():
            break
        state_file = work_dir / f"round_{r}_state.csv"
        subset_file = work_dir / f"round_{r}_subset.txt"
        if state_file.exists() and subset_file.exists():
            state = {}
            for row in csv.DictReader(open(state_file)):
                state[row["layer"]] = int(row["k"])
            subset = [int(x) for x in open(subset_file).read().split()]
            replica.install_full_state(state, kind_of)
            base_err, deltas = replica.measure_shard_deltas(shard, state, kind_of, subset)
            out_json = work_dir / f"round_{r}_result_{args.device}.json"
            tmp = out_json.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump({"base_err": base_err, "deltas": deltas}, f)
            tmp.rename(out_json)
            r += 1
        else:
            time.sleep(0.5)


class PersistentPool:
    """process 모드 + 상주 워커: 워커를 한 번만 띄워두고 라운드마다 파일로 신호를 주고받아
    매 라운드 모델 재빌드 비용(~180초)을 없앤다. 계산 자체는 fresh-process 모드와 동일(이미
    검증됨) - 재사용 여부만 다르므로 결과값은 동일해야 함."""

    def __init__(self, n_gpus: int, work_dir: Path, shards, script_path: str, py: str):
        self.n_gpus = n_gpus
        self.work_dir = work_dir
        self.round = 0
        work_dir.mkdir(parents=True, exist_ok=True)
        for f in work_dir.glob("*"):
            f.unlink()
        self.procs = []
        for i in range(n_gpus):
            shard_file = work_dir / f"shard_{i}.txt"
            shard_file.write_text("\n".join(shards[i]))
            cmd = [py, script_path, "--worker-mode", "--persistent", "--device", str(i),
                   "--shard-file", str(shard_file)]
            log = open(work_dir / f"worker_{i}.log", "w")
            self.procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))

    def measure_round(self, state, subset):
        r = self.round
        state_file = self.work_dir / f"round_{r}_state.csv"
        tmp = state_file.with_suffix(".tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "k"])
            for n, k in state.items():
                w.writerow([n, k])
        tmp.rename(state_file)
        subset_file = self.work_dir / f"round_{r}_subset.txt"
        tmp2 = subset_file.with_suffix(".tmp")
        tmp2.write_text(" ".join(str(i) for i in subset))
        tmp2.rename(subset_file)

        results = []
        for i in range(self.n_gpus):
            out_json = self.work_dir / f"round_{r}_result_{i}.json"
            while not out_json.exists():
                rc = self.procs[i].poll()
                if rc is not None:
                    raise RuntimeError(f"상주 워커 {i}가 죽음(rc={rc}), 로그: "
                                       f"{self.work_dir}/worker_{i}.log")
                time.sleep(0.5)
            results.append(json.loads(out_json.read_text()))
        self.round += 1
        base_err = sum(x["base_err"] for x in results) / len(results)
        deltas = {}
        for x in results:
            deltas.update(x["deltas"])
        return base_err, deltas

    def shutdown(self):
        (self.work_dir / "stop.flag").touch()
        for p in self.procs:
            try:
                p.wait(timeout=30)
            except Exception:
                p.terminate()


def measure_all_shards_process(state, shards, subset, kind_of_name, work_dir: Path, n_gpus: int, script_path: str, py: str):
    """프로세스 기반 병렬: shard/state/subset을 파일로 던지고 GPU마다 워커 프로세스를 새로 띄움."""
    work_dir.mkdir(parents=True, exist_ok=True)
    state_file = work_dir / "state.csv"
    with open(state_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "k"])
        for n, k in state.items():
            w.writerow([n, k])
    subset_file = work_dir / "subset.txt"
    subset_file.write_text(" ".join(str(i) for i in subset))

    procs = []
    out_jsons = []
    for i in range(n_gpus):
        shard_file = work_dir / f"shard_{i}.txt"
        shard_file.write_text("\n".join(shards[i]))
        out_json = work_dir / f"result_{i}.json"
        out_jsons.append(out_json)
        cmd = [py, script_path, "--worker-mode", "--device", str(i),
               "--state-file", str(state_file), "--shard-file", str(shard_file),
               "--subset-file", str(subset_file), "--out-json", str(out_json)]
        log = open(work_dir / f"worker_{i}.log", "w")
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))

    for i, p in enumerate(procs):
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"워커 {i} 실패(rc={rc}), 로그: {work_dir / f'worker_{i}.log'}")

    base_errs, deltas = [], {}
    for out_json in out_jsons:
        r = json.loads(out_json.read_text())
        base_errs.append(r["base_err"])
        deltas.update({k: v for k, v in r["deltas"].items()})
    return sum(base_errs) / len(base_errs), deltas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-avg-bits", type=float, default=1.2)
    ap.add_argument("--refresh-every", type=int, default=50)
    ap.add_argument("--n-gpus", type=int, default=7)
    ap.add_argument("--rollback", action="store_true")
    ap.add_argument("--resume-from", default="")
    ap.add_argument("--out", default=str(QA / "coord_descent_setting1_improved_2026_08_23.csv"))
    ap.add_argument("--checkpoint-step", type=float, default=0.1)
    ap.add_argument("--parallel-mode", choices=["thread", "process"], default="thread")
    ap.add_argument("--persistent", action="store_true",
                    help="process 모드에서 워커를 라운드마다 새로 안 띄우고 상주시켜 재빌드 비용을 없앰")
    ap.add_argument("--subset-size", type=int, default=0, help="0이면 전체, N>0이면 빠른 검증용으로 후보를 N개로 축소")
    # --worker-mode 전용 인자(프로세스 모드에서 자식으로 재호출될 때 씀)
    ap.add_argument("--worker-mode", action="store_true")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--state-file", default="")
    ap.add_argument("--shard-file", default="")
    ap.add_argument("--subset-file", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    if args.worker_mode:
        if args.persistent:
            run_worker_persistent(args)
        else:
            run_worker(args)
        return

    t0 = time.time()

    vlm_candidates = [r["layer"] for r in csv.DictReader(open(ALLOC / "vlm_new11_cg150.csv"))]
    dit_candidates = [r["layer"] for r in csv.DictReader(open(ALLOC / "dit_cg150.csv"))]
    params_by_name = {}
    for r in csv.DictReader(open(ALLOC / "vlm_new11_cg150.csv")):
        params_by_name[r["layer"]] = int(r["params"])
    for r in csv.DictReader(open(ALLOC / "dit_cg150.csv")):
        params_by_name[r["layer"]] = int(r["params"])
    candidates = vlm_candidates + dit_candidates
    if args.subset_size > 0:
        import random as _random
        rng = _random.Random(20260823)
        candidates = rng.sample(candidates, args.subset_size)
    total_params = sum(params_by_name[n] for n in candidates)

    def kind_of(name):
        return "dit" if name.startswith("action_head") else "vlm"

    if args.resume_from:
        state = {}
        for r in csv.DictReader(open(args.resume_from)):
            state[r["layer"]] = int(r["k"])
        missing = [n for n in candidates if n not in state]
        assert not missing, f"resume CSV에 {len(missing)}개 층 누락: {missing[:3]}..."
        print(f"[cd1i] --resume-from {args.resume_from} 에서 state 복원 완료", flush=True)
    else:
        state = {n: LADDER[0] for n in candidates}

    def weighted_avg():
        return sum(params_by_name[n] * BIT_OF_K[state[n]] for n in candidates) / total_params

    start_avg = weighted_avg()

    checkpoints_left = []
    if args.checkpoint_step > 0:
        b = round(int(start_avg / args.checkpoint_step + 1e-9) * args.checkpoint_step, 2)
        if b >= start_avg - 1e-9:
            b = round(b - args.checkpoint_step, 2)
        while b > args.target_avg_bits + 1e-9:
            checkpoints_left.append(round(b, 2))
            b = round(b - args.checkpoint_step, 2)

    print(f"[cd1i] 후보 {len(candidates)}개, GPU {args.n_gpus}개, refresh_every={args.refresh_every}, "
          f"rollback={args.rollback}, parallel={args.parallel_mode}, 시작avg={start_avg:.4f}bit", flush=True)

    work_dir = QA / f"coord_descent_proc_work_{Path(args.out).stem}"
    script_path = str(Path(__file__).resolve())
    py = sys.executable

    shards = [candidates[i::args.n_gpus] for i in range(args.n_gpus)]
    pool = None

    if args.parallel_mode == "thread":
        print(f"[cd1i] 복제본 {args.n_gpus}개 빌드 중...", flush=True)
        replicas = [Replica(i) for i in range(args.n_gpus)]
        print(f"[cd1i] 복제본 빌드 완료 ({time.time()-t0:.0f}초)", flush=True)
    elif args.persistent:
        replicas = None
        print(f"[cd1i] process+상주 모드 - 워커 {args.n_gpus}개를 한 번만 띄우고 계속 재사용 "
              f"(작업 디렉터리: {work_dir})", flush=True)
        pool = PersistentPool(args.n_gpus, work_dir, shards, script_path, py)
        print(f"[cd1i] 상주 워커 {args.n_gpus}개 빌드 시작(백그라운드, 첫 라운드 측정 때 완료 대기) "
              f"({time.time()-t0:.0f}초)", flush=True)
    else:
        replicas = None  # process 모드는 매 라운드 워커를 새로 띄우므로 상주 복제본 없음
        print(f"[cd1i] process 모드 - 라운드마다 워커 {args.n_gpus}개를 새 프로세스로 띄움 "
              f"(작업 디렉터리: {work_dir})", flush=True)

    def save_alloc(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "params", "bit", "k"])
            for n in sorted(candidates):
                w.writerow([n, params_by_name[n], f"{BIT_OF_K[state[n]]:.2f}", state[n]])

    round_idx = 0
    subset0 = rotation_subset(round_idx, POOL_SIZE, ROUND_SAMPLE)
    if args.parallel_mode == "thread":
        for r in replicas:
            r.install_full_state(state, kind_of)
        base_errs = [r.measure(subset0) for r in replicas]
        init_err = sum(base_errs) / len(base_errs)
    elif pool is not None:
        init_err, _ = pool.measure_round(state, subset0)
    else:
        init_err, _ = measure_all_shards_process(state, shards, subset0, kind_of, work_dir,
                                                  args.n_gpus, script_path, py)
    print(f"[cd1i] 초기 proxy err = {init_err:.6f}, avg={weighted_avg():.4f}bit "
          f"({time.time()-t0:.0f}초)", flush=True)

    deltas = {}
    err_history = []  # base_err 기록 (롤백 판단용)
    pre_round_state_snapshot = dict(state)
    step = 0
    steps_since_refresh = 0
    window_size = args.refresh_every
    halved_retry_used = False
    while weighted_avg() > args.target_avg_bits:
        if steps_since_refresh == 0:
            subset = rotation_subset(round_idx, POOL_SIZE, ROUND_SAMPLE)
            if args.parallel_mode == "thread":
                for r in replicas:
                    r.install_full_state(state, kind_of)
                with ThreadPoolExecutor(max_workers=args.n_gpus) as ex:
                    futs = [ex.submit(replicas[i].measure_shard_deltas, shards[i], state, kind_of, subset)
                            for i in range(args.n_gpus)]
                    results = [f.result() for f in futs]
                new_base_err = sum(r[0] for r in results) / len(results)
                new_deltas = {}
                for base_err, shard_deltas in results:
                    new_deltas.update(shard_deltas)
            elif pool is not None:
                new_base_err, new_deltas = pool.measure_round(state, subset)
            else:
                new_base_err, new_deltas = measure_all_shards_process(
                    state, shards, subset, kind_of, work_dir, args.n_gpus, script_path, py)

            if args.rollback and len(err_history) >= 2 and not halved_retry_used:
                recent_ref = sum(err_history[-2:]) / 2
                if new_base_err > recent_ref * 1.5:
                    half_window = max(10, args.refresh_every // 2)
                    print(f"[cd1i][rollback] step{step} base_err={new_base_err:.6f}가 최근평균"
                          f"({recent_ref:.6f})의 1.5배 초과 -> 직전 라운드 되돌리고 "
                          f"refresh_every {args.refresh_every}->{half_window}로 1회 재시도", flush=True)
                    state = dict(pre_round_state_snapshot)
                    halved_retry_used = True
                    window_size = half_window
                    steps_since_refresh = 0
                    continue

            if halved_retry_used:
                window_size = args.refresh_every  # 재시도가 끝났으니 다음 라운드부터 정상 크기로 복귀
            halved_retry_used = False
            err_history.append(new_base_err)
            pre_round_state_snapshot = dict(state)
            deltas = new_deltas
            print(f"[cd1i] step{step} 재측정 완료 (base_err={new_base_err:.6f}, "
                  f"avg={weighted_avg():.4f}bit, deltas={len(deltas)}개, {time.time()-t0:.0f}초 경과, "
                  f"round={round_idx}, window={window_size})", flush=True)
            if args.parallel_mode == "thread":
                for r in replicas:
                    r.cache.clear()
            gc.collect()
            round_idx += 1

        best_n, best_cost = None, 1e18
        for n, d in deltas.items():
            idx = LADDER.index(state[n])
            if idx + 1 >= len(LADDER):
                continue
            bit_saved = BIT_OF_K[state[n]] - BIT_OF_K[LADDER[idx + 1]]
            cost = d / (params_by_name[n] * bit_saved)
            if cost < best_cost:
                best_cost, best_n = cost, n
        if best_n is None:
            print("[cd1i] 더 내릴 층 없음, 종료", flush=True)
            break
        idx = LADDER.index(state[best_n])
        state[best_n] = LADDER[idx + 1]
        del deltas[best_n]
        step += 1
        steps_since_refresh += 1
        if steps_since_refresh >= window_size or not deltas:
            steps_since_refresh = 0
        cur_avg = weighted_avg()
        if step % 50 == 0:
            print(f"[cd1i] step{step}: avg={cur_avg:.4f}bit ({time.time()-t0:.0f}초 경과)", flush=True)
        while checkpoints_left and cur_avg <= checkpoints_left[0]:
            b = checkpoints_left.pop(0)
            ckpt_path = str(Path(args.out).with_suffix("")) + f"_ckpt{b:.2f}.csv"
            save_alloc(ckpt_path)
            print(f"[cd1i] >>> 체크포인트 {b}bit 통과 (실제 avg={cur_avg:.4f}bit, step={step}, "
                  f"{time.time()-t0:.0f}초 경과) -> {ckpt_path}", flush=True)

    final_subset = rotation_subset(round_idx, POOL_SIZE, ROUND_SAMPLE)
    if args.parallel_mode == "thread":
        for r in replicas:
            r.install_full_state(state, kind_of)
        final_err = replicas[0].measure(final_subset)
    elif pool is not None:
        final_err, _ = pool.measure_round(state, final_subset)
    else:
        final_err, _ = measure_all_shards_process(state, shards, final_subset, kind_of, work_dir,
                                                   args.n_gpus, script_path, py)
    print(f"[cd1i] === 최종: avg={weighted_avg():.4f}bit, proxy err={final_err:.6f}, "
          f"총 스텝={step}, 총 소요={time.time()-t0:.0f}초 ===", flush=True)

    if pool is not None:
        pool.shutdown()

    save_alloc(args.out)
    print(f"[cd1i] 배분 저장 -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
