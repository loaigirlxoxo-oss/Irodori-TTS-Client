"""学習から採用・登録までの通し。

サーバーを起動してから実行する。10ステップだけ回すので数分で終わる。
  IRODORI_PORT=<port> python scripts/test_easy_train.py <ref.wav>
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

PORT = os.environ.get("IRODORI_PORT", "8080")
API = f"http://127.0.0.1:{PORT}/api/v1"
DATASET = "easy-train-tmp"
LORA = "easy-train-tmp-lora"


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def get(path: str) -> dict:
    with urllib.request.urlopen(API + path, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def delete(path: str) -> None:
    req = urllib.request.Request(API + path, method="DELETE")
    try:
        urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise


def wait(job_id: str, limit_sec: int = 2400) -> dict:
    deadline = time.time() + limit_sec
    last = ""
    while time.time() < deadline:
        time.sleep(5)
        s = get(f"/easy/jobs/{job_id}")
        state = s.get("state", "")
        if state != last:
            print(f"  {state} {s.get('done', '')}/{s.get('total', '')}", flush=True)
            last = state
        if state in ("done", "failed"):
            return s
    raise AssertionError("時間内に終わらなかった")


def prepare(ref: str) -> None:
    """評価に使う小さなデータセットを作る。"""
    delete(f"/datasets/{DATASET}")
    job = post("/easy/generate", {
        "ref_wav": ref, "limit": 6, "model_type": "v4_1", "dataset": DATASET,
    })
    s = wait(job["job_id"])
    assert s["state"] == "done", f"生成に失敗: {s.get('error')}"


def test_train_and_register(ref: str) -> None:
    delete(f"/loras/{LORA}")
    job = post("/easy/train", {
        "dataset": DATASET, "lora_name": LORA, "ref_wav": ref,
        "max_steps": 10, "save_every": 10,
    })
    s = wait(job["job_id"])
    assert s["state"] == "done", f"state={s['state']} error={s.get('error')}"

    picked = s.get("picked")
    assert picked, "採用したチェックポイントが記録されていない"
    assert "sim" in picked and "cer" in picked, f"評価値がない: {picked}"

    loras = get("/loras")
    items = loras if isinstance(loras, list) else loras.get("loras", [])
    names = [x.get("name") if isinstance(x, dict) else x for x in items]
    assert LORA in names, f"登録されていない。今ある: {names[:5]}"


def test_evaluated_only_late_checkpoints() -> None:
    """60%未満のチェックポイントを評価対象にしていないこと。"""
    # 10ステップ・save_every 5 なら 5 と 10。60% = 6 なので 10 だけが対象。
    s = get(f"/easy/jobs/{LAST_JOB[0]}")
    evaluated = s.get("evaluated") or []
    assert evaluated, "評価の記録がない"
    for row in evaluated:
        assert row["step"] >= 10 * 0.6, f"60%未満を評価している: {row}"


def _vram_used() -> int | None:
    """GPU の使用量（MiB）。nvidia-smi が無ければ None。"""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return int(out.splitlines()[0])


def test_vram_released(before: int | None) -> None:
    """工程が終わったら VRAM を返していること。

    絶対値では判定しない。他のアプリや止め損ねたプロセスが掴んでいる分が
    乗るので、工程の前後の差で見る。モデル1つが約3GB なので、それが残って
    いなければ解放できている。
    """
    if before is None:
        print("  nvidia-smi が無いので VRAM の確認はスキップ")
        return
    after = _vram_used()
    grew = after - before
    print(f"  VRAM: {before} -> {after} MiB（差 {grew:+d}）")
    assert grew < 3000, f"モデル1つ分が残っている（差 {grew} MiB）"


LAST_JOB: list[str] = []
VRAM_BEFORE: list = []


if __name__ == "__main__":
    VRAM_BEFORE.append(_vram_used())
    ref = sys.argv[1] if len(sys.argv) > 1 else ""
    assert ref, "参照音声のパスを引数で渡す"
    print("データセットを用意")
    prepare(ref)
    print("学習と採用")
    job = post("/easy/train", {
        "dataset": DATASET, "lora_name": LORA, "ref_wav": ref,
        "max_steps": 10, "save_every": 10,
    })
    LAST_JOB.append(job["job_id"])
    s = wait(job["job_id"])
    assert s["state"] == "done", f"state={s['state']} error={s.get('error')}"
    picked = s.get("picked")
    assert picked, "採用の記録がない"
    print(f"  採用: {picked['name']} sim={picked['sim']:.3f} cer={picked['cer']:.3f}")
    loras = get("/loras")
    items = loras if isinstance(loras, list) else loras.get("loras", [])
    names = [x.get("name") if isinstance(x, dict) else x for x in items]
    assert LORA in names, f"登録されていない。今ある: {names[:5]}"
    test_evaluated_only_late_checkpoints()
    test_vram_released(VRAM_BEFORE[0])
    print("OK")
