# 外部から呼び出すためのAPIガイド

Electron アプリ（`起動.bat`）を起動すると、内部で FastAPI サーバーが
`0.0.0.0` にバインドされます。このドキュメントは、そのサーバーを **Electron UI を介さず**
外部のスクリプト・別アプリ・別マシンから直接 HTTP で叩くための実用ガイドです。

> [!IMPORTANT]
> **ポートは固定ではありません。** アプリは起動時に 8080 から順に空きポートを探します。
> 8080 が使用中なら 8081、そこも埋まっていれば 8082 …と繰り上がります。
> 実際のポートはアプリ画面右上の **API Base URL** に表示されます。
> 以下の例は既定値の 8080 で書いてあるので、環境に合わせて読み替えてください。

対象読者：
- 自作スクリプト（Python/Node/シェル等）から音声合成を呼びたい人
- 同じ LAN 上の別端末から叩きたい人
- LoRA（キャラ声）を指定して合成を自動化したい人

全パラメータの網羅的なリファレンスは [api-reference.md](api-reference.md) を参照してください。
本ガイドはそこから「外部呼び出しに必要な最短経路」だけを抜き出したものです。

---

## 0. 前提・注意事項

- **認証なし**。誰でも叩けます。CORS も `allow_origins=["*"]` で全許可、かつ `0.0.0.0` バインドのため
  **同じ LAN 上の全端末からアクセス可能**です。信頼できないネットワーク（公共Wi-Fi等）には繋がない、
  ルーターでこのポートを外部公開しない、を徹底してください。
- **サーバーはシングルジョブ前提**。合成リクエストは内部で逐次実行されるため、並列に投げても速くなりません。
  また LoRA 学習ジョブが走っている間は一部エンドポイントが `409` を返します。
- アプリ起動中のみ生きています。落としたら `起動.bat` で再起動してください。

---

## 1. 接続先を決める

### 同じPCから
```
http://127.0.0.1:8080
```

### 同じLAN上の別端末から
まずアプリが動いているPC自身に、外部から見えるIPを聞きます。

```bash
curl http://127.0.0.1:8080/api/v1/network_info
# => {"ip": "192.168.1.10", "port": 8080}
```

返ってきた `ip` を使い、別端末からは `http://192.168.1.10:8080` で叩けます。

### 生存確認
```bash
curl http://127.0.0.1:8080/api/v1/status
```
```json
{
  "status": "online",
  "device": "cuda",
  "available_models": ["v1", "v2", "voice_design", "v3"],
  "registered_voices": [{"id": "voice_xxx", "name": "Marin"}]
}
```

---

## 2. 最短で1音声を作る（LoRAなし）

参照音声を使わず、ベースモデルのデフォルト声で合成する最小構成です。

```bash
curl -X POST http://127.0.0.1:8080/api/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "こんにちは、これはテストです。",
    "model_type": "v3"
  }'
```

**Response**:
```json
{
  "status": "success",
  "seed_used": 42,
  "results": ["/api/v1/outputs/sample_20260722_xxx_001.wav"],
  "timings": ["tokenize_text: 1.4ms", "sample_rf: 800ms"]
}
```

`results[]` は相対パスなので、接続先のホストを付けて取得します。

```bash
curl -o out.wav http://127.0.0.1:8080/api/v1/outputs/sample_20260722_xxx_001.wav
```

---

## 3. LoRA（キャラ声）を指定して合成する

### 3.1 使えるLoRAを確認する
```bash
curl http://127.0.0.1:8080/api/v1/loras
```
```json
{
  "loras": [
    { "name": "Marin", "base": "v3", "imported_at": "...", "source": "...", "notes": "" }
  ]
}
```

`base` は合成時の `model_type` と一致している必要があります（`v3` の LoRA は `model_type: "v3"` でのみ使用可）。

### 3.2 まだ登録されていない場合はインポートする
外部ツール（[Emoji-TTS](https://github.com/iron-mukakin/Emoji-TTS) や本リポジトリの `train.py --config configs/train_500m_v3_lora.yaml`）で
学習した PEFT アダプタディレクトリを、パス指定で取り込みます。

```bash
curl -X POST http://127.0.0.1:8080/api/v1/loras/import \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Marin",
    "base": "v3",
    "source_path": "D:/path/to/checkpoint_10000",
    "notes": "30min, 10kstep"
  }'
```

`source_path` は `adapter_config.json` と `adapter_model.safetensors`（または `.bin`）を含むディレクトリを指します。
同名がすでに存在すると `409` が返ります。

### 3.3 LoRAを指定して合成する
`lora_name` を渡すだけです。LoRA自体が話者性を内包しているため、参照音声（`voice_id` / `ref_wav`）は不要です。

```bash
curl -X POST http://127.0.0.1:8080/api/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "こんにちは、これはLoRA音声のテストです。",
    "model_type": "v3",
    "lora_name": "Marin",
    "num_steps": 32
  }'
```

LoRAの効きが弱い/強すぎる場合の調整は `cfg_scale_speaker`（LoRA有効時は「学習した話者特徴への忠実度」として働く）で行います。詳細は [lora-usage.md](lora-usage.md) の「4. パラメータ別の効きどころ」を参照してください。

---

## 4. 参照音声（ゼロショットクローン）を使う場合

LoRAを使わず、任意の参照音声で声質を寄せる方法です。2通りあります。

**A. 登録済みボイスをIDで指定**
```bash
curl -X POST http://127.0.0.1:8080/api/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "こんにちは", "model_type": "v3", "voice_id": "voice_xxx"}'
```

**B. 参照wavを直接アップロード（multipart）**
```bash
curl -X POST http://127.0.0.1:8080/api/v1/synthesize \
  -F "text=こんにちは" \
  -F "model_type=v3" \
  -F "ref_wav=@/path/to/reference.wav"
```

---

## 5. OpenAI互換エンドポイントを使う場合

既存の OpenAI TTS クライアント（`base_url` を差し替えられるもの）をそのまま向けられます。

```bash
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "こんにちは", "voice": "Marin", "model": "tts-1"}' \
  --output out.wav
```

`voice` または `model` に登録済みボイス名を渡すとそのリファレンス音声で合成されます。バイナリ(`audio/wav`)が直接返ります。

**制限**（`/api/v1/synthesize` との違い）:
- `model_type` は常に `v2` 固定
- `lora_name` 非対応
- `caption` / `sway` などの高度パラメータ非対応

LoRAやv3を使いたい場合は素直に `/api/v1/synthesize` を使ってください。

---

## 6. サンプルコード

### Python (requests)
```python
import requests

BASE = "http://127.0.0.1:8080"

resp = requests.post(f"{BASE}/api/v1/synthesize", json={
    "text": "こんにちは、これはPythonからのテストです。",
    "model_type": "v3",
    "lora_name": "Marin",
})
resp.raise_for_status()
data = resp.json()

wav = requests.get(BASE + data["results"][0])
with open("out.wav", "wb") as f:
    f.write(wav.content)
```

### JavaScript (fetch)
```javascript
const BASE = "http://127.0.0.1:8080";

const res = await fetch(`${BASE}/api/v1/synthesize`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    text: "こんにちは、これはJSからのテストです。",
    model_type: "v3",
    lora_name: "Marin",
  }),
});
const data = await res.json();

const wavRes = await fetch(BASE + data.results[0]);
const wavBuffer = await wavRes.arrayBuffer();
```

---

## 7. エンドポイント早見表

| 用途 | Method / Path |
|-|-|
| 接続先IP取得（LAN用） | `GET /api/v1/network_info` |
| 生存確認 | `GET /api/v1/status` |
| 音声合成 | `POST /api/v1/synthesize` |
| 合成結果の取得 | `GET /api/v1/outputs/{filename}` |
| 登録済みボイス一覧 | `GET /api/v1/voices` |
| LoRA一覧 | `GET /api/v1/loras` |
| LoRAインポート | `POST /api/v1/loras/import` |
| LoRAマージ（複数LoRAを合成） | `POST /api/v1/loras/merge` |
| LoRA削除 | `DELETE /api/v1/loras/{name}` |
| LoRA学習ジョブ起動 | `POST /api/v1/lora/jobs` |
| OpenAI互換モデル一覧 | `GET /v1/models` |
| OpenAI互換合成 | `POST /v1/audio/speech` |

全パラメータ・エラーコード・データセット/学習ジョブ系エンドポイントを含む完全な一覧は
[api-reference.md](api-reference.md) を参照してください。

---

## 8. 関連ドキュメント
- [api-reference.md](api-reference.md) — 全エンドポイントの完全リファレンス
- [lora-usage.md](lora-usage.md) — LoRAのインポート・学習・パラメータ調整の詳細
- [desktop-app.md](desktop-app.md) — アプリ本体のセットアップ・操作
- [parameters.md](parameters.md) — 推論/学習パラメータの詳細
