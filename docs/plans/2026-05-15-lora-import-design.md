# LoRA Import & Selection Design (Electron Desktop App)

**Date**: 2026-05-15
**Status**: Approved (brainstorming complete, awaiting implementation)
**Scope**: Add LoRA adapter import and inference-time selection to the Electron
desktop app at `APP/`. Training is delegated to the upstream
[iron-mukakin/Emoji-TTS](https://github.com/iron-mukakin/Emoji-TTS) fork.

---

## 1. Context & Decision Rationale

The user wants the desktop app to support LoRA adapters end-to-end, but
implementing a full LoRA training pipeline inside Electron would take 1–2
weeks. A community fork (Emoji-TTS) already provides a polished Gradio-based
training UI built on the same Irodori-TTS codebase, with VAD splitting,
Whisper transcription, and PEFT-based LoRA training (with EMA + early
stopping).

**Decision**: split responsibilities.

| App | Role |
|-|-|
| Electron (this repo, `APP/`) | Distribution-friendly inference UI. Adds **import + selection** for LoRAs trained externally. |
| Emoji-TTS (separate repo) | Heavy training/dataset workflow. Run only when needed. |

This keeps the distributable Electron build small and focused while reusing
proven training UX rather than reinventing it.

**Rejected alternatives**:
- *Replace Electron with Emoji-TTS Gradio*: loses our v1–v3 + VoiceDesign +
  Sway-sampling integration and electron-builder-friendly distribution form.
- *Reimplement training inside Electron*: large scope; Emoji-TTS already
  validates the design we'd end up at.

---

## 2. Scope

### In Scope
- Import a trained LoRA adapter directory into the app's user-data area.
- List, browse, and delete imported LoRAs.
- Select a LoRA at inference time via a 2-stage UI:
  radio (`1ショット` | `LoRA`) + LoRA dropdown.
- Filter the LoRA dropdown by base-model compatibility.
- Persist LoRA registry across launches.

### Out of Scope
- Training, dataset preparation, transcription (delegated to Emoji-TTS).
- Real-time loss monitoring / GPU graphs.
- LoRA merging into base safetensors (use `convert_checkpoint_to_safetensors.py`
  CLI from Emoji-TTS or this repo).
- Multi-LoRA stacking (one LoRA per request).

---

## 3. Architecture

### High-level Data Flow

```
[Emoji-TTS or CLI] -> outputs/<run>/checkpoint_NNNNN/  (PEFT adapter dir)
       │
       │  user clicks "Import" in Electron
       ▼
APP/server.py POST /api/v1/loras/import
       │  copies files, writes meta.json
       ▼
%APPDATA%/Irodori-TTS/loras/<name>/
       ├── adapter/                  (adapter_config.json, adapter_model.safetensors)
       └── meta.json

       │  on synthesize with lora_name=<name>
       ▼
RuntimeKey(checkpoint=base, lora_adapter=<userData>/loras/<name>/adapter)
       │  cached via get_cached_runtime
       ▼
InferenceRuntime loads base + applies PEFT adapter -> generates audio
```

### Module Layout

```
APP/
├── server.py             # existing — extended with /api/v1/loras/* endpoints
├── server_lora.py        # NEW — LoRA-specific endpoint handlers, kept separate
│                          for clarity and testability. Mounted onto FastAPI app
│                          from server.py.
├── main.js               # existing — unchanged
├── index.html            # existing — adds radio + LoRA dropdown + Manage modal
├── renderer.js           # existing — adds LoRA state + API calls + filtering
└── style.css             # existing — modal styling
```

### User-data Layout (writable, distribution-safe)

```
%APPDATA%/Irodori-TTS/                 # main.js: app.getPath('userData')
├── loras/
│   └── <name>/
│       ├── adapter/                   # PEFT adapter, copied from import source
│       └── meta.json
├── voices/                            # future: migrate APP/references/ here
├── outputs/                           # future: move APP/outputs/ here
└── hf_cache/                          # HF_HOME redirect (already planned for distribution)
```

Server reads `IRODORI_DATA_DIR` env var to locate the data root. `main.js`
sets it to `app.getPath('userData')` before spawning Python. Falls back to
`APP/` for development if the env var is unset.

---

## 4. Data Structures

### `meta.json` per imported LoRA

```json
{
  "name": "Marin",
  "base": "v3",
  "imported_at": "2026-05-15T10:32:11Z",
  "source": "D:\\Irodori-TTS-Emoji\\outputs\\my_lora\\checkpoint_0010000",
  "notes": "30min dataset, 10k steps, diffusion_attn_mlp preset"
}
```

### `RuntimeKey` extension (in `irodori_tts/inference_runtime.py`)

Add one field:

```python
@dataclass(frozen=True)
class RuntimeKey:
    checkpoint: str
    model_device: str
    codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim"
    model_precision: str = "fp32"
    codec_device: str = "cpu"
    codec_precision: str = "fp32"
    codec_deterministic_encode: bool = True
    codec_deterministic_decode: bool = True
    compile_model: bool = False
    compile_dynamic: bool = False
    lora_adapter: str | None = None      # <-- NEW (None = no adapter)
```

`InferenceRuntime.from_key`: after loading base weights, if
`key.lora_adapter` is set, apply the PEFT adapter via the existing helper
in `irodori_tts/lora.py` (`peft_model_cls.from_pretrained(model, path)`).

Cache invalidation works for free: a different `lora_adapter` value yields
a different key → different cached `InferenceRuntime` instance. Switching
adapters reloads, but each (base, lora) pair stays warm.

---

## 5. API Endpoints

| Method | Path | Body | Returns |
|-|-|-|-|
| GET | `/api/v1/loras` | — | `{ loras: [meta.json contents...] }` |
| POST | `/api/v1/loras/import` | multipart: `name`, `base`, `notes`, `adapter_dir` (path or zip) | `{ status, name }` |
| DELETE | `/api/v1/loras/{name}` | — | `{ status }` |
| POST | `/api/v1/synthesize` | (existing) + new field `lora_name: Optional[str]` | (existing) |

### Validation Rules
- `name`: 1–64 chars, regex `^[A-Za-z0-9_\- 一-龥ぁ-んァ-ヶー]+$`,
  must be unique within `loras/`.
- `base`: must be one of `v3`, `v2`, `voice_design`. (v1 explicitly excluded
  since v1 is not in the distribution.)
- `adapter_dir`: must contain `adapter_config.json` and
  `adapter_model.safetensors`. Reject otherwise.
- On import: copy contents to `loras/<name>/adapter/`, write `meta.json`.

### Synthesize Integration
- When `lora_name` is provided, server resolves
  `userData/loras/<name>/adapter` and verifies `meta.base` matches the
  requested `model_type`. Return 400 on mismatch.
- For `model_type == "v1"`, ignore `lora_name` (subprocess path doesn't
  support adapters; v1 is dev-only anyway).

---

## 6. UI Changes

### Synthesize Tab — model selector area

Current:
```
[Model Checkpoint: ▼ v3 (Latest)        ]
```

New:
```
[Model Checkpoint: ▼ v3 (Latest)        ]   [Manage LoRAs]
○ 1ショット (zero-shot)   ● LoRA
[LoRA Adapter: ▼ Marin (v3, 30min/10kstep)  ]   ← visible when LoRA radio active
```

Behavior:
- Default radio = `1ショット` (current behavior, ref audio shown).
- Switching to `LoRA`: hide ref audio area, show LoRA dropdown.
- Switching model: re-fetch `/api/v1/loras`, filter dropdown by
  `meta.base == model_type`. If 0 results, disable the LoRA radio with
  tooltip "No LoRA imported for this model".
- When a LoRA is selected, `formData.append('lora_name', selected.name)` is
  added in `generateAudio()`.

### LoRA Library Modal (new)

Triggered by `[Manage LoRAs]` button. Single modal, no separate tab.

```
┌─ LoRA Library ──────────────────────────────── × ─┐
│  [+ Import LoRA Folder]                            │
│                                                    │
│  ┌──────┬────────┬───────────────┬─────────┬─────┐│
│  │ Name │ Base   │ Notes         │ Imported│ 🗑   ││
│  ├──────┼────────┼───────────────┼─────────┼─────┤│
│  │Marin │ v3     │ 30min/10kstep │ 5/15    │ [x] ││
│  │Akira │ v2     │ test run      │ 5/14    │ [x] ││
│  └──────┴────────┴───────────────┴─────────┴─────┘│
└────────────────────────────────────────────────────┘
```

Import dialog (sub-modal or inline form):
- File picker: select an adapter directory (Electron's
  `dialog.showOpenDialog` with `properties: ['openDirectory']`).
- Form: Name (text), Base (radio: v3 / v2 / voice_design), Notes (textarea).
- Submit: POST `/api/v1/loras/import` with the selected absolute path; server
  copies files. (Keep upload over network out of scope; we run locally.)

### IPC Wiring (main.js)
- Add `select-folder` IPC handler returning a directory path.
- Renderer calls it for the import flow.

---

## 7. User Workflow

### One-time setup (only for users wanting to train)
1. Clone Emoji-TTS in a separate location, run `uv sync`.
2. Use Emoji-TTS Gradio tabs to slice audio, transcribe, build manifest, train.
3. Output: `outputs/my_lora/checkpoint_0010000/` (PEFT adapter dir).

### Using a trained LoRA in Irodori-TTS Electron
1. Launch app via `Start_Desktop_App.bat`.
2. Click `[Manage LoRAs]` → `[+ Import LoRA Folder]`.
3. Select the `checkpoint_NNNNN/` directory from the Emoji-TTS run.
4. Fill in Name (`Marin`), Base (`v3`), Notes, click Save.
5. Modal closes; new LoRA appears in registry.
6. Back in Synthesize: switch radio to `LoRA`, select `Marin` from dropdown.
7. Type text, click Generate.

### For end-users who never train
- They receive distributed Electron app.
- They can either skip LoRAs entirely (1ショット mode works as before),
  or import community-shared LoRA dirs.

---

## 8. Phased Rollout

### Phase 1 — v3 only (MVP)
- All API endpoints implemented.
- Import accepts `base="v3"` only (other options disabled in UI).
- Synthesize with LoRA works for v3 base.
- Validates the full pipeline including `RuntimeKey` extension.

### Phase 2 — v2 + v2-VoiceDesign bases
- Enable v2 and voice_design as valid `base` values.
- Filtering already works; just expand the validation whitelist.
- No UI rework needed.

### Phase 3 — quality-of-life
- Migrate `APP/references/` → `userData/voices/` for full distribution
  cleanliness.
- Migrate `APP/outputs/` → `userData/outputs/`.
- LoRA preview: show metadata.json in a tooltip on hover in dropdown.
- Optional: drag-and-drop adapter folders into the modal.

These are independent; Phase 2 and 3 can ship in any order after Phase 1.

---

## 9. Open Questions / Risks

| # | Question | Mitigation |
|-|-|-|
| 1 | Does the current `irodori_tts/lora.py` provide a clean adapter-load helper? | Verify during P1 implementation. If not, extract one from `infer.py --lora-adapter` path. |
| 2 | PEFT version compatibility (Emoji-TTS may use a different peft version)? | Pin `peft>=0.18.0` (already in pyproject); load failures should error clearly with a hint to retrain. |
| 3 | adapter dirs from external sources may have arbitrary file layouts. | Validate presence of `adapter_config.json` + at least one `*.safetensors` on import. |
| 4 | Cache thrashing if user switches LoRAs frequently. | Single-runtime cache is OK for typical use. If problematic, add an LRU of size 2–3. |
| 5 | What if user manually drops files into `userData/loras/`? | `GET /api/v1/loras` should accept any well-formed dir; missing meta.json → infer name from dirname, default base to `v3`, mark "untagged". |

---

## 10. Effort Estimate

| Task | Hours |
|-|-|
| `RuntimeKey.lora_adapter` + load wiring in `inference_runtime.py` | 2–3 |
| `server_lora.py` with 3 endpoints + integration into server.py | 2 |
| `server.py /api/v1/synthesize` accepts `lora_name` | 1 |
| `IRODORI_DATA_DIR` env handling + main.js wiring | 1 |
| index.html: radio + LoRA dropdown + Manage modal | 2 |
| renderer.js: state, fetching, filtering, IPC for folder pick | 2–3 |
| E2E smoke test (import → synthesize) | 2 |
| **Total** | **12–14h (≈1.5 dev-days)** |

Fits in a single focused session.

---

## 11. Implementation Order (Suggested)

1. **Backend foundation**: `RuntimeKey.lora_adapter` + `InferenceRuntime`
   adapter load, with a hardcoded test that loads a known adapter and
   synthesizes a clip. Verify via direct Python script before touching API.
2. **Endpoints**: `server_lora.py` with import/list/delete; smoke-test
   via `curl` with a real adapter dir.
3. **Synthesize integration**: extend existing endpoint, smoke-test via
   `curl` to confirm LoRA-conditioned output sounds different.
4. **UI**: index.html + renderer.js, working through the user flow.
5. **End-to-end**: launch Electron, import via Emoji-TTS-trained adapter,
   generate.

---

## Appendix A — Why not modify the existing model dropdown to include LoRA combos?

Considered: `v3 + Marin LoRA` as a single dropdown entry per (base, LoRA)
pair. Rejected because:
- Combinatorial explosion as users accumulate LoRAs.
- Hides the orthogonality of base choice and LoRA choice.
- Harder to communicate "you have 3 LoRAs for v3 and 1 for v2".

The radio + dropdown design makes the relationship explicit and scales
linearly with LoRA count.
