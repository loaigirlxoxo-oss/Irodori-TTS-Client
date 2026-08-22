// State
let voices = [];
let selectedVoice = null;
let currentWavPath = null;

// API サーバーのポートは固定ではない。main プロセスが起動時に空きを探して決め、
// ここで受け取る。init() の前に resolveApiOrigin() で確定させる。
let API_ORIGIN = "http://127.0.0.1:8080";
let API_URL = API_ORIGIN + "/api/v1";

async function resolveApiOrigin() {
  try {
    const port = await window.api.getApiPort();
    if (port) {
      API_ORIGIN = `http://127.0.0.1:${port}`;
      API_URL = API_ORIGIN + "/api/v1";
    }
  } catch (_) {
    // 取得できなければ既定値のまま進む（checkApiStatus が接続失敗を表示する）
  }
}

// ===== 読み辞書（朗読・Synthesize 共通 / localStorage: narrate_dict = {表記:{yomi,memo}}） =====
// 旧形式（値が文字列＝よみ）も読めるよう dictYomiOf で吸収
function loadDict()  { try { return JSON.parse(localStorage.getItem('narrate_dict') || '{}'); } catch { return {}; } }
function saveDict(d) { localStorage.setItem('narrate_dict', JSON.stringify(d)); }
function dictYomiOf(v) { return (v && typeof v === 'object') ? (v.yomi || '') : (v || ''); }
function dictMemoOf(v) { return (v && typeof v === 'object') ? (v.memo || '') : ''; }
function applyDict(text) {
  // キーごとに順番に replace すると、置換結果に別のキーが含まれるとき
  // 二重に当たる（{獅子:しし, しし:シシ} で 獅子舞 → シシ舞）。全キーを
  // 1本の正規表現にまとめて左から1回だけ走査し、出力は再走査しない。
  const d = loadDict();
  const keys = Object.keys(d)
    .filter(k => k && dictYomiOf(d[k]))
    .sort((a, b) => b.length - a.length);   // 最長一致を優先
  if (!keys.length) return text;
  // 表記に . や ( を含む登録がありうるのでエスケープする
  const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(keys.map(esc).join('|'), 'g');
  return text.replace(re, m => dictYomiOf(d[m]));
}

// ── 地の文へのナレーション絵文字 ──
//
// 📖 は v4 で追加された注釈で、公式表では「ナレーション、独白、モノローグ」。
// v3 以前の 41 個の表には入っていないので、v3 に付けても学習されておらず
// 効かない（UI に注記を出す）。
//
// 位置は公式サンプルに倣って対象の文の先頭に置く。上流に「文頭に置く」と
// 明記した規約は無いが、モデルカードの例はいずれも効かせたい箇所の直前に
// 差し込む形で、文全体を囲む例は無い。
// キャプション（声の説明文）で条件づけるモデル。参照音声ではなく文章で
// 声を決めるので、UI の出し分けが他と違う。v2 版に加えて v3 の 600M 版がある。
const VOICE_DESIGN_MODELS = ['voice_design', 'v3_voice_design'];
const isVoiceDesignModel = m => VOICE_DESIGN_MODELS.includes(String(m));

// v4 系（v4-Small と v4.1-Small）。v4.1 は duration predictor だけを
// 差し替えたもので、条件づけの作りは v4 と同じ。
const V4_MODELS = ['v4', 'v4_1'];
const isV4Model = m => V4_MODELS.includes(String(m));

const NARRATION_EMOJI = '📖';

// セリフの目印になる括弧。開き括弧で始まる行はセリフと見なして手を付けない。
// 会話文の慣習に合わせて鉤括弧・二重鉤・引用符・丸括弧（心の声）まで見る。
//
// 分割（splitText）と地の文の判定で同じ表を使う。片方だけ鉤括弧しか見ないと、
// 『はい』。地の文。 のような行が切れずに潰れ、判定まで巻き添えで外れる。
const QUOTE_PAIRS = {
  '「': '」', '『': '』', '（': '）', '(': ')',
  '“': '”', '‘': '’', '〈': '〉', '《': '》',
  '【': '】', '〔': '〕', '[': ']',
};
const QUOTE_OPENERS = Object.keys(QUOTE_PAIRS).join('') + '"\'';
const QUOTE_CLOSERS = Object.values(QUOTE_PAIRS).join('');

// チャンク分割で「いまセリフの中か」を追うのは鉤括弧系だけにする。
// 丸括弧や引用符は本文中に単独で現れる（"a) の形"、閉じない "(" など）。
// これを追跡に混ぜると、閉じない ( ひとつで段落の残り全部が 1 チャンクに
// 潰れ、単独の ) では語中で切れる。上の QUOTE_PAIRS は地の文の判定
// （両端に括弧があるか）に使うので、そちらは広いままでよい。
const DIALOG_PAIRS = { '「': '」', '『': '』' };
const DIALOG_CLOSERS = Object.values(DIALOG_PAIRS).join('');
// 閉じ括弧の直後にこれが続くなら、まだ文の途中（「〜」と言った）。
const DIALOG_TAIL_RE = /[とがをにはでもやからって、]/;

function isNarrationLine(text) {
  const t = String(text || '').trim();
  if (!t) return false;
  // 既に注釈が付いている行は触らない（重ねると尺の予測がずれる）。
  if (t.includes(NARRATION_EMOJI)) return false;
  // 先頭が開き括弧、または末尾が閉じ括弧ならセリフ。両端を見るのは、
  // 「はい」。彼は笑った。 のように括弧で始まる行だけでなく、
  // 彼は言った「行こう」 のように括弧で終わる行も外すため。
  // 途中に括弧があるだけ（彼は「はい」と言った。）は地の文として扱う。
  if (QUOTE_OPENERS.includes(t[0])) return false;
  if (QUOTE_CLOSERS.includes(t[t.length - 1])) return false;
  return true;
}

function withNarrationEmoji(text, enabled) {
  if (!enabled) return text;
  const t = String(text || '');
  if (!isNarrationLine(t)) return t;
  // 行頭の空白は残したまま、本文の直前に差し込む。
  const lead = t.match(/^\s*/)[0];
  return lead + NARRATION_EMOJI + t.slice(lead.length);
}

// DOM Elements
const statusBadge = document.getElementById('status-badge');
const modelSelect = document.getElementById('model-type');
const captionWrapper = document.getElementById('caption-wrapper');
const captionInput = document.getElementById('caption-input');
const textInput = document.getElementById('text-input');

const paramSteps = document.getElementById('param-steps');
const paramCfgText = document.getElementById('param-cfg-text');
const paramDurationScale = document.getElementById('param-duration-scale');
const valDurationScale = document.getElementById('val-duration-scale');

// 実台詞195文（地の文60・喘ぎ交じり60・喘ぎ39・オホ声36）を v4 で生成した実測値の
// 最小二乗フィット。テキストは学習コーパスから層別抽出し、長さは v4 の出力を実測した
// （収録元素材の長さは v4 の出力長と一致しないので使えない）。
// 素の文字数では合わないため2点を補正する:
//   1. 句読点は「間」として尺を食う → 1個を 1.25 文字ぶんとして数える。
//   2. 同じ仮名の連続は1文字ぶんの時間で発音されない。喘ぎ・オホ声で顕著。
//      → 連続 n 文字を 1 + (n-1)*0.70 として数える。
// 未使用の98文で検証して誤差 中央値0.47秒・p90 1.33秒。
// （手書き例文18文だけで作った旧係数はオホ声で中央値1.96秒外していた）
// 長さは duration_scale に比例するので、基準との比で伸縮させる。
// 0.75 は係数を実測したときの scale であって既定値ではない。既定を 1.0 に
// 変えてもここは動かさない（動かすと実測との対応が崩れて予測がずれる）。
const SEC_PER_UNIT = 0.1782;
const SEC_INTERCEPT = 0.327;
const REPEAT_WEIGHT = 0.70;
const PUNCT_WEIGHT = 1.25;
const PUNCT_RE = /[、。！？，．,.!?…]/;
const DURATION_SCALE_BASE = 0.75;

// 表記上の文字数ではなく、発話にかかる「単位数」を数える。
// text[i] は UTF-16 単位なので、絵文字などのサロゲートペアが2単位に割れて
// 連続判定も壊れる。NFC 正規化してコードポイント単位で走査する。
function speechUnits(text) {
  const chars = Array.from(text.normalize('NFC'));
  let total = 0;
  for (let i = 0; i < chars.length;) {
    const ch = chars[i];
    let j = i;
    while (j < chars.length && chars[j] === ch) j++;
    const run = j - i;
    // 句読点も連続すれば詰まって発音される（「……」「！！！」）ので、
    // 文字と同じく連続ぶんを圧縮したうえで句読点の重みを掛ける。
    total += PUNCT_RE.test(ch)
      ? PUNCT_WEIGHT * (1 + (run - 1) * REPEAT_WEIGHT)
      : 1 + (run - 1) * REPEAT_WEIGHT;
    i = j;
  }
  return total;
}

function updateDurationEstimate() {
  const scale = parseFloat(paramDurationScale.value);
  valDurationScale.textContent = scale.toFixed(2);

  // duration_scale が効くのは尺を予測するモデルだけ。v2 系は固定 30 秒で
  // 学習されていて予測を持たないので、倍率を変えても出力が動かない
  // （実測: v2 4.64秒 / voice_design 4.72秒 が 0.5 でも 1.0 でも同値。
  // 対して v3 は 1.56 <-> 3.16秒、v4 は 1.60 <-> 3.24秒と追従する）。
  // 効かないモデルで動かせると「効いたように見えて何も起きない」ので無効化する。
  const model = modelSelect ? modelSelect.value : 'v4';
  const hasDuration = model !== 'v2' && model !== 'voice_design';
  paramDurationScale.disabled = !hasDuration;
  const group = paramDurationScale.closest('.param-group');
  if (group) group.style.opacity = hasDuration ? '' : '0.45';

  // 語尾の捏造は scale ではなく LoRA の性質で決まる（ASR での実測:
  // ある LoRA は 0.60 以上で常に捏造、他の4体は 0.90 でも捏造ゼロ）。
  // 固定の閾値が出せない以上、動的な警告は出さない。注意書きは
  // ツールチップと「はじめにお読みください」に置く。
  const warn = document.getElementById('duration-scale-warn');
  if (warn) warn.textContent = hasDuration ? '' : '（v2 系は尺予測を持たないため効きません）';

  const est = document.getElementById('val-duration-estimate');
  if (!est) return;
  const text = (textInput.value || '').replace(/\s/g, '');
  // 係数は LoRA 音声のコーパスにフィット。素の v4 では誤差が約1.5倍
  // （中央値0.68秒 / p90 2.07秒・長め寄り）になるが、目安としては
  // 有用なので常に出す。ずれの注意は「はじめにお読みください」に記載。
  // 予測式の係数は v4 実測から作ったもの。尺が動く v3 でも当てはまる保証が
  // 無いので、数値を出すのは v4 だけに留める（スライダー自体は v3 でも有効）。
  if (!text || !isV4Model(model)) { est.textContent = ''; return; }
  // 切片が正なので現係数では下限に当たらないが、係数を差し替えたときに
  // 負値やゼロを表示しないためのガードとして残す。
  const sec = Math.max(0.3, SEC_PER_UNIT * speechUnits(text) + SEC_INTERCEPT) * (scale / DURATION_SCALE_BASE);
  est.textContent = ` / 予測 約${sec.toFixed(1)}秒`;
}
const paramCfgSpeaker = document.getElementById('param-cfg-speaker');
const paramSeed = document.getElementById('param-seed');

const valSteps = document.getElementById('val-steps');
const valCfgText = document.getElementById('val-cfg-text');
const valCfgSpeaker = document.getElementById('val-cfg-speaker');

const voiceList = document.getElementById('voice-list');
const addVoiceForm = document.getElementById('add-voice-form');
const selectWavBtn = document.getElementById('select-wav-btn');
const saveVoiceBtn = document.getElementById('save-voice-btn');
const selectedWavPathDisplay = document.getElementById('selected-wav-path');
const newVoiceName = document.getElementById('new-voice-name');

const paramCandidates = document.getElementById('param-candidates');
const valCandidates = document.getElementById('val-candidates');
const paramCfgMode = document.getElementById('param-cfg-mode');
const paramCfgCaption = document.getElementById('param-cfg-caption');
const valCfgCaption = document.getElementById('val-cfg-caption');
const copyApiBtn = document.getElementById('copy-api-btn');
const apiUrlDisplay = document.getElementById('api-url-display');

// Advanced params
const paramCfgOverride = document.getElementById('param-cfg');
const paramCfgMinT = document.getElementById('param-cfg-min-t');
const paramCfgMaxT = document.getElementById('param-cfg-max-t');
const paramMaxTextLen = document.getElementById('param-max-text-len');
const paramDevice    = document.getElementById('param-device');
const valDevice      = document.getElementById('val-device');
if (paramDevice) {
  paramDevice.addEventListener('change', () => {
    localStorage.setItem('synth_device', paramDevice.value);
    invalidateNarrateSession();
  });
  // 保存値の復元は applyDeviceOptions()。ここで入れても、選択肢がまだ
  // auto しか無いので value が空になり、次の保存で auto に戻ってしまう。
}
const paramPrecision = document.getElementById('param-precision');
const valPrecision   = document.getElementById('val-precision');
if (paramPrecision) {
  // 自動のときだけ「→ 実際に使う精度」を出す。明示指定なら見たままなので
  // 何も足さない。値は次の checkApiStatus で GPU 名つきに更新される。
  paramPrecision.addEventListener('change', () => {
    if (valPrecision && paramPrecision.value !== 'auto') valPrecision.textContent = '';
    localStorage.setItem('synth_precision', paramPrecision.value);
    // 朗読も同じ精度を使う。鳴っている最中に変えたら次の行から作り直す。
    invalidateNarrateSession();
  });
  const saved = localStorage.getItem('synth_precision');
  if (saved && ['auto', 'fp32', 'bf16'].includes(saved)) paramPrecision.value = saved;
}
const paramContextKv = document.getElementById('param-context-kv');
const paramTruncFactor = document.getElementById('param-truncation-factor');
const paramRescaleK = document.getElementById('param-rescale-k');
const paramRescaleSigma = document.getElementById('param-rescale-sigma');
const paramSpeakerKvScale = document.getElementById('param-speaker-kv-scale');
const paramSpeakerKvMinT = document.getElementById('param-speaker-kv-min-t');
const paramSpeakerKvMaxLyr = document.getElementById('param-speaker-kv-max-lyr');
const paramTScheduleMode = document.getElementById('param-t-schedule-mode');
const paramSwayCoeff = document.getElementById('param-sway-coeff');
const valSwayCoeff = document.getElementById('val-sway-coeff');

// LoRA elements
const loraSelect = document.getElementById('lora-select');
const condModeRadios = document.getElementsByName('cond-mode');
const condModeLoraRadio = document.getElementById('cond-mode-lora');
const manageLorasBtn = document.getElementById('manage-loras-btn');
const loraModal = document.getElementById('lora-modal');
const loraModalClose = document.getElementById('lora-modal-close');
const loraImportName = document.getElementById('lora-import-name');
const loraImportBase = document.getElementById('lora-import-base');
const loraImportNotes = document.getElementById('lora-import-notes');
const loraImportPickBtn = document.getElementById('lora-import-pick');
const loraImportPath = document.getElementById('lora-import-path');
const loraImportSubmit = document.getElementById('lora-import-submit');
const loraListBody = document.getElementById('lora-list-body');
const loraListEmpty = document.getElementById('lora-list-empty');

let loraRegistry = [];
let loraTotalBytes = 0;
let _serverWasOnline = false;          // Last fetched /api/v1/loras response
let pickedAdapterPath = null;   // Path chosen via folder picker

const currentVoiceName = document.getElementById('current-voice-name');
const dot = document.querySelector('.active-dot');
const clearVoiceBtn = document.getElementById('clear-voice-btn');
const generateBtn = document.getElementById('generate-btn');
const genSpinner = document.getElementById('gen-spinner');
const btnTextSpan = generateBtn.querySelector('span');

const resultContainer = document.getElementById('result-container');
const emojiToolbar = document.getElementById('emoji-toolbar');
const charCounter = document.getElementById('char-counter');

function updateCharCount() {
  // 単位（字）は HTML 側に置いてある。ここは数字だけを入れる。
  charCounter.textContent = String(textInput.value.length);
}

// =====================================================================
// セリフの積み上げ
//   1 行 1 セリフ。合成は行ごとに順へ流す（API は 1 回 1 テキスト）。
//   1 行目の textarea は #text-input のまま使う。文字数・推定秒・タグ挿入が
//   これを見ているので、作り直すとその 3 つを全部書き換えることになる。
// =====================================================================

// いま触っている行。タグはここへ挿す（どれも触っていなければ 1 行目）。
let activeLineTa = null;

// 行を消した後も参照が残ることがある。画面から外れた欄には挿さない。
function ta_alive(ta) { return !!(ta && ta.isConnected); }

const linesEl = () => document.getElementById('lines');

function allLineTextareas() {
  const host = linesEl();
  return host ? [...host.querySelectorAll('textarea')] : [textInput];
}

// 入力欄の高さを中身に合わせる。1 行のときは 1 行分で、
// 増えたぶんだけ伸びる（縦スクロールを出さない）。
function autoGrow(ta) {
  ta.style.height = 'auto';
  ta.style.height = ta.scrollHeight + 'px';
}

function lineSeconds(text) {
  const t = (text || '').replace(/\s/g, '');
  if (!t) return 0;
  const scale = parseFloat(paramDurationScale.value);
  return Math.max(0.3, SEC_PER_UNIT * speechUnits(t) + SEC_INTERCEPT)
       * (scale / DURATION_SCALE_BASE);
}

// 行ごとの「字数・予測秒」と、足の「行数・合計秒」を書き直す。
function refreshLines() {
  const host = linesEl();
  if (!host) return;
  const rows = [...host.querySelectorAll('.line')];
  let total = 0, filled = 0;

  rows.forEach((row, i) => {
    const ta = row.querySelector('textarea');
    row.dataset.line = String(i);
    const no = row.querySelector('.ln');
    if (no) no.textContent = String(i + 1);

    const sec = lineSeconds(ta.value);
    total += sec;
    if (ta.value.trim()) filled++;

    // 1 行目の欄は既存の #char-counter / #val-duration-estimate が使う。
    // 2 行目からは行の中に同じ形で出す。
    if (i > 0) {
      const meta = row.querySelector('.meta');
      if (meta) {
        meta.innerHTML = ta.value
          ? `<b>${ta.value.length}</b> 字${sec ? ` / 予測 約${sec.toFixed(1)}秒` : ''}`
          : '';
      }
    }
    autoGrow(ta);
  });

  // 行が 1 本だけのときは消せないようにする（消すと入力欄が無くなる）
  rows.forEach(r => {
    const del = r.querySelector('.ln-del');
    if (del) del.style.visibility = rows.length > 1 ? '' : 'hidden';
  });

  const sum = document.getElementById('lines-sum');
  if (sum) {
    sum.innerHTML = rows.length > 1 || total
      ? `<b>${filled || rows.length}</b> 行 ・ 合計 <b>${total.toFixed(1)}</b> 秒`
      : '';
  }
  updateCandidateLock(rows.length);
}

// セリフが 2 行以上あるときは候補を作らない。
// 5 行 × 候補 3 で 15 本並んでも選びようがない。
function updateCandidateLock(lineCount) {
  const group = paramCandidates ? paramCandidates.closest('.param-group') : null;
  if (!paramCandidates || !group) return;
  const multi = lineCount > 1;
  paramCandidates.disabled = multi;
  group.classList.toggle('is-locked', multi);
  let why = group.querySelector('.lock-why');
  if (multi) {
    if (!why) {
      why = document.createElement('span');
      why.className = 'lock-why';
      group.appendChild(why);
    }
    why.textContent = '複数セリフのため 1 に固定';
    if (paramCandidates.value !== '1') {
      paramCandidates.value = '1';
      if (valCandidates) valCandidates.textContent = '1';
    }
  } else if (why) {
    why.remove();
  }
}

function bindLine(row) {
  const ta = row.querySelector('textarea');
  ta.addEventListener('input', refreshLines);
  ta.addEventListener('focus', () => { activeLineTa = ta; });
  // Enter で次のセリフへ。改行を入れたいときは Shift+Enter。
  ta.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      addLine(row);
    }
  });
  const del = row.querySelector('.ln-del');
  if (del) {
    del.addEventListener('click', () => {
      const host = linesEl();
      if (host.querySelectorAll('.line').length <= 1) return;
      if (ta === textInput) {
        // 1 行目は他所から参照されているので消さない。
        // 中身だけ 2 行目から引き取って、2 行目のほうを消す。
        const next = row.nextElementSibling;
        if (next) { textInput.value = next.querySelector('textarea').value; next.remove(); }
      } else {
        row.remove();
      }
      if (activeLineTa === ta) activeLineTa = textInput;
      refreshLines(); updateCharCount(); updateDurationEstimate();
    });
  }
}

function addLine(afterRow) {
  const host = linesEl();
  if (!host) return;
  const row = document.createElement('div');
  row.className = 'line';
  row.innerHTML =
    `<span class="ln"></span>` +
    `<div class="editor-field">` +
      `<textarea rows="1" placeholder="読ませたい文章を入力してください。"></textarea>` +
      `<span class="meta"></span>` +
    `</div>` +
    `<button class="ln-del tip" data-tip="この行を削除" aria-label="この行を削除">` +
      `<svg class="i i-sm"><use href="#ic-x"/></svg></button>`;
  if (afterRow && afterRow.nextSibling) host.insertBefore(row, afterRow.nextSibling);
  else host.appendChild(row);
  bindLine(row);
  refreshLines();
  const ta = row.querySelector('textarea');
  ta.focus();
  activeLineTa = ta;
  return row;
}

function setupLines() {
  const host = linesEl();
  if (!host) return;
  host.querySelectorAll('.line').forEach(bindLine);
  activeLineTa = textInput;
  const add = document.getElementById('line-add');
  if (add) add.addEventListener('click', () => addLine(host.lastElementChild));
  refreshLines();
}

// v3 official emoji palette (45). Mirrors irodori_tts/gradio_emoji_palette.py.
// `d` is the longer description, shown as the button tooltip.
const EMOJI_DEFS = [
  { e: '👂',   l: '囁き',         d: '耳元の音' },
  { e: '😮‍💨', l: '吐息',         d: '溜息、寝息' },
  { e: '⏸️',   l: '間',           d: '沈黙' },
  { e: '🤭',   l: '笑い',         d: 'くすくす、含み笑い' },
  { e: '🥵',   l: '喘ぎ',         d: 'うめき声、唸り声' },
  { e: '📢',   l: 'エコー',       d: 'リバーブ' },
  { e: '😏',   l: 'からかう',     d: '甘えるように' },
  { e: '🥺',   l: '震え声',       d: '自信なさげに' },
  { e: '🌬️',   l: '息切れ',       d: '荒い息遣い、呼吸音' },
  { e: '😮',   l: '息をのむ',     d: 'Gasp' },
  { e: '👅',   l: '舐める音',     d: '咀嚼音、水音' },
  { e: '💋',   l: 'リップノイズ', d: 'Lip smack' },
  { e: '🫶',   l: '優しく',       d: 'Tenderly' },
  { e: '😭',   l: '泣き声',       d: '嗚咽、悲しみ' },
  { e: '😱',   l: '悲鳴',         d: '叫び、絶叫' },
  { e: '😪',   l: '眠そう',       d: '気だるげに' },
  { e: '😴',   l: '寝言',         d: 'いびき' },
  { e: '⏩',   l: '早口',         d: '一気に、急いで' },
  { e: '📞',   l: '電話越し',     d: 'スピーカー越し' },
  { e: '🐢',   l: 'ゆっくり',     d: 'Slowly' },
  { e: '🥤',   l: '飲み込む',     d: '唾を飲む音' },
  { e: '🤧',   l: '咳・鼻',       d: '咳き込み、鼻すすり' },
  { e: '😒',   l: '舌打ち',       d: 'Tutting' },
  { e: '😰',   l: '慌てる',       d: '動揺、緊張、どもり' },
  { e: '😆',   l: '喜び',         d: '嬉しそうに' },
  { e: '💥',   l: '勢いよく',     d: '力強い勢い' },
  { e: '😠',   l: '怒り',         d: '不満げ、拗ねる' },
  { e: '😲',   l: '驚き',         d: '感嘆' },
  { e: '🥱',   l: 'あくび',       d: 'Yawn' },
  { e: '😖',   l: '苦しげ',       d: 'Agonizingly' },
  { e: '😟',   l: '心配',         d: '不安そうに' },
  { e: '🫣',   l: '照れ',         d: '恥ずかしそうに' },
  { e: '🙄',   l: '呆れ',         d: 'Exasperatedly' },
  { e: '😊',   l: '楽しげ',       d: '嬉しそうに' },
  { e: '😎',   l: '得意げ',       d: '自信ありげに' },
  { e: '👌',   l: '相槌',         d: '頷く音' },
  { e: '🙏',   l: '懇願',         d: 'お願いするように' },
  { e: '🥴',   l: '酔う',         d: 'Drunkenly' },
  { e: '🎵',   l: '鼻歌',         d: 'Humming' },
  { e: '🤐',   l: '口を塞ぐ',     d: 'Muffled' },
  { e: '😌',   l: '安堵',         d: '満足げに' },
  { e: '🤔',   l: '疑問',         d: 'Questioning' },
  { e: '💪',   l: '力強く',       d: '力を込めて' },
  { e: '👃',   l: '嗅ぐ音',       d: '匂いを嗅ぐ音' },
  { e: '📖',   l: '朗読',         d: 'ナレーション' }
];


// ── 意匠まわりの下ごしらえ ──────────────────────────────
// アイコンの実体。HTML パーサ（innerHTML）に通すと <defs> 内の
// rect / circle が捨てられ、一部のアイコンだけ描かれなくなるので、
// SVG として解釈させてから差し込む。
const ICON_SPRITE = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"0\" height=\"0\" style=\"position:absolute\" aria-hidden=\"true\"><defs>\n  <g id=\"ic-wave\"><path d=\"M2 8h2M6 4v8M10 1.5v13M14 5v6\"/></g>\n  <g id=\"ic-db\"><ellipse cx=\"8\" cy=\"4\" rx=\"5.5\" ry=\"2.2\"/><path d=\"M2.5 4v8c0 1.2 2.5 2.2 5.5 2.2s5.5-1 5.5-2.2V4\"/><path d=\"M2.5 8c0 1.2 2.5 2.2 5.5 2.2s5.5-1 5.5-2.2\"/></g>\n  <g id=\"ic-spark\"><path d=\"M8 1.5l1.6 4.3 4.4 1.7-4.4 1.7L8 13.5l-1.6-4.3L2 7.5l4.4-1.7z\"/></g>\n  <g id=\"ic-book\"><path d=\"M2.5 3.2c2-.9 4-.9 5.5.3 1.5-1.2 3.5-1.2 5.5-.3v9.4c-2-.9-4-.9-5.5.3-1.5-1.2-3.5-1.2-5.5-.3z\"/><path d=\"M8 3.5v9.7\"/></g>\n  <!-- 青空文庫。朗読が開いた本、辞書が線入りの本なので 3 つ目の本は避けて雲にする -->\n  <g id=\"ic-sky\"><path d=\"M11.67 11.5H6a4.67 4.67 0 1 1 4.47-6h1.19a3 3 0 1 1 0 6Z\"/></g>\n  <g id=\"ic-merge\"><path d=\"M4 2v4a4 4 0 004 4h4\"/><path d=\"M4 14v-4\"/><path d=\"M10 8l2.5 2L10 12\"/></g>\n  <g id=\"ic-dict\"><path d=\"M3 2.5h7.5a2 2 0 012 2v9H5a2 2 0 01-2-2z\"/><path d=\"M5.5 6h5M5.5 8.5h3.5\"/></g>\n  <g id=\"ic-chev\"><path d=\"M6 3l5 5-5 5\"/></g>\n  <g id=\"ic-folder\"><path d=\"M1.5 3.5h4.5l1.5 2h7v7a1 1 0 01-1 1h-11a1 1 0 01-1-1z\"/></g>\n  <g id=\"ic-refresh\"><path d=\"M14 8a6 6 0 11-1.8-4.3\"/><path d=\"M14 1.5V5h-3.5\"/></g>\n  <g id=\"ic-save\"><path d=\"M3 2.5h8l2 2v9h-10z\"/><path d=\"M5 2.5v4h5v-4M5 13.5v-4h6v4\"/></g>\n  <g id=\"ic-play2\"><path d=\"M5 3.2l7 4.8-7 4.8z\" fill=\"currentColor\" stroke=\"none\"/></g>\n  <!-- 生成。再生（▶）と同じ形にすると「もう出来たものを鳴らす」に見える。\n       文字から声が立ち上がる図として、縦棒が右へ伸びていく形にする -->\n  <g id=\"ic-gen\"><path d=\"M2.5 8h1.6M6 5.4v5.2M9.4 2.6v10.8M12.8 4.6v6.8\"/><path d=\"M12.8 8h1.7\" stroke-opacity=\".45\"/></g>\n  <g id=\"ic-dl\"><path d=\"M8 2v8M4.5 7L8 10.5 11.5 7M2.5 13.5h11\"/></g>\n  <g id=\"ic-x\"><path d=\"M4 4l8 8M12 4l-8 8\"/></g>\n  <g id=\"ic-plus\"><path d=\"M8 3v10M3 8h10\"/></g>\n  <g id=\"ic-prev\"><path d=\"M12.5 3.2v9.6L5.8 8z\" fill=\"currentColor\" stroke=\"none\"/><path d=\"M3.6 3.2v9.6\"/></g>\n  <g id=\"ic-next\"><path d=\"M3.5 3.2v9.6L10.2 8z\" fill=\"currentColor\" stroke=\"none\"/><path d=\"M12.4 3.2v9.6\"/></g>\n  <g id=\"ic-pause\"><path d=\"M6 3.4v9.2M10 3.4v9.2\" stroke-width=\"2\"/></g>\n  <g id=\"ic-stop\"><rect x=\"4.2\" y=\"4.2\" width=\"7.6\" height=\"7.6\" rx=\"1\" fill=\"currentColor\" stroke=\"none\"/></g>\n  <g id=\"ic-mark\"><path d=\"M4 2.5h8v11l-4-3-4 3z\"/></g>\n  <g id=\"ic-clock\"><circle cx=\"8\" cy=\"8\" r=\"6.2\"/><path d=\"M8 4.4V8l2.6 1.6\"/></g>\n  <g id=\"ic-gear\"><circle cx=\"8\" cy=\"8\" r=\"2.4\"/><path d=\"M8 1.6v1.8M8 12.6v1.8M14.4 8h-1.8M3.4 8H1.6M12.5 3.5l-1.3 1.3M4.8 11.2l-1.3 1.3M12.5 12.5l-1.3-1.3M4.8 4.8L3.5 3.5\"/></g>\n  <g id=\"ic-speaker\"><path d=\"M3 6.2h2.4L8.6 3.4v9.2L5.4 9.8H3z\"/><path d=\"M11 6a3 3 0 010 4\"/></g>\n  <g id=\"ic-search\"><circle cx=\"7.2\" cy=\"7.2\" r=\"4.6\"/><path d=\"M10.6 10.6L14 14\"/></g>\n  <g id=\"ic-moon\"><path d=\"M13.4 9.7A5.9 5.9 0 016.3 2.6a5.95 5.95 0 107.1 7.1z\"/></g>\n  <!-- 役者。人型は肩の弧だけだと椅子に見えるので、首を細く残して頭と胴を離す -->\n  <g id=\"ic-actor\"><circle cx=\"8\" cy=\"5\" r=\"2.8\"/><path d=\"M2.8 14c0-2.9 2.3-4.6 5.2-4.6s5.2 1.7 5.2 4.6\"/></g>\n  <!-- 撮る。写真の枠に山と陽。動画（ic-reel）と紛れないよう枠は角丸の静止画にする -->\n  <g id=\"ic-shot\"><rect x=\"2\" y=\"3.2\" width=\"12\" height=\"9.6\" rx=\"1.4\"/><circle cx=\"5.6\" cy=\"6.4\" r=\"1.1\"/><path d=\"M2.4 11.2l3.4-3 2.6 2.3 2.2-1.9 3 2.6\"/></g>\n  <!-- 仕上げ。フィルムの駒。静止画（ic-shot）と違い縦の抜きで動きを出す -->\n  <g id=\"ic-reel\"><rect x=\"1.8\" y=\"3.4\" width=\"12.4\" height=\"9.2\" rx=\"1.2\"/><path d=\"M5.4 3.4v9.2M10.6 3.4v9.2\"/></g>\n</defs></svg>";

function installIcons() {
  const host = document.getElementById('icon-sprite');
  if (!host) return;
  const svg = new DOMParser().parseFromString(ICON_SPRITE, 'image/svg+xml').documentElement;
  host.appendChild(document.importNode(svg, true));
  // アイコンは 16x16 の座標で描いてある。viewBox が無いと 16px 以外の
  // 大きさにしたとき中身が枠外へ出て消える。viewBox は CSS では指定できない。
  document.querySelectorAll('svg.i:not([viewBox])')
    .forEach((el) => el.setAttribute('viewBox', '0 0 16 16'));
}

// 画面の下角に置く飾り。中身と一緒にスクロールさせないため画面に貼る。
// 右パネル（Voice Library）があるタブでは 4 か所、無いタブでは 2 か所。
function installCorners() {
  [['corner-bl', 'corner-main-l'], ['corner-br', 'corner-main-r'],
   ['corner-bl', 'corner-side-l'], ['corner-br', 'corner-side-r']]
    .forEach(([shape, place]) => {
      const d = document.createElement('div');
      d.className = 'corner ' + shape + ' ' + place;
      document.body.appendChild(d);
    });
  syncCorners();
}

// 右パネルの有無はタブで変わる。切り替えのたびに見直す。
// 柱のクラスはタブで違う（生成は .aside、朗読は .narr-side）。
// 朗読は .narr で一段包んでいるので、直下ではなく子孫まで見る。
function syncCorners() {
  const pane = document.querySelector('.tab-pane.active');
  const hasAside = !!(pane && pane.querySelector('.aside, .narr-side'));
  document.body.classList.toggle('no-aside', !hasAside);
}

// Initialize
async function init() {
  installIcons();      // 先に入れる。以降の描画が <use> を引けるように
  installCorners();
  await resolveApiOrigin(); // 以降の fetch より先にポートを確定させる
  checkApiStatus();
  setInterval(checkApiStatus, 5000); // Check every 5s

  setupTabSwitching();
  renderEmojis();
  setupLines();
  loadVoices();
  setupEventListeners();
  setupLoraEventListeners();
  loadLoras();
  setupDatasetTab();
  setupTrainTab();
  setupSynthSaveFolder();
  // 学習は Train タブを離れても続く。生成側でも状態を持っておく。
  pollTrainingBusy();
  setInterval(pollTrainingBusy, 3000);
}

function setupSynthSaveFolder() {
  const pathEl   = document.getElementById('synth-save-folder-path');
  const btn      = document.getElementById('synth-save-folder-btn');
  const clearBtn = document.getElementById('synth-save-folder-clear');
  if (!pathEl || !btn || !clearBtn) return;
  function update(folder) {
    if (folder) {
      // 見出しの中に置くので、full path だと幅を食い潰す。
      // 末尾のフォルダ名だけ出し、全体は title で見せる。
      pathEl.textContent = '保存先: ' + folder.split(/[\\/]/).filter(Boolean).pop();
      pathEl.title = folder;
      clearBtn.classList.remove('hidden');
    } else {
      pathEl.textContent = '保存先: 未設定';
      pathEl.title = '';
      clearBtn.classList.add('hidden');
    }
  }
  update(localStorage.getItem('synth_save_folder'));
  btn.addEventListener('click', async () => {
    const folder = await window.api.selectSaveFolder();
    if (folder) { localStorage.setItem('synth_save_folder', folder); update(folder); }
  });
  clearBtn.addEventListener('click', () => {
    localStorage.removeItem('synth_save_folder');
    update(null);
  });
}

function setupTabSwitching() {
  const tabs = document.querySelectorAll('.tab-btn');
  const panes = document.querySelectorAll('.tab-pane');
  tabs.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;
      tabs.forEach(t => t.classList.toggle('active', t === btn));
      panes.forEach(p => p.classList.toggle('active', p.dataset.tab === target));
      syncCorners();   // 右パネルの有無で角飾りの数が変わる
      if (target === 'dataset') loadDatasets();
      if (target === 'aozora')   window._aozRefreshSaved && window._aozRefreshSaved();
      if (target === 'loramix') window._loramixRefresh && window._loramixRefresh();
      if (target === 'train') {
        refreshTrainDatasetOptions();
        loadTrainJobs();
      }
    });
  });
}

// 演出タグの分類。左端の色帯で見分けるためのもので、
// 挿入される絵文字そのものには影響しない。
// 分類は EMOJI_DEFS の並びではなく語で決める（並びは変わりうるため）。
const EMOJI_GROUPS = {
  g1: ['喜び', '楽しげ', '安堵', 'からかう', '得意げ', '優しく', '怒り', '驚き',
       '悲鳴', '泣き声', '心配', '慌てる', '照れ', '呆れ', '苦しげ', '懇願', '疑問'],
  g2: ['囁き', '笑い', '震え声', '力強く', '勢いよく', '早口', 'ゆっくり', '眠そう',
       '酔う', '口を塞ぐ', '朗読', '鼻歌', '喘ぎ'],
  g3: ['間', '吐息', '息切れ', '息をのむ', 'あくび', '寝言', '相槌'],
  g4: ['舐める音', 'リップノイズ', '飲み込む', '咳・鼻', '舌打ち', '嗅ぐ音'],
  g5: ['エコー', '電話越し'],
};

function emojiGroupOf(item) {
  for (const [g, words] of Object.entries(EMOJI_GROUPS)) {
    if (words.includes(item.l)) return g;
  }
  return 'g1';   // 分類にない語が増えたときの受け皿
}

// 45 個を 5 つに束ねて出す。ひと並びに全部出すと、
// 探すのに毎回 45 個を読むことになる。
const EMOJI_GROUP_LABELS = {
  g1: '感情', g2: '声の出し方', g3: '息・間', g4: '物音', g5: '空間・加工',
};

function renderEmojis() {
  emojiToolbar.innerHTML = '';

  // 畳めるようにする。既定では開いておく
  const box = document.createElement('details');
  box.className = 'tags';
  box.open = true;
  box.innerHTML =
    `<summary><svg class="i i-sm chev"><use href="#ic-chev"/></svg>` +
    `感情・演出タグ <span class="n">${EMOJI_DEFS.length}</span></summary>` +
    `<div class="tag-wrap"></div>`;
  const wrap = box.querySelector('.tag-wrap');

  Object.entries(EMOJI_GROUP_LABELS).forEach(([g, label]) => {
    const items = EMOJI_DEFS.filter((it) => emojiGroupOf(it) === g);
    if (!items.length) return;

    const group = document.createElement('div');
    group.className = `tag-group ${g}`;
    group.innerHTML = `<span class="tag-cat">${label}</span><div class="tag-row"></div>`;
    const row = group.querySelector('.tag-row');

    items.forEach((item) => {
      const btn = document.createElement('button');
      // 分類の色は CSS 側（.tag.g1〜g5）が付ける。
      // ボタンには絵文字を出さないが、挿入する中身は絵文字のまま。
      // モデルはこの絵文字を演出の指示として読むので変えてはいけない。
      btn.className = 'tag';
      btn.textContent = item.l;
      btn.title = item.d ? `${item.l}: ${item.d}` : item.l;

      btn.addEventListener('click', () => {
        // 最後に触っていた行へ挿す。どれも触っていなければ 1 行目。
        const ta = (activeLineTa && ta_alive(activeLineTa)) ? activeLineTa : textInput;
        const start = ta.selectionStart;
        const end = ta.selectionEnd;
        const text = ta.value;
        ta.value = text.substring(0, start) + item.e + text.substring(end);
        ta.focus();
        ta.selectionStart = ta.selectionEnd = start + item.e.length;
        if (ta === textInput) updateCharCount();
        refreshLines();
      });

      row.appendChild(btn);
    });

    wrap.appendChild(group);
  });

  emojiToolbar.appendChild(box);
}

let isIpUpdated = false;

// サーバー（Python）はモデルの読み込みに数十秒かかる。その間の取得は
// 必ず失敗するが、これは異常ではなく待ちの状態。error で出すと本当の
// 障害と見分けが付かなくなるので、繋がるまでは静かに落とす。
// 一度でも繋がった後の失敗は本物なので、そのまま error で出す。
function reportFetchFailure(tag, err) {
  if (!_serverWasOnline) {
    console.debug(`${tag} サーバーの起動待ち:`, err && err.message ? err.message : err);
    return;
  }
  console.error(tag, err);
}

// 精度の選択肢を GPU に合わせて出し分ける。1 度きりでよいが、サーバを
// 入れ替えたときのために毎回上書きする（同じ値なら見た目は変わらない）。
// 使える device をサーバから受け取って選択肢に流し込む。何が使えるかは
// 環境次第（cuda / xpu / mps / cpu）なので、UI に固定で持たない。
function applyDeviceOptions(status) {
  if (!paramDevice || !status) return;
  const list = status.available_devices || ['auto'];
  const auto = status.auto_device || 'cpu';
  // 選択肢が揃うのはここが初めてなので、保存値の復元もここで行う。
  // 起動直後は select に auto しか無く、そこへ cpu を入れても空になる。
  //
  // 初回だけ保存値を見る。paramDevice.value は初期 option が auto ひとつ
  // なので必ず 'auto' が返り、|| で繋ぐと保存値まで到達しない。2 回目以降は
  // 利用者がいま選んでいる値を尊重する。
  const keep = paramDevice.dataset.filled
    ? (paramDevice.value || 'auto')
    : (localStorage.getItem('synth_device') || 'auto');
  const LABEL = { auto: '自動（既定）', cuda: 'GPU (CUDA)', xpu: 'GPU (Intel XPU)',
                  mps: 'GPU (Apple)', cpu: 'CPU（遅いが確実）' };
  paramDevice.innerHTML = '';
  for (const d of list) {
    const o = document.createElement('option');
    o.value = d;
    o.textContent = LABEL[d] || d;
    paramDevice.appendChild(o);
  }
  // 保存済みの選択が今の環境で使えるなら復元する。使えないなら auto へ。
  paramDevice.value = list.includes(keep) ? keep : 'auto';
  paramDevice.dataset.filled = '1';
  if (valDevice) valDevice.textContent = paramDevice.value === 'auto' ? `→ ${auto}` : '';
}

function applyPrecisionOptions(status) {
  if (!paramPrecision || !status) return;
  const gpu = status.gpu || {};
  const auto = status.auto_precision || 'fp32';
  const bf16Ok = !!gpu.bf16_fast;

  const opt = paramPrecision.querySelector('option[value="bf16"]');
  if (opt) {
    opt.disabled = !bf16Ok;
    // 選べない理由を選択肢そのものに書く。無効なだけだと何が起きているか
    // 分からず、旧 GPU の利用者が原因を追えない。
    opt.textContent = bf16Ok
      ? 'bf16（VRAM 半分・やや遅い）'
      : `bf16（この GPU では非対応${gpu.capability ? ` / CC ${gpu.capability}` : ''}）`;
    // 保存済みの選択が bf16 のまま非対応機に移ったら自動へ戻す。
    if (!bf16Ok && paramPrecision.value === 'bf16') paramPrecision.value = 'auto';
  }
  if (valPrecision) {
    const name = gpu.name ? ` / ${gpu.name}` : '';
    valPrecision.textContent = paramPrecision.value === 'auto' ? `→ ${auto}${name}` : '';
  }
}

async function checkApiStatus() {
  try {
    const res = await fetch(`${API_URL}/status`);
    if (res.ok) {
      statusBadge.textContent = 'Online';
      statusBadge.classList.add('online');

      // 精度の選択肢は GPU 次第。bf16 が速いのは Ampere 以降で、それ未満で
      // 選ぶとエラーも出ないまま遅くなる。サーバの判定に従って出し分ける。
      try {
        const st = await res.clone().json();
        applyDeviceOptions(st);
        applyPrecisionOptions(st);
        // AMD は Windows での学習を公式に対応していない。該当する環境の
        // ときだけ学習タブに断りを出す（それ以外では出さない）。
        const rocmNote = document.getElementById('tr-rocm-notice');
        if (rocmNote) {
          const isRocm = !!(st.gpu && st.gpu.rocm);
          rocmNote.style.display = (isRocm && navigator.platform.startsWith('Win')) ? '' : 'none';
        }
      } catch { /* status が読めなくても生成はできる */ }

      // サーバーが初めてオンラインになったタイミングでデータを取得。
      // init() 直後の初回ロードは Python の起動を待たずに走るため
      // "Failed to fetch" で空振りする。ここで拾い直さないと、その一覧は
      // 二度と読まれないまま空で残る（試聴のプルダウンが空になる原因）。
      if (!_serverWasOnline) {
        _serverWasOnline = true;
        loadVoices();
        // 状態カードの LoRA 件数は loraRegistry を見る。待たずに進むと
        // 初回だけ「0 個」と出て、次のポーリングまで直らない。
        await loadLoras();
        trLoadTestJobs();
      }

      if (!isIpUpdated) {
        updateApiUrl();
      }

      updateStatusCard(await res.clone().json());
    }
  } catch (e) {
    statusBadge.textContent = 'Connecting...';
    statusBadge.classList.remove('online');
  }
}

// 生成結果の隣に出す「状態」。走らせている間に目に入る場所なので、
// 取れる値だけを入れて、取れないものは「—」のままにする。
function updateStatusCard(status) {
  const set = (id, text) => {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  };

  set('stat-device', status && status.device ? String(status.device).toUpperCase() : '—');

  const nVoices = status && Array.isArray(status.registered_voices)
    ? status.registered_voices.length : null;
  set('stat-voices', nVoices == null ? '—' : `${nVoices} 個`);

  // 1 体あたり数百 MB あるので、個数だけでなく占有量も出す。
  if (Array.isArray(loraRegistry)) {
    const gb = loraTotalBytes / (1 << 30);
    const size = loraTotalBytes
      ? (gb >= 1 ? `${gb.toFixed(1)} GB` : `${(loraTotalBytes / (1 << 20)).toFixed(0)} MB`)
      : null;
    set('stat-lora', size ? `${loraRegistry.length} 個 / ${size}` : `${loraRegistry.length} 個`);
  } else {
    set('stat-lora', '—');
  }
}

async function updateApiUrl() {
  try {
    // localhost ではなく 127.0.0.1 を使用して確実にサーバーへ飛ばす
    const ipRes = await fetch(`${API_URL}/network_info`);
    if (ipRes.ok) {
      const netData = await ipRes.json();
      console.log("Network info received:", netData);

      // ポートはサーバーが実際に bind した値を使う（固定ではない）
      const port = netData.port || new URL(API_ORIGIN).port;
      // LAN の IP が取れたらそれを、取れなければ手元用のアドレスを表示する。
      // どちらの場合も必ず埋める（初期値の「接続中...」を残さない）。
      const host = (netData.ip && netData.ip !== '127.0.0.1') ? netData.ip : '127.0.0.1';
      apiUrlDisplay.value = `http://${host}:${port}/v1`;
      const hint = document.getElementById('standard-api-hint');
      if (hint) hint.textContent = `標準エンドポイント: http://${host}:${port}/api/v1/synthesize`;
      if (host !== '127.0.0.1') isIpUpdated = true;
    }
  } catch (e) {
    // 失敗してもエラーで止めない
  }
}

async function loadVoices() {
  voices = await window.api.getVoices();
  renderVoices();
}

// 参照ボイスは選ぶだけのものなので畳む。
// 一覧で並べると数が増えるほど柱が縦に伸びて、他の設定が押し出される。
function renderVoices() {
  voiceList.innerHTML = '';

  const pick = document.createElement('div');
  pick.className = 'vpick';

  const sel = document.createElement('select');
  sel.id = 'voice-select';
  sel.innerHTML = '<option value="">（使わない）</option>' +
    voices.map((v) =>
      `<option value="${v.id}"${selectedVoice && selectedVoice.id === v.id ? ' selected' : ''}>` +
      `${v.name}</option>`).join('');
  sel.addEventListener('change', () => {
    const v = voices.find((x) => x.id === sel.value);
    if (v) selectVoice(v); else clearVoice();
  });

  // 選んでいるものを消す。一覧が無くなったので、削除はここに置く。
  const del = document.createElement('button');
  del.className = 'btn btn-ghost btn-sm btn-tool';
  del.title = '選んでいるボイスを削除';
  del.innerHTML = '<svg class="i i-sm"><use href="#ic-x"/></svg>';
  del.addEventListener('click', async () => {
    if (!selectedVoice) return;
    if (!confirm(`「${selectedVoice.name}」を削除しますか。`)) return;
    await window.api.deleteVoice(selectedVoice.id);
    clearVoice();
    loadVoices();
  });

  pick.appendChild(sel);
  pick.appendChild(del);
  voiceList.appendChild(pick);
}

const previewAudioContainer = document.getElementById('preview-audio-container');
const previewAudio = document.getElementById('preview-audio');
const previewPlayBtn = document.getElementById('preview-play-btn');
const previewWave = document.getElementById('preview-wave');

// 試聴。controls を出さず、線画のボタンと波形で見せる。
if (previewPlayBtn) {
  const useEl = previewPlayBtn.querySelector('use');
  previewPlayBtn.addEventListener('click', () => {
    if (!previewAudio.src) return;
    if (previewAudio.paused) previewAudio.play(); else previewAudio.pause();
  });
  previewAudio.addEventListener('play', () => useEl.setAttribute('href', '#ic-pause'));
  previewAudio.addEventListener('pause', () => useEl.setAttribute('href', '#ic-play2'));
  previewAudio.addEventListener('ended', () => useEl.setAttribute('href', '#ic-play2'));
}

function selectVoice(voice) {
  selectedVoice = voice;
  renderVoices();
  currentVoiceName.textContent = voice.name;
  dot.classList.remove('inactive');
  clearVoiceBtn.classList.remove('hidden');
  previewAudioContainer.classList.remove('hidden');

  // Set preview audio src securely
  previewAudio.src = `file://${voice.path.replace(/\\/g, '/')}`;
  if (previewWave) {
    previewWave.dataset.src = previewAudio.src;
    drawClipWave(previewWave);
  }

  // If user is on a caption-only model (voice_design), switch to v3 since
  // they just picked a ref voice. Otherwise keep their current selection.
  if (isVoiceDesignModel(modelSelect.value)) {
    modelSelect.value = 'v3';
  }
  captionWrapper.classList.add('hidden');
  // 参照音声の有無で「話し方のみ」LoRA の選択可否が変わるので、一覧を作り直す。
  refreshLoraDropdown();
}

function clearVoice() {
  selectedVoice = null;
  renderVoices();
  currentVoiceName.textContent = 'None';
  dot.classList.add('inactive');
  clearVoiceBtn.classList.add('hidden');
  previewAudioContainer.classList.add('hidden');
  previewAudio.src = '';
  refreshLoraDropdown();
}

function setupEventListeners() {
  // 参照ボイスの再読込とフォルダを開く
  const voiceRefreshBtn = document.getElementById('voice-refresh-btn');
  if (voiceRefreshBtn) voiceRefreshBtn.addEventListener('click', () => loadVoices());
  const voiceFolderBtn = document.getElementById('voice-folder-btn');
  if (voiceFolderBtn) {
    voiceFolderBtn.addEventListener('click', async () => {
      try { await window.api.openVoicesFolder(); }
      catch (e) { console.error('フォルダを開けませんでした:', e); }
    });
  }

  modelSelect.addEventListener('change', (e) => {
    if (isVoiceDesignModel(e.target.value)) {
      captionWrapper.classList.remove('hidden');
      clearVoice();
    } else {
      captionWrapper.classList.add('hidden');
    }
    refreshLoraDropdown();
    // 発話速度は v4 専用なので、モデル切替で有効・無効を切り替える
    updateDurationEstimate();
  });

  const dropZone = document.getElementById('drop-zone');
  
  // Drag & Drop logic
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
  dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
  });
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.name.toLowerCase().endsWith('.wav')) {
        currentWavPath = file.path || file.name; // Electron exposes .path for local files
        selectedWavPathDisplay.textContent = file.name;
      } else {
        alert("Please drop a WAV file.");
      }
    }
  });

  copyApiBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(apiUrlDisplay.value);
    const prev = copyApiBtn.textContent;
    copyApiBtn.textContent = 'Copied!';
    setTimeout(() => { copyApiBtn.textContent = prev; }, 1500);
  });

  // 詳細設定は右の柱で群ごとに畳むようになったので、
  // 一括で開け閉めするボタンは置いていない。

  paramCandidates.addEventListener('input', e => valCandidates.textContent = e.target.value);
  paramSteps.addEventListener('input', e => valSteps.textContent = e.target.value);
  paramCfgText.addEventListener('input', e => valCfgText.textContent = e.target.value);
  paramDurationScale.addEventListener('input', updateDurationEstimate);
  textInput.addEventListener('input', updateDurationEstimate);
  updateDurationEstimate();
  paramCfgSpeaker.addEventListener('input', e => valCfgSpeaker.textContent = e.target.value);
  paramCfgCaption.addEventListener('input', e => valCfgCaption.textContent = e.target.value);
  paramSwayCoeff.addEventListener('input', e => valSwayCoeff.textContent = parseFloat(e.target.value).toFixed(1));
  paramTScheduleMode.addEventListener('change', e => {
    paramSwayCoeff.disabled = (e.target.value !== 'sway');
  });

  clearVoiceBtn.addEventListener('click', clearVoice);

  selectWavBtn.addEventListener('click', async () => {
    const path = await window.api.selectFile();
    if (path) {
      currentWavPath = path;
      selectedWavPathDisplay.textContent = path.split('\\').pop() || path.split('/').pop();
    }
  });

  saveVoiceBtn.addEventListener('click', async () => {
    const name = newVoiceName.value.trim();
    if (!name || !currentWavPath) {
      alert("Please provide a name and select a WAV file.");
      return;
    }
    
    await window.api.addVoice({ name, filePath: currentWavPath });
    
    // Clear inputs after save
    newVoiceName.value = '';
    currentWavPath = null;
    selectedWavPathDisplay.textContent = 'No file selected';
    
    loadVoices();
  });

  textInput.addEventListener('input', updateCharCount);

  generateBtn.addEventListener('click', generateAudio);
}

// 生成タブの通知。alert は Electron だとフォーカスが戻らず、
// 閉じたあと本文が打てなくなるので使わない。
function showGenMessage(msg, isError) {
  if (!resultContainer) return;
  const box = document.createElement('div');
  box.className = 'gen-message' + (isError ? ' gen-message-error' : '');
  box.textContent = msg;
  resultContainer.prepend(box);
  setTimeout(() => box.remove(), 8000);
}

// 生成リクエストを1本にまとめる。Synthesize と試聴で別々に組み立てると
// 片方だけパラメータが増えて音が食い違うので、送信内容はここだけで決める。
// 呼び出し側が変えられるのは「何を読ませ、どの LoRA で鳴らすか」だけ。
async function runSynthesis({ text, loraName, modelType, caption, seed, loraHasVoice }) {
  const formData = new FormData();
  formData.append('text', applyDict(text));   // 読み辞書で誤読を矯正（表示は元のまま）
  formData.append('model_type', modelType);
  appendSynthParams(formData);

  if (caption) formData.append('caption', caption);

  const isVoiceDesign = isVoiceDesignModel(modelType);
  if (loraName) {
    formData.append('lora_name', loraName);
    // 話し方のみの LoRA は話者を持たないので、誰の声かを参照音声で渡す。
    // 一覧から隠している一時登録は loraRegistry に載らず meta を引けないので、
    // 呼び出し側が声質の有無を知っているならそちらを優先する。
    const meta = loraRegistry.find(l => l.name === loraName);
    const hasVoice = loraHasVoice != null
      ? loraHasVoice
      : !!(meta && (meta.provides || []).includes('voice'));
    if (!hasVoice && !isVoiceDesign && selectedVoice) {
      await attachRefWav(formData);
    }
  } else if (!isVoiceDesign && selectedVoice) {
    await attachRefWav(formData);
  }

  if (seed != null && seed !== '') formData.append('seed', seed);

  const response = await fetch(`${API_URL}/synthesize`, { method: 'POST', body: formData });
  const json = await response.json();
  if (!response.ok || json.status !== 'success') {
    throw new Error(json.error || 'Server error');
  }
  return json;
}

// 参照音声の添付。file:// から読むのは Synthesize と試聴で共通。
async function attachRefWav(formData) {
  try {
    const audioUrl = `file://${selectedVoice.path.replace(/\\/g, '/')}`;
    const res = await fetch(audioUrl);
    formData.append('ref_wav', await res.blob(), 'ref.wav');
  } catch (err) {
    console.error('Failed to attach ref wav', err);
  }
}

// 生成パラメータ（CFG・steps 等）は生成タブの設定を唯一の出所とする。
function appendSynthParams(formData) {
  formData.append('num_candidates', paramCandidates.value);
  formData.append('num_steps', paramSteps.value);
  formData.append('cfg_guidance_mode', paramCfgMode.value);
  formData.append('cfg_scale_text', paramCfgText.value);
  formData.append('cfg_scale_speaker', paramCfgSpeaker.value);
  formData.append('duration_scale', paramDurationScale.value);
  formData.append('cfg_scale_caption', paramCfgCaption.value);

  // Advanced arguments
  if (paramCfgOverride.value) formData.append('cfg_scale', paramCfgOverride.value);
  formData.append('cfg_min_t', paramCfgMinT.value);
  formData.append('cfg_max_t', paramCfgMaxT.value);
  if (paramMaxTextLen.value) formData.append('max_text_len', paramMaxTextLen.value);

  formData.append('context_kv_cache', paramContextKv.value);
  if (paramPrecision) formData.append('precision', paramPrecision.value);
  if (paramDevice) formData.append('device', paramDevice.value);

  if (paramTruncFactor.value) formData.append('truncation_factor', paramTruncFactor.value);
  if (paramRescaleK.value) formData.append('rescale_k', paramRescaleK.value);
  if (paramRescaleSigma.value) formData.append('rescale_sigma', paramRescaleSigma.value);
  if (paramSpeakerKvScale.value) formData.append('speaker_kv_scale', paramSpeakerKvScale.value);

  formData.append('speaker_kv_min_t', paramSpeakerKvMinT.value);
  if (paramSpeakerKvMaxLyr.value) formData.append('speaker_kv_max_layers', paramSpeakerKvMaxLyr.value);

  formData.append('t_schedule_mode', paramTScheduleMode.value);
  if (paramTScheduleMode.value === 'sway') {
    formData.append('sway_coeff', paramSwayCoeff.value);
  }
}

// =====================================================================
// 生成結果（テイク）
//   番号は入力側のセリフ番号と合わせる。冒頭を添えて、どの行の結果かを
//   波形だけで探さなくて済むようにする。
// =====================================================================

// 「ここに生成した音声が出ます」は結果が 1 つも無いときだけ出す。
// 以前は生成のたびに innerHTML を空にしていたので巻き添えで消えていたが、
// 積む方式では件数を見て自分で切り替える（消してしまうと、× で全部
// 消したときに戻せない）。
function updateResultPlaceholder() {
  const ph = document.getElementById('placeholder-text');
  if (!ph) return;
  const has = resultContainer.querySelector('.take');
  ph.classList.toggle('hidden', !!has);
}

function addRunSeparator() {
  const el = document.createElement('div');
  el.className = 'run-sep';
  const t = new Date();
  const hh = String(t.getHours()).padStart(2, '0');
  const mm = String(t.getMinutes()).padStart(2, '0');
  el.textContent = `${hh}:${mm}`;
  resultContainer.appendChild(el);
}

function addTake(no, srcText) {
  const el = document.createElement('div');
  el.className = 'take';
  el.innerHTML =
    `<span class="no">${escapeHtml(String(no))}</span>` +
    `<button class="play play-sm tip" data-tip="聴く" aria-label="聴く" disabled>` +
      `<svg class="i i-sm"><use href="#ic-play2"/></svg></button>` +
    `<span class="src" title="${escapeHtmlAttr(srcText)}">${escapeHtml(srcText)}</span>` +
    `<canvas class="wave"></canvas>` +
    `<span class="len">生成中</span>` +
    `<span class="acts"></span>`;
  resultContainer.appendChild(el);
  updateResultPlaceholder();
  return el;
}

function setTakeError(take, msg) {
  const len = take.querySelector('.len');
  if (len) { len.textContent = '失敗'; len.classList.add('ng'); }
  take.title = msg || '';
}

function fillTake(take, src, seed) {
  const audio = new Audio(src);
  const btn = take.querySelector('.play');
  const use = btn.querySelector('use');
  const len = take.querySelector('.len');

  btn.disabled = false;
  btn.addEventListener('click', () => {
    // 1 本ずつ鳴らす。複数同時だと何を聴いているか分からない。
    if (fillTake._cur && fillTake._cur !== audio) {
      fillTake._cur.pause();
      if (fillTake._curUse) fillTake._curUse.setAttribute('href', '#ic-play2');
    }
    if (audio.paused) { audio.play(); use.setAttribute('href', '#ic-pause'); }
    else { audio.pause(); use.setAttribute('href', '#ic-play2'); }
    fillTake._cur = audio; fillTake._curUse = use;
  });
  audio.addEventListener('ended', () => use.setAttribute('href', '#ic-play2'));
  audio.addEventListener('loadedmetadata', () => {
    if (isFinite(audio.duration)) len.textContent = `${audio.duration.toFixed(1)}s`;
  });
  if (seed != null) take.title = `seed: ${seed}`;

  // 波形。データセット側と同じ描き方を使う。
  const canvas = take.querySelector('.wave');
  if (canvas) { canvas.dataset.src = src; drawClipWave(canvas); }

  // 保存と削除
  const acts = take.querySelector('.acts');
  acts.innerHTML =
    `<button class="tip" data-tip="保存" aria-label="保存"><svg class="i i-sm"><use href="#ic-dl"/></svg></button>` +
    `<button class="tip" data-tip="消す" aria-label="消す"><svg class="i i-sm"><use href="#ic-x"/></svg></button>`;
  const [dlBtn, delBtn] = acts.querySelectorAll('button');
  dlBtn.addEventListener('click', () => {
    const a = document.createElement('a');
    a.href = src;
    a.download = (take.querySelector('.src').textContent || 'take').slice(0, 40) + '.wav';
    a.click();
  });
  delBtn.addEventListener('click', () => {
    audio.pause();
    // 積む方式では次回生成での一括解放が無くなったので、消すときに自分で
    // 解放する。要素ごと消すので、この blob を参照するものは他に居ない。
    if (src.startsWith('blob:')) URL.revokeObjectURL(src);
    if (fillTake._cur === audio) { fillTake._cur = null; fillTake._curUse = null; }
    take.remove();
    // 最後の 1 つを消したら区切りだけが残るので、それも片付ける。
    resultContainer.querySelectorAll('.run-sep').forEach(sep => {
      let n = sep.nextElementSibling;
      while (n && !n.classList.contains('run-sep')) {
        if (n.classList.contains('take')) return;
        n = n.nextElementSibling;
      }
      sep.remove();
    });
    updateResultPlaceholder();
  });
}

async function generateAudio() {
  if (trainingBusy) {
    // ボタンは無効化してあるが、Enter や外部からの呼び出しでも来うる。
    showGenMessage("LoRA 学習中は生成できません。学習の完了後にお試しください。", true);
    return;
  }
  // 空行は飛ばす。行を足したまま書かずに生成を押すことがあるため。
  // 番号は入力側の行番号を持ち回る。詰め直すと、2 行目を空にしたとき
  // 3 行目の結果が「2」と出て、どの行の結果か分からなくなる。
  const lines = allLineTextareas()
    .map((t, i) => ({ no: i + 1, text: t.value.trim() }))
    .filter(l => l.text);
  const text = lines.length ? lines[0].text : '';
  if (!text) {
    // alert() を閉じたあと Electron はフォーカスを textarea に戻さないため、
    // 「本文が空のまま生成 -> 以後なにも打てない」という行き止まりになる。
    // ダイアログを出さずに入力欄へ差し戻す。
    textInput.focus();
    textInput.placeholder = "読ませたい文章を入力してください";
    textInput.classList.add("input-error");
    setTimeout(() => textInput.classList.remove("input-error"), 1200);
    return;
  }

  const isVoiceDesign = isVoiceDesignModel(modelSelect.value);

  generateBtn.disabled = true;
  btnTextSpan.classList.add('hidden');
  genSpinner.classList.remove('hidden');

  const condMode = getCondMode();
  if (condMode === 'lora') {
    if (!loraSelect.value) {
      showGenMessage("LoRA を選ぶか、1ショットに切り替えてください。", true);
      loraSelect.focus();
      generateBtn.disabled = false;
      btnTextSpan.classList.remove('hidden');
      genSpinner.classList.add('hidden');
      return;
    }
    // A style-only adapter carries no speaker, so it still needs the sample
    // to say who is talking. Only a full adapter can go without one.
    const meta = loraRegistry.find(l => l.name === loraSelect.value);
    const loraHasVoice = !!(meta && (meta.provides || []).includes('voice'));
    if (meta && !loraHasVoice && !isVoiceDesign && !selectedVoice) {
      // 参照なしで送ると no_ref になり seed 次第で別人の声が出る。
      // 送る前に止めて、何をすればよいかを示す。
      showGenMessage("この LoRA は話し方のみです。サンプルボイスを選択してください。", true);
      updateLoraVoiceHint();
      generateBtn.disabled = false;
      btnTextSpan.classList.remove('hidden');
      genSpinner.classList.add('hidden');
      return;
    }
  }

  try {
    // 生成結果は消さずに下へ積む。前回分は聞き比べのために残す。
    //
    // 以前はここで resultContainer を空にし、さらに前回の wav を DELETE して
    // いた。保存フォルダを設定していない場合 outputs/ にしか実体が無いので、
    // 次に生成ボタンを押しただけで前の音声が消えていた。
    // 保存後の後始末は main.js の save-synth-output 側が
    // 「コピーできたときだけ元を消す」形で持っている。ここでの削除は不要。
    //
    // blob URL は要素が生きている間は解放できない（解放すると再生できなく
    // なる）。積む方式では過去の take も再生対象なので、ここでは解放しない。
    // 追跡用の window._lastSynth* は削除処理と対で存在していたもので、
    // 削除をやめた時点で読み手が居なくなったため撤去した。

    const saveFolder = localStorage.getItem('synth_save_folder');

    // 行番号は毎回 1 から振り直されるので、積むと 1,2,1,2 と重複する。
    // どこからが今回かを区切りで示す。
    // placeholder-text は hidden の付け外しで DOM に残り続けるので、
    // children.length で見ると初回にも区切りが入る。take の有無で見る。
    if (resultContainer.querySelector('.take')) addRunSeparator();

    // 行ごとに順へ流す。API は 1 回 1 テキストなので、まとめては送れない。
    // 先に空の行を並べておき、出来た順に埋める（どこまで進んだかが見える）。
    const takes = lines.map((l) => addTake(l.no, l.text));
    // 積むと今回の分が画面外に出ることがあるので、先頭へ寄せる。
    if (takes[0]) takes[0].scrollIntoView({ block: 'nearest', behavior: 'smooth' });

    for (let li = 0; li < lines.length; li++) {
      const take = takes[li];
      take.classList.add('busy');
      let json;
      try {
        json = await runSynthesis({
          text: lines[li].text,
          loraName: condMode === 'lora' ? loraSelect.value : null,
          modelType: modelSelect.value,
          caption: isVoiceDesign ? captionInput.value.trim() : null,
          seed: paramSeed.value ? parseInt(paramSeed.value) : null,
        });
      } catch (e) {
        take.classList.remove('busy');
        setTakeError(take, e.message);
        continue;   // 1 行が失敗しても残りは流す
      }
      take.classList.remove('busy');

      // 保存フォルダへのコピーは outputs/ の元ファイルを消すため、先に音声を
      // メモリへ取り込んでおく。URL 参照のままだと保存後に 404 で再生できない。
      const blobUrls = await Promise.all(json.results.map(async (url) => {
        try {
          const res = await fetch(`${API_ORIGIN}${url}`);
          if (!res.ok) return null;
          return URL.createObjectURL(await res.blob());
        } catch (_) {
          return null;
        }
      }));

      if (saveFolder) {
        json.results.forEach((url, i) => {
          // blob を取れなかった候補は元 URL 参照のままなので、保存に回すと
          // IPC 側が元ファイルを消して再生できなくなる。取れた分だけ保存する。
          if (!blobUrls[i]) return;
          const filename = url.split('/').pop();
          // 行が複数あるときは行番号で分ける。1 行なら従来どおり本文から。
          const base = lines.length > 1
            ? `${String(lines[li].no).padStart(2, '0')}_${lines[li].text}`
            : lines[li].text;
          const saveName = json.results.length > 1 && i > 0 ? `${base}_${i + 1}` : base;
          window.api.saveSynthOutput({ filename, saveFolder, textInput: saveName }).catch(() => {});
        });
      }

      // 候補が複数のときは 1 本目をこの行に、残りは続けて並べる
      fillTake(take, blobUrls[0] || `${API_ORIGIN}${json.results[0]}`, json.seed_used);
      for (let i = 1; i < json.results.length; i++) {
        const extra = addTake(`${lines[li].no}-${i + 1}`, lines[li].text);
        fillTake(extra, blobUrls[i] || `${API_ORIGIN}${json.results[i]}`, json.seed_used);
      }
    }

  } catch (error) {
    showGenMessage("生成に失敗しました: " + error.message, true);
  } finally {
    generateBtn.disabled = false;
    btnTextSpan.classList.remove('hidden');
    genSpinner.classList.add('hidden');
  }
}

// =====================================================================
// LoRA registry & UI
// =====================================================================

function getCondMode() {
  for (const r of condModeRadios) if (r.checked) return r.value;
  return 'oneshot';
}

async function loadLoras() {
  try {
    const res = await fetch(`${API_URL}/loras`);
    if (!res.ok) throw new Error(`status ${res.status}`);
    const json = await res.json();
    loraRegistry = Array.isArray(json.loras) ? json.loras : [];
    loraTotalBytes = Number(json.total_bytes) || 0;
    if (typeof window.onLorasLoaded === 'function') window.onLorasLoaded();
  } catch (err) {
    reportFetchFailure('[lora] レジストリを読めませんでした:', err);
    loraRegistry = [];
    loraTotalBytes = 0;
  }
  refreshLoraDropdown();
  renderLoraList();
}

// 話し方のみの LoRA を選んでいてサンプルボイスが未選択なら、その場で
// 案内を出す。生成ボタンを押して初めてエラーになるより先に気付ける。
// LoRA 学習は同じ GPU にベースモデルを丸ごと載せる。学習開始時に
// clear_cached_runtime() で推論側を解放しているが、学習中に生成を許すと
// その場でモデルが再ロードされ、VRAM を取り合って 8GB 環境では OOM する。
// サーバー側にはこれを拒む口が無いので、止められるのは UI だけ。
let trainingBusy = false;

function applyTrainingLock() {
  const notice = document.getElementById('generate-training-notice');
  if (notice) notice.style.display = trainingBusy ? '' : 'none';
  if (!generateBtn) return;
  if (trainingBusy) {
    generateBtn.disabled = true;
    generateBtn.title = 'LoRA 学習中は生成できません';
  } else {
    // 生成中の disabled は generateAudio 側が管理するので、そちらが
    // 触っていないときだけ戻す。
    if (generateBtn.title === 'LoRA 学習中は生成できません') {
      generateBtn.disabled = false;
      generateBtn.title = '';
    }
  }
}

function setTrainingBusy(busy) {
  const next = !!busy;
  if (next === trainingBusy) return;
  trainingBusy = next;
  applyTrainingLock();
}

// Train タブを開いていなくても状態が要る（合成は生成タブで行う）。
async function pollTrainingBusy() {
  try {
    const res = await fetch(`${API_URL}/lora/jobs`);
    if (!res.ok) return;
    const json = await res.json();
    const jobs = json.jobs || [];
    setTrainingBusy(jobs.some(j =>
      ['pending', 'preparing', 'training', 'stopping'].includes(j.state)));
  } catch (_) {
    // サーバー未起動などは触らない（誤ってロックしたままにしない）
  }
}

function updateLoraVoiceHint() {
  const hint = document.getElementById('lora-voice-hint');
  if (!hint) return;
  let msg = '';
  if (getCondMode() === 'lora' && loraSelect.value && !isVoiceDesignModel(modelSelect.value)) {
    const meta = loraRegistry.find(l => l.name === loraSelect.value);
    const hasVoice = !!(meta && (meta.provides || []).includes('voice'));
    if (meta && !hasVoice && !selectedVoice) {
      msg = 'この LoRA は話し方のみです。サンプルボイスを選択してください。';
    }
  }
  hint.textContent = msg;
  hint.style.display = msg ? '' : 'none';
}

function refreshLoraDropdown() {
  const base = modelSelect.value;
  const compatible = loraRegistry.filter(l => l.base === base);

  loraSelect.innerHTML = '';
  if (compatible.length === 0) {
    const opt = document.createElement('option');
    opt.value = '';
    // v4 の LoRA を v4.1 で選べるようにしても意味が無い。v4.1 は v4 と
    // duration predictor だけが違い、アダプタ側が modules_to_save で
    // それを丸ごと持つので、当てると v4.1 の改良点が上書きされる
    // （実測: v4.1+v4LoRA の出力は v4+v4LoRA とバイト単位で一致）。
    // 弾くのは正しいが、黙って空になると理由が分からないので誘導する。
    const hint = (base === 'v4_1' && loraRegistry.some(l => l.base === 'v4'))
      ? '（既存の LoRA は v4-Small を選ぶと使えます。v4.1 で使うには学習タブで v4.1 を選んで作り直してください）'
      : '';
    opt.textContent = `この base の LoRA はありません${hint}`;
    loraSelect.appendChild(opt);
  } else {
    // 話し方のみの LoRA はサンプルボイスと混ぜて使うのが前提なので、
    // 常に選択できる。サンプル未選択なら選んだ時点で案内を出し、
    // 生成は選ぶまでブロックする（updateLoraVoiceHint / generateAudio）。
    const label = l => {
      const kind = (l.provides || []).includes('voice') ? '完パケ' : '話し方';
      const note = l.notes ? ` — ${l.notes.substring(0, 24)}` : '';
      return `${l.name} [${kind}]${note}`;
    };
    compatible.forEach(l => {
      const opt = document.createElement('option');
      opt.value = l.name;
      opt.textContent = label(l);
      loraSelect.appendChild(opt);
    });
  }
  updateLoraVoiceHint();

  const loraSupported = compatible.length > 0;
  condModeLoraRadio.disabled = !loraSupported;
  if (!loraSupported && getCondMode() === 'lora') {
    document.querySelector('input[name="cond-mode"][value="oneshot"]').checked = true;
    applyCondModeUI();
  }
  if (typeof window.onLorasLoaded === 'function') window.onLorasLoaded();
}

function applyCondModeUI() {
  const mode = getCondMode();
  if (mode === 'lora') {
    loraSelect.classList.remove('hidden');
  } else {
    loraSelect.classList.add('hidden');
  }
  updateLoraVoiceHint();
  updateDurationEstimate();
}

// ベースの見せ方。サーバが受け付ける種類は server_lora.py の ALLOWED_BASES。
// VoiceDesign は帯に入れると長いので VD と略し、意味は tip で補う。
// 見た目の色（cls）は世代で揃える。v4.1 は v4 と同じ扱い。
const LORA_BASES = {
  v4_1:            { cls: 'v4', label: 'v4.1', tip: 'v4.1-Small' },
  v4:              { cls: 'v4', label: 'v4',   tip: 'v4-Small' },
  v3:              { cls: 'v3', label: 'v3',   tip: 'v3' },
  v3_voice_design: { cls: 'vd', label: 'VD3',  tip: 'v3 Voice Design (600M)' },
  v2:              { cls: 'v2', label: 'v2',   tip: 'v2' },
  voice_design:    { cls: 'vd', label: 'VD',   tip: 'v2 Voice Design' },
};
let loraFilterBase = '';   // '' = すべて

function renderLoraList() {
  loraListBody.innerHTML = '';
  const countEl = document.getElementById('lora-count');
  const segEl   = document.getElementById('lora-base-seg');
  const q = (document.getElementById('lora-filter') || {}).value || '';

  if (loraRegistry.length === 0) {
    loraListEmpty.classList.remove('hidden');
    loraListEmpty.textContent = 'まだ LoRA がありません。';
    if (countEl) countEl.textContent = '';
    if (segEl) segEl.innerHTML = '';
    return;
  }
  if (countEl) countEl.innerHTML = `<b>${loraRegistry.length}</b> 体`;

  // 絞り込みの押しボタンは「1 体でも在るベース」だけ出す。
  // 4 種すべて並べると、使っていないベースのボタンが場所を取る。
  if (segEl) {
    const counts = {};
    loraRegistry.forEach(l => { counts[l.base] = (counts[l.base] || 0) + 1; });
    const order = Object.keys(LORA_BASES).filter(b => counts[b]);
    segEl.innerHTML =
      `<button data-base="" aria-pressed="${loraFilterBase === ''}">すべて <i>${loraRegistry.length}</i></button>` +
      order.map(b => {
        const m = LORA_BASES[b] || { label: b };
        return `<button data-base="${escapeHtmlAttr(b)}" aria-pressed="${loraFilterBase === b}">` +
               `${escapeHtml(m.label)} <i>${counts[b]}</i></button>`;
      }).join('');
    segEl.querySelectorAll('button').forEach(btn => {
      btn.addEventListener('click', () => {
        loraFilterBase = btn.dataset.base;
        renderLoraList();
      });
    });
  }

  const shown = loraRegistry.filter(l =>
    (!loraFilterBase || l.base === loraFilterBase) &&
    (!q || (l.name || '').toLowerCase().includes(q.toLowerCase())));

  if (shown.length === 0) {
    loraListEmpty.classList.remove('hidden');
    loraListEmpty.textContent = '当てはまる LoRA がありません。';
    return;
  }
  loraListEmpty.classList.add('hidden');

  shown.forEach(l => {
    const m = LORA_BASES[l.base] || { cls: '', label: l.base || '—', tip: '' };
    const row = document.createElement('div');
    row.className = 'row';
    const dateStr = l.imported_at
      ? new Date(l.imported_at).toLocaleDateString('ja-JP', { year: 'numeric', month: '2-digit', day: '2-digit' })
      : '—';
    row.innerHTML = `
      <span class="nm" title="${escapeHtmlAttr(l.name)}">${escapeHtml(l.name)}</span>
      <span class="tagb ${m.cls} tip" data-tip="${escapeHtmlAttr(m.tip || '')}">${escapeHtml(m.label)}</span>
      <span class="memo" title="${escapeHtmlAttr(l.notes || '')}">${escapeHtml(l.notes || '')}</span>
      <span class="date">${escapeHtml(dateStr)}</span>
      <button class="del tip" data-tip="削除" aria-label="削除"><svg class="i i-sm"><use href="#ic-x"/></svg></button>
    `;
    row.querySelector('.del').addEventListener('click', () => deleteLora(l.name));
    loraListBody.appendChild(row);
  });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[c]);
}

async function deleteLora(name) {
  if (!confirm(`LoRA「${name}」を削除します。元に戻せません。`)) return;
  try {
    const res = await fetch(`${API_URL}/loras/${encodeURIComponent(name)}`, { method: 'DELETE' });
    const json = await res.json();
    if (!res.ok) throw new Error(json.error || `status ${res.status}`);
    await loadLoras();
  } catch (err) {
    alert(`Failed to delete LoRA: ${err.message}`);
  }
}

async function importLora() {
  const name = loraImportName.value.trim();
  const base = loraImportBase.value;
  const notes = loraImportNotes.value.trim();
  if (!name) { alert('Name is required.'); return; }
  if (!pickedAdapterPath) { alert('Pick an adapter folder first.'); return; }

  loraImportSubmit.disabled = true;
  try {
    const res = await fetch(`${API_URL}/loras/import`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, base, source_path: pickedAdapterPath, notes })
    });
    const json = await res.json();
    if (!res.ok) throw new Error(json.detail || json.error || `status ${res.status}`);
    // Reset form
    loraImportName.value = '';
    loraImportNotes.value = '';
    pickedAdapterPath = null;
    loraImportPath.textContent = 'No folder selected';
    await loadLoras();
  } catch (err) {
    alert(`Import failed: ${err.message}`);
  } finally {
    loraImportSubmit.disabled = false;
  }
}

function setupLoraEventListeners() {
  for (const r of condModeRadios) r.addEventListener('change', applyCondModeUI);
  loraSelect.addEventListener('change', () => { updateLoraVoiceHint(); updateDurationEstimate(); });
  manageLorasBtn.addEventListener('click', () => loraModal.classList.remove('hidden'));
  loraModalClose.addEventListener('click', () => loraModal.classList.add('hidden'));
  loraModal.addEventListener('click', (e) => {
    if (e.target === loraModal) loraModal.classList.add('hidden');
  });

  loraImportPickBtn.addEventListener('click', async () => {
    const folder = await window.api.selectFolder();
    if (folder) {
      pickedAdapterPath = folder;
      loraImportPath.textContent = folder;
    }
  });
  loraImportSubmit.addEventListener('click', importLora);

  // 名前での絞り込み。打つたびに引き直す（181 体でも一覧の作り直しは軽い）
  const loraFilter = document.getElementById('lora-filter');
  if (loraFilter) loraFilter.addEventListener('input', renderLoraList);
}

// =====================================================================
// Dataset tab
// =====================================================================

const dsList = () => document.getElementById('ds-list');
const dsListEmpty = () => document.getElementById('ds-list-empty');
const dsRefreshBtn = () => document.getElementById('ds-refresh-btn');
const dsDropZone = () => document.getElementById('ds-drop-zone');
const dsBrowseBtn = () => document.getElementById('ds-browse-btn');
const dsBrowseFolderBtn = () => document.getElementById('ds-browse-folder-btn');
const dsSources = () => document.getElementById('ds-sources');
const dsMinSec = () => document.getElementById('ds-min-sec');
const dsMaxSec = () => document.getElementById('ds-max-sec');
const dsSkipSplit = () => document.getElementById('ds-skip-split');
const dsVadControls = () => document.getElementById('ds-vad-controls');
const dsProcessBtn = () => document.getElementById('ds-process-btn');
const dsProcessStatus = () => document.getElementById('ds-process-status');
const dsClipsBody = () => document.getElementById('ds-clips-body');
const dsClipsEmpty = () => document.getElementById('ds-clips-empty');
const dsSaveName = () => document.getElementById('ds-save-name');
const dsSaveBtn = () => document.getElementById('ds-save-btn');
const dsSaveStatus = () => document.getElementById('ds-save-status');
const dsTargetPick = () => document.getElementById('ds-target-pick');
const dsTargetPath = () => document.getElementById('ds-target-path');
const dsTargetReset = () => document.getElementById('ds-target-reset');

let dsSourceFiles = [];   // [{ path, name }]
let dsCurrentClips = [];  // [{ index, path, text, duration, source, excluded }]
let dsTargetDir = null;   // optional custom save folder; null means use app default

function escapeHtmlAttr(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[c]);
}

async function loadDatasets() {
  try {
    const res = await fetch(`${API_URL}/datasets`);
    if (!res.ok) throw new Error(`status ${res.status}`);
    const json = await res.json();
    renderDatasetList(json.datasets || []);
  } catch (err) {
    reportFetchFailure('[dataset] 一覧を読めませんでした:', err);
    renderDatasetList([]);
  }
}

// 秒 → h:mm:ss。一覧は桁を揃えたいので常に時:分:秒で出す。
function hms(sec) {
  if (sec == null) return '';
  const s = Math.round(sec);
  return `${Math.floor(s / 3600)}:${String(Math.floor((s % 3600) / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

function renderDatasetList(items) {
  const ul = dsList();
  ul.innerHTML = '';
  const sum = document.getElementById('ds-list-sum');
  if (items.length === 0) {
    dsListEmpty().classList.remove('hidden');
    if (sum) sum.textContent = '';
    return;
  }
  dsListEmpty().classList.add('hidden');
  if (sum) {
    const clips = items.reduce((a, d) => a + (d.num_clips || 0), 0);
    const dur = items.reduce((a, d) => a + (d.total_duration || 0), 0);
    sum.innerHTML = `<b>${items.length}</b> 件 ・ <b>${clips.toLocaleString()}</b> クリップ ・ <b>${hms(dur)}</b>`;
  }
  items.forEach(d => {
    const row = document.createElement('div');
    row.className = 'ds-row';
    row.title = d.location || '';
    row.innerHTML = `
      <span class="nm">${escapeHtml(d.name)}</span>
      <span class="num">${(d.num_clips || 0).toLocaleString()} クリップ</span>
      <span class="dur">${hms(d.total_duration)}</span>
      <span class="acts">
        <button class="del tip" data-tip="削除" aria-label="削除"><svg class="i i-sm"><use href="#ic-x"/></svg></button>
      </span>
    `;
    row.querySelector('.del').addEventListener('click', (e) => {
      e.stopPropagation();
      deleteDataset(d.name);
    });
    ul.appendChild(row);
  });
}

async function deleteDataset(name) {
  if (!confirm(`Delete dataset "${name}"?`)) return;
  try {
    const res = await fetch(`${API_URL}/datasets/${encodeURIComponent(name)}`, { method: 'DELETE' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || err.error || `status ${res.status}`);
    }
    await loadDatasets();
  } catch (err) {
    alert(`Failed to delete: ${err.message}`);
  }
}

function addSourceFiles(paths) {
  for (const p of paths) {
    if (!p) continue;
    if (dsSourceFiles.some(s => s.path === p)) continue;
    const name = p.split(/[\\/]/).pop();
    dsSourceFiles.push({ path: p, name });
  }
  renderSources();
}

function renderSources() {
  const ul = dsSources();
  ul.innerHTML = '';
  for (const s of dsSourceFiles) {
    const li = document.createElement('div');
    li.className = 'src-item';
    li.innerHTML = `
      <span class="fn" title="${escapeHtmlAttr(s.path)}">${escapeHtml(s.name)}</span>
      <button class="ds-remove tip" data-tip="外す" aria-label="外す" data-path="${escapeHtmlAttr(s.path)}"><svg class="i i-sm"><use href="#ic-x"/></svg></button>
    `;
    li.querySelector('.ds-remove').addEventListener('click', () => {
      dsSourceFiles = dsSourceFiles.filter(x => x.path !== s.path);
      renderSources();
    });
    ul.appendChild(li);
  }
}

async function processSources() {
  if (dsSourceFiles.length === 0) {
    alert('Add at least one audio file first.');
    return;
  }
  const skipSplit = dsSkipSplit().checked;
  const minSec = parseFloat(dsMinSec().value);
  const maxSec = parseFloat(dsMaxSec().value);
  dsProcessBtn().disabled = true;
  dsCurrentClips = [];
  renderClipsTable();

  const status = dsProcessStatus();
  try {
    for (let i = 0; i < dsSourceFiles.length; i += 1) {
      const src = dsSourceFiles[i];

      let chunkList;
      if (skipSplit) {
        // Per-file mode: each source file is one clip, no Silero VAD.
        status.textContent = `Loading (${i + 1}/${dsSourceFiles.length}): ${src.name}`;
        chunkList = [{ path: src.path, duration: null }];
      } else {
        status.textContent = `Splitting (${i + 1}/${dsSourceFiles.length}): ${src.name}`;
        const splitRes = await fetch(`${API_URL}/audio/split`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: src.path, min_sec: minSec, max_sec: maxSec })
        });
        if (!splitRes.ok) {
          const err = await splitRes.json().catch(() => ({}));
          throw new Error(`split failed for ${src.name}: ${err.detail || err.error || splitRes.status}`);
        }
        const split = await splitRes.json();
        chunkList = split.chunks;
      }

      for (let c = 0; c < chunkList.length; c += 1) {
        const chunk = chunkList[c];
        status.textContent = `Transcribing ${src.name} ${skipSplit ? '' : `chunk ${c + 1}/${chunkList.length}`}…`;
        const trRes = await fetch(`${API_URL}/audio/transcribe`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: chunk.path })
        });
        let text = '';
        let duration = chunk.duration;
        if (trRes.ok) {
          const tr = await trRes.json();
          text = (tr.text || '').trim();
          if (duration == null && tr.duration != null) duration = tr.duration;
        }
        const clip = {
          index: dsCurrentClips.length,
          path: chunk.path,
          text,
          duration,
          source: src.name,
          excluded: false,
        };
        dsCurrentClips.push(clip);
        appendClipRow(clip);
        // Throttle stats recompute (it iterates the whole array each call).
        if (dsCurrentClips.length % 25 === 0) renderClipsStats();
      }
    }
    renderClipsStats();
    status.textContent = `Done. ${dsCurrentClips.length} clips ready.`;
    dsSaveBtn().disabled = dsCurrentClips.length === 0;
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
  } finally {
    dsProcessBtn().disabled = false;
  }
}

function formatDuration(sec) {
  if (sec == null || !isFinite(sec) || sec < 0) return '0s';
  const total = Math.round(sec);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function renderClipsStats() {
  const el = document.getElementById('ds-clips-stats');
  if (!el) return;
  const active = dsCurrentClips.filter(c => !c.excluded && c.duration != null);
  if (active.length === 0) {
    el.innerHTML = '';
    return;
  }
  const total = active.reduce((s, c) => s + (Number(c.duration) || 0), 0);
  const avg = total / active.length;
  el.innerHTML =
    `<span>採用 <b>${active.length}</b> / ${dsCurrentClips.length} クリップ</span>` +
    `<span>合計 <b>${formatDuration(total)}</b></span>` +
    `<span>平均 <b>${avg.toFixed(1)}</b> 秒</span>`;
}

function getDsVol() {
  const el = document.getElementById('ds-vol');
  return el ? parseInt(el.value) / 100 : 1.0;
}

// ── クリップの波形 ───────────────────────────────
// 分割直後は数百行になる。全部デコードすると開いた瞬間に固まるので、
// 画面に入った行だけ描く。描いた結果はパスをキーに覚えておく。
const dsWaveCache = new Map();
let dsWaveCtx = null;

function _waveObserver() {
  if (_waveObserver._o) return _waveObserver._o;
  _waveObserver._o = new IntersectionObserver((entries, obs) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      obs.unobserve(e.target);
      drawClipWave(e.target);
    }
  }, { rootMargin: '150px' });
  return _waveObserver._o;
}

async function drawClipWave(canvas) {
  const src = canvas.dataset.src;
  if (!src) return;
  try {
    let peaks = dsWaveCache.get(src);
    if (!peaks) {
      dsWaveCtx = dsWaveCtx || new (window.AudioContext || window.webkitAudioContext)();
      const buf = await (await fetch(src)).arrayBuffer();
      const audio = await dsWaveCtx.decodeAudioData(buf);
      const ch = audio.getChannelData(0);
      const N = 100;                       // canvas の横幅ぶんだけ間引く
      const step = Math.max(1, Math.floor(ch.length / N));
      peaks = [];
      for (let i = 0; i < N; i++) {
        let max = 0;
        const from = i * step;
        for (let j = from; j < from + step && j < ch.length; j++) {
          const v = Math.abs(ch[j]);
          if (v > max) max = v;
        }
        peaks.push(max);
      }
      dsWaveCache.set(src, peaks);
    }
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 100, h = canvas.clientHeight || 16;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const g = canvas.getContext('2d');
    g.scale(dpr, dpr);
    g.clearRect(0, 0, w, h);
    g.fillStyle = getComputedStyle(canvas).color || '#c9a86a';
    const bw = w / peaks.length;
    for (let i = 0; i < peaks.length; i++) {
      const bh = Math.max(1, peaks[i] * h);
      g.fillRect(i * bw, (h - bh) / 2, Math.max(1, bw - 0.5), bh);
    }
  } catch {
    // 読めない音源は波形なしで通す。行そのものは使える。
  }
}

function _buildClipRow(clip) {
  const tr = document.createElement('tr');
  if (clip.excluded) tr.classList.add('off');
  const audioSrc = `file://${clip.path.replace(/\\/g, '/')}`;
  tr.innerHTML = `
    <td class="c-no">${clip.index + 1}</td>
    <td class="c-dur">${clip.duration}s</td>
    <td class="c-txt"><input type="text" data-idx="${clip.index}" value="${escapeHtmlAttr(clip.text)}"></td>
    <td class="c-aud"><div>
      <button class="play play-sm tip" data-tip="聴く" aria-label="聴く"><svg class="i i-sm"><use href="#ic-play2"/></svg></button>
      <canvas class="mini-wave" data-src="${escapeHtmlAttr(audioSrc)}"></canvas>
    </div></td>
    <td class="c-use"><input type="checkbox" data-idx="${clip.index}" ${clip.excluded ? '' : 'checked'}></td>
  `;

  // 再生。1 つずつ鳴らす（複数同時に鳴ると何を聴いているか分からない）
  const btn = tr.querySelector('.play');
  const use = btn.querySelector('use');
  btn.addEventListener('click', () => {
    if (_buildClipRow._cur && !_buildClipRow._cur.paused) {
      _buildClipRow._cur.pause();
      if (_buildClipRow._curUse) _buildClipRow._curUse.setAttribute('href', '#ic-play2');
      if (_buildClipRow._curBtn === btn) { _buildClipRow._cur = null; return; }
    }
    const a = new Audio(audioSrc);
    a.volume = getDsVol();
    a.addEventListener('ended', () => use.setAttribute('href', '#ic-play2'));
    a.play();
    use.setAttribute('href', '#ic-pause');
    _buildClipRow._cur = a; _buildClipRow._curUse = use; _buildClipRow._curBtn = btn;
  });

  _waveObserver().observe(tr.querySelector('.mini-wave'));

  tr.querySelector('input[type="text"]').addEventListener('input', (e) => {
    clip.text = e.target.value;
  });
  tr.querySelector('input[type="checkbox"]').addEventListener('change', (e) => {
    clip.excluded = !e.target.checked;
    tr.classList.toggle('off', clip.excluded);
    renderClipsStats();
  });
  return tr;
}

function appendClipRow(clip) {
  if (dsClipsEmpty().classList.contains('hidden') === false) {
    dsClipsEmpty().classList.add('hidden');
  }
  dsClipsBody().appendChild(_buildClipRow(clip));
}

function renderClipsTable() {
  const body = dsClipsBody();
  body.innerHTML = '';
  if (dsCurrentClips.length === 0) {
    dsClipsEmpty().classList.remove('hidden');
    renderClipsStats();
    return;
  }
  dsClipsEmpty().classList.add('hidden');
  const frag = document.createDocumentFragment();
  for (const clip of dsCurrentClips) frag.appendChild(_buildClipRow(clip));
  body.appendChild(frag);
  renderClipsStats();
}

async function saveDataset() {
  const name = dsSaveName().value.trim();
  if (!name) { alert('Dataset name is required.'); return; }
  const clips = dsCurrentClips
    .filter(c => !c.excluded)
    .map(c => ({ path: c.path, text: c.text }));
  if (clips.length === 0) { alert('No included clips to save.'); return; }

  dsSaveBtn().disabled = true;
  dsSaveStatus().textContent = `Saving ${clips.length} clips${dsTargetDir ? ' to custom folder' : ' to app data'}…`;
  try {
    const body = {
      name,
      clips,
      source_files: dsSourceFiles.map(s => s.path),
      notes: `${dsSourceFiles.length} source file(s)`,
      overwrite: false,
    };
    if (dsTargetDir) body.target_dir = dsTargetDir;
    const res = await fetch(`${API_URL}/datasets`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const json = await res.json();
    if (!res.ok) {
      throw new Error(json.detail || json.error || `status ${res.status}`);
    }
    dsSaveStatus().textContent = `Saved ${json.num_clips} clips -> ${json.location}`;
    // Reset working state
    dsSaveName().value = '';
    dsCurrentClips = [];
    dsSourceFiles = [];
    dsTargetDir = null;
    dsTargetPath().textContent = 'Default (app data)';
    dsTargetReset().classList.add('hidden');
    renderClipsTable();
    renderSources();
    await loadDatasets();
  } catch (err) {
    dsSaveStatus().textContent = `Save failed: ${err.message}`;
  } finally {
    dsSaveBtn().disabled = false;
  }
}

function setupDatasetTab() {
  // 再生音量スライダー
  const dsVolEl = document.getElementById('ds-vol');
  const dsVolVal = document.getElementById('ds-vol-val');
  if (dsVolEl) {
    dsVolEl.addEventListener('input', () => {
      dsVolVal.textContent = dsVolEl.value;
      // 行の中に audio 要素は持たない（押したときに作る）。
      // いま鳴っているものだけその場で追従させる。
      if (_buildClipRow._cur) _buildClipRow._cur.volume = getDsVol();
    });
  }

  dsRefreshBtn().addEventListener('click', loadDatasets);

  dsBrowseBtn().addEventListener('click', async () => {
    const files = await window.api.selectAudioFiles();
    if (files && files.length) addSourceFiles(files);
  });

  dsBrowseFolderBtn().addEventListener('click', async () => {
    dsProcessStatus().textContent = 'Scanning folder…';
    try {
      const files = await window.api.selectAudioFolder();
      if (files && files.length) {
        addSourceFiles(files);
        dsProcessStatus().textContent = `Added ${files.length} wav files from folder.`;
      } else {
        dsProcessStatus().textContent = 'No wav files found in selected folder.';
      }
    } catch (err) {
      dsProcessStatus().textContent = `Folder scan failed: ${err.message}`;
    }
  });

  const zone = dsDropZone();
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', (e) => { e.preventDefault(); zone.classList.remove('drag-over'); });
  zone.addEventListener('drop', async (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const filePaths = [];
    const folderPaths = [];
    for (const f of e.dataTransfer.files || []) {
      if (!f.path) continue;
      // In Electron, dropped folders show as File with empty `type` and `size`==0.
      // The robust check is to ask main process via IPC; here we use a heuristic:
      // names without a .wav suffix are treated as candidate folders.
      if (/\.(wav|ogg)$/i.test(f.name)) {
        filePaths.push(f.path);
      } else {
        folderPaths.push(f.path);
      }
    }
    if (filePaths.length) addSourceFiles(filePaths);
    if (folderPaths.length) {
      dsProcessStatus().textContent = `Scanning ${folderPaths.length} folder(s)…`;
      try {
        let total = 0;
        for (const p of folderPaths) {
          const found = await window.api.enumerateAudioFolder(p);
          if (found && found.length) {
            addSourceFiles(found);
            total += found.length;
          }
        }
        dsProcessStatus().textContent = total > 0
          ? `Added ${total} wav file(s) from dropped folder(s).`
          : 'No wav files found in dropped folder(s).';
      } catch (err) {
        dsProcessStatus().textContent = `Drop scan failed: ${err.message}`;
      }
    }
  });

  dsProcessBtn().addEventListener('click', processSources);
  dsSaveBtn().addEventListener('click', saveDataset);
  dsSkipSplit().addEventListener('change', (e) => {
    dsVadControls().querySelectorAll('input[type="number"]').forEach(el => {
      el.disabled = e.target.checked;
    });
  });

  dsTargetPick().addEventListener('click', async () => {
    const dir = await window.api.selectFolder();
    if (dir) {
      dsTargetDir = dir;
      dsTargetPath().textContent = dir;
      dsTargetReset().classList.remove('hidden');
    }
  });
  dsTargetReset().addEventListener('click', () => {
    dsTargetDir = null;
    dsTargetPath().textContent = 'Default (app data)';
    dsTargetReset().classList.add('hidden');
  });
}

// =====================================================================
// Train tab
// =====================================================================

const trDataset = () => document.getElementById('tr-dataset');
const trLoraName = () => document.getElementById('tr-lora-name');
const trBase = () => document.getElementById('tr-base');
const trPreset = () => document.getElementById('tr-preset');
const trMaxSteps = () => document.getElementById('tr-max-steps');
const trSaveEvery = () => document.getElementById('tr-save-every');
const trBatchSize = () => document.getElementById('tr-batch-size');
const trGradAccum = () => document.getElementById('tr-grad-accum');
const trLr = () => document.getElementById('tr-lr');
const trStartBtn = () => document.getElementById('tr-start-btn');
const trStopBtn = () => document.getElementById('tr-stop-btn');
const trStartStatus = () => document.getElementById('tr-start-status');

const trNoActive = () => document.getElementById('tr-no-active');
const trActiveDetail = () => document.getElementById('tr-active-detail');
const trJobStateBadge = () => document.getElementById('tr-job-state-badge');
const trJobId = () => document.getElementById('tr-job-id');
const trJobName = () => document.getElementById('tr-job-name');
const trJobDataset = () => document.getElementById('tr-job-dataset');
const trJobStep = () => document.getElementById('tr-job-step');
const trJobLoss = () => document.getElementById('tr-job-loss');
const trJobValLoss = () => document.getElementById('tr-job-val-loss');
const trTestLoraBtn = () => document.getElementById('tr-test-lora-btn');
const trProgressBar = () => document.getElementById('tr-progress-bar');
const trJobLog = () => document.getElementById('tr-job-log');

const trJobsBody = () => document.getElementById('tr-jobs-body');
const trJobsEmpty = () => document.getElementById('tr-jobs-empty');
const trRefreshJobs = () => document.getElementById('tr-refresh-jobs');

let trActiveJobId = null;
let trPollHandle = null;

async function refreshTrainDatasetOptions() {
  try {
    const res = await fetch(`${API_URL}/datasets`);
    const json = await res.json();
    const sel = trDataset();
    sel.innerHTML = '';
    const items = json.datasets || [];
    if (items.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = '(create a dataset first)';
      sel.appendChild(opt);
    } else {
      items.forEach(d => {
        const opt = document.createElement('option');
        opt.value = d.name;
        opt.textContent = `${d.name} (${d.num_clips} clips)`;
        sel.appendChild(opt);
      });
    }
    updateStartBtnState();
  } catch (err) {
    reportFetchFailure('[train] データセットを読めませんでした:', err);
  }
}

async function loadTrainJobs() {
  try {
    const res = await fetch(`${API_URL}/lora/jobs`);
    const json = await res.json();
    const jobs = json.jobs || [];
    renderJobsTable(jobs);

    // Find any active job and set up polling.
    const active = jobs.find(j => ['pending', 'preparing', 'training', 'stopping'].includes(j.state));
    if (active) {
      trActiveJobId = active.id;
      startPolling();
      renderActiveJob(active);
    } else {
      stopPolling();
      trActiveJobId = null;
      renderActiveJob(null);
    }
  } catch (err) {
    reportFetchFailure('[train] 学習記録を読めませんでした:', err);
  }
}

// 状態は色だけに頼らず字でも示す。色の系統は 4 つに畳む
// （pending/preparing/training は「走っている」でひとまとめ）。
const TR_STATE = {
  pending:   { cls: 'run',  label: '待機中' },
  preparing: { cls: 'run',  label: '準備中' },
  training:  { cls: 'run',  label: '学習中' },
  stopping:  { cls: 'stop', label: '停止中' },
  done:      { cls: 'done', label: '完了' },
  failed:    { cls: 'fail', label: '失敗' },
  stopped:   { cls: 'stop', label: '中断' },
};
function trStateBadge(state) {
  const s = TR_STATE[state] || { cls: '', label: state || '—' };
  return `<span class="badge ${s.cls}">${escapeHtml(s.label)}</span>`;
}

function renderJobsTable(jobs) {
  const body = trJobsBody();
  body.innerHTML = '';
  if (jobs.length === 0) {
    trJobsEmpty().classList.remove('hidden');
    const sum0 = document.getElementById('tr-jobs-sum');
    if (sum0) sum0.textContent = '';
    return;
  }
  trJobsEmpty().classList.add('hidden');
  jobs.forEach(j => {
    const tr = document.createElement('tr');
    const dateStr = j.created_at ? new Date(j.created_at).toLocaleString() : '—';
    const stepStr = j.current_step != null && j.max_steps
      ? `${j.current_step}/${j.max_steps}`
      : (j.current_step != null ? String(j.current_step) : '—');
    const lossStr = j.current_loss != null ? j.current_loss.toFixed(4) : '—';
    const valStr = j.valid_loss != null ? j.valid_loss.toFixed(4) : '—';
    // 学習しただけでは何も登録されない。どの回を採ったかは試聴で登録した
    // ときだけ記録されるので、まだのものは「未採用」と出して聴き比べへ促す。
    let adopted;
    if (j.state !== 'done') {
      adopted = '<span class="text-muted">—</span>';
    } else if (j.registered_as) {
      const ck = j.registered_checkpoint ? ` (${escapeHtml(j.registered_checkpoint)})` : '';
      adopted = `${escapeHtml(j.registered_as)}<span class="text-muted">${ck}</span>`;
    } else {
      adopted = '<span class="text-muted">未採用</span>';
    }
    tr.innerHTML = `
      <td class="j-name">${escapeHtml(j.name || j.id)}</td>
      <td>${trStateBadge(j.state)}</td>
      <td class="num">${stepStr}</td>
      <td class="num">${lossStr}</td>
      <td class="num">${valStr}</td>
      <td>${adopted}</td>
      <td class="j-date">${dateStr}</td>
    `;
    body.appendChild(tr);
  });

  // 見出しの右に件数を出す。完了と失敗の内訳が分かると探しやすい。
  const sum = document.getElementById('tr-jobs-sum');
  if (sum) {
    const done = jobs.filter(j => j.state === 'done').length;
    const adoptedN = jobs.filter(j => j.registered_as).length;
    sum.innerHTML = `<b>${jobs.length}</b> 件 ・ 完了 <b>${done}</b> ・ 採用 <b>${adoptedN}</b>`;
  }
}

// ── 実行中の表示まわり ───────────────────────────────
// API は残り時間も損失の履歴も返さない。ここで見えているぶんから作る。
const trLossHistory = { jobId: null, points: [] };   // [{step, loss}]
const trLossPrev = {};                                // 直前の値（増減の計算用）
let trEtaBase = null;                                 // {jobId, t0, step0}

function trFormatEta(job, step, total) {
  if (!['training', 'preparing'].includes(job.state)) return '';
  const now = Date.now();
  if (!trEtaBase || trEtaBase.jobId !== job.id) {
    trEtaBase = { jobId: job.id, t0: now, step0: step };
    return '';
  }
  const dStep = step - trEtaBase.step0;
  const dMs = now - trEtaBase.t0;
  // 開始直後は 1 ステップの時間が安定しない。少し進むまで出さない。
  if (dStep < 5 || dMs < 20000) return '';
  const msPerStep = dMs / dStep;
  const left = Math.max(0, total - step) * msPerStep;
  const min = Math.round(left / 60000);
  if (min < 1) return 'まもなく終了';
  if (min < 60) return `残り 約 ${min} 分`;
  return `残り 約 ${Math.floor(min / 60)} 時間 ${min % 60} 分`;
}

function trRenderLoss(el, value, key) {
  if (!el) return;
  if (value == null) { el.textContent = '—'; return; }
  const prev = trLossPrev[key];
  trLossPrev[key] = value;
  el.textContent = value.toFixed(4);
  if (prev != null && prev !== value) {
    const d = value - prev;
    const span = document.createElement('span');
    // 下がっていれば良い方向。色分けは down / up で持つ。
    span.className = 'd ' + (d < 0 ? 'down' : 'up');
    span.textContent = (d < 0 ? '−' : '+') + Math.abs(d).toFixed(3);
    el.appendChild(span);
  }
}

function trPushLossPoint(jobId, step, loss) {
  if (loss == null) return;
  if (trLossHistory.jobId !== jobId) {
    trLossHistory.jobId = jobId;
    trLossHistory.points = [];
  }
  const last = trLossHistory.points[trLossHistory.points.length - 1];
  if (last && last.step === step) return;      // 同じステップは足さない
  trLossHistory.points.push({ step, loss });
  if (trLossHistory.points.length > 400) trLossHistory.points.shift();
}

// 損失の推移。履歴が無いので、開いている間に貯めた点だけを結ぶ。
// 点が 2 つ未満のときは線にならないので何も描かない。
function trDrawSpark() {
  const c = document.getElementById('tr-loss-spark');
  if (!c) return;
  const pts = trLossHistory.points;
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth || 300, h = c.clientHeight || 40;
  c.width = w * dpr; c.height = h * dpr;
  const g = c.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  if (pts.length < 2) return;

  const ys = pts.map(p => p.loss);
  const min = Math.min(...ys), max = Math.max(...ys);
  const span = (max - min) || 1;
  const x = (i) => (i / (pts.length - 1)) * w;
  const y = (v) => h - 3 - ((v - min) / span) * (h - 6);

  const css = getComputedStyle(c);
  g.strokeStyle = css.color || '#c9a86a';
  g.lineWidth = 1.5;
  g.beginPath();
  pts.forEach((p, i) => (i ? g.lineTo(x(i), y(p.loss)) : g.moveTo(x(i), y(p.loss))));
  g.stroke();
}

function renderActiveJob(job) {
  if (!job) {
    trNoActive().classList.remove('hidden');
    trActiveDetail().classList.add('hidden');
    trStopBtn().classList.add('hidden');
    trJobStateBadge().textContent = '';
    // 次に別の学習が走ったとき、前回の増減や線が残らないようにする
    trLossHistory.jobId = null; trLossHistory.points = [];
    trLossPrev.loss = trLossPrev.val = undefined;
    trEtaBase = null;
    updateStartBtnState();
    return;
  }
  trNoActive().classList.add('hidden');
  trActiveDetail().classList.remove('hidden');
  trStopBtn().classList.remove('hidden');
  updateStartBtnState();

  trJobId().textContent = job.id || '—';
  trJobName().textContent = job.name || '—';
  trJobDataset().textContent = job.dataset || '—';
  trJobStateBadge().innerHTML = trStateBadge(job.state);
  // 走っている枠の中にも状態を出す。表記は履歴と同じものを使う。
  const stateText = document.getElementById('tr-job-state-text');
  if (stateText) {
    stateText.textContent = (TR_STATE[job.state] || {}).label || job.state;
  }

  const step = job.current_step || 0;
  const total = job.max_steps || 1;
  const pct = Math.min(100, (step / total) * 100);
  trJobStep().innerHTML = `${step.toLocaleString()}<span class="of"> / ${total.toLocaleString()}</span>`;

  const pctEl = document.getElementById('tr-job-pct');
  if (pctEl) pctEl.textContent = `${Math.round(pct)}%`;

  // 残り時間。API は返さないので、進んだステップと経過時間から見積もる。
  // 開始直後は 1 ステップの時間が安定しないので、少し進むまで出さない。
  const etaEl = document.getElementById('tr-job-eta');
  if (etaEl) etaEl.textContent = trFormatEta(job, step, total);

  // 損失は増減も添える。下がっているかが一目で分かればよい。
  trRenderLoss(trJobLoss(), job.current_loss, 'loss');
  // 検証ロスは valid_every ごとにしか出ないので、走り始めは '—' のまま。
  // 学習ロスと別枠にしてあるのは、混ぜると検証の瞬間だけ跳ねて見えるため。
  trRenderLoss(trJobValLoss(), job.valid_loss, 'val');

  trPushLossPoint(job.id, step, job.current_loss);
  trDrawSpark();

  // 学習が終わったら次にやることへ繋ぐ。まだ採用していなければ聴き比べへ、
  // 採用済みなら本番タブへ。自動登録をやめたので、終わっただけでは
  // レジストリに何も無く、Synthesize へ飛ばしても選ぶものがない。
  const btn = trTestLoraBtn();
  if (btn) {
    const finished = job.state === 'done';
    btn.classList.toggle('hidden', !finished);
    // 文字だけ差し替える。textContent にすると中の svg ごと消える。
    const setLabel = (t) => {
      const svg = btn.querySelector('svg');
      btn.textContent = '';
      if (svg) btn.appendChild(svg);
      btn.appendChild(document.createTextNode(t));
    };
    if (!finished) {
      btn.onclick = null;
    } else if (job.registered_as) {
      setLabel('この LoRA を試す');
      btn.title = 'この LoRA を選んだ状態で生成タブを開きます';
      btn.onclick = () => testTrainedLora(job.registered_as);
    } else {
      setLabel('チェックポイントを聴き比べる');
      btn.title = '学習記録を選んだ状態で試聴を開きます';
      btn.onclick = () => openCheckpointAudition(job.id);
    }
  }
  trProgressBar().style.width = `${pct}%`;

  const tail = job.log_tail || [];
  trJobLog().textContent = tail.join('\n');
  trJobLog().scrollTop = trJobLog().scrollHeight;
}

async function pollActiveJob() {
  if (!trActiveJobId) return;
  try {
    const res = await fetch(`${API_URL}/lora/jobs/${trActiveJobId}`);
    if (!res.ok) return;
    const job = await res.json();
    renderActiveJob(job);
    setTrainingBusy(['pending', 'preparing', 'training', 'stopping'].includes(job.state));
    if (!['pending', 'preparing', 'training', 'stopping'].includes(job.state)) {
      stopPolling();
      trActiveJobId = null;
      // 完了後だけ全体を再読み込み（チカチカ防止のためポーリング中は呼ばない）
      await loadTrainJobs();
      if (job.state === 'done') {
        loadLoras();
      }
    }
  } catch (err) {
    reportFetchFailure('[train] 進捗を取れませんでした:', err);
  }
}

function startPolling() {
  if (trPollHandle) return;
  trPollHandle = setInterval(pollActiveJob, 2000);
}

function stopPolling() {
  if (trPollHandle) {
    clearInterval(trPollHandle);
    trPollHandle = null;
  }
}

function updateStartBtnState() {
  // Start + Auto Setting are gated by Dataset selection only. LoRA Name
  // can stay blank — when blank we default to the dataset name at submit
  // time. Active job presence still blocks both.
  const dataset = trDataset().value;
  const blockedByActive = !!trActiveJobId;
  const disabled = !dataset || blockedByActive;
  trStartBtn().disabled = disabled;
  const auto = document.getElementById('tr-auto-setting-btn');
  if (auto) auto.disabled = disabled;
}

async function startTrainingJob() {
  const dataset = trDataset().value;
  if (!dataset) {
    trStartStatus().textContent = 'Pick a dataset first.';
    updateStartBtnState();
    return;
  }
  // 空欄ならデータセット名で補う。補った結果は入力欄にも書き戻す
  // ——画面に無い名前で走ると、あとで何のジョブか分からなくなる。
  let name = trLoraName().value.trim();
  if (!name) {
    name = dataset;
    trLoraName().value = name;
  }

  // 同名があっても開始は妨げない。学習しただけでは何も登録されず、
  // レジストリに触るのは採用のときだけなので、ここで上書きは起きない。
  // 置き換えるかどうかは採用の時点で確認する。
  const body = {
    lora_name: name,
    dataset,
    base: trBase().value,
    preset: trPreset().value,
    max_steps: parseInt(trMaxSteps().value, 10) || 2000,
    save_every: parseInt(trSaveEvery().value, 10) || 500,
    batch_size: parseInt(trBatchSize().value, 10) || 4,
    gradient_accumulation_steps: parseInt(trGradAccum().value, 10) || 8,
  };
  const lrRaw = trLr().value.trim();
  if (lrRaw) body.learning_rate = parseFloat(lrRaw);

  trStartBtn().disabled = true;
  trStartStatus().textContent = 'Starting…';
  try {
    const res = await fetch(`${API_URL}/lora/jobs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const json = await res.json();
    if (!res.ok) {
      throw new Error(json.detail || json.error || `status ${res.status}`);
    }
    trStartStatus().textContent = `Started job ${json.job_id}`;
    trActiveJobId = json.job_id;
    setTrainingBusy(true);   // 3秒のポーリングを待たずに生成を止める
    startPolling();
    await loadTrainJobs();
  } catch (err) {
    trStartStatus().textContent = `Start failed: ${err.message}`;
  } finally {
    updateStartBtnState();
  }
}

// 学習が終わったアダプタをそのまま試せるようにする。val_loss は指標でしかなく、
// 良し悪しは聴いて決めるものなので、生成画面へ渡すところまでを受け持つ。
// 学習が終わったジョブを試聴で開く。自動登録をやめたので、聴き比べて
// 採用するまでレジストリには何も載らない。その入口をここから繋ぐ。
async function openCheckpointAudition(jobId) {
  await trLoadTestJobs();
  const sel = trTestJob();
  if (!sel) return;
  sel.value = jobId;
  if (sel.value !== jobId) {
    // 完了したジョブしか一覧に入らない。取れないなら理由を出す。
    trTestStatus().textContent = 'この学習記録は試聴の一覧にありません。';
    return;
  }
  await trLoadCheckpoints();
  sel.scrollIntoView({ block: 'center' });
  trTestText()?.focus();
}

async function testTrainedLora(loraName) {
  await loadLoras();   // 直前に登録されたものを一覧へ取り込む
  const meta = loraRegistry.find(l => l.name === loraName);
  if (!meta) {
    trStartStatus().textContent = `LoRA "${loraName}" が一覧に見つかりません。`;
    return;
  }
  // ベースが違うと候補に出ないので、先にモデルを合わせる。
  if (modelSelect.value !== meta.base) {
    modelSelect.value = meta.base;
    modelSelect.dispatchEvent(new Event('change'));
  }
  const loraRadio = document.querySelector('input[name="cond-mode"][value="lora"]');
  if (loraRadio) {
    loraRadio.checked = true;
    applyCondModeUI();
  }
  refreshLoraDropdown();
  loraSelect.value = loraName;
  loraSelect.dispatchEvent(new Event('change'));
  document.querySelector('.tab-btn[data-tab="synthesize"]')?.click();
  textInput.focus();
}

// ── チェックポイント試聴 ──────────────────────────────────────────
// 学習は save_every ごとにチェックポイントを残すが、登録されるのは1つだけで
// 残りは聴かれないまま消える。どれが良いかは聴かないと分からないので、
// 学習記録とチェックポイントを選んでその場で鳴らせるようにする。
const trTestJob = () => document.getElementById('tr-test-job');
const trTestCkpt = () => document.getElementById('tr-test-ckpt');
const trTestText = () => document.getElementById('tr-test-text');
const trTestStatus = () => document.getElementById('tr-test-status');
const trTestResults = () => document.getElementById('tr-test-results');
const trTestRegName = () => document.getElementById('tr-test-regname');

// 試聴用の一時登録名。生成 API は登録名しか受け付けないため、選んだ
// チェックポイントを毎回この名前で上書き登録してから鳴らす。
const TR_PREVIEW_LORA = '_ckpt_preview';

let trTestJobs = [];

async function trLoadTestJobs() {
  try {
    const res = await fetch(`${API_URL}/lora/jobs`);
    if (!res.ok) {
      trTestStatus().textContent = `一覧の取得に失敗しました (HTTP ${res.status})`;
      return;
    }
    const json = await res.json();
    // 完了したものだけ。失敗・停止はチェックポイントが揃っていない。
    trTestJobs = (json.jobs || []).filter(j => j.state === 'done');
    const sel = trTestJob();
    const prev = sel.value;
    sel.innerHTML = '';
    if (!trTestJobs.length) {
      const o = document.createElement('option');
      o.value = ''; o.textContent = '(完了した学習がありません)';
      sel.appendChild(o);
      trTestCkpt().innerHTML = '';
      return;
    }
    trTestJobs.forEach(j => {
      const o = document.createElement('option');
      o.value = j.id;
      const when = j.created_at ? new Date(j.created_at).toLocaleDateString() : '';
      // base を出す。v3 と v4 が混在しているので、どれを聴いているか分かるようにする。
      const tag = j.base ? ` [${j.base}]` : '';
      o.textContent = `${j.name || j.id}${tag}${when ? ' — ' + when : ''}`;
      sel.appendChild(o);
    });
    if (prev && trTestJobs.some(j => j.id === prev)) sel.value = prev;
    trTestStatus().textContent = '';
    await trLoadCheckpoints();
  } catch (err) {
    // 黙って空のままだと原因が分からない。画面にも出す。
    reportFetchFailure('[train] 試聴用の記録を読めませんでした:', err);
    trTestStatus().textContent = `一覧の取得に失敗: ${err.message}`;
  }
}

async function trLoadCheckpoints() {
  const jobId = trTestJob().value;
  const sel = trTestCkpt();
  sel.innerHTML = '';
  if (!jobId) return;
  try {
    const res = await fetch(`${API_URL}/lora/jobs/${jobId}/checkpoints`);
    const json = await res.json();
    const cks = json.checkpoints || [];
    // 消す前にどれだけ空くか分かるようにしておく。
    trTestSize(json.checkpoints_bytes || 0, cks.length);
    if (!cks.length) {
      const o = document.createElement('option');
      o.value = ''; o.textContent = '(チェックポイントが残っていません)';
      sel.appendChild(o);
      return;
    }
    cks.forEach(c => {
      const o = document.createElement('option');
      o.value = c.name;
      const bits = [];
      if (c.step != null) bits.push(`step ${c.step}`);
      if (c.val_loss != null) bits.push(`val ${c.val_loss.toFixed(6)}`);
      if (c.is_final) bits.push('最終');
      o.textContent = bits.length ? `${c.name}  (${bits.join(' / ')})` : c.name;
      sel.appendChild(o);
    });
    // 登録名の既定は学習時の名前。
    const job = trTestJobs.find(j => j.id === jobId);
    if (job && !trTestRegName().value) trTestRegName().placeholder = `空欄 = ${job.name}`;
  } catch (err) {
    console.error('[train] failed to load checkpoints:', err);
  }
}

// 世代が占めている容量を削除ボタンに出す。何 GB 空くのか分からないまま
// 消させない、というだけの表示。採用済みでなければ「採用以外を削除」は押せない。
function trTestSize(bytes, count) {
  const others = document.getElementById('tr-test-purge-others');
  const one = document.getElementById('tr-test-purge-one');
  const job = trTestJobs.find(j => j.id === trTestJob().value);
  if (others) {
    others.disabled = !count || !(job && job.registered_as);
    others.textContent = count > 1
      ? `採用以外を削除 (${count - 1}件 / ${(bytes / (1 << 30)).toFixed(1)} GB)`
      : '採用以外を削除';
  }
  if (one) one.disabled = !count;
}

// 聴き比べて採用が決まれば、残りの世代は用済みになる。まとめて片付ける。
async function trPurgeOtherCheckpoints() {
  const jobId = trTestJob().value;
  if (!jobId) { trTestStatus().textContent = '学習記録を選んでください。'; return; }
  const job = trTestJobs.find(j => j.id === jobId);
  if (!job || !job.registered_as) {
    trTestStatus().textContent = 'まだ採用していません。先に「この回を採用して登録」を押してください。';
    return;
  }
  const kept = job.registered_checkpoint || 'checkpoint_final';
  if (!confirm(
    `"${job.name || jobId}" の世代を整理します。\n\n` +
    `残す: ${kept}\n` +
    `他の世代は削除されます（登録済みの LoRA と学習記録は残ります）。`
  )) return;
  trTestStatus().textContent = '削除中…';
  try {
    const res = await fetch(
      `${API_URL}/lora/jobs/${jobId}/checkpoints?keep_adopted=true`, { method: 'DELETE' });
    const json = await res.json();
    if (!res.ok) throw new Error(json.detail || json.error || `status ${res.status}`);
    await trLoadCheckpoints();
    trTestStatus().textContent =
      `${json.removed} 世代を削除しました（${(json.freed_bytes / (1 << 30)).toFixed(1)} GB）。`;
  } catch (err) {
    trTestStatus().textContent = `削除に失敗: ${err.message}`;
  }
}

// 個別に落としたいとき。聴いて明らかに要らない回をその場で捨てられる。
async function trPurgeOneCheckpoint() {
  const jobId = trTestJob().value;
  const ckpt = trTestCkpt().value;
  if (!jobId || !ckpt) { trTestStatus().textContent = '学習記録とチェックポイントを選んでください。'; return; }
  const job = trTestJobs.find(j => j.id === jobId);
  const adoptedNow = job && job.registered_checkpoint === ckpt;
  if (!confirm(
    `${ckpt} を削除します。` + (adoptedNow ? '\n\nこれは採用した回です。' : '')
  )) return;
  trTestStatus().textContent = '削除中…';
  try {
    const res = await fetch(
      `${API_URL}/lora/jobs/${jobId}/checkpoints/${encodeURIComponent(ckpt)}`,
      { method: 'DELETE' });
    const json = await res.json();
    if (!res.ok) throw new Error(json.detail || json.error || `status ${res.status}`);
    await trLoadCheckpoints();
    trTestStatus().textContent =
      `${ckpt} を削除しました（${(json.freed_bytes / (1 << 20)).toFixed(0)} MB）。`;
  } catch (err) {
    trTestStatus().textContent = `削除に失敗: ${err.message}`;
  }
}

async function trGenerateFromCheckpoint() {
  const jobId = trTestJob().value;
  const ckpt = trTestCkpt().value;
  const text = (trTestText().value || '').trim();
  if (!jobId || !ckpt) { trTestStatus().textContent = '学習記録とチェックポイントを選んでください。'; return; }
  if (!text) { trTestStatus().textContent = 'テキストを入力してください。'; return; }

  const job = trTestJobs.find(j => j.id === jobId);
  // 既定値に落とすと、v3 で学習したものを v4 で鳴らしても気付けない。
  // 学習時の base が分からないなら鳴らさずに止める。
  const base = job && job.base;
  if (!base) {
    trTestStatus().textContent = 'この学習記録に base が記録されていません（誤ったモデルで鳴るため中止）。';
    return;
  }
  // 試聴は固定名 _ckpt_preview に登録してから鳴らす。続けて押されると
  // 先の生成が読む前に次の登録が上書きし、選んだのと違う回の音が出る。
  // 生成が終わるまで押せないようにして、登録と生成の対を崩さない。
  const genBtn = document.getElementById('tr-test-gen');
  if (genBtn) {
    if (genBtn.disabled) return;
    genBtn.disabled = true;
  }
  trTestStatus().textContent = '生成中…';
  try {
    // 一時名で登録してから生成する（生成 API は登録名しか受け取らない）。
    const reg = await fetch(`${API_URL}/lora/jobs/${jobId}/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ checkpoint: ckpt, lora_name: TR_PREVIEW_LORA }),
    });
    const regJson = await reg.json().catch(() => ({}));
    if (!reg.ok) {
      throw new Error(regJson.detail || regJson.error || `register failed (${reg.status})`);
    }
    await loadLoras();

    // 生成は Synthesize と同じ経路を通す。違うのは LoRA が
    // 一時登録したチェックポイントであることと、seed を固定することだけ。
    // 声質の有無は登録の応答から渡す。この名前は一覧に出さないので
    // loraRegistry から引けず、渡さないと完パケでも参照音声が付く。
    const json = await runSynthesis({
      text,
      loraName: TR_PREVIEW_LORA,
      modelType: base,
      seed: 1234,   // 世代間の比較のため固定
      loraHasVoice: (regJson.provides || []).includes('voice'),
    });
    const url = json.results[0];
    const wav = await fetch(`${API_ORIGIN}${url}`);
    const blobUrl = URL.createObjectURL(await wav.blob());

    // 聴き比べたいので、前の結果は消さずに積む。
    const row = document.createElement('div');
    row.style.cssText = 'display:flex; gap:8px; align-items:center; margin-top:6px; flex-wrap:wrap;';
    const label = document.createElement('span');
    label.className = 'text-xs';
    label.textContent = ckpt;
    const audio = document.createElement('audio');
    audio.controls = true;
    audio.src = blobUrl;
    audio.style.height = '32px';
    row.appendChild(label);
    row.appendChild(audio);
    trTestResults().prepend(row);
    trTestStatus().textContent = '';
  } catch (err) {
    trTestStatus().textContent = `失敗: ${err.message}`;
  } finally {
    if (genBtn) genBtn.disabled = false;
  }
}

async function trRegisterCheckpoint() {
  const jobId = trTestJob().value;
  const ckpt = trTestCkpt().value;
  if (!jobId || !ckpt) { trTestStatus().textContent = '学習記録とチェックポイントを選んでください。'; return; }
  const job = trTestJobs.find(j => j.id === jobId);
  const name = (trTestRegName().value || '').trim() || (job && job.name) || '';
  if (!name) { trTestStatus().textContent = '登録名を入力してください。'; return; }
  // 置き換えになるときだけ確認する。毎回同じ文言を出すと読まれなくなる。
  const clash = loraRegistry.find(l => l.name === name);
  const msg = clash
    ? `"${name}" は既にあります。この回で置き換えますか？`
    : `"${name}" として登録します。`;
  if (!confirm(msg)) return;
  trTestStatus().textContent = '登録中…';
  try {
    const res = await fetch(`${API_URL}/lora/jobs/${jobId}/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ checkpoint: ckpt, lora_name: name }),
    });
    const json = await res.json();
    if (!res.ok) throw new Error(json.detail || json.error || `status ${res.status}`);
    await loadLoras();
    // 採用はジョブ側にも記録される。History の「採用」列へ反映する。
    await loadTrainJobs();
    // 試聴タブが持つ trTestJobs も取り直す。これを忘れると registered_as が
    // 古いままで、「採用以外を削除」が押せないままになる。
    await trLoadTestJobs();
    trTestStatus().textContent = `${name} として登録しました（${ckpt}）。`;
  } catch (err) {
    trTestStatus().textContent = `登録に失敗: ${err.message}`;
  }
}

// 推奨値を入力欄に入れるだけで、学習は開始しない。
// 開始まで続けていたころは、画面に残った値と実際に走る値が食い違い
// （50 と入れたのに 350 で走る）、何で回っているのか分からなくなっていた。
// 走るのは常に画面に見えている値、という一点を崩さない。
async function applyAutoSetting() {
  const dataset = trDataset().value;
  if (!dataset) {
    trStartStatus().textContent = 'Pick a dataset first.';
    updateStartBtnState();
    return;
  }
  trStartStatus().textContent = `Analyzing dataset "${dataset}"…`;
  try {
    const res = await fetch(`${API_URL}/datasets/${encodeURIComponent(dataset)}/auto_config`);
    const json = await res.json();
    if (!res.ok) {
      throw new Error(json.detail || json.error || `status ${res.status}`);
    }
    const r = json.recommended || {};
    if (r.max_steps != null) trMaxSteps().value = r.max_steps;
    if (r.save_every != null) trSaveEvery().value = r.save_every;
    if (r.preset) {
      trPreset().value = r.preset;
      // <select> に無い値を入れると .value は黙って空文字になり、
      // そのまま送信されて 400 で落ちる。ここで止めて理由を出す。
      if (trPreset().value !== r.preset) {
        throw new Error(`unknown preset "${r.preset}" (this build accepts: ` +
          Array.from(trPreset().options).map(o => o.value).join(', ') + ')');
      }
    }
    trStartStatus().textContent =
      `推奨値を入れました: ${r.preset} / max_steps=${r.max_steps} / save_every=${r.save_every} ` +
      `(${json.num_clips} clips, avg ${json.avg_duration}s, total ${formatDuration(json.total_duration)})。` +
      `内容を確認して Start Training を押してください。`;
  } catch (err) {
    trStartStatus().textContent = `Auto config failed: ${err.message}`;
  }
}

async function stopTrainingJob() {
  if (!trActiveJobId) return;
  if (!confirm('Stop the running training job?')) return;
  try {
    const res = await fetch(`${API_URL}/lora/jobs/${trActiveJobId}/stop`, { method: 'POST' });
    const json = await res.json();
    if (!res.ok) throw new Error(json.error || `status ${res.status}`);
    trStartStatus().textContent = 'Stop requested…';
  } catch (err) {
    alert(`Stop failed: ${err.message}`);
  }
}

function setupTrainTab() {
  // 配線のどこかで例外が出ると以降の登録が全部飛ぶ。プルダウンが空のまま
  // ステータスも空という症状は「ここで死んで trLoadTestJobs に届いていない」
  // ときにも起きるので、理由が画面に残るようにする。
  try {
    trStartBtn().addEventListener('click', startTrainingJob);
    trStopBtn().addEventListener('click', stopTrainingJob);
    trRefreshJobs().addEventListener('click', loadTrainJobs);
    document.getElementById('tr-test-reload')?.addEventListener('click', trLoadTestJobs);
    trTestJob()?.addEventListener('change', trLoadCheckpoints);
    document.getElementById('tr-test-gen')?.addEventListener('click', trGenerateFromCheckpoint);
    document.getElementById('tr-test-register')?.addEventListener('click', trRegisterCheckpoint);
    document.getElementById('tr-test-purge-others')?.addEventListener('click', trPurgeOtherCheckpoints);
    document.getElementById('tr-test-purge-one')?.addEventListener('click', trPurgeOneCheckpoint);
    trLoadTestJobs();
    const autoBtn = document.getElementById('tr-auto-setting-btn');
    if (autoBtn) autoBtn.addEventListener('click', applyAutoSetting);
    // Keep Start/Auto enabled state in sync with required fields.
    trDataset().addEventListener('change', updateStartBtnState);
  } catch (err) {
    console.error('[train] setupTrainTab failed:', err);
    const st = trTestStatus();
    if (st) st.textContent = `Train タブの初期化に失敗: ${err.message}`;
  }
}

// Start
init();

// =====================================================================
// 朗読タブ — ボイス/モデル/LoRA は生成タブの設定をそのまま使用
// =====================================================================
(function () {

let nChunks = [], nIndex = 0, nPlaying = false, nAudio = null;
let nBuffer = {}, nSession = null, nRawText = '', nSynthBusy = false, gapTimer = null, nFilePath = '';

const $ = id => document.getElementById(id);
const openBtn      = $('narrate-open-btn');
const filePathInput = $('narrate-file-path');
const chunkInfo  = $('narrate-chunk-info');
const posText    = $('narrate-pos-text');
const display    = $('narrate-text-display');
const prevBtn    = $('narrate-prev-btn');
const playBtn    = $('narrate-play-btn');
const stopBtn    = $('narrate-stop-btn');
const nextBtn    = $('narrate-next-btn');
const progBar    = $('narrate-progress-bar');
const statusEl   = $('narrate-status');
const stepsR     = $('narrate-steps');    const stepsV   = $('narrate-steps-val');
const cfgTextR   = $('narrate-cfg-text');  const cfgTextV = $('narrate-cfg-text-val');
const cfgSpkR    = $('narrate-cfg-spk');   const cfgSpkV  = $('narrate-cfg-spk-val');
const gapR       = $('narrate-gap');       const gapV     = $('narrate-gap-val');
const volR       = $('narrate-vol');       const volV     = $('narrate-vol-val');
const rateR      = $('narrate-rate');      const rateV    = $('narrate-rate-val');
const narrEmojiC = $('narrate-emoji');     const narrEmojiN = $('narrate-emoji-note');
const narrEmojiOn = () => !!(narrEmojiC && narrEmojiC.checked);
const prefR      = $('narrate-prefetch');  const prefV    = $('narrate-prefetch-val');

stepsR.addEventListener('input',   () => stepsV.textContent   = stepsR.value);
cfgTextR.addEventListener('input', () => cfgTextV.textContent = parseFloat(cfgTextR.value).toFixed(1));
cfgSpkR.addEventListener('input',  () => cfgSpkV.textContent  = parseFloat(cfgSpkR.value).toFixed(1));
gapR.addEventListener('input',     () => gapV.textContent     = parseFloat(gapR.value).toFixed(1));
volR.addEventListener('input',     () => {
  volV.textContent = volR.value;
  if (nAudio) nAudio.volume = volR.value / 100;
  localStorage.setItem('narrate_vol', volR.value);
});
rateR.addEventListener('input', () => {
  rateV.textContent = parseFloat(rateR.value).toFixed(2);
  // 尺は生成時に決まるので、鳴っている音には遡って効かない。
  localStorage.setItem('narrate_duration_scale', rateR.value);
  invalidateNarrateSession();
});
prefR.addEventListener('input', () => {
  prefV.textContent = prefR.value;
  localStorage.setItem('narrate_prefetch', prefR.value);
});
// 📖 は v4 で追加された注釈。v3 以前の表には無いので、選んでも効かない。
// 黙って無視されると原因を追えないので、選んだモデルに応じて注記を出す。
function refreshNarrEmojiNote() {
  if (!narrEmojiN) return;
  const m = modelSelect ? modelSelect.value : '';
  // 📖 は v4 で追加された注釈。v4.1 は本体が v4 と同じなので同様に効く。
  narrEmojiN.textContent = (narrEmojiOn() && !isV4Model(m)) ? '（v4 系のみ有効）' : '';
}
if (narrEmojiC) {
  narrEmojiC.addEventListener('change', () => {
    localStorage.setItem('narrate_emoji', narrEmojiOn() ? 'on' : 'off');
    refreshNarrEmojiNote();
    invalidateNarrateSession();
  });
}
if (modelSelect) modelSelect.addEventListener('change', refreshNarrEmojiNote);

// 保存済みの設定を復元
(function () {
  const saved = localStorage.getItem('narrate_vol');
  if (saved !== null) { volR.value = saved; volV.textContent = saved; }
  // 以前この枠は再生速度（0.5〜2.0）で、いまは生成する尺（0.4〜1.5）。
  // 意味が逆向きなので旧値は引き継がない（旧 0.5 は「半速＝ゆっくり」だが、
  // 尺として読むと「半分の長さ＝速い」になり、真逆に化ける）。範囲だけ見て
  // 通すと 0.5〜1.5 が黙って反転するので、キーごと分けて旧値は捨てる。
  const rate = localStorage.getItem('narrate_duration_scale');
  if (rate !== null) {
    const r = parseFloat(rate);
    const ok = Number.isFinite(r) && r >= 0.4 && r <= 1.5;
    rateR.value = ok ? rate : '1.0';
  }
  rateV.textContent = parseFloat(rateR.value).toFixed(2);
  // 旧キーはもう読まないので片付ける。残しても誤解のもとにしかならない。
  localStorage.removeItem('narrate_rate');
  const pref = localStorage.getItem('narrate_prefetch');
  if (pref !== null) { prefR.value = pref; prefV.textContent = pref; }
  const ne = localStorage.getItem('narrate_emoji');
  if (narrEmojiC && (ne === 'on' || ne === 'off')) narrEmojiC.checked = (ne === 'on');
  refreshNarrEmojiNote();
})();

// チャンク末尾の記号で間（無音ms）を決める。倍率スライダーを掛ける。
function gapMsFor(text) {
  const base = /[。！？]\s*$/.test(text) ? 450
             : /」\s*$/.test(text)       ? 300
             : /—\s*$/.test(text)        ? 150
             :                              300;
  return base * parseFloat(gapR.value);
}

// ── 分割 ──
function splitText(text) {
  text = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  // まず段落（空行区切り）に分ける
  const paragraphs = text.split(/\n{2,}/);
  const result = [];

  paragraphs.forEach((para, pIdx) => {
    para = para.trim();
    if (!para) return;

    // 段落内をチャンクに分割
    //
    // 会話の追跡は鉤括弧系（「」『』）だけ。丸括弧や引用符まで追うと、
    // 閉じない ( ひとつで段落の残り全部が 1 チャンクに潰れ、単独の ) では
    // 語中で切れる。地の文かどうかの判定は別（isNarrationLine）で広く見る。
    const raw = [];
    let cur = '', quoteCloser = '';
    for (let i = 0; i < para.length; i++) {
      const c = para[i], next = para[i + 1] || '';
      cur += c;
      if (!quoteCloser) {
        const q = DIALOG_PAIRS[c];
        if (q) { quoteCloser = q; continue; }
      }
      let cut = false;
      if (quoteCloser) {
        if (c === quoteCloser) {
          quoteCloser = '';
          // 閉じ括弧の直後の句読点は取り込んでから切る。先に切ると
          // 「。」だけの片が残り、末尾記号を落とすと空になって落ちる。
          if (/[。、]/.test(next)) continue;
          // 助詞が続くなら「〜」と言った の途中なので切らない。
          cut = !DIALOG_TAIL_RE.test(next);
        }
      } else {
        if (/[。！？]/.test(c)) cut = true;
        else if (DIALOG_CLOSERS.includes(c)) cut = !DIALOG_TAIL_RE.test(next);
        else if (c === '—' && next === '—') { cur += next; i++; cut = true; }
      }
      if (cut || i === para.length - 1) { const t = cur.trim(); if (t) raw.push(t); cur = ''; }
    }

    // 地の文で20字以下なら次チャンクと結合（合計100字以内）
    const merged = [];
    for (let i = 0; i < raw.length; i++) {
      const ch = raw[i];
      const next = raw[i + 1];
      // セリフは短くても単独で残す。地の文と混ぜると演技が引きずられる。
      if (!QUOTE_OPENERS.includes(ch[0]) && ch.length <= 20 && next && ch.length + next.length <= 100) {
        merged.push(ch + next);
        i++;
      } else {
        merged.push(ch);
      }
    }

    // 150字超を強制分割
    const finals = [];
    for (const ch of merged) {
      if (ch.length <= 150) { finals.push(ch); continue; }
      let rem = ch;
      while (rem.length > 150) {
        let cut = -1;
        for (let j = Math.min(149, rem.length - 1); j >= 20; j--) { if (/[。！？」]/.test(rem[j])) { cut = j; break; } }
        if (cut < 0) cut = 149;
        finals.push(rem.slice(0, cut + 1).trim());
        rem = rem.slice(cut + 1).trim();
      }
      if (rem) finals.push(rem);
    }

    // 段落の先頭チャンクに newPara フラグ（最初の段落は除く）
    finals.forEach((t, idx) => {
      result.push({ text: t, newPara: (pIdx > 0 && idx === 0) });
    });
  });

  return result;
}
// 再生ボタンは線画。textContent で書き換えると中の svg が消えるので、
// use の参照先だけ差し替える。
function setPlayIcon(playing) {
  const use = playBtn.querySelector('use');
  if (use) use.setAttribute('href', playing ? '#ic-pause' : '#ic-play2');
  playBtn.dataset.tip = playing ? '一時停止' : '読み上げる';
  playBtn.setAttribute('aria-label', playing ? '一時停止' : '読み上げる');
}

function chunkClass(i) {
  return 'chunk' +
    (nChunks[i].newPara ? ' para' : '') +
    (i < nIndex ? ' done' : '') +
    (i === nIndex ? ' now' : '');
}
function renderText() {
  display.innerHTML = '';
  nChunks.forEach((chunk, i) => {
    const p = document.createElement('p');
    p.className = chunkClass(i);
    p.textContent = chunk.text.replace(/\n/g, ' ');
    p.addEventListener('click', () => jumpTo(i));
    display.appendChild(p);
  });
  scrollToActive(); updateMeta();
}
function refreshChunkClasses() {
  display.querySelectorAll('.chunk').forEach((el, i) => {
    el.className = chunkClass(i);
  });
  scrollToActive(); updateMeta();
}
function scrollToActive() {
  const el = display.querySelector('.chunk.now');
  if (el) el.scrollIntoView({ block: 'center', behavior: 'smooth' });
}
function updateMeta() {
  progBar.style.width = (nChunks.length ? (nIndex / nChunks.length) * 100 : 0) + '%';
  // 行数はファイル行に、現在位置は進み具合の下に置く（モックの配置）
  chunkInfo.innerHTML = nChunks.length ? `<b>${nChunks.length}</b> 行` : '';
  if (posText) {
    posText.innerHTML = nChunks.length
      ? `<b>${nIndex + 1}</b> / ${nChunks.length} 行` : '';
  }
}

// ── 現在の設定を読む（生成タブの選択＋朗読の合成パラメータ） ──
function currentSettings() {
  const condLora = (typeof getCondMode === 'function' && getCondMode() === 'lora');
  const selfrefEl = document.getElementById('narrate-selfref');
  const seedEl = document.getElementById('narrate-selfref-seed');
  const loraName = (condLora && loraSelect.value) ? loraSelect.value : null;
  const meta = loraName ? loraRegistry.find(l => l.name === loraName) : null;
  const loraHasVoice = !!(meta && (meta.provides || []).includes('voice'));
  return {
    model:     modelSelect.value,
    loraName,
    loraHasVoice,
    // A style-only adapter needs the sample to hold the speaker steady across
    // chunks; only a full adapter can narrate without one.
    voicePath: (selectedVoice && !loraHasVoice) ? selectedVoice.path : null,
    steps:     parseInt(stepsR.value),
    cfgText:   parseFloat(cfgTextR.value),
    cfgSpk:    parseFloat(cfgSpkR.value),
    selfrefOn: !!(selfrefEl && selfrefEl.checked),
    seedText:  (seedEl && seedEl.value.trim()) || '',
    // 尺は朗読タブ自身の「読む速さ」で決める。生成タブの値を黙って
    // borrow していたが、朗読側に設定が見えないまま尺が変わるのは
    // 追えない。ここが唯一の出どころ。
    durationScale: rateR ? rateR.value : null,
    // 精度は生成タブと共通の設定を使う。朗読だけ別の精度で回ると、
    // 同じ声のはずが章によって変わってしまう。
    precision: paramPrecision ? paramPrecision.value : 'auto',
    device: paramDevice ? paramDevice.value : 'auto',
    // 地の文へのナレーション注釈。付けると尺の予測も変わるので、
    // 切り替えたら作り直せるよう settingsKey にも入れる。
    narrEmoji: narrEmojiOn(),
  };
}
// 設定が変わったか判定するキー（seed/refBlobは除外）
function settingsKey(s) {
  return [s.model, s.loraName, s.voicePath, s.steps, s.cfgText, s.cfgSpk, s.selfrefOn, s.seedText, s.durationScale, s.precision, s.device, s.narrEmoji].join('|');
}

// 合成の世代。設定を変えるたびに進める。
//
// 先読みは複数の合成を同時に走らせるので、設定を変えた瞬間に走っていた分は
// 止められない。世代を持たせず nBuffer を空にするだけだと、そのあと古い設定で
// 終わった応答が新しい nBuffer に入り、次の再生でそれが使われてしまう
// （変えたはずの速さや精度が反映されない）。開始時の世代を覚えておいて、
// 戻ってきたときに世代が変わっていたら結果を捨てる。
let nGeneration = 0;

function invalidateNarrateSession() {
  nGeneration++;
  nBuffer = {};
  // 鳴っている音は最後まで流す。途中で切ると設定を触るたびに音が飛ぶ。
  // 次の行から新しい設定で作り直される。
  //
  // currentSettings() で丸ごと差し替えると、play() が足した seed / key /
  // refBlob が消える。seed が undefined のまま送られるとサーバが 400 を返し、
  // playChunk はそこで止まるので朗読全体が死ぬ。話者を保つ意味でも seed は
  // 引き継ぐ。key は新しい設定のものに更新する（次の play() で作り直すか
  // どうかの判定に使う）。
  if (nPlaying && nSession) {
    const s = currentSettings();
    nSession = Object.assign({}, s, {
      key: settingsKey(s),
      seed: nSession.seed,
      refBlob: nSession.refBlob,
    });
  }
}

// ── 合成（1本ずつ） ──
async function synthesize(index, session) {
  if (nBuffer[index]) return true;
  // 開始時の世代を覚える。待っている間に設定が変わったら、この合成の結果は
  // もう古いので捨てる。
  const gen = nGeneration;
  if (nSynthBusy) {
    await new Promise(r => { const t = setInterval(() => { if (!nSynthBusy) { clearInterval(t); r(); } }, 100); });
    if (gen !== nGeneration) return false;
    if (nBuffer[index]) return true;
  }
  nSynthBusy = true;
  try {
    const fd = new FormData();
    // 末尾記号除去（幻発話防止。途中の……や♡は残す）
    const rawText = nChunks[index].text;
    let ttsText = rawText
      .replace(/…+(\s*」?\s*)$/, '$1')  // 末尾……（」の直前も含む）
      .replace(/。(」?)\s*$/, '$1')      // 末尾。
      .replace(/！+\s*$/, '');           // 末尾！（」なし）→ TTS の末尾カット防止
    // 「……」「！！」「。」のように記号だけの行は、除去すると空になって
    // サーバが 400 を返す。読み上げる中身が無いので、そのまま飛ばす
    // （…… は元々「間」なので無音で構わない）。
    if (!ttsText.trim()) return false;
    ttsText = applyDict(ttsText);   // 読み辞書で誤読を矯正（表示は元のまま）
    // 地の文なら先頭に 📖 を足す。辞書の後に置くのは、辞書の見出しが
    // 行頭に当たっている場合でも置換を邪魔しないようにするため。
    ttsText = withNarrationEmoji(ttsText, session.narrEmoji);
    fd.append('text', ttsText);
    fd.append('model_type', session.model);
    fd.append('num_steps', session.steps);
    fd.append('cfg_scale_text', session.cfgText);
    fd.append('cfg_scale_speaker', session.cfgSpk);
    if (session.durationScale) fd.append('duration_scale', session.durationScale);
    if (session.precision) fd.append('precision', session.precision);
    if (session.device) fd.append('device', session.device);
    fd.append('seed', session.seed);   // セッション固定seedで話者を一貫させる
    if (session.loraName) fd.append('lora_name', session.loraName);
    // The voice comes from the sample when there is one; a full adapter
    // carries its own and needs nothing. Self-reference is the fallback for
    // a style-only adapter with no sample: generate once, then reuse that
    // take so the speaker stops drifting between chunks.
    if (session.voicePath) {
      const r = await fetch('file://' + session.voicePath.replace(/\\/g, '/'));
      fd.append('ref_wav', await r.blob(), 'ref.wav');
    } else if (session.refBlob) {
      fd.append('ref_wav', session.refBlob, 'selfref.wav');
    }
    const res = await fetch(`${API_URL}/synthesize`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await res.text());
    const json = await res.json();
    const wav = await fetch(API_ORIGIN + json.results[0]);
    // 合成している間に設定が変わっていたら、これは古い設定の音。
    // 書き戻すと新しい設定で作り直されなくなるので捨てる。
    if (gen !== nGeneration) return false;
    nBuffer[index] = await wav.blob();
    // blob に読み込んだのでサーバ側ファイルは不要
    const fname = json.results[0].split('/').pop();
    fetch(`${API_URL}/outputs/${fname}`, { method: 'DELETE' }).catch(() => {});
    return true;
  } catch (e) {
    console.error('[narrate]', index, e);
    return false;
  } finally {
    nSynthBusy = false;
  }
}

// 再生中の行より先を、指定された本数だけ作り置きする。
// synthesize 自身が nSynthBusy で直列化するので、まとめて呼んでも
// GPU を奪い合うことはない。順に埋まっていく。
function prefetchAhead(index, session) {
  const ahead = nPrefetchCount();
  for (let i = 1; i <= ahead; i++) {
    const target = index + i;
    if (target >= nChunks.length) break;
    if (nBuffer[target]) continue;
    synthesize(target, session);
  }
}

function nPrefetchCount() {
  const n = prefR ? parseInt(prefR.value, 10) : 1;
  return Number.isFinite(n) ? Math.max(1, Math.min(10, n)) : 1;
}

async function playChunk(index) {
  if (index < 0 || index >= nChunks.length) { stop(); return; }
  const gen = nGeneration;   // 失敗時に進めてよいかの判定に使う
  nIndex = index; refreshChunkClasses(); setStatus('合成中...');
  const ok = await synthesize(index, nSession);
  // 合成待ちの間にスキップ/停止された場合、この古い呼び出しはここで抜ける
  if (!nPlaying || index !== nIndex) return;
  if (!ok) {
    // 1 行の失敗で朗読全体を止めない。以前はここで return しており、
    // 「……」だけの行（末尾記号を落とすと空になり 400）や、通信の瞬断、
    // OOM のいずれでもその場で止まっていた。飛ばして次へ進む。
    // 世代が変わっているとき（設定変更）は次の再生で作り直すので進めない。
    if (gen !== nGeneration) return;
    setStatus(`${index + 1} 行目を飛ばしました`);
    if (index + 1 < nChunks.length) {
      gapTimer = setTimeout(() => { if (nPlaying) playChunk(index + 1); }, 120);
    } else {
      stop();
    }
    return;
  }
  setStatus('再生中');
  // 先読み。3080 のように合成が遅いと1行ぶんでは間に合わず、
  // 行が変わるたびに無音の待ちが入る。何行先まで作るかを選べるようにした。
  prefetchAhead(index, nSession);
  const url = URL.createObjectURL(nBuffer[index]);
  nAudio = new Audio(url);
  nAudio.volume = volR.value / 100;
  // 再生速度はいじらない。速さは生成時の尺（duration_scale）で決める。
  // playbackRate での早回しはタイムストレッチが掛かって音が濁るうえ、
  // モデルが実際に話す速さは変わらない。
  nAudio.onended = () => {
    URL.revokeObjectURL(url);
    if (!nPlaying || index !== nIndex) return;
    const gap = gapMsFor(nChunks[index].text);
    gapTimer = setTimeout(() => { if (nPlaying) playChunk(nIndex + 1); }, gap);
  };
  nAudio.play();
  saveBookmark(index);   // 再生位置を自動保存（しおり）
  playBtn.classList.add('playing'); setPlayIcon(true);
}

async function play() {
  if (!nChunks.length) return;
  const s = currentSettings();
  if (!s.loraName && !s.voicePath) {
    setStatus('⚠ 生成タブでボイスか LoRA を選択してください');
    nSession = null;
    return;
  }
  // 話し方のみの LoRA は声を持たない。参照も自己参照も無いまま進めると
  // チャンクごとに話者が揺れるので、どちらかを用意してもらう。
  if (s.loraName && !s.loraHasVoice && !s.voicePath && !s.selfrefOn) {
    setStatus('⚠ この LoRA は話し方のみです。サンプルボイスを選ぶか自己参照を ON にしてください');
    nSession = null;
    return;
  }
  // 停止/一時停止のあと再生した時、設定が変わっていれば作り直す
  // （ライブ反映はしない。変わっていなければそのまま再開＝高速）
  if (!nSession || nSession.key !== settingsKey(s)) {
    nSession = Object.assign({}, s, {
      key: settingsKey(s),
      seed: Math.floor(Math.random() * 1000000000), // 全チャンク同一話者
      refBlob: null,
    });
    nBuffer = {};   // 旧設定で作った音声を破棄
    // 自己参照ブートストラップ: LoRAで種セリフを1回合成し、その声を参照に固定。
    // サンプルがあればそれが話者を決めるので不要。完パケLoRAも自前で持つ。
    if (s.loraName && s.selfrefOn && !s.voicePath && !s.loraHasVoice) {
      setStatus('自己参照を生成中…');
      try {
        nSession.refBlob = await buildSelfRef(nSession);
      } catch (e) {
        console.error('[narrate] selfref failed', e);
        setStatus('自己参照の生成に失敗（LoRAのみで続行）');
      }
    }
  }
  nPlaying = true;
  playChunk(nIndex);
}

// 種セリフをLoRAで1回合成し、その音声Blobを返す（自己参照）
async function buildSelfRef(session) {
  const seedEl = document.getElementById('narrate-selfref-seed');
  const seed = (seedEl && seedEl.value.trim()) || 'こんにちは。今日はとてもいい天気ね。';
  const fd = new FormData();
  fd.append('text', seed);
  fd.append('model_type', session.model);
  fd.append('num_steps', session.steps);
  fd.append('cfg_scale_text', session.cfgText);
  fd.append('cfg_scale_speaker', session.cfgSpk);
  if (session.durationScale) fd.append('duration_scale', session.durationScale);
  if (session.precision) fd.append('precision', session.precision);
  if (session.device) fd.append('device', session.device);
  fd.append('seed', session.seed);
  fd.append('lora_name', session.loraName);   // 参照は付けない（no_ref）= LoRA素の声
  const res = await fetch(`${API_URL}/synthesize`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(await res.text());
  const json = await res.json();
  const wav = await fetch(API_ORIGIN + json.results[0]);
  const blob = await wav.blob();
  const fname = json.results[0].split('/').pop();
  fetch(`${API_URL}/outputs/${fname}`, { method: 'DELETE' }).catch(() => {});
  return blob;
}
function pause() {
  nPlaying = false;
  if (nAudio) { nAudio.pause(); nAudio = null; }
  playBtn.classList.remove('playing'); setPlayIcon(false);
  if (gapTimer) { clearTimeout(gapTimer); gapTimer = null; }
  setStatus('一時停止');
}
function stop() {
  nPlaying = false; nSession = null; nBuffer = {};
  if (nAudio) { nAudio.pause(); nAudio = null; }
  playBtn.classList.remove('playing'); setPlayIcon(false);
  if (gapTimer) { clearTimeout(gapTimer); gapTimer = null; }
  nIndex = 0; refreshChunkClasses(); setStatus('停止中');
}
function jumpTo(i) {
  const was = nPlaying; pause(); nBuffer = {};
  nIndex = Math.max(0, Math.min(i, nChunks.length - 1));
  refreshChunkClasses(); if (was) play();
}
function setStatus(msg) { statusEl.textContent = msg; }

playBtn.addEventListener('click', () => { if (nPlaying) pause(); else play(); });
stopBtn.addEventListener('click', stop);
prevBtn.addEventListener('click', () => jumpTo(nIndex - 1));
nextBtn.addEventListener('click', () => jumpTo(nIndex + 1));

// 自己参照の種セリフ（localStorage保存・デフォルトは音素広めの会話文）
(function initSelfRefSeed() {
  const el = document.getElementById('narrate-selfref-seed');
  if (!el) return;
  const DEF = '「あらゆる現実を、すべて自分のほうへねじ曲げたのよ。ねえ、今日はとてもいい天気ね」';
  el.value = localStorage.getItem('narrate_selfref_seed') || DEF;
  el.addEventListener('input', () => localStorage.setItem('narrate_selfref_seed', el.value));
})();

function loadNarrateContent(filePath, content) {
  stop();
  nFilePath = filePath;
  nRawText = content;
  filePathInput.value = filePath;
  localStorage.setItem('narrate_last_path', filePath);
  nChunks = splitText(nRawText);
  renderText();
  setStatus(nChunks.length + ' 行 — 再生で開始');
  showResumeIfAny();   // しおりがあれば再開ボタン表示
}

openBtn.addEventListener('click', async () => {
  const result = await window.api.openTextFile();
  if (!result) return;
  loadNarrateContent(result.path, result.content);
});

filePathInput.addEventListener('keydown', async (e) => {
  if (e.key !== 'Enter') return;
  const p = filePathInput.value.trim();
  if (!p) return;
  const r = await window.api.readTextFile(p);
  if (r) loadNarrateContent(r.path, r.content);
  else setStatus('ファイルを開けませんでした: ' + p);
});

// 起動時に前回パスを復元
(function () {
  const saved = localStorage.getItem('narrate_last_path');
  if (saved) filePathInput.value = saved;
})();

// 他のタブ（青空文庫など）から保存ファイルを朗読で直接開くためのフック
window.openNarrateFile = async (filePath) => {
  const r = await window.api.readTextFile(filePath);
  if (r) loadNarrateContent(r.path, r.content);
};

// =====================================================================
// 読み辞書（localStorage: narrate_dict = { 表記: よみ }）
// =====================================================================
const dictWord  = $('narrate-dict-word');
const dictYomi  = $('narrate-dict-yomi');
const dictMemo  = $('narrate-dict-memo');
const dictAdd   = $('narrate-dict-add');
const dictBody  = $('narrate-dict-body');
const dictEmpty = $('narrate-dict-empty');
const dictCount = $('narrate-dict-count');
// loadDict/saveDict/applyDict/dictYomiOf/dictMemoOf はモジュール先頭で定義済み

function renderDict() {
  const d = loadDict();
  const keys = Object.keys(d);
  dictBody.innerHTML = '';
  dictEmpty.classList.toggle('hidden', keys.length > 0);
  dictCount.textContent = String(keys.length);
  keys.forEach(k => {
    const tr = document.createElement('tr');
    const td1 = document.createElement('td'); td1.className = 'd-w'; td1.textContent = k;
    const td2 = document.createElement('td'); td2.className = 'd-y'; td2.textContent = dictYomiOf(d[k]);
    const td3 = document.createElement('td'); td3.className = 'd-m'; td3.textContent = dictMemoOf(d[k]);
    const td4 = document.createElement('td'); td4.className = 'd-x';
    const del = document.createElement('button');
    del.className = 'tip';
    del.dataset.tip = '削除';
    del.setAttribute('aria-label', '削除');
    del.innerHTML = '<svg class="i i-sm"><use href="#ic-x"/></svg>';
    del.addEventListener('click', () => { const dd = loadDict(); delete dd[k]; saveDict(dd); renderDict(); });
    td4.appendChild(del);
    tr.append(td1, td2, td3, td4);
    dictBody.appendChild(tr);
  });
}

dictAdd.addEventListener('click', () => {
  const w = dictWord.value.trim(), y = dictYomi.value.trim(), m = (dictMemo ? dictMemo.value.trim() : '');
  if (!w || !y) return;
  const d = loadDict(); d[w] = { yomi: y, memo: m }; saveDict(d);
  dictWord.value = ''; dictYomi.value = ''; if (dictMemo) dictMemo.value = '';
  renderDict();
});
renderDict();

// =====================================================================
// しおり（localStorage: narrate_bookmarks = { ファイルパス: チャンク番号 }）
// =====================================================================
const bookmarkBtn = $('narrate-bookmark-btn');
const resumeBtn   = $('narrate-resume-btn');
const resumeText  = $('narrate-resume-text');

// しおりの鍵はファイルのパス。Windows は大文字小文字を区別しないので、
// 同じ本でも開いた経路で D:\ と d:\ に割れ、別の本として二重に記録される。
// 鍵を作るときも引くときも同じ形に正規化して、割れないようにする。
function bookmarkKey(p) {
  return String(p || '').replace(/\//g, '\\').toLowerCase();
}
function loadBookmarks()  { try { return JSON.parse(localStorage.getItem('narrate_bookmarks') || '{}'); } catch { return {}; } }
function saveBookmark(idx) {
  if (!nFilePath) return;
  const k = bookmarkKey(nFilePath);
  const b = loadBookmarks();
  // 正規化前の鍵で入っているものは、ここで畳んで 1 本にする。
  // 残したままだと読まれないゴミが本の数だけ溜まる。
  for (const key of Object.keys(b)) {
    if (key !== k && bookmarkKey(key) === k) delete b[key];
  }
  b[k] = idx;
  localStorage.setItem('narrate_bookmarks', JSON.stringify(b));
}
// 旧い鍵（正規化前）で入っているものを拾う。書き込み時に畳むので、
// 一度しおりを挟めば 1 本にまとまる。
function findBookmark(b, p) {
  const k = bookmarkKey(p);
  if (k in b) return b[k];
  for (const [key, v] of Object.entries(b)) {
    if (bookmarkKey(key) === k) return v;
  }
  return undefined;
}
function showResumeIfAny() {
  const b = loadBookmarks();
  const saved = findBookmark(b, nFilePath);
  if (typeof saved === 'number' && saved > 0 && saved < nChunks.length) {
    resumeText.innerHTML = `しおり: <b>${saved + 1}</b> 行目`;
    resumeBtn.classList.remove('hidden');
    resumeBtn.onclick = () => { jumpTo(saved); resumeBtn.classList.add('hidden'); };
  } else {
    resumeText.textContent = '';
    resumeBtn.classList.add('hidden');
  }
}

bookmarkBtn.addEventListener('click', () => {
  if (!nFilePath || !nChunks.length) { setStatus('ファイル未選択'); return; }
  saveBookmark(nIndex);
  setStatus(`しおりを挟みました（${nIndex + 1} / ${nChunks.length}）`);
  showResumeIfAny();
});

})();

// =====================================================================
// 青空文庫タブ — カタログCSVをローカルキャッシュしてキーワード検索→保存
// =====================================================================
(function () {
  const $ = (id) => document.getElementById(id);
  const titleInput  = $('aoz-title');
  const authorInput = $('aoz-author');
  const searchBtn   = $('aoz-search-btn');
  const urlInput    = $('aoz-url');
  const urlBtn      = $('aoz-url-btn');
  const resultsEl   = $('aoz-results');
  const detailEl    = $('aoz-detail');
  const statusEl    = $('aoz-status');

  if (!searchBtn) return;

  let busy = false;
  let savedNovels = new Set();

  async function refreshSaved() {
    const list = await window.api.getSavedNovels().catch(() => []);
    savedNovels = new Set(list);
  }

  function sanitizeName(author, title) {
    return String([author, title].filter(Boolean).join('_') || 'aozora')
      .replace(/[\\/:*?"<>|]/g, '_').replace(/\s+/g, ' ').trim().slice(0, 120);
  }
  function isSaved(author, title) { return savedNovels.has(sanitizeName(author, title)); }

  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const setStatus = (m) => { if (statusEl) statusEl.textContent = m || ''; };

  // ── カタログ確認・更新 ──
  async function ensureCatalog() {
    const st = await window.api.aozoraCatalogStatus();
    if (st.ready) return true;
    setStatus('カタログを初回ダウンロード中… (約1〜2MB、30秒ほどかかります)');
    const r = await window.api.aozoraUpdateCatalog();
    setStatus(`カタログ準備完了（${r.count.toLocaleString()}作品）`);
    return true;
  }

  // ── 検索 ──
  async function doSearch() {
    const title  = titleInput.value.trim();
    const author = authorInput.value.trim();
    if (!title && !author || busy) return;
    busy = true;
    setStatus('検索中…');
    resultsEl.innerHTML = '';
    detailEl.innerHTML = '';
    try {
      await refreshSaved();
      await ensureCatalog();
      const { error, items } = await window.api.aozoraSearch({ title, author });
      if (error) { setStatus('エラー: ' + error); return; }
      setStatus(items.length ? `${items.length}件` : '見つかりませんでした');
      renderResults(items);
    } catch (e) {
      setStatus('検索失敗: ' + e.message);
    } finally { busy = false; }
  }

  function renderResults(items) {
    resultsEl.innerHTML = '';
    const sum = $('aoz-results-sum');
    if (!items.length) {
      resultsEl.innerHTML = '<p class="hits-empty note">該当する作品がありません。</p>';
      if (sum) sum.textContent = '';
      return;
    }
    if (sum) sum.innerHTML = `<b>${items.length}</b> 件`;
    items.forEach((it) => {
      // 青空文庫はあらすじが無いので 1 行に収める（題名と著者だけ）
      const div = document.createElement('div');
      div.className = 'hit one';
      const badge = isSaved(it.author, it.title) ? '<span class="got">保存済み</span>' : '';
      div.innerHTML =
        `<span class="t aoz-result-title">${esc(it.title)}</span>` +
        `<span class="a">${esc(it.author)}</span>${badge}`;
      div.addEventListener('click', () => {
        resultsEl.querySelectorAll('.hit.on').forEach(e => e.classList.remove('on'));
        div.classList.add('on');
        renderDetail(it);
      });
      resultsEl.appendChild(div);
    });
  }

  // カタログに txtUrl が入っているので card ページ取得不要
  function renderDetail(it) {
    detailEl.innerHTML =
      `<div class="pick">` +
        `<div class="pick-t">${esc(it.title)}</div>` +
        `<div class="pick-a">${esc(it.author)}</div>` +
        `<div class="pick-act">` +
          `<button id="aoz-dl-btn" class="btn btn-primary"><svg class="i i-sm"><use href="#ic-dl"/></svg>テキストを保存</button>` +
        `</div>` +
        `<div id="aoz-saved" class="saved hidden"></div>` +
      `</div>`;
    $('aoz-dl-btn').addEventListener('click', () => doDownload(it));
  }

  async function doDownload(it) {
    if (busy) return;
    busy = true;
    const btn = $('aoz-dl-btn');
    if (btn) btn.disabled = true;
    setStatus('ダウンロード中…');
    try {
      const res = await window.api.aozoraDownload({
        txtUrl: it.txtUrl, title: it.title, author: it.author, encoding: it.encoding
      });
      const savedEl = $('aoz-saved');
      if (savedEl) {
        savedEl.classList.remove('hidden');
        savedEl.innerHTML =
          `<span class="ok"><svg class="i i-sm"><use href="#ic-save"/></svg></span>` +
          `保存しました <code>${esc(res.path)}</code>` +
          `<span class="sp"></span>` +
          `<button id="aoz-to-narrate" class="btn btn-ghost btn-sm"><svg class="i i-sm"><use href="#ic-book"/></svg>朗読で開く</button>`;
        $('aoz-to-narrate').addEventListener('click', () => {
          document.querySelector('.tab-btn[data-tab="narrate"]')?.click();
          if (window.openNarrateFile) window.openNarrateFile(res.path);
        });
      }
      await refreshSaved();
      // 題名・著者・印はそれぞれ別の要素。題名と著者だけを見て判定する。
      resultsEl.querySelectorAll('.hit').forEach(el => {
        const titleEl = el.querySelector('.t');
        const authorEl = el.querySelector('.a');
        if (!titleEl || !authorEl) return;
        if (isSaved(authorEl.textContent.trim(), titleEl.textContent.trim())
            && !el.querySelector('.got')) {
          el.insertAdjacentHTML('beforeend', '<span class="got">保存済み</span>');
        }
      });
      setStatus(`保存完了: ${res.name}`);
    } catch (e) {
      setStatus('保存失敗: ' + e.message);
    } finally {
      if (btn) btn.disabled = false;
      busy = false;
    }
  }

  // ── 図書カードURLから直接開く ──
  async function openByUrl() {
    const url = urlInput.value.trim();
    if (!url || busy) return;
    busy = true;
    setStatus('作品を探しています…');
    resultsEl.innerHTML = '';
    detailEl.innerHTML = '';
    try {
      await refreshSaved();
      await ensureCatalog();
      const { error, item } = await window.api.aozoraByUrl(url);
      if (error === 'BAD_URL') {
        setStatus('図書カードのURLとして読み取れません（例: https://www.aozora.gr.jp/cards/000148/card789.html）');
        return;
      }
      if (error === 'CATALOG_MISSING') { setStatus('カタログが未取得です。先に検索してください。'); return; }
      if (error === 'NOT_FOUND') {
        setStatus('カタログに見つかりません。著作権が存続している作品、またはテキストファイルが無い作品の可能性があります。');
        return;
      }
      if (error) { setStatus('エラー: ' + error); return; }
      setStatus(`${item.title}（${item.author}）`);
      renderResults([item]);
      renderDetail(item);
    } catch (e) {
      setStatus('取得失敗: ' + e.message);
    } finally { busy = false; }
  }

  searchBtn.addEventListener('click', doSearch);
  [titleInput, authorInput].forEach(el => el.addEventListener('keydown', (e) => { if (e.key === 'Enter') doSearch(); }));
  if (urlBtn)   urlBtn.addEventListener('click', openByUrl);
  if (urlInput) urlInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') openByUrl(); });
  window._aozRefreshSaved = () => refreshSaved().catch(() => {});
})();

// =====================================================================
// LoRAマージタブ
// =====================================================================
(function () {
  const $ = (id) => document.getElementById(id);
  const slotsEl   = $('loramix-slots');
  const barEl     = $('loramix-bar');
  const keyEl     = $('loramix-key');
  const addBtn    = $('loramix-add-btn');
  const methodSel = $('loramix-method');
  const nameInput = $('loramix-name');
  const notesInput= $('loramix-notes');
  const runBtn    = $('loramix-run-btn');
  const statusEl  = $('loramix-status');

  if (!slotsEl) return;

  const MAX_SLOTS = 3;
  let slots = [];       // [{id, loraName, weight}]
  let loraList = [];    // [{name, base, notes}]

  function setStatus(msg, err) {
    statusEl.textContent = msg;
    statusEl.style.color = err ? '#f87171' : '';
  }

  async function loadLoraList() {
    try {
      const res = await fetch(`${API_URL}/loras`);
      const json = await res.json();
      loraList = json.loras || [];
    } catch (e) {
      loraList = [];
    }
    renderSlots();
  }

  // 取り合いを 1 本の帯で見せる。数字を追わなくても、ひとつ動かすと
  // 他が押し出される様子が分かる。色は .mixbar i:nth-child が持つ。
  function renderMix() {
    if (!barEl) return;
    barEl.innerHTML = '';
    keyEl.innerHTML = '';
    slots.forEach(s => {
      const i = document.createElement('i');
      i.style.width = `${s.weight}%`;
      barEl.appendChild(i);

      const span = document.createElement('span');
      span.appendChild(document.createElement('i'));
      span.appendChild(document.createTextNode(`${s.loraName || '未選択'} ${s.weight}%`));
      keyEl.appendChild(span);
    });
  }

  function buildSlotEl(slot, index) {
    const div = document.createElement('div');
    div.className = 'slot loramix-slot';
    div.dataset.id = slot.id;

    const no = document.createElement('span');
    no.className = 'no';
    no.textContent = String(index + 1);

    const pick = document.createElement('span');
    pick.className = 'pick';
    const sel = document.createElement('select');
    sel.className = 'loramix-lora-sel';
    sel.innerHTML = '<option value="">-- LoRAを選択 --</option>' +
      loraList.map(l => `<option value="${l.name}"${l.name === slot.loraName ? ' selected' : ''}>${l.name} (${l.base})</option>`).join('');
    sel.addEventListener('change', () => { slot.loraName = sel.value; updateMethodOption(); renderMix(); });
    pick.appendChild(sel);

    // 比率はスライダーと数字を縦に積む。横に置くと数字が動いて読みにくい
    const ratio = document.createElement('span');
    ratio.className = 'ratio';
    const ratioT = document.createElement('span');
    ratioT.className = 'ratio-t';
    const weightLabel = document.createElement('b');
    weightLabel.className = 'loramix-weight-label';
    weightLabel.textContent = `${Math.round(slot.weight)}%`;
    ratioT.appendChild(document.createTextNode('比率'));
    ratioT.appendChild(weightLabel);

    const maxVal = 100 - (slots.length - 1);  // 他スロットが最低1%ずつ確保できる上限
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = 1; slider.max = maxVal; slider.value = Math.min(slot.weight, maxVal);
    slider.className = 'loramix-slider';
    slider.addEventListener('input', () => {
      slot.weight = parseInt(slider.value);
      weightLabel.textContent = `${slot.weight}%`;
      // 他スロットを比例配分
      const others = slots.filter(s => s.id !== slot.id);
      const remaining = 100 - slot.weight;
      const othersTotal = others.reduce((s, o) => s + o.weight, 0);
      let assigned = 0;
      others.forEach((o, i) => {
        const w = i === others.length - 1
          ? remaining - assigned
          : Math.max(1, Math.round((o.weight / (othersTotal || 1)) * remaining));
        o.weight = Math.max(1, w);
        assigned += o.weight;
      });
      // 他スロットの表示を更新（DOMを再作成せず値だけ更新）
      slotsEl.querySelectorAll('.loramix-slot').forEach(el => {
        const s = slots.find(x => String(x.id) === el.dataset.id);
        if (!s || s.id === slot.id) return;
        const sl = el.querySelector('.loramix-slider');
        const lb = el.querySelector('.loramix-weight-label');
        if (sl) sl.value = s.weight;
        if (lb) lb.textContent = `${s.weight}%`;
      });
      renderMix();
    });
    ratio.append(ratioT, slider);

    const delBtn = document.createElement('button');
    delBtn.className = 'del tip loramix-del';
    delBtn.dataset.tip = '外す';
    delBtn.setAttribute('aria-label', '外す');
    delBtn.innerHTML = '<svg class="i i-sm"><use href="#ic-x"/></svg>';
    delBtn.addEventListener('click', () => {
      slots = slots.filter(s => s.id !== slot.id);
      renderSlots();
    });

    div.append(no, pick, ratio, delBtn);
    return div;
  }

  function renderSlots() {
    slotsEl.innerHTML = '';
    slots.forEach((s, i) => slotsEl.appendChild(buildSlotEl(s, i)));
    addBtn.disabled = slots.length >= MAX_SLOTS;
    updateMethodOption();
    renderMix();
  }

  function updateMethodOption() {
    const slerp = methodSel.querySelector('option[value="slerp"]');
    if (slerp) slerp.disabled = slots.length !== 2;
    if (slots.length !== 2 && methodSel.value === 'slerp') methodSel.value = 'linear';
  }

  let slotId = 0;
  function addSlot() {
    if (slots.length >= MAX_SLOTS) return;
    slots.push({ id: ++slotId, loraName: '', weight: Math.round(100 / (slots.length + 1)) });
    // 既存スロットの比率を均等に
    const eq = Math.round(100 / slots.length);
    slots.forEach((s, i) => { s.weight = i < slots.length - 1 ? eq : 100 - eq * (slots.length - 1); });
    renderSlots();
  }

  addBtn.addEventListener('click', addSlot);

  runBtn.addEventListener('click', async () => {
    const newName = nameInput.value.trim();
    if (!newName) { setStatus('新しいLoRA名を入力してください', true); return; }
    const sources = slots.map(s => ({ name: s.loraName, weight: s.weight }));
    if (sources.some(s => !s.name)) { setStatus('すべてのスロットでLoRAを選択してください', true); return; }
    if (sources.length < 2) { setStatus('LoRAを2つ以上選択してください', true); return; }

    runBtn.disabled = true;
    setStatus('マージ中…');
    try {
      const res = await fetch(`${API_URL}/loras/merge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newName,
          sources,
          method: methodSel.value,
          notes: notesInput.value.trim(),
        }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || json.error || `status ${res.status}`);
      setStatus(`✅ 完了: ${json.name}（${json.keys_merged}レイヤーをマージ）`);
      nameInput.value = '';
      notesInput.value = '';
      await loadLoraList();
      loadLoras(); // 生成タブのドロップダウンも更新
    } catch (e) {
      setStatus('失敗: ' + e.message, true);
    } finally {
      runBtn.disabled = false;
    }
  });

  // タブ切替時に LoRA リストを更新
  window._loramixRefresh = loadLoraList;

  // 初期スロット2つ
  addSlot(); addSlot();
})();