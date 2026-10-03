// 初回起動でやること。setup.bat がやっていた3工程をアプリの中に移したもの。
//
//   1. ランタイムを作る   同梱の埋め込み Python を data\runtime\python へ写して pip を入れる
//   2. ライブラリを入れる GPU を見て cu128 / cu126 / rocm / cpu を選び、requirements を入れる
//   3. モデルを取得       取得するモデルを選んでもらい（model_catalog.py）、fetch_models.py で取る
//
// 進捗は onProgress で返す。黒い画面に流れるだけだと、止まっているのか
// 進んでいるのか分からないため。
//
// venv は使わない。埋め込み版には venv も ensurepip も入っていないので、
// 一式を丸ごと写して、そこへ直接入れる。アプリ専用なので隔離としては同じで、
// アプリを入れ替えてもランタイムとモデルは data 側に残る。
//
// 工程ごとに印（マーカー）を置き、その印だけで「済み」を判断する。
// 生成物の有無で判断すると、途中で落ちた中途半端な状態を済みと誤判定する。
// 実際、models\hub は最初の1バイトを落とす前に作られるので、30GB のうち
// 数百MBで中断しても「取得済み」になっていた。

const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn, execFileSync } = require('child_process');

// 済み判定の版。ここを上げると、更新後に依存を入れ直させられる。
const SETUP_SCHEMA = 2;

// 同梱した Python の置き場。パッケージ版は resources の下、開発中は APP/build。
function bundledPython(isPackaged) {
  return isPackaged
    ? path.join(process.resourcesPath, 'python-embed')
    : path.join(__dirname, 'build', 'python-embed');
}

function runtimeDir(dataRoot) { return path.join(dataRoot, 'runtime', 'python'); }
function runtimePython(dataRoot) { return path.join(runtimeDir(dataRoot), 'python.exe'); }
function modelsDir(dataRoot) { return path.join(dataRoot, 'models'); }

// 工程ごとの印。何で済ませたかを中に書き、次の起動で照合する。
function runtimeMarker(dataRoot) { return path.join(runtimeDir(dataRoot), '.runtime-ready'); }
function depsMarker(dataRoot) { return path.join(runtimeDir(dataRoot), '.deps-ready'); }
function modelsMarker(dataRoot) { return path.join(modelsDir(dataRoot), '.models-ready'); }

function readMarker(file) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')); } catch (e) { return null; }
}

function writeMarker(file, data) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  // 書き切ってから置き換える。途中で落ちた印を残さない。
  const tmp = `${file}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify({ ...data, at: new Date().toISOString() }, null, 1));
  fs.renameSync(tmp, file);
}

// 2枚ある。上流のモデル側（root）と、この画面が使う FastAPI 側（APP）。
function requirementFiles(backendDir) {
  return [path.join(backendDir, '..', 'requirements.txt'),
          path.join(backendDir, 'requirements.txt')];
}

// 同梱した Python と pip の版。ビルド時に fetch-python.js が控えを残す。
// ここを ABI タグ（python312）で済ませると、3.12.10 から 3.12.11 へ上げても
// 値が変わらず、古いランタイムを使い続ける。
function embedTag(dir) {
  try {
    const m = JSON.parse(fs.readFileSync(path.join(dir, '.embed.json'), 'utf8'));
    return `python-${m.python}+pip-${m.pip}`;
  } catch (e) { /* 控えの無い古い同梱物 */ }
  try {
    const pth = fs.readdirSync(dir).find((f) => /^python\d+\._pth$/.test(f));
    if (pth) return pth.replace('._pth', '');
  } catch (e) { /* 同梱が見つからない */ }
  return 'python-unknown';
}

function fileBytes(file) {
  return fs.existsSync(file) ? fs.readFileSync(file) : Buffer.from('missing');
}

// requirements の中身と同梱 Python の版から指紋を作る。
// アプリを更新してライブラリの一覧や Python の版が変わったのに、
// 古い依存のまま起動する事故を防ぐ。
function depsFingerprint({ backendDir, isPackaged, backend }) {
  const h = crypto.createHash('sha256');
  h.update(`schema=${SETUP_SCHEMA}\n`);
  for (const req of requirementFiles(backendDir)) h.update(fileBytes(req));
  h.update(embedTag(bundledPython(isPackaged)));
  // 導入手順そのもの（wheel の URL・固定する版・requirements から外す行）も
  // 入れる。ここを入れないと、手順だけ直したときに既存の利用者へ届かない。
  h.update(backend ? JSON.stringify(torchPlan(backend)) : 'backend-unknown');
  return h.digest('hex').slice(0, 16);
}

// モデル側は取得スクリプトの中身で見る。取得対象を足したのに
// 「取得済み」のまま起動されると、足したモデルだけ永久に来ない。
function modelsFingerprint({ backendDir }) {
  const h = crypto.createHash('sha256');
  h.update(`schema=${SETUP_SCHEMA}\n`);
  h.update(fileBytes(path.join(backendDir, 'fetch_models.py')));
  return h.digest('hex').slice(0, 16);
}

function state({ dataRoot, backendDir, isPackaged }) {
  const wantModels = modelsFingerprint({ backendDir });
  const wantEmbed = embedTag(bundledPython(isPackaged));
  const rt = readMarker(runtimeMarker(dataRoot));
  const deps = readMarker(depsMarker(dataRoot));
  const models = readMarker(modelsMarker(dataRoot));
  // GPU を載せ替えた・外した・IRODORI_BACKEND を変えたときは入れ直す。
  // 判定は実測で 60〜90ms（NVIDIA 機）。GPU が無い機では PowerShell を
  // 待つぶん1〜2秒かかるが、合わない torch で起動するほうが高くつく。
  // 印が無いときは、どうせ入れ直すので見に行かない。
  const backend = deps ? detectBackend() : null;
  const wantDeps = depsFingerprint({ backendDir, isPackaged, backend });
  return {
    wantDeps,
    wantModels,
    wantEmbed,
    hasRuntime: !!rt && rt.schema === SETUP_SCHEMA && rt.embed === wantEmbed
      && fs.existsSync(runtimePython(dataRoot)),
    hasDeps: !!deps && deps.fingerprint === wantDeps && deps.backend === backend,
    hasModels: !!models && models.fingerprint === wantModels,
  };
}

function isReady(opts) {
  const s = state(opts);
  return s.hasRuntime && s.hasDeps && s.hasModels;
}

// GPU を見てどの PyTorch を入れるか決める。setup.bat と同じ判断。
//   NVIDIA → cu128（compute capability 7 未満なら cu126）
//   Radeon → rocm
//   どちらも無ければ cpu
//
// 毎回の起動では呼ばない。nvidia-smi と PowerShell を待つので数秒かかる。
// 済み判定には使わず、入れ直しが必要になったときだけ見る。
function detectBackend() {
  if (process.env.IRODORI_BACKEND) return process.env.IRODORI_BACKEND;
  try {
    const caps = execFileSync('nvidia-smi',
      ['--query-gpu=compute_cap', '--format=csv,noheader'],
      { encoding: 'utf8', timeout: 20000 }).trim();
    if (caps) {
      const major = Math.min(...caps.split('\n').map((c) => parseInt(c, 10) || 0));
      return major < 7 ? 'cu126' : 'cu128';
    }
  } catch (e) { /* NVIDIA が無いだけ */ }
  try {
    // wmic は Windows 11 の新しいビルドで消えたので PowerShell で見る
    const names = execFileSync('powershell',
      ['-NoProfile', '-Command', '(Get-CimInstance Win32_VideoController).Name'],
      { encoding: 'utf8', timeout: 30000 });
    // AMD の Windows 版 ROCm 7.2.1 は Windows 11 のみが対象。
    // 10 で選ぶと 1GB 以上落としたうえで動かない。
    // なお対応 GPU の一覧までは見ていない。古い Radeon では入れたあとに
    // 失敗しうる。その場合は IRODORI_BACKEND=cpu で逃がせる。
    // https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html
    if (/radeon|\bamd\b/i.test(names) && isWindows11()) return 'rocm';
  } catch (e) { /* 取れなければ cpu 扱い */ }
  return 'cpu';
}

// Windows 11 は 10.0.22000 以降。os.release() が "10.0.26100" のように返す。
function isWindows11() {
  const m = /^10\.0\.(\d+)/.exec(os.release());
  return !!m && Number(m[1]) >= 22000;
}

// ROCm の wheel はリリースの直下にある。torch/ のような下位ディレクトリは
// 無い（そう書いていて 404 になっていた）。
const ROCM_BASE = 'https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1';
const ROCM_VERSION = '7.2.1';
const ROCM_TORCH = '2.9.1+rocm7.2.1';
// AMD の手順は2段構え。先に ROCm SDK を入れ、そのあとに torch を入れる。
// SDK を抜かすと、torch は入っても DLL が無くて import で落ちる。
// 手順のコードブロックは4つ並んでいる。3つの wheel と、まとめ役の tar.gz。
// https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/windows/install-pytorch.html
//
// 手順にある torchvision は入れない。このアプリは使っておらず、
// requirements にも無いため（意図的にここだけ外している）。
const ROCM_SDK = [
  ...['rocm_sdk_core', 'rocm_sdk_devel', 'rocm_sdk_libraries_custom']
    .map((n) => `${ROCM_BASE}/${n}-${ROCM_VERSION}-py3-none-win_amd64.whl`),
  `${ROCM_BASE}/rocm-${ROCM_VERSION}.tar.gz`,
];

// バックエンドごとの PyTorch の入れ方と、入れた版の固定。
function torchPlan(backend) {
  if (backend === 'rocm') {
    const whl = (name) =>
      `${ROCM_BASE}/${name}-${ROCM_TORCH.replace('+', '%2B')}-cp312-cp312-win_amd64.whl`;
    return {
      // AMD の手順どおり、SDK → torch の順に分けて入れる。
      // --no-cache-dir も手順にある（同名 wheel の取り違えを避けるため）。
      pre: [[...ROCM_SDK, '--no-cache-dir']],
      args: [whl('torch'), whl('torchaudio'), '--no-cache-dir'],
      pins: [`torch==${ROCM_TORCH}`, `torchaudio==${ROCM_TORCH}`],
      // AMD が出しているのは 2.9.1 まで。requirements は torch>=2.10 を
      // 要求するので、そのまま続けると入れたばかりの ROCm 版が PyPI の
      // 通常版 2.10 に差し替えられ、GPU 対応を失う。ROCm のときだけ
      // torch 系を requirements から外す。
      // torchcodec は torch の ABI に合わせた作りで ROCm 版が無いが、
      // 読み書きは soundfile へ落ちる（irodori_tts/codec.py）。
      // Triton は CUDA 専用（学習の高速化に使う）。Radeon では使えないので入れない。
      drop: ['torch', 'torchaudio', 'torchcodec', 'triton-windows', 'torchao'],
    };
  }
  const version = backend === 'cpu' ? '2.10.0' : `2.10.0+${backend}`;
  const args = [`torch==${version}`, `torchaudio==${version}`];
  if (backend !== 'cpu') args.push('--index-url', `https://download.pytorch.org/whl/${backend}`);
  return {
    pre: [],
    args,
    pins: [`torch==${version}`, `torchaudio==${version}`],
    // cu128 / cu126 / cpu は requirements の torch>=2.10.0 を満たすので外さない。
    // 版は constraints で固定し、別の依存が引き上げるのを止める。
    // Triton は CUDA 専用（学習の高速化に使う）。CPU では使えないので入れない。
    // torchao は量子化版 v4-Large 用。量子化版は NVIDIA の GPU でしか動かない。
    drop: backend === 'cpu' ? ['triton-windows', 'torchao'] : [],
  };
}

// requirements から指定の行を外した写しを作る。元のファイルは触らない。
function filteredRequirements(src, drop, outDir, label) {
  if (!drop.length) return src;
  const text = fs.readFileSync(src, 'utf8');
  const nameOf = (line) => (line.split(/[\s=<>!~;[@]/)[0] || '').trim().toLowerCase();
  const out = text.split(/\r?\n/).map((line) => {
    const bare = line.trim();
    if (!bare || bare.startsWith('#')) return line;
    return drop.includes(nameOf(bare)) ? `# (このバックエンドでは別に入れる) ${line}` : line;
  }).join('\n');
  fs.mkdirSync(outDir, { recursive: true });
  const dest = path.join(outDir, `requirements-${label}.txt`);
  fs.writeFileSync(dest, out);
  return dest;
}

// 埋め込み版でも %APPDATA%\Roaming\Python\Python312\site-packages が
// 勝手に sys.path へ入る（._pth では止められない）。利用者の環境に入っている
// 別バージョンのライブラリを拾うと依存解決が崩れるので、環境変数で切る。
// PYTHONPATH / PYTHONHOME も同じ理由で外す。
function pyEnv(extra) {
  const env = { ...process.env, ...extra, PYTHONNOUSERSITE: '1' };
  delete env.PYTHONPATH;
  delete env.PYTHONHOME;
  return env;
}

// pip は利用者の pip.ini と PIP_* も読む。user=true が書かれていれば
// アプリ専用のランタイムではなく利用者の site-packages へ入れてしまう。
// --isolated はその両方を無視させる。
function pipInstall(...rest) {
  return ['-m', 'pip', 'install', '--isolated', '--no-warn-script-location', ...rest];
}

// 埋め込み版は python3xx._pth で sys.path を固定する。その副作用で、通常の
// Python なら必ず先頭に入る「実行したスクリプト自身のディレクトリ」が入らない。
// そのため server.py が server_lora を、train.py が irodori_tts を読めずに落ちる。
// PYTHONPATH は ._pth があると無視されるので、site が読む sitecustomize.py で戻す。
const PATH_FIX = `# Restore the entry script's own directory on sys.path.
# The bundled runtime pins sys.path through python3xx._pth, which drops the
# directory CPython normally puts first. Sibling imports (server_lora,
# irodori_tts) fail without it. PYTHONPATH is ignored while a ._pth exists.
import os
import sys


def _entry_dir():
    script = sys.argv[0] if sys.argv else ""
    # For -c / -m, argv[0] is a flag or a module file inside the runtime.
    # CPython adds the cwd in those cases and site already handles it.
    if not script or script in ("-c", "-"):
        return None
    if not os.path.isfile(script):
        return None
    return os.path.dirname(os.path.abspath(script))


def _inside(child, parent):
    # Compare whole path segments. A plain startswith() would treat
    # "C:\\\\app\\\\python-tools" as living inside "C:\\\\app\\\\python".
    try:
        child = os.path.abspath(child)
        parent = os.path.abspath(parent)
        return os.path.commonpath([child, parent]) == parent
    except ValueError:  # different drives
        return False


try:
    _here = _entry_dir()
    _runtime = os.path.dirname(os.path.abspath(sys.executable))
    if _here and not _inside(_here, _runtime) and _here not in sys.path:
        sys.path.insert(0, _here)
except Exception:  # never let path repair break the interpreter
    pass
`;

// sitecustomize.py を置く（無ければ作る／違っていれば書き直す）。
// 何度呼んでも同じ結果になるので、起動のたびに通してよい。
function ensurePathFix(dataRoot) {
  const dest = path.join(runtimeDir(dataRoot), 'Lib', 'site-packages', 'sitecustomize.py');
  if (!fs.existsSync(path.dirname(dest))) return false;
  if (fs.existsSync(dest) && fs.readFileSync(dest, 'utf8') === PATH_FIX) return false;
  fs.writeFileSync(dest, PATH_FIX);
  return true;
}

// セットアップで起こした子。画面を閉じたときに置き去りにしない。
const children = new Set();

function stopAll() {
  for (const child of children) {
    try {
      // pip は自分の子を起こす。直接の子だけ止めても孫が残るので、
      // Windows では木ごと落とす（サーバの停止と同じやり方）。
      if (process.platform === 'win32') {
        spawn('taskkill', ['/pid', String(child.pid), '/f', '/t']);
      } else {
        child.kill();
      }
    } catch (e) { /* すでに終わっている */ }
  }
  children.clear();
}

function run(exe, args, opts, onLine) {
  return new Promise((resolve, reject) => {
    const p = spawn(exe, args, { ...opts, shell: false });
    children.add(p);
    const feed = (buf) => {
      for (const line of buf.toString().split(/\r?\n/)) {
        if (line.trim()) onLine(line.trim());
      }
    };
    p.stdout.on('data', feed);
    p.stderr.on('data', feed);
    p.on('error', (err) => { children.delete(p); reject(err); });
    p.on('close', (code) => {
      children.delete(p);
      if (code === 0) resolve();
      else reject(new Error(`${path.basename(exe)} ${args[0] || ''} が終了コード ${code} で終わりました`));
    });
  });
}

// 学習の高速化（Triton）が使う C の開発用ファイル。1.2.2 より前に入れたランタイムには無いので、
// 起動のたびに見て、足りなければ同梱の Python から写す。ランタイム全体は作り直さない
// （作り直すとライブラリの入れ直しで数分〜十数分かかる）。
function ensurePythonDevFiles({ dataRoot, isPackaged }) {
  const src = bundledPython(isPackaged);
  const dest = runtimeDir(dataRoot);
  if (!fs.existsSync(path.join(dest, 'python.exe'))) return;
  for (const name of ['include', 'libs']) {
    if (fs.existsSync(path.join(src, name)) && !fs.existsSync(path.join(dest, name))) {
      copyDir(path.join(src, name), path.join(dest, name));
    }
  }
}

function copyDir(src, dest) {
  fs.mkdirSync(dest, { recursive: true });
  for (const ent of fs.readdirSync(src, { withFileTypes: true })) {
    const s = path.join(src, ent.name);
    const d = path.join(dest, ent.name);
    if (ent.isDirectory()) copyDir(s, d);
    else fs.copyFileSync(s, d);
  }
}

/**
 * 初回セットアップを通す。すでに済んでいる工程は飛ばす。
 * @param {object} o
 * @param {string} o.dataRoot   書き込み先（<install>\data）
 * @param {string} o.backendDir Python のソース置き場
 * @param {boolean} o.isPackaged
 * @param {(p:{step:string,index:number,total:number,detail?:string,final?:boolean})=>void} o.onProgress
 */
async function ensureRuntime({ dataRoot, backendDir, isPackaged, onProgress, chooseModels }) {
  const total = 3;
  const s = state({ dataRoot, backendDir, isPackaged });
  const py = runtimePython(dataRoot);
  const log = (step, index) => (line) => onProgress({ step, index, total, detail: line });
  // ランタイムを作り直すと、その中にある .deps-ready もライブラリも消える。
  // 作り直したときは、判定が済みでも依存を入れ直す。
  let rebuilt = false;

  if (!s.hasRuntime) {
    onProgress({ step: 'ランタイムを用意しています', index: 1, total });
    const src = bundledPython(isPackaged);
    if (!fs.existsSync(path.join(src, 'python.exe'))) {
      throw new Error(`同梱の Python が見つかりません: ${src}`);
    }
    // 印が無い＝前回が途中で落ちている。残骸の上に重ねず、作り直す。
    fs.rmSync(runtimeDir(dataRoot), { recursive: true, force: true });
    copyDir(src, runtimeDir(dataRoot));
    // 埋め込み版には pip が入っていない。同梱した wheel を site-packages へ
    // そのまま展開して据える。
    //   get-pip.py は使わない。bootstrap.pypa.io の中身が入れ替わり続けて
    //   ハッシュを固定できないため。
    //   wheel の中の pip に自分自身を入れさせる手も使えない。いまの pip は
    //   「pip を書き換えるときは -m pip で」と言って止まる。
    //   wheel は zip で、中身は site-packages に置く形そのままなので、
    //   標準ライブラリの zipfile で展開すれば入れたのと同じ状態になる。
    const whl = fs.readdirSync(runtimeDir(dataRoot)).find((f) => /^pip-.+\.whl$/.test(f));
    if (!whl) throw new Error(`pip の wheel が見つかりません: ${runtimeDir(dataRoot)}`);
    const sitePackages = path.join(runtimeDir(dataRoot), 'Lib', 'site-packages');
    await run(py, ['-c', 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])',
                   path.join(runtimeDir(dataRoot), whl), sitePackages],
      { cwd: runtimeDir(dataRoot), env: pyEnv() }, log('ランタイムを用意しています', 1));
    await run(py, ['-m', 'pip', '--version'],
      { env: pyEnv() }, log('ランタイムを用意しています', 1));
    // dacvae はソース配布なので、ビルドに setuptools が要る。
    // 埋め込み版には入っていないので先に入れる。
    await run(py, pipInstall('setuptools', 'wheel'),
      { env: pyEnv() }, log('ランタイムを用意しています', 1));
    ensurePathFix(dataRoot);
    writeMarker(runtimeMarker(dataRoot), { schema: SETUP_SCHEMA, embed: s.wantEmbed });
    rebuilt = true;
  } else {
    ensurePathFix(dataRoot);
  }

  if (!s.hasDeps || rebuilt) {
    const backend = detectBackend();
    const plan = torchPlan(backend);
    onProgress({ step: `PyTorch を入れています（${backend}）`, index: 2, total });
    for (const step of plan.pre) {
      await run(py, pipInstall(...step), { env: pyEnv() },
        log(`PyTorch を入れています（${backend}）`, 2));
    }
    await run(py, pipInstall(...plan.args), { env: pyEnv() },
      log(`PyTorch を入れています（${backend}）`, 2));

    // 入れた版を固定する。別の依存が引き上げて、バックエンド用に選んだ
    // ビルドを通常版へ差し替えるのを止める。
    const work = path.join(dataRoot, 'runtime');
    fs.mkdirSync(work, { recursive: true });
    const constraints = path.join(work, 'constraints.txt');
    fs.writeFileSync(constraints, `${plan.pins.join('\n')}\n`);

    onProgress({ step: '残りのライブラリを入れています', index: 2, total });
    const reqs = requirementFiles(backendDir);
    for (const req of reqs) {
      if (!fs.existsSync(req)) throw new Error(`requirements が見つかりません: ${req}`);
    }
    for (const [i, req] of reqs.entries()) {
      const use = filteredRequirements(req, plan.drop, work, `${backend}-${i}`);
      await run(py, pipInstall('-c', constraints, '-r', use),
        { env: pyEnv() }, log('残りのライブラリを入れています', 2));
    }
    writeMarker(depsMarker(dataRoot), {
      schema: SETUP_SCHEMA, backend, torch: plan.pins,
      // 判定時はバックエンド未確定のことがある。実際に入れた構成で作り直す。
      fingerprint: depsFingerprint({ backendDir, isPackaged, backend }),
    });
  }

  if (!s.hasModels) {
    const modelEnv = pyEnv({ HF_HOME: modelsDir(dataRoot), IRODORI_MODELS_DIR: modelsDir(dataRoot),
                             PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8', HF_HUB_OFFLINE: '1' });
    // 取得するモデルを選んでもらう。一覧（容量・VRAM・この GPU で動くか・取得済みか）は
    // model_catalog.py が出す。設定画面の表と同じ中身。
    onProgress({ step: '取得する音声モデルを選んでください', index: 3, total, detail: '' });
    let catalogLine = '';
    await run(py, [path.join(backendDir, 'model_catalog.py')], { cwd: backendDir, env: modelEnv },
      (line) => { if (line.startsWith('{')) catalogLine = line; });
    const ids = await chooseModels(JSON.parse(catalogLine));

    const step = '音声モデルを取得しています';
    onProgress({ step, index: 3, total, detail: '' });
    const fetchEnv = { ...modelEnv };
    delete fetchEnv.HF_HUB_OFFLINE;
    await run(py, [path.join(backendDir, 'fetch_models.py'), '--models', ids.join(',')],
      { cwd: backendDir, env: fetchEnv }, log(step, 3));
    // fetch_models.py が 0 で終わったときだけ印を置く。途中で落ちれば
    // 印が無いままなので、次の起動で選び直して続きから取りに行く。
    writeMarker(modelsMarker(dataRoot), { schema: SETUP_SCHEMA, fingerprint: s.wantModels, models: ids });
  }

  // final を立てて、工程名の上書きではなく「全部済み」の表示に使わせる。
  // これが無いと、飛ばした工程の行が完了の文言で塗り潰される。
  onProgress({ step: '準備ができました', index: total, total, detail: '', final: true });
  return py;
}

module.exports = {
  ensureRuntime, ensurePathFix, isReady, runtimePython, modelsDir,
  // torchPlan は「どの wheel をどの順で入れるか」を返すだけの関数。
  // 中身を見て確かめられるように出している。
  detectBackend, torchPlan, pyEnv, stopAll, ensurePythonDevFiles,
};
