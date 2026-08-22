const { app, BrowserWindow, ipcMain, dialog, nativeTheme, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const net = require('net');
const crypto = require('crypto');
const zlib = require('zlib');
const { TextDecoder } = require('util');

let mainWindow;
let pythonProcess;
let apiPort = null; // 実際に使うポート。起動時に空きを探して決める。

// 8080 から順に空いているポートを探す。
// 固定ポートだと、二重起動や無関係なサービスと衝突したときに起動できない。
// 他人のプロセスを落として席を空けるのではなく、こちらが空いている席に座る。
const DEFAULT_PORT = 8080;
const PORT_SCAN_LIMIT = 50; // 8080-8129 まで見る

function isPortFree(port) {
  return new Promise((resolve) => {
    const tester = net.createServer();
    tester.once('error', () => resolve(false));
    tester.once('listening', () => tester.close(() => resolve(true)));
    // サーバー本体と同じ 0.0.0.0 で試す（127.0.0.1 だけ空いていても意味がない）
    tester.listen(port, '0.0.0.0');
  });
}

async function findFreePort() {
  for (let i = 0; i < PORT_SCAN_LIMIT; i++) {
    const port = DEFAULT_PORT + i;
    if (await isPortFree(port)) return port;
  }
  throw new Error(
    `空きポートが見つかりません（${DEFAULT_PORT}〜${DEFAULT_PORT + PORT_SCAN_LIMIT - 1} をすべて確認しました）`
  );
}

// Writable data lives next to the launcher so users can find files easily.
//   Dev (`npm start`):  D:\Irodori-TTS\APP\        (i.e. __dirname)
//   Packaged exe:       <install_dir>\data\        (writable area next to .exe)
//
// Python's server reads IRODORI_DATA_DIR and uses the same convention.
// Legacy `APP/references` + `APP/metadata.json` get migrated to
// `<data_root>/voices/` automatically on first server start.
function getDataRoot() {
  return app.isPackaged
    ? path.join(path.dirname(app.getPath('exe')), 'data')
    : __dirname;
}
function getRefsDir()     { return path.join(getDataRoot(), 'voices'); }
function getMetadataPath(){ return path.join(getRefsDir(), 'metadata.json'); }
function getOutputsDir()  { return path.join(getDataRoot(), 'outputs'); }

function ensureDataLayout() {
  for (const d of [getRefsDir(), getOutputsDir(), getNovelsDir()]) {
    if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true });
  }
  const meta = getMetadataPath();
  if (!fs.existsSync(meta)) {
    fs.writeFileSync(meta, JSON.stringify({ voices: [] }, null, 2));
  }
}

function startPythonServer(port) {
  console.log('[Electron] Starting Python FastAPI server...');

  // Use the project's local .venv python directly (no uv needed).
  const venvPython = path.join(__dirname, '..', '.venv', 'Scripts', 'python.exe');
  const pythonExe = fs.existsSync(venvPython) ? venvPython : 'python';
  console.log(`[Electron] Python executable: ${pythonExe}`);

  // Writable data dir co-located with the launcher (portable layout).
  const dataDir = getDataRoot();
  console.log(`[Electron] IRODORI_DATA_DIR: ${dataDir}`);
  console.log(`[Electron] IRODORI_PORT: ${port}`);

  pythonProcess = spawn(pythonExe, ['server.py'], {
    cwd: __dirname,
    shell: false,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      IRODORI_DATA_DIR: dataDir,
      IRODORI_PORT: String(port),
      // モデルはアプリ配下（<app>/models）に置いてある。その場所を知って
      // いるのが setup.bat と 起動.bat だけだったので、npm start や
      // electron-builder で作った .exe から起動すると既定の
      // ~/.cache/huggingface を見に行き、新規インストール機では全モデルが
      // 「見つかりません」になる。
      // 起動.bat 経由のときはそちらの指定を尊重する。
      HF_HOME: process.env.HF_HOME || path.join(__dirname, '..', 'models'),
    }
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`[Python] ${data.toString()}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    // uvicorn と logging は通常の情報も stderr に書く。まとめて "Error" と
    // 付けると、正常な起動バナーがエラーに見えてログを読む人を惑わせる。
    // 深刻そうな語が含まれるときだけ error として出す。
    const text = data.toString();
    if (/\b(ERROR|CRITICAL|Traceback|Exception)\b/.test(text)) {
      console.error(`[Python] ${text}`);
    } else {
      console.log(`[Python] ${text}`);
    }
  });

  pythonProcess.on('close', (code) => {
    console.log(`[Python] API process exited with code ${code}`);
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    title: "Irodori-TTS Desktop",
    // 指定しないと Electron の既定アイコンのままウィンドウとタスクバーに出る。
    // .ico は 16〜256 を束ねてあり、Windows が場面ごとに使い分ける。
    icon: path.join(__dirname, 'build', 'icon.ico'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: false // Allow fetch('file://...') locally
    },
    show: false, // show later when ready
    backgroundColor: '#0f172a'
  });

  mainWindow.loadFile('index.html');

  // レンダラ側の例外は DevTools を開かないと見えず、UI が黙って半分死ぬ
  // 事故につながる。メインプロセスのログに転送して端末から追えるようにする。
  mainWindow.webContents.on('console-message', (_e, level, message, line, sourceId) => {
    // Electron の level は 0=verbose 1=info 2=warning 3=error。
    // 並びを 1 つずらすと、いちばん見たい error が DEBUG に化けて
    // 「UI が黙って半分死ぬ事故を追う」という目的が潰れる。
    const tag = ['DEBUG', 'LOG', 'WARN', 'ERROR'][level] || String(level);
    const where = sourceId ? ` (${path.basename(sourceId)}:${line})` : '';
    console.log(`[Renderer/${tag}] ${message}${where}`);
  });

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });
}

app.whenReady().then(async () => {
  // 暗い前提であることを Electron 側にも伝える。
  // 既定では OS の設定に関わらず prefers-color-scheme が light を返し
  // （electron/electron#21427）、<select> のようにブラウザが自前で描く
  // 部品が「明るい地に暗い字」で描かれる。暗い地に置くと字が読めなくなる。
  nativeTheme.themeSource = 'dark';

  ensureDataLayout();
  // Migration is handled by Python (data_paths.migrate_legacy_voices)
  // on server startup, so the dev and Electron paths converge.
  try {
    apiPort = await findFreePort();
  } catch (err) {
    dialog.showErrorBox('Irodori-TTS を起動できません', String(err.message || err));
    app.quit();
    return;
  }
  console.log(`[Electron] API port: ${apiPort}`);
  startPythonServer(apiPort);
  createWindow();

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

// レンダラーは自分でポートを決められないので、main が確定させた値を渡す。
ipcMain.handle('get-api-port', () => apiPort);

// Full cleanup on exit
app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});

app.on('will-quit', () => {
  if (pythonProcess) {
    console.log("[Electron] Killing Python process...");
    // Force kill child process on Windows
    spawn("taskkill", ["/pid", pythonProcess.pid, '/f', '/t']);
  }
});

// IPC MAIN HANDLERS
ipcMain.handle('get-voices', async () => {
  try {
    const data = fs.readFileSync(getMetadataPath(), 'utf-8');
    return JSON.parse(data).voices || [];
  } catch (err) {
    console.error(err);
    return [];
  }
});

// 参照ボイスの置き場を開く。一覧の取得元（getRefsDir）と同じ場所を必ず開くこと。
// 開く先がずれると「置いたのに出ない」の原因が追えなくなる。
ipcMain.handle('open-voices-folder', async () => {
  const dir = getRefsDir();
  fs.mkdirSync(dir, { recursive: true });
  const err = await shell.openPath(dir);
  if (err) throw new Error(err);
  return dir;
});

ipcMain.handle('add-voice', async (event, { name, filePath }) => {
  if (!fs.existsSync(filePath)) throw new Error("File not found");

  const id = 'voice_' + crypto.randomBytes(4).toString('hex');
  const ext = path.extname(filePath);
  const destPath = path.join(getRefsDir(), `${id}${ext}`);

  fs.copyFileSync(filePath, destPath);

  const voiceInfo = {
    id,
    name,
    path: destPath,
    created_at: new Date().toISOString()
  };

  const metaPath = getMetadataPath();
  const data = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
  data.voices.push(voiceInfo);
  fs.writeFileSync(metaPath, JSON.stringify(data, null, 2));

  return voiceInfo;
});

ipcMain.handle('delete-voice', async (event, id) => {
  const metaPath = getMetadataPath();
  const data = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
  const voice = data.voices.find(v => v.id === id);
  if (!voice) return false;

  if (fs.existsSync(voice.path)) {
    fs.unlinkSync(voice.path);
  }

  data.voices = data.voices.filter(v => v.id !== id);
  fs.writeFileSync(metaPath, JSON.stringify(data, null, 2));
  return true;
});

ipcMain.handle('select-file', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [{ name: 'Audio Files', extensions: ['wav'] }]
  });
  if (canceled) return null;
  return filePaths[0];
});

ipcMain.handle('select-folder', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
    title: 'Select PEFT adapter directory (contains adapter_config.json)'
  });
  if (canceled) return null;
  return filePaths[0];
});

ipcMain.handle('select-audio-files', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile', 'multiSelections'],
    filters: [{ name: 'Audio (wav / ogg)', extensions: ['wav', 'ogg'] }],
    title: 'Select source audio file(s) for the dataset'
  });
  if (canceled) return [];
  return filePaths;
});

// Recursively collect wav files under `dir`, capped to prevent runaways.
function enumerateWavs(dir, depth, maxDepth, out, cap) {
  if (out.length >= cap) return;
  let entries;
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch (e) {
    return;
  }
  for (const ent of entries) {
    if (out.length >= cap) return;
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) {
      if (depth < maxDepth) enumerateWavs(full, depth + 1, maxDepth, out, cap);
    } else if (ent.isFile() && /\.(wav|ogg)$/i.test(ent.name)) {
      out.push(full);
    }
  }
}

ipcMain.handle('select-audio-folder', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
    title: 'Select a folder of wav files (scanned recursively up to 5 levels)'
  });
  if (canceled || !filePaths[0]) return [];
  const out = [];
  enumerateWavs(filePaths[0], 0, 5, out, 5000);
  return out;
});

ipcMain.handle('open-text-file', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [{ name: 'Text Files', extensions: ['txt', 'md'] }],
    defaultPath: getNovelsDir(),
  });
  if (canceled || !filePaths[0]) return null;
  const content = fs.readFileSync(filePaths[0], 'utf-8');
  return { path: filePaths[0], content };
});

ipcMain.handle('enumerate-audio-folder', async (event, dirPath) => {
  if (!dirPath || typeof dirPath !== 'string') return [];
  try {
    const stat = fs.statSync(dirPath);
    if (!stat.isDirectory()) return [];
  } catch (e) {
    return [];
  }
  const out = [];
  enumerateWavs(dirPath, 0, 5, out, 5000);
  return out;
});

// Synthesize タブ: 生成済みwavを保存フォルダへコピーし、outputs/ の元ファイルを削除
// filename = outputs/ 内のファイル名のみ（例: sample_20260603_001.wav）
ipcMain.handle('save-synth-output', async (event, { filename, saveFolder, textInput }) => {
  const srcPath = path.join(getOutputsDir(), String(filename || ''));
  if (!fs.existsSync(srcPath)) return { saved: null };
  let saved = null;
  if (saveFolder && fs.existsSync(saveFolder)) {
    const safeName = String(textInput || 'output')
      .replace(/[\\/:*?"<>|]/g, '_')
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, 60) || 'output';
    const dest = path.join(saveFolder, safeName + '.wav');
    fs.copyFileSync(srcPath, dest);
    saved = dest;
  }
  // コピーできた場合だけ元を消す。保存先が USB やネットワークドライブで
  // 切断されていると saved が null になり、消してしまうと音声が完全に失われる。
  if (saved) {
    try { fs.unlinkSync(srcPath); } catch (_) {}
  }
  return { saved };
});

// 保存フォルダ選択ダイアログ
ipcMain.handle('select-save-folder', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
    title: '合成音声の保存先フォルダを選択'
  });
  if (canceled || !filePaths[0]) return null;
  return filePaths[0];
});

// 任意パスのテキストを読む（保存済みの話を朗読タブで開くため）
ipcMain.handle('read-text-file', async (event, filePath) => {
  if (!filePath || typeof filePath !== 'string' || !fs.existsSync(filePath)) return null;
  return { path: filePath, content: fs.readFileSync(filePath, 'utf-8') };
});

// =====================================================================
// 保存済みテキストの共通処理（青空文庫の保存先・朗読タブの一覧が使う）
// =====================================================================
function getNovelsDir() {
  // 書き込み先はデータルート直下に置く（配布物ごと持ち運べるように）。
  return path.join(getDataRoot(), 'novels');
}

function nocSanitizeFilename(name) {
  return String(name || '')
    .replace(/[\\/:*?"<>|]/g, '_')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 120) || 'novel';
}

ipcMain.handle('get-saved-novels', async () => {
  const dir = getNovelsDir();
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir)
    .filter(f => f.endsWith('.txt'))
    .map(f => f.slice(0, -4)); // 拡張子なしのファイル名
});

// =====================================================================
// 青空文庫タブ — 著作権切れ作品を検索→取得→正規化→保存（朗読で読む）
//   カタログCSV（全作品一覧）をローカルキャッシュしてキーワード検索
//   個人利用のみ・再配布なし
// =====================================================================
const AOZ_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) IrodoriTTS/1.0 (personal use)';
const AOZ_CATALOG_URL = 'https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip';

async function aozFetch(url) {
  const res = await fetch(url, { headers: { 'User-Agent': AOZ_UA } });
  if (!res.ok) throw new Error('取得エラー: ' + res.status + ' ' + url);
  return res;
}

// ZIPの最初の非ディレクトリファイルをデコードして返す
function aozUnzipFirst(buf, enc) {
  let offset = 0;
  while (offset < buf.length - 30) {
    if (buf.readUInt32LE(offset) !== 0x04034b50) { offset++; continue; }
    const method = buf.readUInt16LE(offset + 8);
    const compSize = buf.readUInt32LE(offset + 18);
    const fnLen = buf.readUInt16LE(offset + 26);
    const extLen = buf.readUInt16LE(offset + 28);
    const fname = buf.slice(offset + 30, offset + 30 + fnLen).toString('binary');
    const dataStart = offset + 30 + fnLen + extLen;
    if (!fname.endsWith('/') && compSize > 0) {
      const compData = buf.slice(dataStart, dataStart + compSize);
      const raw = method === 0 ? compData : zlib.inflateRawSync(compData);
      return new TextDecoder(enc).decode(raw);
    }
    offset = dataStart + Math.max(compSize, 1);
  }
  throw new Error('ZIPにファイルが見つかりません');
}

// CSVの1行をパース（RFC 4180準拠）
function aozParseCsvLine(line) {
  const cells = []; let cur = ''; let inQ = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') { if (inQ && line[i + 1] === '"') { cur += '"'; i++; } else inQ = !inQ; }
    else if (c === ',' && !inQ) { cells.push(cur); cur = ''; }
    else cur += c;
  }
  cells.push(cur);
  return cells;
}

let _aozCatalog = null; // メモリキャッシュ

function aozCatalogCachePath() {
  const dir = getNovelsDir();
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  return path.join(dir, '_aozora_catalog.json');
}

async function aozGetCatalog() {
  if (_aozCatalog) return _aozCatalog;
  const cachePath = aozCatalogCachePath();
  if (fs.existsSync(cachePath)) {
    try {
      const age = Date.now() - fs.statSync(cachePath).mtimeMs;
      if (age < 30 * 24 * 60 * 60 * 1000) {
        _aozCatalog = JSON.parse(fs.readFileSync(cachePath, 'utf-8'));
        return _aozCatalog;
      }
    } catch {}
  }
  return null;
}

ipcMain.handle('aozora-catalog-status', async () => {
  const cat = await aozGetCatalog();
  if (cat) return { ready: true, count: cat.length };
  return { ready: false, count: 0 };
});

ipcMain.handle('aozora-update-catalog', async () => {
  const res = await aozFetch(AOZ_CATALOG_URL);
  const buf = Buffer.from(await res.arrayBuffer());
  const csvText = aozUnzipFirst(buf, 'utf-8');
  const lines = csvText.split(/\r?\n/);
  const headers = aozParseCsvLine(lines[0]);
  const col = {
    title:     headers.indexOf('作品名'),
    copyright: headers.indexOf('作品著作権フラグ'),
    cardUrl:   headers.indexOf('図書カードURL'),
    lastName:  headers.indexOf('姓'),
    firstName: headers.indexOf('名'),
    txtUrl:    headers.indexOf('テキストファイルURL'),
    encoding:  headers.indexOf('テキストファイル文字集合'),
  };
  const items = [];
  for (let i = 1; i < lines.length; i++) {
    const cells = aozParseCsvLine(lines[i]);
    if (cells.length < 10) continue;
    if (cells[col.copyright] !== 'なし') continue; // 'なし'=著作権なし（パブリックドメイン）
    const txtUrl = cells[col.txtUrl] || '';
    if (!txtUrl) continue;
    const author = [cells[col.lastName], cells[col.firstName]].filter(Boolean).join(' ');
    const cardUrl = cells[col.cardUrl] || '';
    let cardPath = '';
    try { cardPath = cardUrl ? new URL(cardUrl).pathname : ''; } catch {}
    items.push({
      title: cells[col.title] || '',
      author,
      cardPath,
      txtUrl,
      encoding: cells[col.encoding] || '',
    });
  }
  fs.writeFileSync(aozCatalogCachePath(), JSON.stringify(items), 'utf-8');
  _aozCatalog = items;
  return { count: items.length };
});

ipcMain.handle('aozora-search', async (event, opts) => {
  const title  = String((opts && opts.title)  || '').trim();
  const author = String((opts && opts.author) || '').trim();
  if (!title && !author) return { error: null, items: [] };
  const catalog = await aozGetCatalog();
  if (!catalog) return { error: 'CATALOG_MISSING', items: [] };
  const items = catalog.filter(it =>
    (!title  || it.title.includes(title)) &&
    (!author || it.author.includes(author))
  ).slice(0, 60);
  return { error: null, items };
});

// 図書カードURL（例: https://www.aozora.gr.jp/cards/000148/card789.html）から
// カタログの1件を引く。カタログは図書カードURLのパスを持っているのでそれで突合する。
// 末尾スラッシュ・大文字小文字・www有無の揺れを吸収するため、
// 「cards/<著者ID>/card<作品ID>」の形に正規化してから比較する。
function aozNormalizeCardPath(s) {
  const t = String(s || '').trim();
  if (!t) return '';
  let pathname = t;
  try {
    if (/^https?:\/\//i.test(t)) pathname = new URL(t).pathname;
  } catch {
    return '';
  }
  const m = pathname.match(/cards\/(\d+)\/card(\d+)/i);
  return m ? `cards/${m[1]}/card${m[2]}` : '';
}

ipcMain.handle('aozora-by-url', async (event, url) => {
  const key = aozNormalizeCardPath(url);
  if (!key) return { error: 'BAD_URL', item: null };
  const catalog = await aozGetCatalog();
  if (!catalog) return { error: 'CATALOG_MISSING', item: null };
  const item = catalog.find((it) => aozNormalizeCardPath(it.cardPath) === key);
  if (!item) return { error: 'NOT_FOUND', item: null };
  return { error: null, item };
});

function aozNormalizeText(text) {
  // 先頭付近の区切り線（-------）を全て飛ばす
  // ヘッダー＋「テキスト中に現れる記号について」セクション両方を除去
  const sep = '-------';
  const limit = 5000;
  let pos = 0;
  while (pos < limit) {
    const idx = text.indexOf(sep, pos);
    if (idx === -1 || idx >= limit) break;
    const lineEnd = text.indexOf('\n', idx);
    pos = lineEnd !== -1 ? lineEnd + 1 : idx + sep.length;
  }
  if (pos > 0) text = text.slice(pos);

  const footIdx = text.search(/^底本[：:]/m);
  if (footIdx !== -1) text = text.slice(0, footIdx);
  return text
    .replace(/｜([^《\n]+)《[^》\n]*》/g, '$1')
    .replace(/([^\s《])《[^》\n]*》/g, '$1')
    .replace(/《[^》\n]*》/g, '')
    .replace(/｜/g, '')
    .replace(/［＃[^］]*］/g, '')
    .replace(/^[　 ]+/gm, '')
    .replace(/\n{4,}/g, '\n\n\n')
    .trim();
}

ipcMain.handle('aozora-download', async (event, arg) => {
  const { txtUrl, title, author, encoding } = arg || {};
  const res = await aozFetch(txtUrl);
  const buf = Buffer.from(await res.arrayBuffer());
  const enc = String(encoding || '').includes('Unicode') ? 'utf-8' : 'shift-jis';
  const rawText = aozUnzipFirst(buf, enc);
  const normalized = aozNormalizeText(rawText);
  const dir = getNovelsDir();
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  const name = nocSanitizeFilename([author, title].filter(Boolean).join('_') || 'aozora');
  const file = path.join(dir, name + '.txt');
  fs.writeFileSync(file, normalized, 'utf-8');
  return { path: file, name };
});
