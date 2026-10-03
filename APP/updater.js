// アプリ本体のアップデート。設定画面の「アップデートを確認」「ダウンロードして更新」。
//
// GitHub の最新リリースを見て、新しい版があればインストーラを一時フォルダへ落とし、
// 起動してからアプリを閉じる。更新は「旧版を消して入れ直す」作りで、data\ は残る
// （build/installer.nsh）。開発版（パッケージしていない起動）では確認だけして入れない。

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');

const REPO = 'loaigirlxoxo-oss/Irodori-TTS-Client';
const API = `https://api.github.com/repos/${REPO}/releases/latest`;
const ASSET_PATTERN = /^Irodori-TTS-Client-App-Setup-.*\.exe$/;
const CHECK_TIMEOUT_MS = 15000;
// 回線が止まったまま待ち続けないための上限。進んでいる間は延長する。
const STALL_TIMEOUT_MS = 60000;

function parseVersion(v) {
  return String(v || '').replace(/^v/i, '').split('.').map((x) => parseInt(x, 10) || 0);
}

function isNewer(latest, current) {
  const a = parseVersion(latest);
  const b = parseVersion(current);
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    if ((a[i] || 0) !== (b[i] || 0)) return (a[i] || 0) > (b[i] || 0);
  }
  return false;
}

async function fetchLatest() {
  const res = await fetch(API, {
    headers: { Accept: 'application/vnd.github+json', 'User-Agent': 'Irodori-TTS-Client-App' },
    signal: AbortSignal.timeout(CHECK_TIMEOUT_MS),
  });
  if (!res.ok) throw new Error(`GitHub が ${res.status} を返しました`);
  return res.json();
}

async function checkUpdate({ currentVersion, isPackaged }) {
  try {
    const rel = await fetchLatest();
    const latest = String(rel.tag_name || '').replace(/^v/i, '');
    const asset = (rel.assets || []).find((a) => ASSET_PATTERN.test(a.name));
    const newer = isNewer(latest, currentVersion);
    let reason = '';
    if (newer && !isPackaged) reason = '開発版では自動で入れ替えません。リポジトリを更新してください。';
    else if (newer && !asset) reason = 'このリリースにはインストーラがありません。';
    return {
      latest,
      newer,
      published: rel.published_at ? String(rel.published_at).slice(0, 10) : '',
      canInstall: newer && isPackaged && !!asset,
      reason,
    };
  } catch (err) {
    return { error: err.message || String(err) };
  }
}

async function downloadInstaller(onProgress) {
  const rel = await fetchLatest();
  const asset = (rel.assets || []).find((a) => ASSET_PATTERN.test(a.name));
  if (!asset) throw new Error('このリリースにはインストーラがありません。');
  const dest = path.join(os.tmpdir(), asset.name);
  const ctrl = new AbortController();
  let stall = setTimeout(() => ctrl.abort(), STALL_TIMEOUT_MS);
  try {
    const res = await fetch(asset.browser_download_url, { signal: ctrl.signal, headers: { 'User-Agent': 'Irodori-TTS-Client-App' } });
    if (!res.ok) throw new Error(`ダウンロードが ${res.status} で失敗しました`);
    const total = Number(res.headers.get('content-length')) || asset.size || 0;
    const out = fs.createWriteStream(dest);
    let done = 0;
    for await (const chunk of res.body) {
      clearTimeout(stall);
      stall = setTimeout(() => ctrl.abort(), STALL_TIMEOUT_MS);
      done += chunk.length;
      if (!out.write(chunk)) await new Promise((r) => out.once('drain', r));
      onProgress({ done, total });
    }
    await new Promise((resolve, reject) => out.end((err) => (err ? reject(err) : resolve())));
    if (asset.size && fs.statSync(dest).size !== asset.size) {
      throw new Error('ダウンロードしたファイルの大きさが合いません。もう一度お試しください。');
    }
    return dest;
  } finally {
    clearTimeout(stall);
  }
}

// インストーラを起動してアプリを閉じる。インストーラは自分で旧版を閉じて入れ替える。
async function runUpdate({ isPackaged, onProgress, quit }) {
  if (!isPackaged) return { error: '開発版では自動で入れ替えません。' };
  try {
    const file = await downloadInstaller(onProgress);
    spawn(file, [], { detached: true, stdio: 'ignore' }).unref();
    setTimeout(quit, 500);
    return { ok: true };
  } catch (err) {
    return { error: err.name === 'AbortError' ? 'ダウンロードが止まったので中断しました。' : (err.message || String(err)) };
  }
}

module.exports = { checkUpdate, runUpdate, isNewer };
