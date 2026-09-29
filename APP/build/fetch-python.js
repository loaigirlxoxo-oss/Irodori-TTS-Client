// 配布物に同梱する Python を取ってくる（ビルド時に一度だけ走る）。
//
// 埋め込み版（embeddable）を使う。インストーラに入れても10MBちょっとで、
// ユーザーの PATH も既存の Python も触らない。
//
// 3.12 を選んでいる理由:
//   AMD(Radeon) 版 PyTorch が Windows では cp312 でしか配られていない。
//   3.10 を同梱すると Radeon 機が最初から動かせない。NVIDIA 側は cu128 に
//   cp312 の wheel があるので、3.12 ならランタイムを1本にまとめられる。
//
// 埋め込み版には venv も ensurepip も入っていない。そのため初回起動では
// この一式を丸ごとコピーして、そこへ直接 pip で入れる（runtime.js）。
//
// pip の導入に get-pip.py は使わない。あれは bootstrap.pypa.io で中身が
// 入れ替わり続けるのでハッシュを固定できない。代わりに PyPI の pip の
// wheel を版とハッシュで固定して持ってくる。wheel は zip で、中身は
// site-packages に置く形そのままなので、初回起動時に展開すれば入る
// （runtime.js。pip 自身に pip を入れさせる手は、いまの pip が
// 「pip を書き換えるときは -m pip で」と言って止めるので使えない）。

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const https = require('https');
const { execFileSync } = require('child_process');

const PY_VERSION = '3.12.10';
const PY_URL = `https://www.python.org/ftp/python/${PY_VERSION}/python-${PY_VERSION}-embed-amd64.zip`;
// 2026-09-24 に取得したものの SHA-256。版ごとに中身は変わらないので、
// 以降は取り違えや壊れたダウンロードをここで弾ける。
// PY_VERSION を上げるときは、この値も取り直して書き換えること。
const PY_SHA256 = '4acbed6dd1c744b0376e3b1cf57ce906f9dc9e95e68824584c8099a63025a3c3';

const PIP_VERSION = '26.2.1';
const PIP_FILE = `pip-${PIP_VERSION}-py3-none-any.whl`;
const PIP_URL = 'https://files.pythonhosted.org/packages/f3/6e/'
  + '1736e5b4ae2b778ef2f81c47d797de9f891d4d8acb047a24ca37a60294dd/' + PIP_FILE;
// PyPI が公開している digest そのもの。
const PIP_SHA256 = '71138adf1f4ca900cdb7d289c21b7494329f2332b6d85f0e1c42108c0384ed3e';

const OUT = path.join(__dirname, 'python-embed');
// 何を入れたかの控え。次のビルドはこれを見て、同じ版・同じハッシュで
// 揃っているときだけ作り直しを省く。
const MANIFEST = path.join(OUT, '.embed.json');

function download(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    https.get(url, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        file.close();
        return download(res.headers.location, dest).then(resolve, reject);
      }
      if (res.statusCode !== 200) return reject(new Error(`${url}: HTTP ${res.statusCode}`));
      res.pipe(file);
      file.on('finish', () => file.close(resolve));
    }).on('error', reject);
  });
}

function verify(file, expected, label) {
  const got = crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
  if (got !== expected) {
    fs.unlinkSync(file);
    throw new Error(`${label} のハッシュが違う\n  期待: ${expected}\n  実際: ${got}`);
  }
  console.log(`[fetch-python] ${label} のハッシュ一致`);
}

// 前回の取得が最後まで通っていて、いまの版と一致しているか。
// python.exe の有無だけで判断すると、展開まで済んで pip の取得前に
// 落ちた残骸を「取得済み」と見なし、ビルドは通って利用者の初回起動で
// 初めて失敗する。版やハッシュを変えたときも同じ。
function alreadyDone() {
  try {
    const m = JSON.parse(fs.readFileSync(MANIFEST, 'utf8'));
    if (m.python !== PY_VERSION || m.pythonSha256 !== PY_SHA256
      || m.pip !== PIP_VERSION || m.pipSha256 !== PIP_SHA256) return false;
    if (!fs.existsSync(path.join(OUT, 'python.exe'))) return false;
    // wheel は 1.8MB なので毎回ハッシュを取り直す。控えの文字列だけで
    // 済ませると、壊れた・差し替えられたものをそのまま梱包してしまう。
    // Python 側は展開後の数千ファイルなので、ここでは再計算していない
    // （元の zip は展開後に消している）。
    const whl = path.join(OUT, PIP_FILE);
    if (!fs.existsSync(whl)) return false;
    return crypto.createHash('sha256').update(fs.readFileSync(whl)).digest('hex') === PIP_SHA256;
  } catch (e) {
    return false;
  }
}

async function main() {
  if (alreadyDone()) {
    console.log('[fetch-python] すでにある。何もしない');
    return;
  }
  // 中途半端な残骸の上に重ねない
  fs.rmSync(OUT, { recursive: true, force: true });
  fs.mkdirSync(OUT, { recursive: true });
  const zip = path.join(OUT, 'python-embed.zip');
  console.log(`[fetch-python] ${PY_URL} を取得`);
  await download(PY_URL, zip);
  verify(zip, PY_SHA256, 'Python');

  // Windows 10 以降の tar.exe は zip を展開できる。外部ツールを足さない。
  // PATH の "tar" だと Git Bash 同梱のものが先に来ることがあり、そちらは
  // D:\… を「ホスト名 D」と読んで失敗する。System32 のものを名指しする。
  const winTar = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'tar.exe');
  if (fs.existsSync(winTar)) {
    execFileSync(winTar, ['-xf', zip, '-C', OUT], { stdio: 'inherit' });
  } else {
    // -Command は「以降すべてが本文」なので、引数として外から渡せない。
    // シングルクォートの中では '' が ' 1つを表す。その規則で埋める。
    // 作業パスに ' が入っても壊れない。
    const q = (s) => `'${s.replace(/'/g, "''")}'`;
    execFileSync('powershell', ['-NoProfile', '-Command',
      `Expand-Archive -LiteralPath ${q(zip)} -DestinationPath ${q(OUT)} -Force`],
      { stdio: 'inherit' });
  }
  fs.unlinkSync(zip);

  // 埋め込み版は既定で site が無効。pip を使うので有効にして、
  // site-packages を検索パスに足す。
  const pth = fs.readdirSync(OUT).find((f) => /^python\d+\._pth$/.test(f));
  if (!pth) throw new Error('._pth が見つからない');
  const p = path.join(OUT, pth);
  let text = fs.readFileSync(p, 'utf8').replace('#import site', 'import site');
  if (!text.includes('Lib\\site-packages')) text = `${text.trimEnd()}\nLib\\site-packages\n`;
  fs.writeFileSync(p, text);

  console.log(`[fetch-python] pip ${PIP_VERSION} の wheel を取得`);
  const whl = path.join(OUT, PIP_FILE);
  await download(PIP_URL, whl);
  verify(whl, PIP_SHA256, 'pip');

  fs.writeFileSync(MANIFEST, JSON.stringify({
    python: PY_VERSION, pythonSha256: PY_SHA256,
    pip: PIP_VERSION, pipSha256: PIP_SHA256,
    at: new Date().toISOString(),
  }, null, 1));
  console.log(`[fetch-python] 用意できた: ${OUT}`);
}

main().catch((e) => {
  console.error('[fetch-python] 失敗:', e.message);
  process.exit(1);
});
