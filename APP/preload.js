const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  // main が決めた API サーバーのポート（固定ではない）
  getApiPort: () => ipcRenderer.invoke('get-api-port'),
  // 設定画面のアップデート
  appVersion: () => ipcRenderer.invoke('app-version'),
  checkUpdate: () => ipcRenderer.invoke('check-update'),
  runUpdate: () => ipcRenderer.invoke('run-update'),
  onUpdateProgress: (fn) => {
    const h = (_e, p) => fn(p);
    ipcRenderer.on('update-progress', h);
    return () => ipcRenderer.removeListener('update-progress', h);
  },
  getVoices: () => ipcRenderer.invoke('get-voices'),
  openVoicesFolder: () => ipcRenderer.invoke('open-voices-folder'),
  openEasyLines: () => ipcRenderer.invoke('open-easy-lines'),
  selectEasyVoice: () => ipcRenderer.invoke('select-easy-voice'),
  selectEasyFolder: () => ipcRenderer.invoke('select-easy-folder'),
  addVoice: (data) => ipcRenderer.invoke('add-voice', data),
  deleteVoice: (id) => ipcRenderer.invoke('delete-voice', id),
  selectFile: () => ipcRenderer.invoke('select-file'),
  selectFolder: () => ipcRenderer.invoke('select-folder'),
  selectAudioFiles: () => ipcRenderer.invoke('select-audio-files'),
  selectAudioFolder: () => ipcRenderer.invoke('select-audio-folder'),
  enumerateAudioFolder: (dirPath) => ipcRenderer.invoke('enumerate-audio-folder', dirPath),
  openTextFile: () => ipcRenderer.invoke('open-text-file'),
  saveSynthOutput: (arg) => ipcRenderer.invoke('save-synth-output', arg),
  selectSaveFolder: () => ipcRenderer.invoke('select-save-folder'),
  getOutputsDir: () => ipcRenderer.invoke('get-outputs-dir'),
  openSaveFolder: (folder) => ipcRenderer.invoke('open-save-folder', folder),
  readTextFile: (filePath) => ipcRenderer.invoke('read-text-file', filePath),
  // 青空文庫
  aozoraCatalogStatus: () => ipcRenderer.invoke('aozora-catalog-status'),
  aozoraUpdateCatalog: () => ipcRenderer.invoke('aozora-update-catalog'),
  aozoraSearch: (keyword) => ipcRenderer.invoke('aozora-search', keyword),
  aozoraByUrl: (url) => ipcRenderer.invoke('aozora-by-url', url),
  aozoraDownload: (arg) => ipcRenderer.invoke('aozora-download', arg),
  aozoraPreview: (arg) => ipcRenderer.invoke('aozora-preview', arg),
  // 保存済みテキストの一覧（青空文庫・朗読タブが使う）
  getSavedNovels: () => ipcRenderer.invoke('get-saved-novels')
});

// 初回起動の画面（setup.html）だけが使う。進捗と失敗を受け取る。
contextBridge.exposeInMainWorld('irodoriSetup', {
  onProgress: (fn) => ipcRenderer.on('setup-progress', (_e, p) => fn(p)),
  onError: (fn) => ipcRenderer.on('setup-error', (_e, m) => fn(m)),
  // 取得する音声モデルを選ぶ（一覧を受け取り、選んだ id を返す）
  onChooseModels: (fn) => ipcRenderer.on('setup-choose-models', (_e, c) => fn(c)),
  sendChosenModels: (ids) => ipcRenderer.send('setup-models-chosen', ids)
});
