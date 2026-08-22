const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  // main が決めた API サーバーのポート（固定ではない）
  getApiPort: () => ipcRenderer.invoke('get-api-port'),
  getVoices: () => ipcRenderer.invoke('get-voices'),
  openVoicesFolder: () => ipcRenderer.invoke('open-voices-folder'),
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
  readTextFile: (filePath) => ipcRenderer.invoke('read-text-file', filePath),
  // 青空文庫
  aozoraCatalogStatus: () => ipcRenderer.invoke('aozora-catalog-status'),
  aozoraUpdateCatalog: () => ipcRenderer.invoke('aozora-update-catalog'),
  aozoraSearch: (keyword) => ipcRenderer.invoke('aozora-search', keyword),
  aozoraByUrl: (url) => ipcRenderer.invoke('aozora-by-url', url),
  aozoraDownload: (arg) => ipcRenderer.invoke('aozora-download', arg),
  // 保存済みテキストの一覧（青空文庫・朗読タブが使う）
  getSavedNovels: () => ipcRenderer.invoke('get-saved-novels')
});
