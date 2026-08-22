@echo off
setlocal
cd /d "%~dp0"

REM ---------- モデルの置き場 ----------
REM 既定では C:\Users\<ユーザー名>\.cache\huggingface に入る。
REM アプリを別ドライブに置いてもモデルだけ C を占有するため、
REM アプリ配下へ向ける。フォルダごと移動しても壊れない。
set "HF_HOME=%~dp0models"

REM ---------- セットアップ済みかの確認 ----------
if not exist "APP\node_modules" (
  echo [エラー] セットアップがまだのようです。
  echo         先に setup.bat を実行してください。
  echo.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [エラー] Python 環境 ^(.venv^) が見つかりません。
  echo         先に setup.bat を実行してください。
  echo.
  pause
  exit /b 1
)

REM ポートはアプリが起動時に空きを自動で探すため、ここでは何もしない。
REM （他のプロセスを終了させることは一切しない）

echo ==============================================
echo  Irodori-TTS を起動します...
echo ==============================================

cd /d "%~dp0APP"
set ELECTRON_RUN_AS_NODE=
call npm start

endlocal
