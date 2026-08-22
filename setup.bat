@echo off
setlocal
cd /d "%~dp0"

REM ---------- モデルの置き場 ----------
REM 既定では C:\Users\<ユーザー名>\.cache\huggingface に入る。
REM アプリを別ドライブに置いてもモデルだけ C を占有するため、
REM アプリ配下へ向ける。フォルダごと移動しても壊れない。
set "HF_HOME=%~dp0models"

echo ==============================================
echo  Irodori-TTS  セットアップ
echo ==============================================
echo.
echo  このスクリプトは以下を構築します。
echo    1. Node パッケージ       (APP\node_modules)
echo    2. Python 環境           (.venv)
echo    3. 音声モデル            (約 17GB)
echo.
echo  初回は 30～60 分ほどかかります（回線速度による）。
echo.

REM ---------- 前提コマンドの確認 ----------
where npm >nul 2>nul
if errorlevel 1 (
  echo [エラー] npm が見つかりません。
  echo         Node.js LTS を入れてから再実行してください。
  echo         https://nodejs.org/
  goto :fail
)

REM requirements.txt の dacvae と silentcipher は git+https で取得する。
REM git が無いと pip の途中で分かりにくいエラーになるので、先に弾く。
where git >nul 2>nul
if errorlevel 1 (
  echo [エラー] git が見つかりません。
  echo         一部のライブラリを GitHub から取得するため必要です。
  echo         Git for Windows を入れてから再実行してください。
  echo         https://git-scm.com/download/win
  goto :fail
)

REM ---------- GPU の判別 ----------
REM   NVIDIA なら CUDA 版、AMD(Radeon) なら ROCm 版を入れる。どちらも
REM   無ければ CPU 版。IRODORI_BACKEND で上書きできる
REM   （cu128 / cu126 / rocm / cpu）。
REM
REM   Radeon を分けるのは、AMD が配る Windows 版 PyTorch が
REM   cp312 のみの配布で、Python 3.10 では入らないため。
set "BACKEND=%IRODORI_BACKEND%"
if not "%BACKEND%"=="" goto :gpu_done

set "BACKEND=cpu"
nvidia-smi -L >nul 2>nul
if not errorlevel 1 (
  REM cu128 のビルドに入っているカーネルは sm_70 以上。GTX 10xx などの
  REM Pascal（sm_61）では torch.cuda.is_available() が True を返すのに、
  REM 行列積ひとつで no kernel image is available になる。しかも pip は
  REM 成功するのでセットアップは正常終了し、生成して初めて壊れが分かる。
  REM compute capability を見て、7.0 未満なら cu126 に落とす。
  set "BACKEND=cu128"
  for /f "delims=" %%C in ('nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader 2^>nul') do (
    for /f "tokens=1 delims=." %%M in ("%%C") do (
      if %%M LSS 7 set "BACKEND=cu126"
    )
  )
  goto :gpu_done
)
REM AMD の GPU があるかを見る。名前に Radeon / AMD を含むものを探す。
REM wmic は Windows 11 の新しいビルドで削除された。無いままだとループが
REM 空回りして BACKEND が cpu のままになり、Radeon 機でも CPU 版が入る。
REM PowerShell の CIM で引く。
for /f "delims=" %%G in ('powershell -NoProfile -Command "(Get-CimInstance Win32_VideoController).Name" 2^>nul ^| findstr /i "Radeon AMD"') do (
  set "BACKEND=rocm"
  goto :gpu_done
)

:gpu_done
echo [確認] GPU    : %BACKEND%
if "%BACKEND%"=="rocm" (
  echo.
  echo         Radeon 版は試験対応です。生成は動く見込みですが、
  echo         AMD は Windows での学習を公式に対応していません。
  echo         学習を使うときは NVIDIA の GPU か WSL2 をご利用ください。
  echo.
)

REM ---------- Python の選択 ----------
REM   通常は 3.10。sentencepiece 0.1.99 の Windows wheel が cp311 までで、
REM   3.12 だとソースビルドに落ちて CMake で失敗するため。
REM   ただし Radeon 版 PyTorch は cp312 のみの配布なので、そちらは 3.12 を
REM   使う（requirements.txt の sentencepiece は上限を外してあり、
REM   3.12 では cp312 の wheel がある 0.2.x が入る）。
set "PYLAUNCH="
set "PYWANT=3.10"
if "%BACKEND%"=="rocm" set "PYWANT=3.12"

py -%PYWANT% -c "import sys" >nul 2>nul
if not errorlevel 1 goto :py_ok_launcher
python -c "import sys; v='%PYWANT%'.split('.'); sys.exit(0 if sys.version_info[:2]==(int(v[0]),int(v[1])) else 1)" >nul 2>nul
if not errorlevel 1 goto :py_ok_plain
goto :py_missing

:py_ok_launcher
set "PYLAUNCH=py -%PYWANT%"
goto :py_done

:py_ok_plain
set "PYLAUNCH=python"
goto :py_done

:py_missing
echo [エラー] Python %PYWANT% が見つかりません。
echo.
if "%BACKEND%"=="rocm" (
  echo         Radeon 版 PyTorch は Python 3.12 でのみ配布されています。
  echo         https://www.python.org/downloads/release/python-3129/
) else (
  echo         このアプリは Python 3.10 が必要です。
  echo         3.11 以降では音声処理ライブラリの導入に失敗します。
  echo         https://www.python.org/downloads/release/python-31011/
)
echo.
echo         インストーラの「Add python.exe to PATH」に
echo         チェックを入れてください。
goto :fail

:py_done

echo [確認] npm    : OK
echo [確認] git    : OK
echo [確認] Python : %PYLAUNCH%
echo.

REM ---------- 1. Node パッケージ ----------
echo ----------------------------------------------
echo  [1/3] Electron を取得します...
echo ----------------------------------------------
pushd APP
call npm install
if errorlevel 1 (
  popd
  echo [エラー] npm install に失敗しました。
  goto :fail
)
popd
echo [完了] Electron
echo.

REM ---------- 旧版の名残 ----------
REM 以前は v4 を別ツリー・別 venv で動かしていた。いまは統合したので
REM そのフォルダは使わない。消すのは利用者の判断なので案内だけ出す。
if exist "Irodori-TTS-v4\.venv" (
  echo [お知らせ] 旧版の Irodori-TTS-v4 フォルダ（約 5GB）が残っています。
  echo            いまは使いません。削除して構いません。
  echo.
)

REM ---------- 2. Python 環境 ----------
echo ----------------------------------------------
echo  [2/3] Python 環境を作ります...
echo ----------------------------------------------
REM 既にある .venv が今回ほしい版かを確かめる。存在だけ見て使い回すと、
REM 3.10 で作った環境に Radeon 版（cp312 のみ）を入れようとして失敗し、
REM CPU 版へ落ちる。どこで壊れたか分からない形になるので、版が違えば
REM 作り直す。
set "VENV_OK="
if exist ".venv\Scripts\python.exe" (
  call ".venv\Scripts\python.exe" -c "import sys; v='%PYWANT%'.split('.'); sys.exit(0 if sys.version_info[:2]==(int(v[0]),int(v[1])) else 1)" >nul 2>nul
  if not errorlevel 1 set "VENV_OK=1"
)
if not defined VENV_OK (
  if exist ".venv\Scripts\python.exe" (
    echo  - 既存の .venv は Python %PYWANT% ではないので作り直します。
    rmdir /s /q ".venv"
  )
  %PYLAUNCH% -m venv .venv
  if errorlevel 1 (
    echo [エラー] .venv の作成に失敗しました。
    goto :fail
  )
)
call ".venv\Scripts\python.exe" -m pip install --upgrade pip

REM torch は GPU 版を専用の配布元から先に入れる。
REM   通常の PyPI には CPU 版しか無く、素直に requirements.txt を入れると
REM   GPU があっても CPU 動作になり、生成が桁違いに遅くなる。
REM   （pyproject.toml の [tool.uv.sources] は uv 専用の設定で、pip では効かない）
REM
REM   版を固定するのは、専用インデックス側の最新が PyPI 側より古いため。
REM   固定しないと requirements.txt の torch>=2.10.0 を PyPI の新しい CPU 版が
REM   先に満たしてしまい、あとから GPU 版を入れようとしても
REM   「already satisfied」で黙って飛ばされる。
if "%BACKEND%"=="rocm" goto :torch_rocm
if "%BACKEND%"=="cpu"  goto :torch_cpu

echo  - GPU 版 PyTorch (%BACKEND%) を取得します（数分かかります）...
REM Pascal 世代は cu128 のカーネルを持たないので BACKEND が cu126 になる。
REM 版はどちらも 2.10.0 で揃える（requirements の torch>=2.10.0 を満たす）。
call ".venv\Scripts\python.exe" -m pip install torch==2.10.0+%BACKEND% torchaudio==2.10.0+%BACKEND% --index-url https://download.pytorch.org/whl/%BACKEND%
if errorlevel 1 goto :torch_cpu_fallback
echo  - GPU 版 PyTorch: OK
goto :v23_deps

:torch_rocm
REM Radeon 版。AMD が配る Windows 用 wheel を直接指定して入れる。
REM   PyTorch 公式インデックス（download.pytorch.org/whl/rocm*）には
REM   Windows 版が置かれておらず、AMD 側の repo.radeon.com にしか無い。
REM   そのため index ではなく wheel の URL を直に渡す。
REM   版が 2.9.1 なのは AMD がそこまでしか出していないため。生成・学習とも
REM   2.9.1 で動くことは確認済み（学習は AMD が Windows で非対応）。
echo  - Radeon 版 PyTorch を取得します（約 800MB。数分かかります）...
set "ROCMBASE=https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
call ".venv\Scripts\python.exe" -m pip install "%ROCMBASE%/torch-2.9.1%%2Brocm7.2.1-cp312-cp312-win_amd64.whl" "%ROCMBASE%/torchaudio-2.9.1%%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
if errorlevel 1 (
  echo [警告] Radeon 版 PyTorch を取得できませんでした。CPU 版に切り替えます。
  goto :torch_cpu
)
echo  - Radeon 版 PyTorch: OK
goto :v23_deps

:torch_cpu_fallback
echo [警告] GPU 版 PyTorch を取得できませんでした。CPU 版に切り替えます。

:torch_cpu
REM GPU 版を使わない（使えない）ときは、ここで CPU 版を明示的に入れて確定させる。
REM （requirements.txt 任せにすると「入ったのか入っていないのか」が曖昧になる）
REM 版は GPU 経路と同じ 2.10.0 に固定する。範囲指定にすると PyPI の
REM 新しい版を拾い、torchcodec 0.10 と組み合わせが合わなくなる。
call ".venv\Scripts\python.exe" -m pip install torch==2.10.0 torchaudio==2.10.0
if errorlevel 1 (
  echo [エラー] PyTorch を導入できませんでした。回線を確認してください。
  goto :fail
)
echo [警告] CPU 版で続行します。生成はできますが、かなり遅くなります。

:v23_deps
REM requirements.txt は torch>=2.10.0 を要求する。Radeon 版は AMD が 2.9.1
REM までしか出していないので、そのまま入れると pip が「条件を満たさない」と
REM 判断して PyPI の新しい torch（Windows では CPU 版）へ差し替えてしまう。
REM 導入直後は「Radeon 版: OK」と出るのに、最終的な環境は CPU という
REM 分かりにくい壊れ方をする。Radeon 経路では torch 系を外した一覧を使う。
if not "%BACKEND%"=="rocm" goto :deps_normal

echo  - 依存関係を導入します（torch 系は導入済みのものを使います）...
REM 前方一致で torch を弾くと torchcodec と torchdata まで落ちる。
REM train.py は torchdata を import するので、Radeon 環境では学習が
REM 「非対応です」ではなく ModuleNotFoundError で落ちていた。
REM 版を固定して入れた torch と torchaudio の 2 つだけを外す。
findstr /v /r /i /c:"^torch>" /c:"^torchaudio>" requirements.txt > "%TEMP%\irodori_req_rocm.txt"
call ".venv\Scripts\python.exe" -m pip install -r "%TEMP%\irodori_req_rocm.txt"
set "DEPS_ERR=%errorlevel%"
del "%TEMP%\irodori_req_rocm.txt" >nul 2>nul
if not "%DEPS_ERR%"=="0" (
  echo [エラー] 依存関係の導入に失敗しました。
  goto :fail
)
goto :deps_app

:deps_normal
call ".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo [エラー] 依存関係の導入に失敗しました。
  goto :fail
)

:deps_app
call ".venv\Scripts\python.exe" -m pip install -r APP\requirements.txt
if errorlevel 1 (
  echo [エラー] アプリ側の依存関係の導入に失敗しました。
  goto :fail
)
echo [完了] v2/v3 環境
echo.

REM ---------- 3. モデルの取得 ----------
REM 取得しておかないと初回生成時に数分間、無反応に見える。
REM ここで落としておけば進捗が見える状態で待てる。
echo ----------------------------------------------
echo  [3/3] 音声モデルを取得します（約 17GB）...
echo ----------------------------------------------
echo  ※ 途中で止めても、再実行すれば続きから取得します。
call ".venv\Scripts\python.exe" APP\fetch_models.py
if errorlevel 1 (
  echo.
  echo [エラー] モデルの取得に失敗しました。
  echo         モデルが無いと音声を生成できません。
  echo         回線を確認して、もう一度 setup.bat を実行してください。
  goto :fail
)
echo.

:done
echo ==============================================
echo  セットアップが終わりました。
echo.
echo  起動.bat をダブルクリックしてください。
echo ==============================================
echo.
pause
exit /b 0

:fail
echo.
echo ==============================================
echo  セットアップを中断しました。
echo  上のエラーを確認してから再実行してください。
echo ==============================================
echo.
pause
exit /b 1
