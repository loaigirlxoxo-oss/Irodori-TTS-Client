; NSIS の追加設定。electron-builder が build/installer.nsh を自動で取り込む。
;
; ここでやっていること
;   1. data\ を消さない          更新は「旧版をアンインストール→入れ直し」なので、
;                                既定のままだと $INSTDIR ごと消えてモデル30GBと
;                                利用者の声・生成物が毎回吹き飛ぶ
;   2. 削除の選択肢              アンインストール時だけ、data\ も消すかを聞く
;                                （既定はチェックなし。意図して消したい人だけ消せる）
;   3. 字体                      既定の ＭＳ Ｐゴシック をやめ、アプリと同じ
;                                Yu Gothic UI に合わせる

!include "nsDialogs.nsh"
!include "LogicLib.nsh"

; アンインストーラ側だけで使う。インストーラ側で宣言すると
; 「未使用の変数」の警告になり、electron-builder はそれを失敗として扱う。
!ifdef BUILD_UNINSTALLER
  ; 1 = data\ も消す。既定は 0。更新（/S）のときは触らないので 0 のまま。
  Var UnDeleteData
  Var UnDataCheckbox
  ; 消せなかった数。レジスタだと Pop の後で読めないので専用に持つ。
  Var UnRemoveFailed
!endif

; ── 字体 ──────────────────────────────────────────────
; 既定はダイアログの字体を言語ファイル任せにしていて、日本語だと
; 「ＭＳ Ｐゴシック 9」になる。アプリ（夜永オールド明朝／Yu Gothic UI）と
; 並べると明らかに古い。
; LangString ^Font での上書きは見出しの太字にしか効かない（実測）。
; 各コントロールの字体はダイアログ資源に焼き込まれるので、SetFont で指定する。
; Yu Gothic UI は Windows 8.1 以降に標準で入っている。
!macro customHeader
  SetFont "Yu Gothic UI" 9
!macroend

!ifdef BUILD_UNINSTALLER

; 無人アンインストール（/S）では画面が出ないので、既定値をここで入れる。
; 空のままだと「消す」条件に一致しないので実害は無いが、意図を明示する。
!macro customUnInit
  StrCpy $UnDeleteData "0"
!macroend

; ── アンインストール時に出す選択画面 ──────────────────
; customUnWelcomePage は既定の「ようこそ」を置き換える位置に入る。
; 削除が始まる前なので、ここで聞いた答えを customRemoveFiles が使える。
!macro customUnWelcomePage
  UninstPage custom un.dataPageCreate un.dataPageLeave

  Function un.dataPageCreate
    StrCpy $UnDeleteData "0"
    ; 更新のときは黙って通す（そもそも /S で走るので表示されない）
    ${If} ${isUpdated}
      Abort
    ${EndIf}

    !insertmacro MUI_HEADER_TEXT "${PRODUCT_NAME} のアンインストール" "削除する範囲を選んでください。"

    nsDialogs::Create 1018
    Pop $0
    ${If} $0 == error
      Abort
    ${EndIf}

    ${NSD_CreateLabel} 0 0 100% 24u "${PRODUCT_NAME} をこのパソコンから削除します。"
    Pop $1

    ${NSD_CreateCheckbox} 0 32u 100% 12u "生成したデータと学習データも削除する"
    Pop $UnDataCheckbox
    ${NSD_SetState} $UnDataCheckbox ${BST_UNCHECKED}
    ${NSD_CreateLabel} 12u 48u 94% 56u "チェックを入れないと、data フォルダはそのまま残ります。$\r$\n音声モデル（約30GB）、登録した声、生成した音声、データセット、LoRA が入っています。$\r$\n入れ直したときは、そのまま続きから使えます。"
    Pop $3

    nsDialogs::Show
  FunctionEnd

  Function un.dataPageLeave
    ${NSD_GetState} $UnDataCheckbox $0
    ${If} $0 == ${BST_CHECKED}
      StrCpy $UnDeleteData "1"
    ${Else}
      StrCpy $UnDeleteData "0"
    ${EndIf}
  FunctionEnd
!macroend

; ── 削除する中身 ──────────────────────────────────────
; 既定は $INSTDIR をまるごと消す。data\ だけ残すため自前で回す。
;
; 既定にあった「更新中に使えないファイルがあれば退避して巻き戻す」処理は
; 使えない。あれは $INSTDIR を丸ごと $PLUGINSDIR へ移すので、data\ の
; 十数GBまで一緒に動かすことになる（%TEMP% が別ドライブなら実コピー）。
; 代わりに、消せたかどうかを数え、更新時に1件でも残ったら中止する。
; 呼び出し元は終了コードを見て5回まで待ち直す（installUtil.nsh の UninstallLoop）。
!macro customRemoveFiles
  Push $R0
  Push $R1

  StrCpy $UnRemoveFailed 0

  FindFirst $R1 $R0 "$INSTDIR\*.*"
  removeLoop:
    StrCmp $R0 "" removeDone
    StrCmp $R0 "." removeNext
    StrCmp $R0 ".." removeNext
    StrCmp $R0 "data" removeNext
    IfFileExists "$INSTDIR\$R0\*.*" 0 removeFile
      ClearErrors
      RMDir /r "$INSTDIR\$R0"
      IfErrors 0 removeNext
      IntOp $UnRemoveFailed $UnRemoveFailed + 1
      DetailPrint "消せませんでした: $INSTDIR\$R0"
      Goto removeNext
    removeFile:
      ClearErrors
      Delete "$INSTDIR\$R0"
      IfErrors 0 removeNext
      IntOp $UnRemoveFailed $UnRemoveFailed + 1
      DetailPrint "消せませんでした: $INSTDIR\$R0"
    removeNext:
      FindNext $R1 $R0
      Goto removeLoop
  removeDone:
  FindClose $R1

  Pop $R1
  Pop $R0

  ${If} $UnRemoveFailed <> 0
  ${AndIf} ${isUpdated}
    ; 新旧のファイルが混ざったまま起動させない。呼び出し元が待ち直す。
    SetErrorLevel 1
    Abort "旧版のファイルを $UnRemoveFailed 件消せませんでした。アプリを閉じてからやり直してください。"
  ${EndIf}

  ${IfNot} ${isUpdated}
    ${If} $UnDeleteData == "1"
      RMDir /r "$INSTDIR\data"
      ; インストール先が書けない場所だったときは、本体が利用者ごとの領域へ
      ; 逃がしている（main.js getDataRoot）。そちらも消さないと残る。
      SetShellVarContext current
      RMDir /r "$APPDATA\${APP_FILENAME}\data"
      !ifdef APP_PRODUCT_FILENAME
        RMDir /r "$APPDATA\${APP_PRODUCT_FILENAME}\data"
      !endif
      ; Electron の userData は package.json の name を使う（productName は
      ; build ブロックの中なので Electron からは見えない）。実際の退避先は
      ; %APPDATA%\irodori-desktop\data になる。
      !ifdef APP_PACKAGE_NAME
        RMDir /r "$APPDATA\${APP_PACKAGE_NAME}\data"
      !endif
      ${If} $installMode == "all"
        SetShellVarContext all
      ${EndIf}
    ${EndIf}
    ; data を残したときは中身があるので消えない。それでよい。
    RMDir "$INSTDIR"
  ${EndIf}
!macroend

!endif ; BUILD_UNINSTALLER
