!include "MUI2.nsh"
!define APP_NAME "Strata"
!define COMPANY  "Jadevit"
!define VERSION  "0.0.1"

Name "${APP_NAME} ${VERSION}"
OutFile "dist\Strata_${VERSION}_x64_setup.exe"
InstallDir "$PROGRAMFILES64\${COMPANY}\${APP_NAME}"
RequestExecutionLevel admin
SetCompress auto
SetCompressor /SOLID lzma

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_LANGUAGE "English"

Section "Install"
  SetOutPath "$INSTDIR"

  ; Copy app payload (relative to this script's working dir)
  File /r "payload\*.*"

  ; Shortcuts
  CreateDirectory "$SMPROGRAMS\${APP_NAME}"
  CreateShortCut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\strata.exe"
  CreateShortCut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\strata.exe"

  ; Run runtime-installer during install
  DetailPrint "Installing runtimes..."
  nsExec::ExecToStack '"$INSTDIR\runtime-installer.exe" --prefer auto'
  Pop $0
  Pop $1
  StrCmp $0 0 +3
    MessageBox MB_ICONEXCLAMATION "Runtime installation reported an error (code $0). Strata may still run in CPU mode."
SectionEnd

Section "Uninstall"
  Delete "$DESKTOP\${APP_NAME}.lnk"
  RMDir /r "$SMPROGRAMS\${APP_NAME}"
  RMDir /r "$INSTDIR"
SectionEnd