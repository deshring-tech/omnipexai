@echo off
rem ---------------------------------------------------------------
rem  Omnipex launcher - double-click to open the menu, or run a
rem  single action directly:  launch.bat 2
rem ---------------------------------------------------------------
setlocal EnableExtensions
cd /d "%~dp0"
title Omnipex

set "PY=python"
where python >nul 2>nul || set "PY=py"

set "BASE=examples\rag_docs_qa\manifest.json"
set "V2=examples\rag_docs_qa\manifest_v2.json"

if "%~1"=="" goto :menu
call :run %~1
exit /b %errorlevel%

:menu
echo.
echo   =====================================================
echo     OMNIPEX  -  regression CI for AI pipelines
echo   =====================================================
echo     1  show     demo pipeline graph
echo     9  doctor   config vs. what was actually built (instant)
echo     2  diff     change-impact analysis (the CI gate)
echo     3  blame    bisect the diff, name the culprit
echo     4  test     run the full test suite
echo     5  check    validate value-test scenarios
echo     6  session  run a value-test session (facilitator)
echo     7  timing   DeepEval timing rig (dry run)
echo     8  gui      open the OmniPex GUI in your browser
echo     q  quit
echo.
set /p "choice=  choose: "
if /i "%choice%"=="q" exit /b 0
call :run %choice%
goto :menu

:run
if "%~1"=="1" %PY% -m omnipex.cli show "%BASE%"
if "%~1"=="2" %PY% -m omnipex.cli diff "%BASE%" "%V2%"
if "%~1"=="3" %PY% -m omnipex.cli blame "%BASE%" "%V2%"
if "%~1"=="4" %PY% tests\test_invalidation.py
if "%~1"=="5" %PY% -m valuetest.cli check
if "%~1"=="6" goto :session
if "%~1"=="7" %PY% timing\time_eval.py --dry-run
if "%~1"=="8" %PY% -m omnipex.cli ui
if "%~1"=="9" %PY% -m omnipex.cli doctor "%V2%"
exit /b %errorlevel%

:session
set /p "pname=  participant id: "
if "%pname%"=="" (
    echo   participant id is required.
    exit /b 1
)
%PY% -m valuetest.cli run --participant %pname%
exit /b %errorlevel%
