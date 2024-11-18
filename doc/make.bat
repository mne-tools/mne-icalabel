@echo off

REM Minimal makefile for Sphinx documentation

REM Set default options and commands
set SPHINXOPTS=-nWT --keep-going
set SPHINXBUILD=sphinx-build

if "%1" == "html" goto html
if "%1" == "html-noplot" goto html-noplot
if "%1" == "clean" goto clean
if "%1" == "linkcheck" goto linkcheck
if "%1" == "linkcheck-grep" goto linkcheck-grep
if "%1" == "view" goto view

REM Define targets
:help
echo Please use `make ^<target^>` where ^<target^> is one of
echo   html             to make standalone HTML files
echo   html-noplot      to make standalone HTML files without plotting
echo   clean            to clean HTML files
echo   linkcheck        to check all external links for integrity
echo   linkcheck-grep   to grep the linkcheck result
echo   view             to view the built HTML
goto :eof

:html
%SPHINXBUILD% . _build\html -b html %SPHINXOPTS%
goto :eof

:html-noplot
%SPHINXBUILD% . _build\html -b html %SPHINXOPTS% -D plot_gallery=0
goto :eof

:clean
rmdir /s /q _build generated
goto :eof

:linkcheck
%SPHINXBUILD% . _build\linkcheck -b linkcheck -D plot_gallery=0
goto :eof

:linkcheck-grep
findstr /C:"[broken]" _build\linkcheck\output.txt > nul
if %errorlevel% equ 0 (
    echo Lines with [broken]:
    findstr /C:"[broken]" _build\linkcheck\output.txt
) else (
    echo No lines with [broken] found.
)
goto :eof

:view
python -c "import webbrowser; webbrowser.open_new_tab(r'file:///%cd%\_build\html\index.html')"
goto :eof
