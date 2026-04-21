@echo off
setlocal

rem One-shot pipeline: compile every Slang permutation to DXBC, then
rem repack the results into the Warcraft III v1.8 BLS bundles under
rem bls_out/. Forwards any extra args to compile_all_slang.py so callers
rem can narrow the sweep, e.g. `build.bat --family hd_ps --target d3d11`.

pushd "%~dp0"

py compile_all_slang.py %*
if errorlevel 1 goto :error

py build_bls.py --templates war3.w3mod\shaders --output bls_out --strip
if errorlevel 1 goto :error

popd
endlocal
exit /b 0

:error
popd
endlocal
exit /b 1
