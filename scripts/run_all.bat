@echo off
REM Windows batch script to run common tasks

echo Dry Eye Assessment - Task Runner
echo.

:menu
echo Select task:
echo 1. Standardize data
echo 2. Build OLAP aggregates
echo 3. Run backend
echo 4. Run frontend
echo 5. Exit
echo.
set /p choice="Enter choice (1-5): "

if "%choice%"=="1" goto standardize
if "%choice%"=="2" goto olap
if "%choice%"=="3" goto backend
if "%choice%"=="4" goto frontend
if "%choice%"=="5" goto end
goto menu

:standardize
echo Running standardization...
call .venv\Scripts\activate.bat
python scripts/standardize.py --input DryEyeDisease/Dry_Eye_Dataset.csv --output data/standardized/clean_assessments.parquet --report data/standardized/data_quality_report.json
goto menu

:olap
echo Building OLAP aggregates...
call .venv\Scripts\activate.bat
python -m backend.scripts.olap_build --input data/standardized/clean_assessments.parquet --outdir analytics/duckdb/agg
goto menu

:backend
echo Starting backend...
call .venv\Scripts\activate.bat
python backend/run.py
goto menu

:frontend
echo Starting frontend...
cd frontend
call npm run dev
cd ..
goto menu

:end
echo Goodbye!

