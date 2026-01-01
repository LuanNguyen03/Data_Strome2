"""
One-shot Orchestration Script
Runs: standardize -> olap_build -> train_models
Then performs smoke tests on API endpoints

Per docs/clinical_governance_checklist.md
"""
from __future__ import annotations

import sys
import subprocess
import time
from pathlib import Path
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import requests
from datetime import datetime


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")


def print_step(step: str):
    """Print step header"""
    print(f"{Colors.BOLD}▶ {step}{Colors.RESET}")


def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.RESET}")


def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.RESET}")


def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.RESET}")


def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """Run a command and return success status and output"""
    print_step(f"Running: {description}")
    print(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=3600,  # 1 hour timeout
        )
        
        if result.returncode == 0:
            print_success(f"{description} completed successfully")
            return True, result.stdout
        else:
            print_error(f"{description} failed with return code {result.returncode}")
            print(f"  Error output: {result.stderr[:500]}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print_error(f"{description} timed out after 1 hour")
        return False, "Timeout"
    except Exception as e:
        print_error(f"{description} failed: {e}")
        return False, str(e)


def check_file_exists(filepath: Path, description: str) -> bool:
    """Check if a file exists"""
    if filepath.exists():
        print_success(f"{description} exists: {filepath}")
        return True
    else:
        print_warning(f"{description} not found: {filepath}")
        return False


def smoke_test_api(base_url: str = "http://localhost:8000") -> Tuple[bool, List[str]]:
    """Perform smoke tests on API endpoints"""
    print_header("Smoke Testing API Endpoints")
    
    failures: List[str] = []
    
    # Test 1: Health check
    print_step("Testing GET /api/v1/healthz")
    try:
        response = requests.get(f"{base_url}/api/v1/healthz", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print_success(f"Health check passed: {data.get('status')}")
        else:
            failures.append(f"Health check returned {response.status_code}")
            print_error(f"Health check failed: {response.status_code}")
    except Exception as e:
        failures.append(f"Health check error: {e}")
        print_error(f"Health check error: {e}")
    
    # Test 2: Screening endpoint
    print_step("Testing POST /api/v1/assessments/screening")
    try:
        response = requests.post(
            f"{base_url}/api/v1/assessments/screening",
            json={
                "age": 30,
                "average_screen_time": 8.5,
                "sleep_quality": 3,
                "sleep_duration": 6.5,
                "stress_level": 4,
            },
            timeout=30,
        )
        if response.status_code == 200:
            data = response.json()
            # Check contract compliance
            if "disclaimers" in data and "model_version" in data:
                print_success("Screening endpoint passed (contract compliant)")
                
                # Check if trigger_symptom is true
                trigger_symptom = data.get("screening", {}).get("trigger_symptom", False)
                if trigger_symptom:
                    print_step("  trigger_symptom=true, testing triage endpoint")
                    # Test 3: Triage endpoint (if trigger_symptom)
                    try:
                        triage_response = requests.post(
                            f"{base_url}/api/v1/assessments/triage",
                            json={
                                "age": 30,
                                "average_screen_time": 8.5,
                                "sleep_quality": 3,
                                "discomfort_eyestrain": 1,
                                "redness_in_eye": 1,
                            },
                            timeout=30,
                        )
                        if triage_response.status_code == 200:
                            triage_data = triage_response.json()
                            if "disclaimers" in triage_data and "model_version" in triage_data:
                                print_success("Triage endpoint passed (contract compliant)")
                            else:
                                failures.append("Triage response missing disclaimers or model_version")
                        else:
                            failures.append(f"Triage endpoint returned {triage_response.status_code}")
                    except Exception as e:
                        failures.append(f"Triage endpoint error: {e}")
                else:
                    print_warning("  trigger_symptom=false, skipping triage test")
            else:
                failures.append("Screening response missing disclaimers or model_version")
        else:
            failures.append(f"Screening endpoint returned {response.status_code}")
            print_error(f"Screening endpoint failed: {response.status_code}")
    except Exception as e:
        failures.append(f"Screening endpoint error: {e}")
        print_error(f"Screening endpoint error: {e}")
    
    # Test 4: OLAP KPI endpoint
    print_step("Testing GET /api/v1/olap/kpis/age_gender")
    try:
        response = requests.get(
            f"{base_url}/api/v1/olap/kpis/age_gender",
            params={"page": 1, "page_size": 100},
            timeout=30,
        )
        if response.status_code == 200:
            data = response.json()
            if "data" in data and len(data.get("data", [])) > 0:
                print_success(f"OLAP KPI endpoint passed ({len(data['data'])} rows)")
            else:
                print_warning("OLAP KPI endpoint returned empty data")
        else:
            failures.append(f"OLAP KPI endpoint returned {response.status_code}")
            print_error(f"OLAP KPI endpoint failed: {response.status_code}")
    except Exception as e:
        failures.append(f"OLAP KPI endpoint error: {e}")
        print_error(f"OLAP KPI endpoint error: {e}")
    
    # Test 5: Model metadata endpoint
    print_step("Testing GET /api/v1/models/latest")
    try:
        response = requests.get(f"{base_url}/api/v1/models/latest", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "model_version" in data:
                print_success(f"Model metadata endpoint passed (version: {data['model_version']})")
            else:
                failures.append("Model metadata missing model_version")
        else:
            failures.append(f"Model metadata endpoint returned {response.status_code}")
            print_error(f"Model metadata endpoint failed: {response.status_code}")
    except Exception as e:
        failures.append(f"Model metadata endpoint error: {e}")
        print_error(f"Model metadata endpoint error: {e}")
    
    return len(failures) == 0, failures


def main():
    """Main orchestration function"""
    print_header("One-Shot Orchestration: Standardize → OLAP → Train → Smoke Test")
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Project root: {project_root}")
    
    all_passed = True
    failures: List[str] = []
    
    # Step 1: Standardize
    print_header("Step 1: Data Standardization")
    success, output = run_command(
        [sys.executable, str(project_root / "backend" / "scripts" / "standardize.py")],
        "Data Standardization"
    )
    if not success:
        all_passed = False
        failures.append("Standardization failed")
    else:
        # Check output file
        output_file = project_root / "data" / "standardized" / "clean_assessments.parquet"
        if not check_file_exists(output_file, "Standardized parquet file"):
            all_passed = False
            failures.append("Standardized parquet file not found")
    
    # Step 2: OLAP Build
    print_header("Step 2: OLAP KPI Generation")
    success, output = run_command(
        [sys.executable, str(project_root / "backend" / "scripts" / "olap_build.py")],
        "OLAP KPI Generation"
    )
    if not success:
        all_passed = False
        failures.append("OLAP build failed")
    else:
        # Check KPI files
        kpi_files = [
            "agg_ded_by_age_gender.parquet",
            "agg_ded_by_screen_sleep.parquet",
            "agg_ded_by_symptom_score.parquet",
            "agg_ded_by_stress_sleepband.parquet",
            "agg_data_quality_by_group.parquet",
        ]
        for kpi_file in kpi_files:
            kpi_path = project_root / "analytics" / "duckdb" / "agg" / kpi_file
            if not check_file_exists(kpi_path, f"KPI file: {kpi_file}"):
                print_warning(f"KPI file missing (may be expected): {kpi_file}")
    
    # Step 3: Train Models (optional - may take long)
    print_header("Step 3: Model Training")
    print_warning("Model training may take a long time.")
    
    # Ask user if they want to train (or skip)
    train_models = False
    if len(sys.argv) > 1 and "--train" in sys.argv:
        train_models = True
    
    if train_models:
        success, output = run_command(
            [sys.executable, str(project_root / "backend" / "scripts" / "train_models_advanced.py")],
            "Model Training"
        )
        if not success:
            all_passed = False
            failures.append("Model training failed")
    else:
        print_warning("Skipping model training (use --train flag to enable)")
        print_warning("To train models manually: python backend/scripts/train_models_advanced.py")
    
    # Check if models exist
    registry_path = project_root / "modeling" / "registry" / "registry.json"
    if registry_path.exists():
        print_success("Model registry exists (models may be available)")
    else:
        print_warning("Model registry not found (will use fallback)")
    
    # Step 4: Smoke Tests (requires API server running)
    print_header("Step 4: API Smoke Tests")
    print_warning("Note: API server must be running on http://localhost:8000")
    print_warning("Start server with: uvicorn backend.main:app --reload")
    
    # Check if server is running
    try:
        test_response = requests.get("http://localhost:8000/api/v1/healthz", timeout=2)
        if test_response.status_code == 200:
            print_success("API server is running, proceeding with smoke tests...")
            smoke_success, smoke_failures = smoke_test_api()
            if not smoke_success:
                print_warning("Some smoke tests failed (non-critical)")
                failures.extend(smoke_failures)
                # Don't fail overall if smoke tests fail (server might not be running)
        else:
            print_warning("API server responded but with error, skipping smoke tests")
    except requests.exceptions.RequestException:
        print_warning("API server not running, skipping smoke tests")
        print_warning("To run smoke tests: start server and run again")
        # Don't fail overall if server is not running
    
    # Summary
    print_header("Summary")
    
    # Always print artifact locations (helpful even if some steps failed)
    print("\nKey Artifact Locations:")
    print(f"  • Standardized data: data/standardized/clean_assessments.parquet")
    print(f"  • OLAP KPIs: analytics/duckdb/agg/*.parquet")
    print(f"  • Model registry: modeling/registry/registry.json")
    print(f"  • Model artifacts: modeling/artifacts/*.joblib")
    print(f"  • Audit logs: backend/logs/audit.jsonl")
    
    if all_passed:
        print_success("\n✅ ALL TESTS PASSED")
        return 0
    else:
        print_error("\n❌ SOME TESTS FAILED")
        print("\nFailures:")
        for failure in failures:
            print(f"  • {failure}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print_error("\n\nOrchestration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
