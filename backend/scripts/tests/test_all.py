"""
Run all backend tests
Per docs/clinical_governance_checklist.md
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def run_tests():
    """Run all test modules"""
    print("=" * 70)
    print("Running All Backend Tests")
    print("=" * 70)
    
    test_modules = [
        "backend.scripts.tests.test_standardize",
        "backend.scripts.tests.test_api_contract",
        "backend.scripts.tests.test_leakage",
    ]
    
    results = {}
    
    for module_name in test_modules:
        print(f"\n▶ Running: {module_name}")
        try:
            module = __import__(module_name, fromlist=[""])
            if hasattr(module, "__main__") or hasattr(module, "if __name__"):
                # Run the module's main block
                exec(f"import {module_name}")
            results[module_name] = "PASSED"
            print(f"✅ {module_name} passed")
        except AssertionError as e:
            results[module_name] = f"FAILED: {e}"
            print(f"❌ {module_name} failed: {e}")
        except Exception as e:
            results[module_name] = f"ERROR: {e}"
            print(f"❌ {module_name} error: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for r in results.values() if r == "PASSED")
    total = len(results)
    
    for module, result in results.items():
        status = "✅" if result == "PASSED" else "❌"
        print(f"{status} {module}: {result}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    # Also run individual test modules directly
    import subprocess
    
    print("Running individual test modules...\n")
    
    test_files = [
        "backend/scripts/tests/test_standardize.py",
        "backend/scripts/tests/test_api_contract.py",
        "backend/scripts/tests/test_leakage.py",
    ]
    
    all_passed = True
    for test_file in test_files:
        test_path = project_root / test_file
        if test_path.exists():
            print(f"Running {test_file}...")
            result = subprocess.run(
                [sys.executable, str(test_path)],
                cwd=project_root,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"✅ {test_file} passed\n")
            else:
                print(f"❌ {test_file} failed")
                print(result.stdout)
                print(result.stderr)
                all_passed = False
        else:
            print(f"⚠️  {test_file} not found")
    
    if all_passed:
        print("✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
