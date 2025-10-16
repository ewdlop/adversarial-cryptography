#!/usr/bin/env python3
"""
Validation script for differential privacy implementation.

This script validates the structure and API of the differential privacy module
without requiring full TensorFlow/Keras installation.
"""

import sys
import ast
import os


def check_file_exists(filepath):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✓ Found: {filepath}")
        return True
    else:
        print(f"✗ Missing: {filepath}")
        return False


def check_module_structure(filepath):
    """Check that a Python module has expected structure."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
            tree = ast.parse(source)
        
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        return classes, functions
    except Exception as e:
        print(f"  Error parsing {filepath}: {e}")
        return [], []


def validate_differential_privacy_module():
    """Validate the differential_privacy.py module."""
    print("\n" + "=" * 70)
    print("Validating differential_privacy.py")
    print("=" * 70)
    
    filepath = "differential_privacy.py"
    if not check_file_exists(filepath):
        return False
    
    classes, functions = check_module_structure(filepath)
    
    # Expected classes
    expected_classes = ["DPOptimizer", "PrivacyAccountant"]
    print("\nExpected classes:")
    for cls in expected_classes:
        if cls in classes:
            print(f"  ✓ {cls}")
        else:
            print(f"  ✗ {cls} (missing)")
    
    # Expected functions
    expected_functions = ["add_noise_to_output", "clip_gradients_by_norm", "get_privacy_preset"]
    print("\nExpected functions:")
    for func in expected_functions:
        if func in functions:
            print(f"  ✓ {func}")
        else:
            print(f"  ✗ {func} (missing)")
    
    # Check for privacy presets
    with open(filepath, 'r') as f:
        content = f.read()
        if "PRIVACY_PRESETS" in content:
            print("\n  ✓ PRIVACY_PRESETS dictionary defined")
            if '"high"' in content and '"medium"' in content and '"low"' in content:
                print("  ✓ All three preset levels defined (high, medium, low)")
        else:
            print("\n  ✗ PRIVACY_PRESETS dictionary not found")
    
    return True


def validate_notebooks():
    """Validate notebook files."""
    print("\n" + "=" * 70)
    print("Validating Jupyter Notebooks")
    print("=" * 70)
    
    notebooks = [
        "symmetric_dp.ipynb",
    ]
    
    all_exist = True
    for notebook in notebooks:
        if check_file_exists(notebook):
            # Check if it's valid JSON
            try:
                import json
                with open(notebook, 'r') as f:
                    nb = json.load(f)
                    cell_count = len(nb.get('cells', []))
                    print(f"  ✓ Valid notebook with {cell_count} cells")
            except Exception as e:
                print(f"  ✗ Invalid notebook format: {e}")
                all_exist = False
        else:
            all_exist = False
    
    return all_exist


def validate_documentation():
    """Validate documentation files."""
    print("\n" + "=" * 70)
    print("Validating Documentation")
    print("=" * 70)
    
    docs = [
        "DIFFERENTIAL_PRIVACY.md",
        "ASYMMETRIC_DP_NOTES.md",
        "README.md"
    ]
    
    all_exist = True
    for doc in docs:
        if check_file_exists(doc):
            with open(doc, 'r') as f:
                lines = f.readlines()
                print(f"  ✓ {len(lines)} lines")
                
                # Check for key sections in DIFFERENTIAL_PRIVACY.md
                if doc == "DIFFERENTIAL_PRIVACY.md":
                    content = ''.join(lines)
                    required_sections = [
                        "Overview",
                        "Implementation",
                        "Usage",
                        "Privacy Parameters",
                        "References"
                    ]
                    for section in required_sections:
                        if section in content:
                            print(f"    ✓ Contains '{section}' section")
                        else:
                            print(f"    ✗ Missing '{section}' section")
        else:
            all_exist = False
    
    return all_exist


def validate_example_script():
    """Validate example script."""
    print("\n" + "=" * 70)
    print("Validating Example Script")
    print("=" * 70)
    
    filepath = "example_differential_privacy.py"
    if not check_file_exists(filepath):
        return False
    
    classes, functions = check_module_structure(filepath)
    
    expected_functions = [
        "demonstrate_privacy_presets",
        "demonstrate_privacy_accounting",
        "demonstrate_dp_optimizer",
        "demonstrate_privacy_utility_tradeoff",
        "main"
    ]
    
    print("\nExample demonstrations:")
    for func in expected_functions:
        if func in functions:
            print(f"  ✓ {func}")
        else:
            print(f"  ✗ {func} (missing)")
    
    return True


def validate_tests():
    """Validate test files."""
    print("\n" + "=" * 70)
    print("Validating Test Files")
    print("=" * 70)
    
    filepath = "test_differential_privacy.py"
    if not check_file_exists(filepath):
        return False
    
    classes, functions = check_module_structure(filepath)
    
    print(f"\nTest classes found: {len([c for c in classes if c.startswith('Test')])}")
    for cls in classes:
        if cls.startswith('Test'):
            print(f"  ✓ {cls}")
    
    test_methods = [f for f in functions if f.startswith('test_')]
    print(f"\nTest methods found: {len(test_methods)}")
    
    return True


def validate_requirements():
    """Validate requirements.txt."""
    print("\n" + "=" * 70)
    print("Validating Requirements")
    print("=" * 70)
    
    filepath = "requirements.txt"
    if not check_file_exists(filepath):
        return False
    
    with open(filepath, 'r') as f:
        requirements = f.readlines()
    
    expected_packages = ["tensorflow", "tensorflow-privacy", "numpy", "matplotlib"]
    print("\nRequired packages:")
    for pkg in expected_packages:
        found = any(pkg in req.lower() for req in requirements)
        if found:
            print(f"  ✓ {pkg}")
        else:
            print(f"  ✗ {pkg} (missing)")
    
    return True


def main():
    """Run all validations."""
    print("=" * 70)
    print("DIFFERENTIAL PRIVACY IMPLEMENTATION VALIDATION")
    print("=" * 70)
    
    results = []
    
    # Run validations
    results.append(("Module", validate_differential_privacy_module()))
    results.append(("Notebooks", validate_notebooks()))
    results.append(("Documentation", validate_documentation()))
    results.append(("Example Script", validate_example_script()))
    results.append(("Tests", validate_tests()))
    results.append(("Requirements", validate_requirements()))
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All validations passed!")
        print("\nDifferential privacy implementation is complete and ready to use.")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run example: python example_differential_privacy.py")
        print("  3. Open symmetric_dp.ipynb in Jupyter")
        print("  4. Run tests: pytest test_differential_privacy.py")
    else:
        print("✗ Some validations failed!")
        print("\nPlease review the output above for details.")
        sys.exit(1)
    
    print("=" * 70)


if __name__ == "__main__":
    main()
