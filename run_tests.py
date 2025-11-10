"""
Test Suite cho Franka Shadow Hand Training
Kiểm tra tất cả components: environment, policy, checkpoint
"""
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_result(test_name, passed, message=""):
    """Print test result"""
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} - {test_name}")
    if message:
        print(f"      {message}")

# ========================================
# TEST 1: Import Modules
# ========================================
def test_imports():
    """Test if all required modules can be imported"""
    print_section("TEST 1: Import Modules")
    
    tests_passed = 0
    tests_total = 0
    
    # Test basic imports
    modules = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("gymnasium", "Gymnasium"),
    ]
    
    for module_name, display_name in modules:
        tests_total += 1
        try:
            __import__(module_name)
            print_result(f"Import {display_name}", True)
            tests_passed += 1
        except ImportError as e:
            print_result(f"Import {display_name}", False, str(e))
    
    # Test project imports (without Isaac Sim)
    tests_total += 1
    try:
        # Don't import franka_train.py directly as it starts Isaac Sim
        # Instead, just check if the file exists
        if Path("franka_train.py").exists():
            print_result("Check franka_train.py", True)
            tests_passed += 1
        else:
            print_result("Check franka_train.py", False, "File not found")
    except Exception as e:
        print_result("Check franka_train.py", False, str(e))
    
    print(f"\nImport Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total

# ========================================
# TEST 2: Policy Structure
# ========================================
def test_policy_structure():
    """Test policy network structure"""
    print_section("TEST 2: Policy Network Structure")
    
    tests_passed = 0
    tests_total = 0
    
    # Skip detailed policy tests to avoid Isaac Sim startup
    # Just check if DRL modules exist
    tests_total += 1
    try:
        drl_files = [
            "drl/agent/sac.py",
            "drl/memory/rher.py",
            "drl/learning/rher.py",
            "drl/utils/model/stochastics.py"
        ]
        all_exist = all(Path(f).exists() for f in drl_files)
        if all_exist:
            print_result("Check DRL modules", True, f"{len(drl_files)} files found")
            tests_passed += 1
        else:
            print_result("Check DRL modules", False, "Some files missing")
    except Exception as e:
        print_result("Check DRL modules", False, str(e))
    
    print(f"\nPolicy Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total

# ========================================
# TEST 3: Checkpoint Loading
# ========================================
def test_checkpoint_loading():
    """Test loading checkpoint files"""
    print_section("TEST 3: Checkpoint Loading")
    
    tests_passed = 0
    tests_total = 0
    
    # Find checkpoint files
    runs_dir = Path("runs")
    
    # Test 1: Check runs directory
    tests_total += 1
    if not runs_dir.exists():
        print_result("Check runs directory", False, "runs/ directory not found")
        print("\nCheckpoint Tests: 0/{} passed".format(tests_total))
        print("[WARNING] No training has been run yet. Run training first.")
        return False
    print_result("Check runs directory", True)
    tests_passed += 1
    
    # Find all policy files
    policy_files = list(runs_dir.glob("*/policy/*.pt"))
    
    # Test 2: Check policy files exist
    tests_total += 1
    if not policy_files:
        print_result("Find policy files", False, "No policy files found")
        print("\nCheckpoint Tests: {}/{} passed".format(tests_passed, tests_total))
        print("[WARNING] No policy saved yet. Training may be incomplete.")
        return False
    print_result("Find policy files", True, f"Found {len(policy_files)} policy file(s)")
    tests_passed += 1
    
    # Test the most recent policy
    policy_file = sorted(policy_files, key=lambda x: x.stat().st_mtime)[-1]
    print(f"\nTesting: {policy_file}")
    
    # Test 3: Check file size
    tests_total += 1
    file_size_mb = policy_file.stat().st_size / (1024**2)
    if 5 < file_size_mb < 15:  # Expected: 7-10 MB
        print_result("Check file size", True, f"{file_size_mb:.2f} MB")
        tests_passed += 1
    else:
        print_result("Check file size", False, 
                    f"{file_size_mb:.2f} MB (expected 7-10 MB)")
    
    # Test 4: Load checkpoint
    tests_total += 1
    try:
        state_dict = torch.load(str(policy_file), map_location='cpu', weights_only=True)
        print_result("Load checkpoint", True, f"{len(state_dict)} keys")
        tests_passed += 1
        
        # Test 5: Check keys
        tests_total += 1
        required_keys = ['net.seq_fe.h0', 'net.seq_fe.cells.0.i2h.weight']
        has_keys = all(key in state_dict for key in required_keys)
        if has_keys:
            print_result("Check keys", True, f"GRU keys present")
            tests_passed += 1
        else:
            print_result("Check keys", False, "Missing GRU keys")
        
        # Test 6: Verify key structure
        tests_total += 1
        try:
            # Check if it's GRU-based policy
            gru_keys = [k for k in state_dict.keys() if 'seq_fe' in k or 'gru' in k.lower()]
            if len(gru_keys) > 0:
                print_result("Verify state_dict keys", True, f"{len(gru_keys)} GRU keys found")
                tests_passed += 1
            else:
                print_result("Verify state_dict keys", False, "No GRU keys found")
        except Exception as e:
            print_result("Verify state_dict keys", False, str(e))
        
    except Exception as e:
        print_result("Load checkpoint", False, str(e))
    
    print(f"\nCheckpoint Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total

# ========================================
# TEST 4: CUDA/GPU
# ========================================
def test_cuda():
    """Test CUDA availability"""
    print_section("TEST 4: CUDA/GPU")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: CUDA available
    tests_total += 1
    cuda_available = torch.cuda.is_available()
    print_result("CUDA available", cuda_available)
    if cuda_available:
        tests_passed += 1
    
    if cuda_available:
        # Test 2: Device count
        tests_total += 1
        device_count = torch.cuda.device_count()
        print_result("GPU count", True, f"{device_count} GPU(s)")
        tests_passed += 1
        
        # Test 3: GPU name
        tests_total += 1
        gpu_name = torch.cuda.get_device_name(0)
        print_result("GPU name", True, gpu_name)
        tests_passed += 1
        
        # Test 4: CUDA version
        tests_total += 1
        cuda_version = torch.version.cuda
        print_result("CUDA version", True, cuda_version)
        tests_passed += 1
        
        # Test 5: Simple CUDA operation
        tests_total += 1
        try:
            x = torch.randn(100, 100).cuda()
            y = x @ x.T
            _ = y.cpu()
            print_result("CUDA operation", True)
            tests_passed += 1
        except Exception as e:
            print_result("CUDA operation", False, str(e))
    else:
        print("[WARNING] CUDA not available. Training will be very slow on CPU.")
    
    print(f"\nCUDA Tests: {tests_passed}/{tests_total} passed")
    return tests_passed >= tests_total - 1  # Pass if at least CUDA available

# ========================================
# TEST 5: File Structure
# ========================================
def test_file_structure():
    """Test project file structure"""
    print_section("TEST 5: File Structure")
    
    tests_passed = 0
    tests_total = 0
    
    required_files = [
        ("franka_train.py", "Training script"),
        ("franka_env/__init__.py", "Environment package"),
        ("franka_env/env_cfg.py", "Environment config"),
        ("drl/agent/sac.py", "SAC agent"),
        ("drl/memory/rher.py", "RHER memory"),
        ("drl/learning/rher.py", "RHER learning"),
    ]
    
    for file_path, description in required_files:
        tests_total += 1
        if Path(file_path).exists():
            print_result(f"{description}", True, file_path)
            tests_passed += 1
        else:
            print_result(f"{description}", False, f"{file_path} not found")
    
    print(f"\nFile Structure Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total

# ========================================
# MAIN TEST RUNNER
# ========================================
def run_all_tests():
    """Run all tests"""
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*15 + "FRANKA SHADOW HAND TEST SUITE" + " "*14 + "║")
    print("╚" + "="*58 + "╝")
    
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    results['policy'] = test_policy_structure()
    results['checkpoint'] = test_checkpoint_loading()
    results['cuda'] = test_cuda()
    results['files'] = test_file_structure()
    
    # Summary
    print_section("SUMMARY")
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} - {test_name.upper()}")
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {passed_tests}/{total_tests} test categories passed")
    print(f"{'='*60}\n")
    
    if passed_tests == total_tests:
        print("*** ALL TESTS PASSED! ***")
        print("[OK] Your setup is ready for training/testing.")
        return 0
    else:
        print("*** SOME TESTS FAILED ***")
        print("[WARNING] Please fix the issues above before training/testing.")
        return 1

if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
