import os
import sys
import unittest
from unittest.mock import MagicMock

# 1. Set environment variables before imports to configure settings
os.environ["NEUROSETTLE_ADMIN_TOKEN"] = "test-admin-token"
os.environ["DEFAULT_MODEL_NAME"] = "TCN_aug_weighted_v1"
os.environ["ENABLE_TRAINING"] = "0"
os.environ["MLFLOW_DISABLED"] = "1"

# 2. Mock heavy ML/DL modules to keep the execution lightweight
sys.modules['torch'] = MagicMock()
sys.modules['autogluon'] = MagicMock()
sys.modules['autogluon.tabular'] = MagicMock()
sys.modules['mlflow'] = MagicMock()
sys.modules['mlflow.tracking'] = MagicMock()

# Mock inference_service singleton module to prevent cold starts and loading checkpoints
mock_module = MagicMock()
mock_module.inference_service = MagicMock()
sys.modules['app.inference_service'] = mock_module

# Set path for importing app modules
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# 3. Custom test result formatter for concise outputs (showing PASS/FAIL status directly)
class ConciseTestResult(unittest.TextTestResult):
    def addSuccess(self, test):
        desc = test.shortDescription() or test._testMethodName
        print(f"[PASS] {desc}")

    def addFailure(self, test, err):
        desc = test.shortDescription() or test._testMethodName
        print(f"[FAIL] {desc}")
        super().addFailure(test, err)

    def addError(self, test, err):
        desc = test.shortDescription() or test._testMethodName
        print(f"[ERROR] {desc}")
        super().addError(test, err)

class ConciseTestRunner(unittest.TextTestRunner):
    resultclass = ConciseTestResult
    def run(self, test):
        result = super().run(test)
        print("\n" + "="*50)
        if result.wasSuccessful():
            print(f"Status: PASS ({result.testsRun} tests passed)")
        else:
            failures = len(result.failures)
            errors = len(result.errors)
            print(f"Status: FAIL ({failures} failed, {errors} errors)")
        return result

if __name__ == "__main__":
    # Disable the default dot stream outputs
    devnull = open(os.devnull, 'w')
    runner = ConciseTestRunner(stream=devnull, verbosity=0)
    
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.join(backend_dir, "tests"), pattern="test_*.py")
    
    print("Running Backend Unit Tests...")
    print("="*50)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
