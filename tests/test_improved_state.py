#!/usr/bin/env python3
"""
Test to verify improved state management functionality
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_state_size_limits():
    """Test that state size limits work correctly"""
    try:
        from llamia_v3_2.state import LlamiaState, Message, ExecResult, CodePatch
        
        state = LlamiaState()
        
        # Test message limit
        for i in range(150):  # Add more than the limit of 100
            state.add_message("user", f"Message {i}", "test_node")
        
        assert len(state.messages) <= 100, "Messages should be limited to 100"
        print("✓ State message limit test passed")
        
        # Test trace limit
        for i in range(1500):  # Add more than the limit of 1000
            state.log(f"Log entry {i}")
        
        assert len(state.trace) <= 1000, "Trace entries should be limited to 1000"
        print("✓ State trace limit test passed")
        
        # Test execution results limit
        for i in range(150):  # Add more than the limit of 100
            result = ExecResult(f"command {i}", 0, "stdout", "stderr")
            state.add_exec_result(result)
        
        assert len(state.exec_results) <= 100, "Execution results should be limited to 100"
        print("✓ State execution results limit test passed")
        
        return True
    except Exception as e:
        print(f"✗ State size limits test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_state_size_limits()
    sys.exit(0 if success else 1)
