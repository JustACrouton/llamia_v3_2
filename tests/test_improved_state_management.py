#!/usr/bin/env python3
"""
Test to verify improved state management functionality with size limits
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_state_size_limits():
    """Test that state size limits work correctly"""
    try:
        # Import the improved state implementation
        from llamia_v3_2.state_improved import LlamiaState, ExecResult
        
        # Create a state instance
        state = LlamiaState()
        
        # Test message limit
        for i in range(150):  # Add more than the limit of 100
            state.add_message("user", f"Message {i}", "test_node")
        
        # Check that messages are limited
        assert len(state.messages) <= 100, f"Messages should be limited to 100, but got {len(state.messages)}"
        
        # Check that we have the most recent messages
        if len(state.messages) == 100:
            assert state.messages[0]["content"] == "Message 50", "Should have the most recent messages"
            assert state.messages[-1]["content"] == "Message 149", "Should have the most recent messages"
        
        print("✓ State message limit test passed")
        
        # Test trace limit
        for i in range(1500):  # Add more than the limit of 1000
            state.log(f"Log entry {i}")
        
        assert len(state.trace) <= 1000, f"Trace entries should be limited to 1000, but got {len(state.trace)}"
        
        # Check that we have the most recent trace entries
        if len(state.trace) == 1000:
            assert state.trace[0] == "Log entry 500", "Should have the most recent trace entries"
            assert state.trace[-1] == "Log entry 1499", "Should have the most recent trace entries"
        
        print("✓ State trace limit test passed")
        
        # Test execution results limit
        for i in range(150):  # Add more than the limit of 100
            result = ExecResult(f"command {i}", 0, "stdout", "stderr")
            state.add_exec_result(result)
        
        assert len(state.exec_results) <= 100, f"Execution results should be limited to 100, but got {len(state.exec_results)}"
        
        # Check that we have the most recent execution results
        if len(state.exec_results) == 100:
            assert state.exec_results[0].command == "command 50", "Should have the most recent execution results"
            assert state.exec_results[-1].command == "command 149", "Should have the most recent execution results"
        
        print("✓ State execution results limit test passed")
        
        return True
    except Exception as e:
        print(f"✗ State size limits test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_state_size_limits()
    sys.exit(0 if success else 1)
