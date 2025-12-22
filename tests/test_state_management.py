#!/usr/bin/env python3
"""
Test to verify state management functionality
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_state_initialization():
    """Test that the state can be initialized correctly"""
    try:
        from llamia_v3_2.state import LlamiaState
        state = LlamiaState()
        assert state is not None, "State should be initialized successfully"
        assert state.messages == [], "Messages should be empty initially"
        assert state.turn_id == 0, "Turn ID should be 0 initially"
        print("✓ State initialization test passed")
        return True
    except Exception as e:
        print(f"✗ State initialization test failed: {e}")
        return False

def test_state_message_adding():
    """Test that messages can be added to the state"""
    try:
        from llamia_v3_2.state import LlamiaState
        state = LlamiaState()
        state.add_message("user", "Hello, world!", "test_node")
        
        assert len(state.messages) == 1, "Should have one message"
        assert state.messages[0]["role"] == "user", "Message role should be 'user'"
        assert state.messages[0]["content"] == "Hello, world!", "Message content should match"
        assert state.messages[0]["node"] == "test_node", "Message node should match"
        print("✓ State message adding test passed")
        return True
    except Exception as e:
        print(f"✗ State message adding test failed: {e}")
        return False

def test_state_logging():
    """Test that logging works correctly"""
    try:
        from llamia_v3_2.state import LlamiaState
        state = LlamiaState()
        state.log("Test log message")
        
        assert len(state.trace) == 1, "Should have one trace entry"
        assert state.trace[0] == "Test log message", "Trace entry should match"
        print("✓ State logging test passed")
        return True
    except Exception as e:
        print(f"✗ State logging test failed: {e}")
        return False

if __name__ == "__main__":
    results = [
        test_state_initialization(),
        test_state_message_adding(),
        test_state_logging()
    ]
    success = all(results)
    sys.exit(0 if success else 1)
