#!/usr/bin/env python3
"""
Test to verify improved graph routing functionality with error handling
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_routing_error_handling():
    """Test that routing functions handle errors gracefully"""
    try:
        # Import the improved graph implementation
        from llamia_v3_2.graph_improved import (
            _route_from_intent, 
            _route_from_planner, 
            _route_from_research,
            _route_from_research_web,
            _route_from_coder,
            _route_from_critic
        )
        
        # Test routing with None state (should not crash)
        result = _route_from_intent(None)
        assert result == "chat", f"Expected 'chat' but got {result}"
        print("✓ Intent router handles None state correctly")
        
        # Test routing with empty dict state
        result = _route_from_intent({})
        assert result == "chat", f"Expected 'chat' but got {result}"
        print("✓ Intent router handles empty dict correctly")
        
        # Test planner routing with None state
        result = _route_from_planner(None)
        assert result == "coder", f"Expected 'coder' but got {result}"
        print("✓ Planner router handles None state correctly")
        
        # Test research routing with None state
        result = _route_from_research(None)
        assert result == "chat", f"Expected 'chat' but got {result}"
        print("✓ Research router handles None state correctly")
        
        # Test research_web routing with None state
        result = _route_from_research_web(None)
        assert result == "executor", f"Expected 'executor' but got {result}"
        print("✓ Research web router handles None state correctly")
        
        # Test coder routing with None state
        result = _route_from_coder(None)
        assert result == "executor", f"Expected 'executor' but got {result}"
        print("✓ Coder router handles None state correctly")
        
        # Test critic routing with None state
        result = _route_from_critic(None)
        assert result == "chat", f"Expected 'chat' but got {result}"
        print("✓ Critic router handles None state correctly")
        
        return True
    except Exception as e:
        print(f"✗ Routing error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_routing_error_handling()
    sys.exit(0 if success else 1)
