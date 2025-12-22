#!/usr/bin/env python3
"""
Test to verify the graph building functionality
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_graph_build():
    """Test that the graph can be built without errors"""
    try:
        from llamia_v3_2.graph import build_llamia_graph
        graph = build_llamia_graph()
        assert graph is not None, "Graph should be built successfully"
        print("✓ Graph building test passed")
        return True
    except Exception as e:
        print(f"✗ Graph building test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_graph_build()
    sys.exit(0 if success else 1)
