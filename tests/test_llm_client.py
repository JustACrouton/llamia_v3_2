#!/usr/bin/env python3
"""
Test to verify LLM client functionality
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_client_import():
    """Test that the LLM client can be imported"""
    try:
        from llamia_v3_2.llm_client import get_client
        client = get_client()
        # We're just testing import and basic instantiation here
        # Not actually calling the API to avoid external dependencies
        assert client is not None, "Client should be created successfully"
        print("✓ LLM client import test passed")
        return True
    except Exception as e:
        print(f"✗ LLM client import test failed: {e}")
        return False

def test_model_config():
    """Test that model configuration works"""
    try:
        from llamia_v3_2.config import DEFAULT_CONFIG
        chat_model = DEFAULT_CONFIG.chat_model
        assert chat_model is not None, "Chat model should exist"
        assert chat_model.model == "qwen3:32b", "Chat model should be qwen3:32b"
        assert chat_model.provider == "openai_compatible", "Provider should be openai_compatible"
        print("✓ Model configuration test passed")
        return True
    except Exception as e:
        print(f"✗ Model configuration test failed: {e}")
        return False

if __name__ == "__main__":
    results = [
        test_client_import(),
        test_model_config()
    ]
    success = all(results)
    sys.exit(0 if success else 1)
