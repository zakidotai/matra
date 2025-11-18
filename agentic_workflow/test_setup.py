#!/usr/bin/env python3
"""
Test script to verify the agentic workflow setup
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from agentic_workflow import config, tool_registry, agent, workflow
        from agentic_workflow.tools import (
            crossref_tool, deduplication_tool, 
            download_tool, database_tool, organize_tool
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_tool_registry():
    """Test that tools are registered"""
    print("\nTesting tool registry...")
    try:
        from agentic_workflow.tools import (
            crossref_tool, deduplication_tool,
            download_tool, database_tool, organize_tool
        )
        from agentic_workflow.tool_registry import get_registry
        
        registry = get_registry()
        tools = registry.list_tools()
        
        expected_tools = [
            'crossref_search',
            'combine_and_deduplicate',
            'download_articles',
            'build_database',
            'organize_xmls'
        ]
        
        missing = set(expected_tools) - set(tools)
        if missing:
            print(f"✗ Missing tools: {missing}")
            return False
        
        print(f"✓ All {len(tools)} tools registered: {', '.join(tools)}")
        return True
    except Exception as e:
        print(f"✗ Tool registry test failed: {e}")
        return False

def test_config():
    """Test configuration"""
    print("\nTesting configuration...")
    try:
        from agentic_workflow.config import Config
        
        config = Config()
        errors = config.validate()
        
        if errors:
            print(f"⚠ Configuration warnings (expected if env vars not set):")
            for error in errors:
                print(f"  - {error}")
        else:
            print("✓ Configuration valid")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_workflow():
    """Test workflow initialization"""
    print("\nTesting workflow...")
    try:
        from agentic_workflow.config import Config
        from agentic_workflow.workflow import Workflow
        
        config = Config()
        workflow = Workflow(config)
        print("✓ Workflow initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Workflow test failed: {e}")
        return False

def main():
    print("="*60)
    print("Agentic Workflow Setup Test")
    print("="*60)
    
    tests = [
        test_imports,
        test_tool_registry,
        test_config,
        test_workflow
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*60)
    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

