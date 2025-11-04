#!/usr/bin/env python3
"""
Test script for Material Analyzer

This script tests the material analyzer with a small sample of data
"""

import json
import pickle
from material_analyzer import MaterialAnalyzer, LLMClient

def test_llm_connection():
    """Test connection to the LLM server"""
    print("Testing LLM connection...")
    
    try:
        client = LLMClient("http://localhost:8000")
        
        # Test with a simple prompt
        test_table = [
            ["Sample", "Composition", "Spall Strength (MPa)"],
            ["A1", "Al-5%Mg", "450"],
            ["A2", "Al-10%Mg", "520"]
        ]
        
        result = client.analyze_table(test_table, "Test table for spall strength", "10.1000/test.001")
        
        if "error" in result:
            print(f"❌ LLM Error: {result['error']}")
            return False
        else:
            print("✅ LLM connection successful!")
            print(f"Response: {json.dumps(result, indent=2)}")
            return True
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_analyzer():
    """Test the analyzer with sample data"""
    print("\nTesting Material Analyzer...")
    
    try:
        # Load the actual data
        analyzer = MaterialAnalyzer(LLMClient("http://localhost:8000"))
        data = analyzer.load_tables("corpus_spall/pii_table_dict.pkl")
        
        if not data:
            print("❌ Failed to load data")
            return False
        
        print(f"✅ Loaded {len(data)} papers")
        
        # Test with first paper only
        first_pii = list(data.keys())[0]
        first_tables = data[first_pii]
        
        print(f"Testing with paper: {first_pii}")
        print(f"Number of tables: {len(first_tables)}")
        
        # Analyze first paper
        result = analyzer.analyze_paper(first_pii, first_tables)
        
        print("✅ Analysis completed!")
        print(f"Results: {json.dumps(result, indent=2)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Material Analyzer Test Suite")
    print("=" * 50)
    
    # Test LLM connection
    if not test_llm_connection():
        print("\n❌ LLM connection test failed. Please check:")
        print("1. LLM server is running on localhost:8000")
        print("2. Server is accessible")
        print("3. Model is loaded correctly")
        return
    
    # Test analyzer
    if not test_analyzer():
        print("\n❌ Analyzer test failed. Please check:")
        print("1. pii_table_dict.pkl file exists")
        print("2. File contains valid data")
        return
    
    print("\n🎉 All tests passed!")
    print("\nYou can now run:")
    print("1. Command line: python material_analyzer.py --pkl_path corpus_spall/pii_table_dict.pkl")
    print("2. Dashboard: streamlit run streamlit_dashboard.py")

if __name__ == "__main__":
    main()
