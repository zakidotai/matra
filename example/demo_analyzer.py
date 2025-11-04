#!/usr/bin/env python3
"""
Simplified Demo for Material Analyzer

This demo shows the material analyzer working with actual data
without requiring the LLM server to be perfectly configured.
"""

import pickle
import json
import pandas as pd
from typing import Dict, List, Any

def analyze_table_simple(table_data: List[List[str]], caption: str = "") -> Dict[str, Any]:
    """
    Simple rule-based analysis for demonstration
    This mimics what the LLM would do
    """
    if not table_data:
        return {"error": "Empty table"}
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(table_data)
    
    # Look for composition indicators
    composition_keywords = ['composition', 'element', 'wt%', 'weight', 'content', 'alloy']
    spall_keywords = ['spall', 'strength', 'dynamic', 'impact', 'fragmentation', 'resistance']
    
    has_composition = any(
        any(keyword.lower() in str(cell).lower() for keyword in composition_keywords)
        for cell in df.values.flatten()
    )
    
    has_spall_strength = any(
        any(keyword.lower() in str(cell).lower() for keyword in spall_keywords)
        for cell in df.values.flatten()
    )
    
    # Extract materials (simplified)
    materials = []
    
    if has_composition and len(df) > 1:
        # Look for element data
        for i, row in df.iterrows():
            if i == 0:  # Skip header
                continue
            
            # Check if row contains numeric values (likely composition)
            numeric_values = []
            for cell in row:
                try:
                    val = float(str(cell).replace('%', '').replace('Bal.', '0'))
                    numeric_values.append(val)
                except:
                    pass
            
            if len(numeric_values) >= 2:  # At least 2 elements
                material = {
                    "material_id": f"Sample_{i}",
                    "composition": {},
                    "spall_strength": None,
                    "units": None,
                    "confidence": 0.8,
                    "notes": f"Extracted from row {i}"
                }
                
                # Map elements to values (simplified)
                if len(df.columns) > 1:
                    for j, col in enumerate(df.columns):
                        if j < len(numeric_values):
                            element = str(col).strip()
                            material["composition"][element] = numeric_values[j]
                
                materials.append(material)
    
    return {
        "has_composition": has_composition,
        "has_spall_strength": has_spall_strength,
        "materials": materials,
        "table_type": "both" if (has_composition and has_spall_strength) else 
                     "composition_only" if has_composition else
                     "property_only" if has_spall_strength else "neither",
        "common_identifiers": ["Sample", "ID", "Name", "Element"]
    }

def demo_analysis():
    """Run a demonstration analysis"""
    print("🔬 Material Science Table Analyzer Demo")
    print("=" * 50)
    
    # Load the actual data
    print("Loading pii_table_dict.pkl...")
    try:
        with open('corpus_spall/pii_table_dict.pkl', 'rb') as f:
            data = pickle.load(f)
        print(f"✅ Loaded {len(data)} papers")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Analyze first few papers
    print("\nAnalyzing first 3 papers...")
    results = []
    
    for i, (pii, tables) in enumerate(list(data.items())[:3]):
        print(f"\n📄 Paper {i+1}: {pii}")
        print(f"   Tables: {len(tables)}")
        
        paper_result = {
            "pii": pii,
            "doi": tables[0].get("doi", "") if tables else "",
            "total_tables": len(tables),
            "analyzed_tables": [],
            "matched_materials": [],
            "unmatched_compositions": [],
            "unmatched_properties": []
        }
        
        for j, table_info in enumerate(tables):
            table_data = table_info.get("act_table", [])
            caption = table_info.get("caption", "")
            
            if not table_data:
                continue
            
            print(f"   📊 Analyzing table {j+1}: {caption[:50]}...")
            
            # Analyze table
            analysis = analyze_table_simple(table_data, caption)
            analysis["table_index"] = j
            analysis["table_caption"] = caption
            
            paper_result["analyzed_tables"].append(analysis)
            
            # Extract materials
            for material in analysis.get("materials", []):
                if material.get("spall_strength"):
                    paper_result["matched_materials"].append(material)
                elif material.get("composition"):
                    paper_result["unmatched_compositions"].append(material)
        
        results.append(paper_result)
        
        # Show summary
        print(f"   ✅ Analyzed: {len(paper_result['analyzed_tables'])} tables")
        print(f"   🔗 Matched materials: {len(paper_result['matched_materials'])}")
        print(f"   📋 Unmatched compositions: {len(paper_result['unmatched_compositions'])}")
    
    # Overall summary
    print("\n" + "=" * 50)
    print("📊 DEMO SUMMARY")
    print("=" * 50)
    
    total_papers = len(results)
    total_tables = sum(r["total_tables"] for r in results)
    analyzed_tables = sum(len(r["analyzed_tables"]) for r in results)
    matched_materials = sum(len(r["matched_materials"]) for r in results)
    unmatched_compositions = sum(len(r["unmatched_compositions"]) for r in results)
    
    print(f"Total papers analyzed: {total_papers}")
    print(f"Total tables: {total_tables}")
    print(f"Successfully analyzed: {analyzed_tables}")
    print(f"Matched materials: {matched_materials}")
    print(f"Unmatched compositions: {unmatched_compositions}")
    print(f"Success rate: {analyzed_tables/total_tables:.1%}")
    
    # Show sample results
    print("\n📋 SAMPLE RESULTS:")
    for result in results:
        if result["matched_materials"] or result["unmatched_compositions"]:
            print(f"\nPaper: {result['pii']}")
            print(f"DOI: {result['doi']}")
            
            if result["matched_materials"]:
                print("Matched materials:")
                for material in result["matched_materials"][:2]:  # Show first 2
                    print(f"  - {material['material_id']}: {material['composition']}")
            
            if result["unmatched_compositions"]:
                print("Unmatched compositions:")
                for material in result["unmatched_compositions"][:2]:  # Show first 2
                    print(f"  - {material['material_id']}: {material['composition']}")
    
    # Save results
    output_file = "demo_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nTo run the full LLM-powered analysis:")
    print("1. Ensure LLM server is running on localhost:8000")
    print("2. Run: python material_analyzer.py --pkl_path corpus_spall/pii_table_dict.pkl")
    print("3. Or launch dashboard: streamlit run streamlit_dashboard.py")

if __name__ == "__main__":
    demo_analysis()
