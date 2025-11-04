#!/usr/bin/env python3
"""
Streamlit Dashboard for Material Science Table Analysis

This dashboard provides a web interface for analyzing material science tables
using the hosted LLM to extract composition and spall strength data.
"""

import streamlit as st
import json
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import time
import os
from material_analyzer import MaterialAnalyzer, LLMClient

# Page configuration
st.set_page_config(
    page_title="Material Science Table Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analysis_stats' not in st.session_state:
        st.session_state.analysis_stats = None
    if 'is_analyzing' not in st.session_state:
        st.session_state.is_analyzing = False
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None

def load_sample_data():
    """Load sample data for demonstration"""
    sample_data = {
        "total_papers": 153,
        "total_tables": 333,
        "analyzed_tables": 280,
        "matched_materials": 45,
        "unmatched_compositions": 23,
        "unmatched_properties": 12,
        "success_rate": 0.84
    }
    return sample_data

def display_header():
    """Display the main header"""
    st.markdown('<h1 class="main-header">🔬 Material Science Table Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Analyze research paper tables to extract material compositions and spall strength data using AI
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display sidebar with configuration options"""
    st.sidebar.header("⚙️ Configuration")
    
    # LLM Server Configuration
    st.sidebar.subheader("LLM Server")
    llm_url = st.sidebar.text_input(
        "Server URL", 
        value="http://localhost:8000",
        help="URL of the hosted LLM server"
    )
    
    # File Upload
    st.sidebar.subheader("📁 Data Input")
    uploaded_file = st.sidebar.file_uploader(
        "Upload pii_table_dict.pkl",
        type=['pkl'],
        help="Upload the pickle file containing extracted table data"
    )
    
    # Analysis Options
    st.sidebar.subheader("🔍 Analysis Options")
    max_papers = st.sidebar.slider(
        "Max Papers to Analyze",
        min_value=1,
        max_value=200,
        value=10,
        help="Limit the number of papers to analyze (for testing)"
    )
    
    return llm_url, uploaded_file, max_papers

def display_metrics(stats: Dict[str, Any]):
    """Display key metrics in cards"""
    if not stats:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📄 Total Papers",
            value=stats.get("total_papers", 0),
            delta=None
        )
    
    with col2:
        st.metric(
            label="📊 Total Tables",
            value=stats.get("total_tables", 0),
            delta=None
        )
    
    with col3:
        st.metric(
            label="✅ Analyzed Tables",
            value=stats.get("analyzed_tables", 0),
            delta=f"{stats.get('success_rate', 0):.1%} success rate"
        )
    
    with col4:
        st.metric(
            label="🔗 Matched Materials",
            value=stats.get("matched_materials", 0),
            delta=None
        )

def create_analysis_charts(stats: Dict[str, Any]):
    """Create visualization charts"""
    if not stats:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Success rate pie chart
        success_rate = stats.get("success_rate", 0)
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Successfully Analyzed', 'Failed Analysis'],
            values=[success_rate, 1 - success_rate],
            marker_colors=['#2E8B57', '#DC143C']
        )])
        fig_pie.update_layout(
            title="Analysis Success Rate",
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Material matching bar chart
        categories = ['Matched', 'Unmatched Comp', 'Unmatched Prop']
        values = [
            stats.get("matched_materials", 0),
            stats.get("unmatched_compositions", 0),
            stats.get("unmatched_properties", 0)
        ]
        
        fig_bar = px.bar(
            x=categories,
            y=values,
            title="Material Matching Results",
            color=categories,
            color_discrete_map={
                'Matched': '#2E8B57',
                'Unmatched Comp': '#FF8C00',
                'Unmatched Prop': '#DC143C'
            }
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

def display_detailed_results(results: List[Dict[str, Any]]):
    """Display detailed analysis results"""
    if not results:
        return
    
    st.subheader("📋 Detailed Results")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Summary", "🔍 Paper Details", "📈 Material Data"])
    
    with tab1:
        # Summary table
        summary_data = []
        for result in results:
            summary_data.append({
                "Paper ID": result.get("pii", ""),
                "DOI": result.get("doi", "")[:50] + "..." if len(result.get("doi", "")) > 50 else result.get("doi", ""),
                "Tables": result.get("total_tables", 0),
                "Analyzed": len(result.get("analyzed_tables", [])),
                "Matched": len(result.get("matched_materials", [])),
                "Unmatched Comp": len(result.get("unmatched_compositions", [])),
                "Unmatched Prop": len(result.get("unmatched_properties", []))
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
    
    with tab2:
        # Paper selection and details
        paper_options = [f"{r['pii']} - {r['doi'][:30]}..." for r in results]
        selected_paper = st.selectbox("Select Paper", paper_options)
        
        if selected_paper:
            paper_idx = paper_options.index(selected_paper)
            paper_data = results[paper_idx]
            
            st.write(f"**Paper ID:** {paper_data['pii']}")
            st.write(f"**DOI:** {paper_data['doi']}")
            st.write(f"**Total Tables:** {paper_data['total_tables']}")
            
            # Display analyzed tables
            if paper_data.get("analyzed_tables"):
                st.write("**Analyzed Tables:**")
                for i, table in enumerate(paper_data["analyzed_tables"]):
                    with st.expander(f"Table {i+1}: {table.get('table_caption', 'No caption')[:50]}..."):
                        st.json(table)
    
    with tab3:
        # Material data export
        all_materials = []
        for result in results:
            for material in result.get("matched_materials", []):
                material["paper_id"] = result["pii"]
                material["doi"] = result["doi"]
                all_materials.append(material)
        
        if all_materials:
            st.write(f"**Found {len(all_materials)} matched materials**")
            
            # Create DataFrame for export
            df_materials = pd.DataFrame(all_materials)
            
            # Display key columns
            display_cols = ["paper_id", "material_id", "composition", "spall_strength", "units", "confidence"]
            available_cols = [col for col in display_cols if col in df_materials.columns]
            
            st.dataframe(df_materials[available_cols], use_container_width=True)
            
            # Export options
            col1, col2 = st.columns(2)
            with col1:
                csv = df_materials.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="material_data.csv",
                    mime="text/csv"
                )
            
            with col2:
                json_data = df_materials.to_json(orient="records", indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name="material_data.json",
                    mime="application/json"
                )

def run_analysis(llm_url: str, file_data: bytes, max_papers: int):
    """Run the analysis process"""
    try:
        # Initialize analyzer
        llm_client = LLMClient(llm_url)
        analyzer = MaterialAnalyzer(llm_client)
        
        # Save uploaded file temporarily
        temp_path = "temp_data.pkl"
        with open(temp_path, "wb") as f:
            f.write(file_data)
        
        # Load and analyze data
        data = analyzer.load_tables(temp_path)
        
        if not data:
            st.error("Failed to load data from uploaded file")
            return None, None
        
        # Limit papers for analysis
        limited_data = dict(list(data.items())[:max_papers])
        
        # Run analysis
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        total_papers = len(limited_data)
        
        for i, (pii, tables) in enumerate(limited_data.items()):
            status_text.text(f"Analyzing paper {i+1}/{total_papers}: {pii}")
            
            try:
                # Extract DOI
                doi = ""
                if tables and tables[0].get("doi"):
                    doi = tables[0]["doi"]
                
                paper_result = analyzer.analyze_paper(pii, tables, doi)
                results.append(paper_result)
                
            except Exception as e:
                st.warning(f"Error analyzing paper {pii}: {e}")
                continue
            
            progress_bar.progress((i + 1) / total_papers)
            time.sleep(0.1)  # Small delay for UI updates
        
        # Clean up temp file
        os.remove(temp_path)
        
        # Get summary stats
        analyzer.results = results
        stats = analyzer.get_summary_stats()
        
        status_text.text("Analysis completed!")
        progress_bar.progress(1.0)
        
        return results, stats
        
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return None, None

def main():
    """Main dashboard function"""
    initialize_session_state()
    display_header()
    
    # Sidebar configuration
    llm_url, uploaded_file, max_papers = display_sidebar()
    
    # Main content area
    if uploaded_file is not None:
        st.success("✅ File uploaded successfully!")
        
        # Analysis button
        if st.button("🚀 Start Analysis", type="primary"):
            st.session_state.is_analyzing = True
            
            with st.spinner("Running analysis..."):
                results, stats = run_analysis(llm_url, uploaded_file.getvalue(), max_papers)
                
                if results is not None and stats is not None:
                    st.session_state.analysis_results = results
                    st.session_state.analysis_stats = stats
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.is_analyzing = False
                    
                    st.success("🎉 Analysis completed successfully!")
                else:
                    st.session_state.is_analyzing = False
                    st.error("❌ Analysis failed. Please check your configuration and try again.")
    
    # Display results if available
    if st.session_state.analysis_results and st.session_state.analysis_stats:
        st.markdown("---")
        
        # Display metrics
        display_metrics(st.session_state.analysis_stats)
        
        # Display charts
        st.markdown("---")
        create_analysis_charts(st.session_state.analysis_stats)
        
        # Display detailed results
        st.markdown("---")
        display_detailed_results(st.session_state.analysis_results)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🔬 Material Science Table Analyzer | Powered by LLM</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
