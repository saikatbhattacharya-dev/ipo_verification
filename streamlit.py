import streamlit as st
import tempfile
import os
from pathlib import Path
import logging
from typing import List

# Import your existing modules
from agno.document import Document 
from agno.knowledge.document import DocumentKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from agno.vectordb.lancedb import LanceDb
from agno.models.google import Gemini
from agno.embedder.google import GeminiEmbedder
from youtube_transcript_api import YouTubeTranscriptApi
from agno.agent import Agent
from llama_parse import LlamaParse
import os
from dotenv import load_dotenv
load_dotenv()

LLAMA_KEY = os.getenv("LLAMA_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure logging
logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Your existing functions (keeping them as they are)
def parsing_using_llamaparse(filepath):
    parser = LlamaParse(api_key=LLAMA_KEY, result_type="markdown")
    docs = parser.load_data(filepath)
    return docs

def convert_llama_docs_to_agno(llama_docs):
    agno_docs = []
    for doc in llama_docs:
        agno_doc = Document(
            id=doc.id_,
            content=doc.text_resource.text,
            meta_data={'source': "prospectus"}
        )
        agno_docs.append(agno_doc)
    return agno_docs

# def push_into_kb(agno_docs):
#     knowledge_base = DocumentKnowledgeBase(
#         documents=agno_docs,
#         vector_db=ChromaDb(collection="documents", path="tmp/chromadb",embedder=GeminiEmbedder(api_key=GOOGLE_API_KEY)),
#         embedder=GeminiEmbedder(api_key=GOOGLE_API_KEY)
#     )
#     knowledge_base.load(recreate=True)
#     knowledge_base.num_documents = 100
#     return knowledge_base

def push_into_kb(agno_docs):
    knowledge_base = DocumentKnowledgeBase(
        documents=agno_docs,
        vector_db=LanceDb(table_name="documents", uri="tmp/lancedb",embedder=GeminiEmbedder(api_key=GOOGLE_API_KEY)),
        embedder=GeminiEmbedder(api_key=GOOGLE_API_KEY)
    )
    knowledge_base.load(recreate=True)
    knowledge_base.num_documents = 100
    return knowledge_base

def create_prospectus_agent(knowledge_base):
    return Agent(
        model=Gemini(id="gemini-2.0-flash-exp", api_key=GOOGLE_API_KEY),
        knowledge=knowledge_base,
        instructions=[
            "You are a specialized financial document verification expert with access to company prospectus data.",
            "Your primary role is to cross-verify claims made in video transcripts against official company documentation.",
            
            "Core Responsibilities:",
            "- Systematically verify each claim extracted from video transcripts",
            "- Search the prospectus knowledge base for corresponding official information",
            "- Identify discrepancies, inconsistencies, or unsubstantiated claims",
            "- Provide authoritative fact-checking based on official documents",
            
            "Verification Process:",
            "1. Receive extracted claims from the YouTube transcript analysis",
            "2. For each claim, search the knowledge base using relevant keywords and concepts",
            "3. Compare transcript statements with official prospectus information",
            "4. Document exact matches, partial matches, and contradictions",
            "5. Note any claims not addressed in the prospectus",
            
            "Search Strategy:",
            "- Use multiple search terms for each claim (synonyms, related concepts)",
            "- Search for financial figures, dates, strategic initiatives separately",
            "- Look for both explicit statements and implied information in documents",
            "- Cross-reference different sections of the prospectus for comprehensive verification",
            
            "Analysis Categories:",
            "- Financial Data: Revenue, turnover, profit margins, growth rates",
            "- IPO Information: Listing details, offering size, valuation, timeline",
            "- Company Vision: Mission statements, strategic objectives, market positioning",
            "- Forward-Looking Statements: Future plans, projections, expansion goals",
            "- Regulatory Compliance: Disclosures, risk factors, legal obligations",
            
            "Response Framework:",
            "For each verified claim, provide:",
            "- **CLAIM**: Original statement from transcript",
            "- **PROSPECTUS REFERENCE**: Exact section/page where information is found",
            "- **VERIFICATION STATUS**: ✅ Confirmed / ⚠️ Partially Verified / ❌ Contradicted / ❓ Not Found",
            "- **DETAILS**: Specific comparison between transcript claim and prospectus data",
            "- **DISCREPANCY ANALYSIS**: If differences exist, explain the nature and potential reasons",
            
            "Quality Standards:",
            "- Cite specific document sections, page numbers, or paragraph references when available",
            "- Distinguish between exact matches and reasonable interpretations",
            "- Flag any material discrepancies that could affect investor decisions",
            "- Highlight missing information that should be in the prospectus",
            "- Provide confidence levels for your verification assessments",
            
            "Red Flags to Identify:",
            "- Financial figures that don't match official records",
            "- Timeline discrepancies for IPO or business milestones",
            "- Overstated market positions or competitive advantages",
            "- Unrealistic projections not supported by prospectus data",
            "- Missing mandatory disclosures or risk factors",
            
            "Final Deliverable:",
            "- Comprehensive verification report with summary of findings",
            "- Overall credibility assessment of the video transcript",
            "- List of claims requiring further investigation",
            "- Recommendations for investors based on discrepancy analysis"
        ],
        search_knowledge=True,
        show_tool_calls=True,
        markdown=True,
    )

def get_yt_transcript(url):
    id = url.split("=")[-1]
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(id)
    return transcript

def get_formatted_transcript(transcript):
    text = ""
    for snippet in transcript:
        text += snippet.text
    return text

def create_yt_agent():
    return Agent(
        model=Gemini(id="gemini-2.0-flash-exp", api_key=GOOGLE_API_KEY),
        instructions=[
            "You are an AI agent specialized in analyzing video transcripts from company representatives.",
            "Your primary objective is to extract and structure key corporate information from speech transcripts.",
            
            "Key Information to Extract:",
            "- Company vision and mission statements",
            "- Annual turnover and revenue figures", 
            "- IPO details (listing date, exchange, offering size)",
            "- Financial metrics and performance indicators",
            "- Strategic initiatives and future plans",
            "- Market position and competitive advantages",
            
            "Analysis Guidelines:",
            "- Carefully read through the entire transcript before extracting information",
            "- Focus on factual statements made by company representatives",
            "- Distinguish between confirmed facts and forward-looking statements",
            "- Note any specific numbers, dates, and quantitative data mentioned",
            
            "Response Format:",
            "- Structure your response with clear headings and bullet points",
            "- For each category, provide the extracted information with relevant context",
            "- If specific information is not mentioned in the transcript, clearly state 'Not provided in transcript'",
            "- Include direct quotes when relevant to support your findings",
            
            "Verification Note:",
            "- Flag any claims that would benefit from cross-verification with official company prospectus",
            "- Highlight discrepancies or unusually bold claims that merit fact-checking",
            "- Provide a confidence level for each piece of extracted information"
        ],
        show_tool_calls=True,
        markdown=True,
    )
def create_quality_agent():
    return Agent(
        model=Gemini(id="gemini-2.0-flash-exp", api_key=GOOGLE_API_KEY),
        instructions=[
            "You are a quality assurance agent for AI-generated financial analysis reports.",
            "Your job is to assess whether the transcript analysis and verification report are:",
            "- Factually consistent",
            "- Free from hallucinations",
            "- Complete and well-structured",
            "",
            "Scoring Guidelines (0–100):",
            "- 90–100: Excellent (clear, accurate, complete, no hallucinations)",
            "- 70–89: Good (mostly accurate, minor gaps)",
            "- 50–69: Fair (some inaccuracies or incomplete sections)",
            "- Below 50: Poor (hallucinated, unclear, or missing key info)",
            "",
            "Output Format (strict JSON):",
            "{",
            '  "quality_score": <integer>,',
            '  "issues": "<brief description of problems found>"',
            "}"
        ],
        show_tool_calls=True,
        markdown=False,
    )

def workflow_streamlit(file_path: str, video_urls: List[str], progress_bar, status_text):
    """Modified workflow function for Streamlit with progress tracking"""
    
    try:
        status_text.text("🔄 Parsing PDF document...")
        progress_bar.progress(10)
        llama_docs = parsing_using_llamaparse(file_path)
        
        status_text.text("🔄 Converting documents to knowledge base format...")
        progress_bar.progress(20)
        agno_docs = convert_llama_docs_to_agno(llama_docs)
        
        status_text.text("🔄 Building knowledge base...")
        progress_bar.progress(30)
        pdf_kb = push_into_kb(agno_docs)
        
        status_text.text("🔄 Creating document verification agent...")
        progress_bar.progress(40)
        pdf_agent = create_prospectus_agent(pdf_kb)
        
        final_formatted_transcript = ""
        for i, video_url in enumerate(video_urls):
            status_text.text(f"🔄 Processing video {i+1}/{len(video_urls)}...")
            progress_bar.progress(50 + (i * 20 // len(video_urls)))
            
            transcript = get_yt_transcript(video_url)
            formatted_transcript = get_formatted_transcript(transcript)
            final_formatted_transcript += formatted_transcript + "\n\n"
        
        status_text.text("🔄 Creating YouTube transcript agent...")
        progress_bar.progress(70)
        yt_agent = create_yt_agent()
        
        if len(final_formatted_transcript) == 0:
            return "⚠️ No transcript content found to process"
        
        status_text.text("🔄 Analyzing transcript content...")
        progress_bar.progress(80)
        yt_agent_res = yt_agent.run(final_formatted_transcript)
        
        # status_text.text("🔄 Cross-verifying with document...")
        # progress_bar.progress(90)
        # pdf_agent_res = pdf_agent.run(yt_agent_res.content)
        
        # progress_bar.progress(100)
        # status_text.text("✅ Analysis completed successfully!")
        
        # return {
        #     "transcript_analysis": yt_agent_res.content,
        #     "verification_report": pdf_agent_res.content
        # }
        status_text.text("🔄 Cross-verifying with document...")
        progress_bar.progress(90)
        pdf_agent_res = pdf_agent.run(yt_agent_res.content)
        
        # Step 3: Quality Check
        status_text.text("🔎 Checking response quality...")
        progress_bar.progress(95)
        quality_agent = create_quality_agent()
        quality_res = quality_agent.run({
            "transcript_analysis": yt_agent_res.content,
            "verification_report": pdf_agent_res.content
        })

        import json
        try:
            quality_data = json.loads(quality_res.content)
            score = quality_data.get("quality_score", 0)
        except Exception:
            score = 0

        # Retry if quality is poor
        if score < 50:
            logging.warning("⚠️ Low quality score detected. Retrying process once...")
            status_text.text("⚠️ Low quality detected. Retrying...")
            
            yt_agent_res = yt_agent.run(final_formatted_transcript)
            pdf_agent_res = pdf_agent.run(yt_agent_res.content)

            # Re-check quality
            quality_res = quality_agent.run({
                "transcript_analysis": yt_agent_res.content,
                "verification_report": pdf_agent_res.content
            })
            try:
                quality_data = json.loads(quality_res.content)
                score = quality_data.get("quality_score", 0)
            except Exception:
                score = 0
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis completed successfully!")
        
        return {
            "transcript_analysis": yt_agent_res.content,
            "verification_report": pdf_agent_res.content,
            "quality_score": score,
            "quality_feedback": quality_res.content
        }

    except Exception as e:
        logging.error(f"❌ Error in workflow: {e}", exc_info=True)
        return f"❌ Error occurred: {str(e)}"


# Streamlit App
def main():
    st.set_page_config(
        page_title="IPO Verification System",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 IPO Verification System")
    st.markdown("**Cross-verify YouTube video claims against official company prospectus documents**")
    
    # Sidebar for inputs
    st.sidebar.header("📋 Input Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Company Prospectus (PDF)",
        type=['pdf'],
        help="Upload the official company prospectus document"
    )
    
    # YouTube URLs input
    st.sidebar.subheader("🎥 YouTube Video URLs")
    video_urls = []
    
    # Dynamic URL input
    if "url_count" not in st.session_state:
        st.session_state.url_count = 1
    
    for i in range(st.session_state.url_count):
        url = st.sidebar.text_input(
            f"YouTube URL {i+1}:",
            key=f"url_{i}",
            placeholder="https://www.youtube.com/watch?v=example"
        )
        if url:
            video_urls.append(url)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("➕ Add URL"):
            st.session_state.url_count += 1
            st.rerun()
    
    with col2:
        if st.button("➖ Remove URL") and st.session_state.url_count > 1:
            st.session_state.url_count -= 1
            st.rerun()
    
    # Process button
    process_btn = st.sidebar.button(
        "🚀 Start Verification",
        type="primary",
        disabled=not (uploaded_file and video_urls)
    )
    
    # Main content area
    if not uploaded_file or not video_urls:
        st.info("👆 Please upload a PDF document and provide at least one YouTube URL to get started.")
        
        # Show demo/instructions
        st.subheader("🔍 How it works:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **1. Upload Document**
            - Upload company prospectus PDF
            - System extracts and indexes content
            - Creates searchable knowledge base
            """)
        
        with col2:
            st.markdown("""
            **2. Analyze Videos**
            - Fetches YouTube transcripts
            - Extracts key claims and statements
            - Identifies financial data and projections
            """)
        
        with col3:
            st.markdown("""
            **3. Cross-Verify**
            - Compares video claims with document
            - Identifies discrepancies
            - Generates verification report
            """)
    
    else:
        if process_btn:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_file_path = tmp_file.name
            
            try:
                # Progress tracking
                progress_container = st.container()
                with progress_container:
                    st.subheader("🔄 Processing Status")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                # Run workflow
                results = workflow_streamlit(temp_file_path, video_urls, progress_bar, status_text)
                
                if isinstance(results, dict):
                    # Display results in tabs
                    st.success("✅ Analysis completed successfully!")
                    
                    tab1, tab2 = st.tabs(["📹 Transcript Analysis", "🔍 Verification Report"])
                    
                    with tab1:
                        st.subheader("YouTube Transcript Analysis")
                        st.markdown(results["transcript_analysis"])
                    
                    with tab2:
                        st.subheader("Document Verification Report")
                        st.markdown(results["verification_report"])
                        
                    st.subheader("📊 Quality Check")
                    st.write(f"**Quality Score:** {results['quality_score']} / 100")
                    st.markdown(f"**Quality Feedback:** {results['quality_feedback']}")
                    
                    # Download option
                    st.subheader("💾 Export Results")
                    
                    # Combine results for download
                    combined_report = f"""# Document Verification Report

## YouTube Transcript Analysis
{results["transcript_analysis"]}

---

## Document Verification Report
{results["verification_report"]}

---
*Generated by Document Verification System*
"""
                    
                    st.download_button(
                        label="📥 Download Full Report",
                        data=combined_report,
                        file_name="verification_report.md",
                        mime="text/markdown"
                    )
                else:
                    st.error(f"❌ Processing failed: {results}")
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                logging.error(f"Streamlit error: {e}", exc_info=True)
            
            finally:
                # Cleanup temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Tip:** Ensure YouTube videos have available transcripts for analysis.")

if __name__ == "__main__":
    main()
