import streamlit as st
import fitz
import concurrent.futures
import time
import shutil
import os
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
import uuid
from datetime import datetime
import gc
import re
from typing import List, Dict, Tuple

# --- For Re-ranking ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

# --- FIX: Call model_rebuild() for FlashrankRerank ---
FlashrankRerank.model_rebuild() 

# --- Page Configuration ---
st.set_page_config(
    page_title="Enhanced Specification Analyzer",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
PARENT_DB_DIRECTORY = "chroma_dbs_temp"

# --- Enhanced PDF Text Extraction ---
def extract_text_from_pdf_enhanced(pdf_bytes) -> List[Document]:
    """
    Enhanced PDF extraction that preserves page structure and metadata.
    Extracts text page by page with better formatting preservation.
    """
    documents = []
    
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            # Extract text with layout preservation
            text_blocks = page.get_text("blocks")  # Get text blocks with position info
            
            # Sort blocks by vertical then horizontal position for reading order
            text_blocks = sorted(text_blocks, key=lambda b: (b[1], b[0]))
            
            page_text = ""
            for block in text_blocks:
                if block[6] == 0:  # Check if it's a text block (not image)
                    block_text = block[4].strip()
                    if block_text:
                        page_text += block_text + "\n"
            
            # Also get dictionary-based text as fallback
            dict_text = page.get_text("dict")
            
            # Create document with rich metadata
            if page_text.strip():
                doc_metadata = {
                    "page": page_num,
                    "source": "pdf_upload",
                    "total_pages": len(doc),
                    "char_count": len(page_text)
                }
                
                documents.append(Document(
                    page_content=page_text,
                    metadata=doc_metadata
                ))
    
    return documents

def clean_and_normalize_text(text: str) -> str:
    """Clean and normalize extracted text for better search."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might interfere
    text = re.sub(r'[^\w\s\-\.\,\:\;\(\)\[\]\/]', '', text)
    return text.strip()

def get_unique_db_directory(pdf_name):
    """Generates a unique directory path for a Chroma DB for a given PDF."""
    safe_name = pdf_name.replace(" ", "_").replace(".pdf", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"{PARENT_DB_DIRECTORY}/{safe_name}_{timestamp}_{unique_id}"

# --- Keywords and Specifications Structure ---
KEYWORDS_AND_SPECS = {
    "DIVISION 06/09 (FINISHES)": {
        "06 41 00 - ARCHITECTURAL WOOD CASEWORK": {
            "keywords": "ARCHITECTURAL WOOD CASEWORK",
            "specifications": [
                {"question": "Type of unit finish", "related_terms": "finish type, surface material, laminate type, veneer, paint, coating, construction material, material finish, casework finish, SOLID SURFACE, P-LAM, PLASTIC LAMINATE, HIGH PRESSURE LAMINATE, THERMOFOIL"},
                {"question": "Color and manufacturer of the finish", "related_terms": "finish color, material supplier, brand, shade, hue, maker, WILSONART, OMNOVA, FORMICA, finish manufacturer, color selection, finish brand"},
                {"question": "3form material requirements", "related_terms": "3form, translucent panels, resin panels, specific 3form series, Varia, Chroma, Koda, specialty panels, decorative resin"},
                {"question": "Any details related to headwalls in this section", "related_terms": "headwall specifications, architectural integration, mounting details, finishes on headwalls, casework around headwalls, headwall integration"}
            ]
        }
    },
    "DIVISION 10/11 (SPECIALTIES)": {
        "10 25 13 - PATIENT BED SERVICE WALLS": {
            "keywords": "PATIENT BED SERVICE WALLS (PBSW), PRE-FABRICATION, HEADWALL, BED LOCATOR, MODULAR, AMICO, medical gas service panel, patient room console, bed services unit, patient headwall",
            "specifications": [
                {"question": "Type of Headwall", "related_terms": "headwall type, configuration, design, integrated, surface mount, recessed, vertical, horizontal, custom, patient care wall, bed wall unit, headwall model"},
                {"question": "Preferred Manufacturer", "related_terms": "manufacturer, brand, preferred vendor, Amico, Stryker, Hill-Rom, Modular Services Company, Continental Metal Products (CMP), specified brand, approved manufacturer"},
                {"question": "Headwall Dimensions", "related_terms": "dimensions, size, length, height, width, unit size, overall footprint, measurements, headwall size"},
                {"question": "Headwall Finish details", "related_terms": "headwall finish, material, color, aesthetic, laminate, powder coat, solid surface, facing material, headwall cladding"},
                {"question": "List of specified Headwall Services", "related_terms": "services provided, gas outlets, electrical receptacles, data ports, nurse call, patient monitoring, communication services, utilities, power, oxygen, vacuum, medical air, general services, bed services"},
                {"question": "Headwall Accessories and scope", "related_terms": "accessories, components, medical rails, equipment management, IV poles, monitor mounts, storage bins, support arms, shelves, headwall attachments"}
            ]
        }
    },
    "DIVISION 22 (PLUMBING)": {
        "22 60 00 / 22 61 00 / 22 62 00 - GAS AND VACUUM SYSTEMS": {
            "keywords": "OXYGEN, MEDICAL AIR, VACUUM, GAS OUTLET, BEACON, D.I.S.S., COPPER PIPE, DISS, medical gas, central vacuum, compressed air, nitrous oxide, nitrogen, WAGD, AGSS, medical gas piping, gas supply",
            "specifications": [
                {"question": "Gas connection type", "related_terms": "connection type, DISS, Diameter Index Safety System, quick connect, Puritan-Bennett, Schrader, Chemetron, Ohmeda, NFPA 99 connection, type of outlet, gas terminal unit, gas connection fittings"},
                {"question": "Mention of 'BEACON' brand", "related_terms": "Beacon, brand, manufacturer, medical gas supplier, alarm panels, BeaconMedaes, Beacon Medical, specified gas supplier"},
                {"question": "Pipe sizes for Oxygen, Med Air, Vacuum", "related_terms": "pipe size, tubing diameter, oxygen pipe, medical air pipe, vacuum pipe, supply lines, medical gas line dimensions, pipe diameter, sizing of medical gas piping"},
                {"question": "Copper pipe type (K or L)", "related_terms": "copper pipe type, K-type, L-type, M-type, hard drawn copper, ASTM B819, medical gas tubing specification, copper tubing grade"}
            ]
        }
    },
    "DIVISION 26 (ELECTRICAL)": {
        "26 05 33 - RACEWAYS AND BOXES": {
            "keywords": "RACEWAYS, BACKBOX, CONDUIT, STAINLESS STEEL, junction box, pull box, electrical enclosure, metallic conduit, non-metallic conduit, wiring channels, electrical raceways",
            "specifications": [
                {"question": "What are the conduit size requirements for typical branch circuits?", "related_terms": "conduit size, low voltage, minimum conduit, EMT, raceway size, ½ inch, ¾ inch, 1 inch"},
                {"question": "What is the back box size specified for 1-gang and 2-gang devices?", "related_terms": "back box, mudring, gang box, device box, junction box, outlet box size"},
                {"question": "What is the specified back box and mudring size for low voltage systems?", "related_terms": "low voltage box, communication box, data outlet, mud ring size"},
                {"question": "Is there any specific information on EMT, FMC, or Metal Clad (MC) cable types?", "related_terms": "Electrical Metallic Tubing, Flexible Metal Conduit, MC Cable, armored cable"},
                {"question": "What are the approved brands for wiring devices and receptacles?", "related_terms": "manufacturer, approved vendor, Leviton, Hubbell, Pass & Seymour, Cooper"},
                {"question": "What types of receptacles are specified?", "related_terms": "outlet type, standard, GFCI, Ground Fault, tamperproof, tamper resistant, hospital grade, USB"},
                {"question": "What are the faceplate color and material requirements?", "related_terms": "wall plate, cover plate, color, ivory, white, red, material, nylon, stainless steel"},
                {"question": "Are there project-specific device notes, such as for VA or Children's hospital projects?", "related_terms": "self-illuminated receptacles, tamperproof devices, project specific, VA, hospital"},
                {"question": "What switch types and configurations are required?", "related_terms": "light switch, SPST, 3-way, 4-way, momentary, dimmer, keyed switch"},
                {"question": "What are the voltage and amperage ratings for receptacles and switches?", "related_terms": "voltage, amperage, 120V, 277V, 15A, 20A, rating"}
            ]
        },
        "26 27 26 - WIRING DEVICES": {
            "keywords": "WIRING DEVICES, TAMPERPROOF, GFCI, HOSPITAL GRADE, RECEPTACLE, SWITCH, HUBBELL, LEVITON, outlets, power receptacles, wall plates, dimmers, electrical outlets, wiring accessories",
            "specifications": [
                {"question": "Receptacle brand", "related_terms": "receptacle manufacturer, outlet brand, Hubbell, Leviton, Pass & Seymour, Cooper, specific brand, electrical outlet manufacturer"},
                {"question": "Receptacle type (Tamperproof, GFCI, etc.)", "related_terms": "receptacle functionality, tamper-resistant, ground fault circuit interrupter, surge protective, isolated ground, standard, hospital grade, patient care area receptacle, duplex outlet, outlet type"},
                {"question": "Faceplate color and material", "related_terms": "faceplate finish, wall plate color, material, stainless steel, plastic, ivory, white, almond, cover plate, finish of faceplate"},
                {"question": "Switch types", "related_terms": "switch functionality, toggle switch, rocker switch, dimmer switch, occupancy sensor switch, 3-way, 4-way, single pole, double pole, light switch types"}
            ]
        }
    },
    "DIVISION 27 (COMMUNICATIONS)": {
        "27 09 04 / 27 52 23 - LOW VOLTAGE & NURSE CALL": {
            "keywords": "LOW VOLTAGE, NURSE CALL, CODE BLUE, CONDUIT, BACKBOX, data cabling, communication systems, patient communication, emergency call, call systems, telecom infrastructure",
            "specifications": [
                {"question": "Low voltage conduit size", "related_terms": "low voltage raceway size, data conduit diameter, communication conduit, telecom conduit, low voltage piping dimensions"},
                {"question": "Nurse Call system brand", "related_terms": "nurse call manufacturer, patient call system brand, Rauland-Borg, Hill-Rom, Ascom, Dukane, Tektone, specified nurse call vendor, call system brand"},
                {"question": "Code Blue system brand and requirements", "related_terms": "Code Blue system, emergency call system, stat call, requirements, functionality, call for help, urgent call, code blue activation"}
            ]
        }
    }
}

# --- Utility Function to Clear Chroma DB ---
def clear_parent_chroma_db_directory(directory=PARENT_DB_DIRECTORY):
    """Clears all temporary Chroma DB directories."""
    if os.path.exists(directory):
        try:
            shutil.rmtree(directory)
        except Exception as e:
            st.warning(f"Could not clear old DB directory: {e}")
    os.makedirs(directory, exist_ok=True)

def format_docs(docs):
    """Formats retrieved document chunks with page numbers."""
    formatted = []
    for doc in docs:
        page_num = doc.metadata.get("page", "N/A")
        content = doc.page_content
        formatted.append(f"[Page {page_num}]\n{content}")
    return "\n\n---\n\n".join(formatted)

def run_analysis_for_section_single_spec(
    rag_chain, 
    division_title, 
    section_title, 
    spec_question, 
    retrieval_query_for_this_spec,
    thinking_mode: str
):
    """Runs the RAG chain for a single specification question."""
    input_data = {
        "query": retrieval_query_for_this_spec, 
        "questions": spec_question,
        "thinking_mode": thinking_mode
    }
    try:
        analysis_result = rag_chain.invoke(input_data)
        return {
            "Division": division_title,
            "Section": section_title,
            "Specification": spec_question,
            "Result": analysis_result
        }
    except Exception as e:
        return {
            "Division": division_title,
            "Section": section_title,
            "Specification": spec_question,
            "Result": f"Error during analysis: {str(e)}"
        }

def process_and_analyze_pdf(
    pdf_file, 
    vector_store, 
    llm, 
    embeddings,
    chunk_size: int,
    chunk_overlap: int,
    retriever_k: int,
    rerank_top_n: int,
    thinking_mode: str,
    temperature: float
):
    """Orchestrates the PDF processing and analysis workflow."""
    timings = []
    pdf_filename = pdf_file.name
    
    # Step 1: Enhanced Text Extraction
    with st.status(f"Step 1/4: Extracting text from '{pdf_filename}' with enhanced parser...", expanded=True) as status:
        s1_start = time.perf_counter()
        pdf_bytes = pdf_file.getvalue()
        page_documents = extract_text_from_pdf_enhanced(pdf_bytes)
        s1_end = time.perf_counter()
        duration1 = s1_end - s1_start
        timings.append(("Step 1: Enhanced Text Extraction", duration1))
        
        if not page_documents:
            status.update(label="Text extraction failed. No content found.", state="error")
            return None, timings
        
        total_chars = sum(len(doc.page_content) for doc in page_documents)
        status.update(
            label=f"Step 1/4: Extracted {len(page_documents)} pages ({total_chars:,} characters) in {duration1:.2f}s", 
            state="complete"
        )

    # Step 2: Smart Chunking
    with st.status("Step 2/4: Intelligent document chunking...", expanded=True) as status:
        s2_start = time.perf_counter()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        all_chunks = []
        for page_doc in page_documents:
            chunks = text_splitter.split_text(page_doc.page_content)
            for chunk in chunks:
                cleaned_chunk = clean_and_normalize_text(chunk)
                if cleaned_chunk:  # Only add non-empty chunks
                    all_chunks.append(Document(
                        page_content=cleaned_chunk,
                        metadata=page_doc.metadata.copy()
                    ))
        
        s2_end = time.perf_counter()
        duration2 = s2_end - s2_start
        timings.append(("Step 2: Smart Chunking", duration2))
        status.update(
            label=f"Step 2/4: Created {len(all_chunks)} optimized chunks in {duration2:.2f}s", 
            state="complete"
        )

    # Step 3: Embedding Generation
    with st.status("Step 3/4: Creating semantic embeddings...", expanded=True) as status:
        s3_start = time.perf_counter()
        
        if all_chunks:
            # Add in batches for better performance
            batch_size = 100
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i+batch_size]
                vector_store.add_documents(batch)
                if i % 500 == 0 and i > 0:
                    st.write(f"Processed {i}/{len(all_chunks)} chunks...")
        
        s3_end = time.perf_counter()
        duration3 = s3_end - s3_start
        timings.append(("Step 3: Embedding & Storage", duration3))
        status.update(
            label=f"Step 3/4: Embeddings created in {duration3:.2f}s", 
            state="complete"
        )

    # Step 4: Analysis with Enhanced Prompting
    with st.status("Step 4/4: Analyzing specifications with enhanced AI...", expanded=True) as status:
        s4_start = time.perf_counter()
        
        # Enhanced prompt based on thinking mode
        if thinking_mode == "Deep Analysis":
            template = """You are an expert construction specification analyst with deep knowledge of building codes and standards.

**CRITICAL INSTRUCTIONS:**
1. Read the ENTIRE CONTEXT carefully and thoroughly. Do not skip any part.
2. Look for EXACT matches, partial matches, synonyms, and related information.
3. Check for information in tables, lists, notes, and running text.
4. If you find the information, provide SPECIFIC details including:
   - Exact specifications, model numbers, or brand names
   - Dimensions, sizes, or quantities when mentioned
   - Any relevant standards, codes, or requirements
5. Include the page number reference for each finding: [Page X]
6. If information is found but incomplete, state what IS found and note what's missing.
7. Only say "Information not found" if you've thoroughly checked and found nothing relevant.

**THINKING MODE:** Deep Analysis - Be thorough and check multiple times.

CONTEXT:
{context}

QUESTION: {questions}

YOUR DETAILED ANSWER:"""
        
        elif thinking_mode == "Balanced":
            template = """You are an expert construction specification analyst.

**INSTRUCTIONS:**
1. Carefully review all provided CONTEXT.
2. Answer based on explicit information or clear implications in the CONTEXT.
3. Look for exact terms, synonyms, and related specifications.
4. Provide specific details with page references: [Page X]
5. If partially found, state what IS available and what's missing.
6. Only state "Information not found" after thorough review.

CONTEXT:
{context}

QUESTION: {questions}

YOUR ANSWER:"""
        
        else:  # Fast Mode
            template = """Extract specific information from the construction specifications provided.

CONTEXT: {context}

QUESTION: {questions}

Provide a concise answer with page reference [Page X] or state "Information not found":"""
        
        prompt = PromptTemplate.from_template(template)
        
        # Setup retriever with reranking
        base_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={'k': retriever_k}
        )
        
        compressor = FlashrankRerank(
            model="ms-marco-TinyBERT-L-2-v2", 
            top_n=rerank_top_n
        )
        
        retriever_with_rerank = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        # Update LLM temperature
        llm_dynamic = OllamaLLM(model=st.session_state.generative_model, temperature=temperature)
        
        setup_and_retrieval = RunnableParallel(
            context=(lambda x: x["query"]) | retriever_with_rerank | format_docs,
            questions=(lambda x: x["questions"]),
            thinking_mode=(lambda x: x.get("thinking_mode", "Balanced"))
        )
        
        rag_chain = setup_and_retrieval | prompt | llm_dynamic | StrOutputParser()

        all_results_flat = []
        
        # Progress tracking
        total_specs = sum(
            len(details['specifications']) 
            for division in KEYWORDS_AND_SPECS.values() 
            for details in division.values()
        )
        
        progress_bar = st.progress(0)
        progress_text = st.empty()
        completed = 0
        
        tasks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for division, sections in KEYWORDS_AND_SPECS.items():
                for section_title, details in sections.items():
                    section_keywords = details['keywords']
                    for spec_item in details['specifications']:
                        spec_question = spec_item['question']
                        related_terms = spec_item['related_terms']

                        retrieval_query = (
                            f"{section_title} {section_keywords} {spec_question} "
                            f"{related_terms}"
                        )
                        
                        tasks.append(executor.submit(
                            run_analysis_for_section_single_spec,
                            rag_chain,
                            division,
                            section_title,
                            spec_question,
                            retrieval_query,
                            thinking_mode
                        ))
            
            for future in concurrent.futures.as_completed(tasks):
                result_item = future.result()
                all_results_flat.append(result_item)
                completed += 1
                progress = completed / total_specs
                progress_bar.progress(progress)
                progress_text.text(f"Analyzing: {completed}/{total_specs} specifications")
        
        progress_bar.empty()
        progress_text.empty()
        
        s4_end = time.perf_counter()
        duration4 = s4_end - s4_start
        timings.append(("Step 4: Parallel Analysis with Reranking", duration4))
        status.update(
            label=f"Step 4/4: Analyzed {len(all_results_flat)} specifications in {duration4:.2f}s", 
            state="complete"
        )
    
    return all_results_flat, timings

# --- PDF Generation Function ---
def generate_pdf_with_table(report_df, timings_data, config_info, filename_prefix="specification_analysis_report"):
    """Generates a PDF report from the analysis DataFrame."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Enhanced Specification Analyzer Report", styles["Title"]))
    story.append(Spacer(1, 12))
    
    # Configuration Info
    story.append(Paragraph("Analysis Configuration", styles["Heading2"]))
    for key, value in config_info.items():
        story.append(Paragraph(f"<b>{key}:</b> {value}", styles["Normal"]))
    story.append(Spacer(1, 12))
    
    # Performance Breakdown
    story.append(Paragraph("Performance Breakdown", styles["Heading2"]))
    total_time = sum(d for _, d in timings_data)
    for label, duration in timings_data:
        story.append(Paragraph(f"{label}: {duration:.2f} seconds", styles["Normal"]))
    story.append(Paragraph(f"<b>Total Processing Time:</b> {total_time:.2f} seconds", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Analysis Report", styles["Heading2"]))
    story.append(Spacer(1, 12))

    if not report_df.empty:
        for division, division_df in report_df.groupby("Division"):
            story.append(Paragraph(f"<b>{division}</b>", styles["Heading3"]))
            story.append(Spacer(1, 6))

            table_data = [["Section", "Specification", "Result"]]
            
            section_df_sorted = division_df.sort_values(by=["Section", "Specification"])
            
            for index, row in section_df_sorted.iterrows():
                result_text = row["Result"].replace("\n", "<br/>")
                
                if "Information not found" in row["Result"]:
                    table_data.append([
                        Paragraph(row["Section"], styles["Normal"]),
                        Paragraph(row["Specification"], styles["Normal"]),
                        Paragraph(f"<font color='red'>{result_text}</font>", styles["Normal"])
                    ])
                else:
                    table_data.append([
                        Paragraph(row["Section"], styles["Normal"]),
                        Paragraph(row["Specification"], styles["Normal"]),
                        Paragraph(result_text, styles["Normal"])
                    ])
            
            if len(table_data) > 1:
                col_widths = [1.5*inch, 2.5*inch, 4*inch]
                table = Table(table_data, colWidths=col_widths)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#ADD8E6')),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.black),
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0,0), (-1,0), 12),
                    ('BACKGROUND', (0,1), (-1,-1), colors.white),
                    ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#D3D3D3')),
                    ('LEFTPADDING', (0,0), (-1,-1), 6),
                    ('RIGHTPADDING', (0,0), (-1,-1), 6),
                    ('TOPPADDING', (0,0), (-1,-1), 6),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ]))
                story.append(table)
                story.append(Spacer(1, 24))

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# --- Streamlit UI ---
st.title("📋 Enhanced Specification Analyzer V3")
st.markdown("Advanced AI-powered construction specification analysis with configurable settings")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Model Settings")
    embedding_model = st.selectbox(
        "Embedding Model",
        ["mxbai-embed-large", "nomic-embed-text", "all-minilm"],
        index=0,
        help="Model for converting text to embeddings"
    )
    
    generative_model = st.selectbox(
        "Generative Model",
        ["llama3:8b", "llama3:70b", "mistral", "mixtral"],
        index=0,
        help="Model for analysis and text generation"
    )
    
    st.subheader("Chunking Parameters")
    chunk_size = st.slider(
        "Chunk Size",
        min_value=300,
        max_value=1500,
        value=700,
        step=50,
        help="Size of text chunks for processing"
    )
    
    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=150,
        step=25,
        help="Overlap between chunks to preserve context"
    )
    
    st.subheader("Retrieval Settings")
    retriever_k = st.slider(
        "Initial Retrieval (K)",
        min_value=10,
        max_value=50,
        value=30,
        step=5,
        help="Number of chunks to retrieve before reranking"
    )
    
    rerank_top_n = st.slider(
        "Rerank Top N",
        min_value=3,
        max_value=15,
        value=5,
        step=1,
        help="Number of top chunks after reranking"
    )
    
    st.subheader("Analysis Mode")
    thinking_mode = st.radio(
        "Thinking Mode",
        ["Fast", "Balanced", "Deep Analysis"],
        index=1,
        help="Controls thoroughness vs speed of analysis"
    )
    
    temperature = st.slider(
        "Model Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Lower = more focused, Higher = more creative"
    )
    
    st.divider()
    
    if st.button("🗑️ Clear All Temp DBs", use_container_width=True):
        clear_parent_chroma_db_directory()
        st.success("Cleared temporary databases")
    
    st.divider()
    st.caption(f"**Current Settings:**")
    st.caption(f"Chunks: {chunk_size} / Overlap: {chunk_overlap}")
    st.caption(f"Retrieval: K={retriever_k} / Top N={rerank_top_n}")
    st.caption(f"Mode: {thinking_mode} / Temp: {temperature}")

# Store in session state
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = embedding_model
if 'generative_model' not in st.session_state:
    st.session_state.generative_model = generative_model

st.session_state.embedding_model = embedding_model
st.session_state.generative_model = generative_model

# Initialize LLM and embeddings with current settings
@st.cache_resource
def get_llm_and_embeddings(_embedding_model, _generative_model):
    """Initializes and caches the LLM and Embedding models."""
    llm = OllamaLLM(model=_generative_model, temperature=0.0)
    embeddings = OllamaEmbeddings(model=_embedding_model)
    return llm, embeddings

llm, embeddings = get_llm_and_embeddings(embedding_model, generative_model)

# Initialize session state variables
if 'report_df' not in st.session_state:
    st.session_state.report_df = None
if 'timings' not in st.session_state:
    st.session_state.timings = None
if 'config_info' not in st.session_state:
    st.session_state.config_info = None
if 'analysis_stats' not in st.session_state:
    st.session_state.analysis_stats = None

# --- MAIN INTERFACE ---
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📄 Upload Construction Specification PDF",
        type="pdf",
        help="Upload a complete specification document"
    )

with col2:
    st.metric("Thinking Mode", thinking_mode)
    st.metric("Chunk Size", f"{chunk_size} chars")

if uploaded_file is not None:
    st.info(f"📁 **File:** {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.2f} MB)")
    
    if st.button(
        f"🚀 Analyze '{uploaded_file.name}'",
        type="primary",
        use_container_width=True,
        help="Start comprehensive specification analysis"
    ):
        # Clear old DBs
        clear_parent_chroma_db_directory()

        # Generate unique directory
        current_session_db_dir = get_unique_db_directory(uploaded_file.name)
        os.makedirs(current_session_db_dir, exist_ok=True)
        
        # Initialize Chroma
        vector_store = Chroma(
            persist_directory=current_session_db_dir,
            embedding_function=embeddings
        )
        
        # Store configuration info
        config_info = {
            "Embedding Model": embedding_model,
            "Generative Model": generative_model,
            "Chunk Size": chunk_size,
            "Chunk Overlap": chunk_overlap,
            "Retriever K": retriever_k,
            "Rerank Top N": rerank_top_n,
            "Thinking Mode": thinking_mode,
            "Temperature": temperature,
            "Analysis Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Process and analyze
        all_results_flat, timings_data = process_and_analyze_pdf(
            uploaded_file,
            vector_store,
            llm,
            embeddings,
            chunk_size,
            chunk_overlap,
            retriever_k,
            rerank_top_n,
            thinking_mode,
            temperature
        )
        
        if all_results_flat:
            st.session_state.report_df = pd.DataFrame(all_results_flat)
            
            # Calculate statistics
            total_specs = len(all_results_flat)
            found_specs = sum(1 for r in all_results_flat if "Information not found" not in r["Result"])
            not_found_specs = total_specs - found_specs
            success_rate = (found_specs / total_specs * 100) if total_specs > 0 else 0
            
            st.session_state.analysis_stats = {
                "total": total_specs,
                "found": found_specs,
                "not_found": not_found_specs,
                "success_rate": success_rate
            }
        else:
            st.session_state.report_df = pd.DataFrame(columns=["Division", "Section", "Specification", "Result"])
            st.session_state.analysis_stats = {"total": 0, "found": 0, "not_found": 0, "success_rate": 0}
        
        st.session_state.timings = timings_data
        st.session_state.config_info = config_info
        
        # Cleanup
        try:
            vector_store.delete_collection()
            del vector_store
            gc.collect()
            shutil.rmtree(current_session_db_dir)
        except Exception as e:
            st.warning(f"Cleanup warning: {e}")
        
        st.rerun()

# --- DISPLAY RESULTS ---
if st.session_state.report_df is not None and not st.session_state.report_df.empty:
    
    # Statistics Dashboard
    st.header("📊 Analysis Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    stats = st.session_state.analysis_stats
    
    with col1:
        st.metric("Total Specifications", stats["total"])
    with col2:
        st.metric("Found", stats["found"], delta=f"{stats['success_rate']:.1f}%")
    with col3:
        st.metric("Not Found", stats["not_found"])
    with col4:
        total_time = sum(d for _, d in st.session_state.timings)
        st.metric("Processing Time", f"{total_time:.1f}s")
    
    # Performance Breakdown
    st.header("⚡ Performance Breakdown")
    with st.container(border=True):
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            for i, (label, duration) in enumerate(st.session_state.timings):
                if i < len(st.session_state.timings) // 2 + 1:
                    st.markdown(f"**{label}:** `{duration:.2f}s`")
        
        with perf_col2:
            for i, (label, duration) in enumerate(st.session_state.timings):
                if i >= len(st.session_state.timings) // 2 + 1:
                    st.markdown(f"**{label}:** `{duration:.2f}s`")
        
        st.divider()
        total_time = sum(d for _, d in st.session_state.timings)
        st.metric("**Total Processing Time**", f"{total_time:.2f} seconds")

    # Analysis Report
    st.header("📋 Detailed Analysis Report")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_option = st.selectbox(
            "Filter Results",
            ["Show All", "Found Only", "Not Found Only"],
            index=0
        )
    
    with col2:
        division_filter = st.multiselect(
            "Filter by Division",
            options=st.session_state.report_df["Division"].unique(),
            default=st.session_state.report_df["Division"].unique()
        )
    
    with col3:
        search_term = st.text_input(
            "Search in Results",
            placeholder="Enter search term..."
        )
    
    # Apply filters
    filtered_df = st.session_state.report_df.copy()
    
    if division_filter:
        filtered_df = filtered_df[filtered_df["Division"].isin(division_filter)]
    
    if filter_option == "Found Only":
        filtered_df = filtered_df[~filtered_df["Result"].str.contains("Information not found", na=False)]
    elif filter_option == "Not Found Only":
        filtered_df = filtered_df[filtered_df["Result"].str.contains("Information not found", na=False)]
    
    if search_term:
        filtered_df = filtered_df[
            filtered_df["Result"].str.contains(search_term, case=False, na=False) |
            filtered_df["Specification"].str.contains(search_term, case=False, na=False)
        ]
    
    st.markdown(f"**Showing {len(filtered_df)} of {len(st.session_state.report_df)} results**")
    
    # Display grouped results
    for division, division_df in filtered_df.groupby("Division"):
        with st.expander(f"**{division}**", expanded=True):
            for section, section_df in division_df.groupby("Section"):
                st.subheader(section)
                
                # Enhanced styling
                def highlight_results(s):
                    colors = []
                    for v in s:
                        v_str = str(v)
                        if 'Information not found' in v_str:
                            colors.append('background-color: #ffe0e0')
                        elif '[Page' in v_str:  # Has page reference
                            colors.append('background-color: #e0ffe0')
                        else:
                            colors.append('')
                    return colors
                
                styled_df = section_df.style.apply(
                    highlight_results,
                    subset=['Result'],
                    axis=0
                )
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Division": st.column_config.TextColumn("Division", width="small"),
                        "Section": st.column_config.TextColumn("Section", width="medium"),
                        "Specification": st.column_config.TextColumn("Specification", width="medium"),
                        "Result": st.column_config.TextColumn("Result", width="large")
                    }
                )
                st.markdown("---")
    
    # Export Options
    st.header("📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV Download
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📊 Download as CSV",
            data=csv,
            file_name=f"{uploaded_file.name.replace('.pdf', '')}_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel Download (if openpyxl available)
        try:
            import openpyxl
            from io import BytesIO
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='Analysis')
            excel_data = excel_buffer.getvalue()
            
            st.download_button(
                label="📑 Download as Excel",
                data=excel_data,
                file_name=f"{uploaded_file.name.replace('.pdf', '')}_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except ImportError:
            st.button("📑 Excel Export (Not Available)", disabled=True, use_container_width=True)
    
    with col3:
        # PDF Download
        if uploaded_file and st.session_state.config_info:
            pdf_bytes = generate_pdf_with_table(
                filtered_df,
                st.session_state.timings,
                st.session_state.config_info,
                uploaded_file.name.replace(".pdf", "")
            )
            st.download_button(
                label="📄 Download as PDF Report",
                data=pdf_bytes,
                file_name=f"{uploaded_file.name.replace('.pdf', '')}_analysis_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

elif st.session_state.report_df is not None and st.session_state.report_df.empty:
    st.info("🔍 No analysis results to display. Please upload a PDF and run the analysis.")
elif uploaded_file is None:
    st.info("👆 Upload a construction specification PDF to begin analysis.")
    
    # Show feature highlights
    st.markdown("---")
    st.subheader("✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 Enhanced Accuracy**
        - Advanced PDF parsing
        - Smart text extraction
        - Context preservation
        """)
    
    with col2:
        st.markdown("""
        **⚙️ Full Configurability**
        - Adjustable chunking
        - Multiple AI models
        - Custom thinking modes
        """)
    
    with col3:
        st.markdown("""
        **📊 Comprehensive Reports**
        - Detailed findings
        - Page references
        - Multiple export formats
        """)