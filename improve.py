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
import gc # Import garbage collector

# --- For Re-ranking ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

# --- FIX: Call model_rebuild() for FlashrankRerank ---
# This resolves Pydantic's "class not fully defined" error
FlashrankRerank.model_rebuild() 
# --- END OF FIX ---

# --- Page Configuration ---
st.set_page_config(
    page_title="Specification Analyzer",
    page_icon="",
    layout="wide"
)

# --- Configuration ---
PARENT_DB_DIRECTORY = "chroma_dbs_temp"
EMBEDDING_MODEL = "mxbai-embed-large"
GENERATIVE_MODEL = "llama3:8b" # Keep temperature 0 for consistent extraction

# You might need to adjust these for optimal performance vs. recall
# Smaller chunks mean more chunks but more granular matches.
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150
RETRIEVER_K = 30 # Retrieve more documents initially for the reranker to sort
RERANK_TOP_N = 5 # How many top documents to pass to the LLM after reranking

def get_unique_db_directory(pdf_name):
    """Generates a unique directory path for a Chroma DB for a given PDF."""
    safe_name = pdf_name.replace(" ", "_").replace(".pdf", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"{PARENT_DB_DIRECTORY}/{safe_name}_{timestamp}_{unique_id}"

# --- Keywords and Specifications Structure (Enhanced with related_terms) ---
KEYWORDS_AND_SPECS = {
    "DIVISION 06/09 (FINISHES)": {
        "06 41 00 - ARCHITECTURAL WOOD CASEWORK": {
            "keywords": "ARCHITECTURAL WOOD CASEWORK",
            "specifications": [
                {"question": "Type of unit finish", "related_terms": "finish type, surface material, laminate type, veneer, paint, coating, construction material, material finish, casework finish,SOLID SURFACE, P-LAM,PLASTIC LAMINATE, HIGH PRESSURE LAMINATE, THERMOFOIL "},
                {"question": "Color and manufacturer of the finish", "related_terms": "finish color, material supplier, brand, shade, hue, maker, , WILSONART, OMNOVA, FORMICA, finish manufacturer, color selection, finish brand"},
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
                # {"question": "Conduit size requirements", "related_terms": "conduit diameter, size of conduit, pipe size for electrical, raceway size, trade size, electrical piping, conduit dimensions"},
                # {"question": "Backbox size and type requirements", "related_terms": "backbox dimensions, backbox material, gang box, electrical box, mounting box, flush, surface, device box, outlet box, backbox type"}
            
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

# --- Utility Function to Clear Chroma DB (parent directory) ---
def clear_parent_chroma_db_directory(directory=PARENT_DB_DIRECTORY):
    """Clears all temporary Chroma DB directories within the specified parent directory."""
    if os.path.exists(directory):
        st.info(f"Clearing old temporary Chroma DBs in '{directory}'...")
        try:
            shutil.rmtree(directory)
            st.success(f"Successfully cleared directory: {directory}")
        except PermissionError as e:
            st.error(f"Could not clear Chroma DB parent directory: {e}. Please ensure no other process is using it.")
        except Exception as e:
            st.error(f"An unexpected error occurred while clearing DB directory: {e}")
    os.makedirs(directory, exist_ok=True) # Recreate parent directory

# --- Caching for Performance ---
@st.cache_resource
def get_llm_and_embeddings():
    """Initializes and caches the LLM and Embedding models."""
    llm = OllamaLLM(model=GENERATIVE_MODEL, temperature=0.0) # Temperature 0 for consistent answers
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return llm, embeddings

# --- Core Functions ---
def extract_text_from_pdf(pdf_bytes):
    """Extracts all text from a PDF given its bytes."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "".join(page.get_text() for page in doc)

def format_docs(docs):
    """Formats retrieved document chunks into a single string for the LLM context."""
    # Sort for consistent context presentation to the LLM (optional with reranking, but good practice)
    docs = sorted(docs, key=lambda d: d.metadata.get("source", "") + str(d.metadata.get("page", "")))
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def run_analysis_for_section_single_spec(rag_chain, division_title, section_title, spec_question, retrieval_query_for_this_spec):
    """
    Runs the RAG chain for a single specification question.
    Constructs input data for the retriever (via 'query') and the LLM (via 'questions').
    """
    input_data = {"query": retrieval_query_for_this_spec, "questions": spec_question}
    try:
        analysis_result = rag_chain.invoke(input_data)
        return {
            "Division": division_title,
            "Section": section_title,
            "Specification": spec_question,
            "Result": analysis_result
        }
    except Exception as e:
        print(f"Error during analysis for '{spec_question}' in {section_title}: {str(e)}") # Log for debugging
        return {
            "Division": division_title,
            "Section": section_title,
            "Specification": spec_question,
            "Result": f"Error during analysis: Could not process question. {str(e)}"
        }

def process_and_analyze_pdf(pdf_file, vector_store, llm, embeddings):
    """
    Orchestrates the PDF processing and analysis workflow:
    text extraction -> chunking -> embedding -> RAG analysis.
    Integrates re-ranking for improved context quality.
    """
    timings = []
    pdf_filename = pdf_file.name
    
    with st.status(f"Step 1/4: Extracting text from '{pdf_filename}'...", expanded=True) as status:
        s1_start = time.perf_counter()
        pdf_bytes = pdf_file.getvalue()
        full_text = extract_text_from_pdf(pdf_bytes)
        s1_end = time.perf_counter()
        duration1 = s1_end - s1_start
        timings.append(("Step 1: Text Extraction", duration1))
        
        if not full_text:
            status.update(label="Text extraction failed. No content found.", state="error")
            return None, timings
        status.update(label=f"Step 1/4: Text Extraction complete in {duration1:.2f}s", state="complete")

    with st.status("Step 2/4: Chunking document...", expanded=True) as status:
        s2_start = time.perf_counter()
        # Adjusted chunk_size and overlap for potentially better granularity with large documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = text_splitter.split_text(full_text)
        documents = [Document(page_content=chunk, metadata={"source": pdf_filename}) for chunk in chunks]
        s2_end = time.perf_counter()
        duration2 = s2_end - s2_start
        timings.append(("Step 2: Document Chunking", duration2))
        status.update(label=f"Step 2/4: Document chunking complete ({len(chunks)} chunks) in {duration2:.2f}s", state="complete")

    with st.status("Step 3/4: Creating semantic embeddings and storing in vector database...", expanded=True) as status:
        s3_start = time.perf_counter()
        if documents: # Only add if there are documents
            vector_store.add_documents(documents)
        else:
            st.warning("No document chunks to embed.")
        s3_end = time.perf_counter()
        duration3 = s3_end - s3_start
        timings.append(("Step 3: Embedding Generation & DB Storage", duration3))
        status.update(label=f"Step 3/4: Embeddings created and stored in {duration3:.2f}s", state="complete")

    with st.status("Step 4/4: Analyzing all document sections in parallel...", expanded=True) as status:
        s4_start = time.perf_counter()
        
        # --- NEW ENHANCED PROMPT TEMPLATE for thorough checking ---
        template = """You are an expert at extracting specific technical information from specifications documents. 
        Your goal is to accurately identify and extract details related to the user's question from the provided CONTEXT. 
        Perform a thorough check within the context.

        **Instructions:**
        1.  Carefully read and **thoroughly check** all provided CONTEXT, and try to understand the CONTEXT format.
        2.  Answer the QUESTION based **ONLY** on the information explicitly stated or clearly implied in the CONTEXT. Do not make assumptions or infer beyond what is directly supported.
        3.  Look for synonyms, related terms, different phrasing, or descriptions that convey the same meaning as the requested information.
        4.  If the exact information is not found, or cannot be reasonably inferred *even after a thorough check*, state: "Information not found."
        5.  Be concise and provide the most direct answer possible. Do not invent information.
        6.  If a list of items is requested, provide a clear, bulleted list.

        CONTEXT: {context}

        QUESTION: {questions}

        YOUR ANSWER:"""
        prompt = PromptTemplate.from_template(template)
        
        # --- Retriever with Re-ranking ---
        base_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={'k': RETRIEVER_K, 'filter': {'source': pdf_filename}} # Retrieve more for reranker
        )
        
        # Initialize FlashrankRerank compressor
        compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2", top_n=RERANK_TOP_N)
        
        # Create a ContextualCompressionRetriever that uses the base retriever and the compressor
        # This will retrieve K docs, then rerank them, and pass only the top N to the LLM.
        retriever_with_rerank = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        setup_and_retrieval = RunnableParallel(
            context=(lambda x: x["query"]) | retriever_with_rerank | format_docs, # Use the reranking retriever
            questions=(lambda x: x["questions"])
        )
        rag_chain = setup_and_retrieval | prompt | llm | StrOutputParser()

        all_results_flat = []
        
        tasks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for division, sections in KEYWORDS_AND_SPECS.items():
                for section_title, details in sections.items():
                    section_keywords = details['keywords']
                    for spec_item in details['specifications']:
                        spec_question = spec_item['question']
                        related_terms = spec_item['related_terms']

                        # Construct a comprehensive search query for the retriever
                        # This query goes to the `base_retriever` before reranking
                        retrieval_query_for_this_spec = (
                            f"In section {section_title} of {division}, concerning {section_keywords}: {spec_question} {related_terms}"
                        )
                        
                        tasks.append(executor.submit(
                            run_analysis_for_section_single_spec,
                            rag_chain,
                            division,
                            section_title,
                            spec_question,
                            retrieval_query_for_this_spec
                        ))
            
            for future in concurrent.futures.as_completed(tasks):
                result_item = future.result()
                all_results_flat.append(result_item)
        
        s4_end = time.perf_counter()
        duration4 = s4_end - s4_start
        timings.append(("Step 4: Parallel Analysis (RAG with Re-ranking)", duration4))
        status.update(label=f"Step 4/4: Analysis complete in {duration4:.2f}s", state="complete")
    
    return all_results_flat, timings

# --- Streamlit UI ---
st.title("Specification Analyzer V2")
st.markdown("Upload a PDF specification document to extract key information using an advanced RAG system with re-ranking.")

# Initialize LLM and embeddings once per session
llm, embeddings = get_llm_and_embeddings()

# Initialize session state variables
if 'report_df' not in st.session_state:
    st.session_state.report_df = None
if 'timings' not in st.session_state:
    st.session_state.timings = None

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if st.button(f"Analyze '{uploaded_file.name}'", help="Click to start the analysis. This will generate a new report."):
        
        # Clear any old temporary DBs before starting a new analysis
        clear_parent_chroma_db_directory()

        # Generate a unique directory for the current PDF's vector store
        current_session_db_dir = get_unique_db_directory(uploaded_file.name)
        
        # Ensure the unique directory exists
        os.makedirs(current_session_db_dir, exist_ok=True)
        
        # Initialize Chroma with the unique, session-specific directory
        vector_store = Chroma(persist_directory=current_session_db_dir, embedding_function=embeddings)
        
        # Process and analyze the PDF
        all_results_flat, timings_data = process_and_analyze_pdf(uploaded_file, vector_store, llm, embeddings)
        
        if all_results_flat:
            st.session_state.report_df = pd.DataFrame(all_results_flat)
        else:
            st.session_state.report_df = pd.DataFrame(columns=["Division", "Section", "Specification", "Result"]) # Empty DataFrame
        
        st.session_state.timings = timings_data
        
        # Explicitly close the vector store connection and delete its content
        vector_store.delete_collection()
        del vector_store # Remove reference to allow garbage collection
        gc.collect() # Force garbage collection

        # Clean up the specific temporary directory after use
        try:
            shutil.rmtree(current_session_db_dir)
            st.success(f"Cleaned up temporary database directory: {current_session_db_dir}")
        except Exception as e:
            st.warning(f"Could not remove temporary DB directory {current_session_db_dir}: {e}")
        
        st.rerun() # Rerun to display the results

# --- DISPLAY SECTION ---
if st.session_state.report_df is not None and not st.session_state.report_df.empty:
    st.header("Performance Breakdown")
    total_time = 0
    with st.container(border=True):
        for label, duration in st.session_state.timings:
            st.markdown(f"**{label}:** `{duration:.2f}` seconds")
            total_time += duration
        st.divider()
        st.metric(label="**Total Processing Time**", value=f"{total_time:.2f} seconds")

    st.header("Analysis Report")
    st.markdown("Results are grouped by Division and Section. Expand each section to view the detailed table.")

    # Group by 'Division' first for expanders
    for division, division_df in st.session_state.report_df.groupby("Division"):
        with st.expander(f"**{division}**", expanded=True):
            # Then group by 'Section' within each division
            for section, section_df in division_df.groupby("Section"):
                st.subheader(section)
                
                # Apply conditional styling for 'Information not found.'
                def highlight_not_found(s):
                    # Check if 'Information not found.' is in the string content
                    return ['background-color: #ffe0e0' if 'Information not found' in str(v) else '' for v in s]
                
                # Apply styling to the DataFrame for Streamlit display
                styled_df = section_df.style.apply(
                    highlight_not_found,
                    subset=['Result'], # Only apply styling to the 'Result' column
                    axis=0
                )
                st.dataframe(styled_df, width='stretch', hide_index=True)
                st.markdown("---") # Separator between sections

elif st.session_state.report_df is not None and st.session_state.report_df.empty:
    st.info("No analysis results to display. Please upload a PDF and run the analysis.")
elif uploaded_file is None:
    st.info("Upload a PDF to begin the analysis.")


# --- DOWNLOAD PDF BUTTON ---
def generate_pdf_with_table(report_df, timings_data, filename_prefix="specification_analysis_report"):
    """Generates a PDF report from the analysis DataFrame and timings data."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Specification Analyzer Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Performance Breakdown", styles["Heading2"]))
    total_time = sum(d for _, d in timings_data)
    for label, duration in timings_data:
        story.append(Paragraph(f"{label}: {duration:.2f} seconds", styles["Normal"]))
    story.append(Paragraph(f"<b>Total Processing Time:</b> {total_time:.2f} seconds", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Analysis Report", styles["Heading2"]))
    story.append(Spacer(1, 12))

    if not report_df.empty:
        # Group by Division for sections in PDF
        for division, division_df in report_df.groupby("Division"):
            story.append(Paragraph(f"<b>{division}</b>", styles["Heading3"]))
            story.append(Spacer(1, 6))

            # Table for this division
            # Ensure consistent column order
            table_data = [["Section", "Specification", "Result"]]
            
            # Sort by Section, then by Specification question for consistent output
            section_df_sorted = division_df.sort_values(by=["Section", "Specification"])
            
            for index, row in section_df_sorted.iterrows():
                result_text = row["Result"].replace("\n", "<br/>") # Replace newlines for ReportLab
                
                # Apply color for "Information not found." in PDF table
                if "Information not found" in row["Result"]: # Check for substring
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
            
            if len(table_data) > 1: # If there are actual results (header + at least one data row)
                col_widths = [1.5*inch, 2.5*inch, 4*inch] # Adjust as needed for better fit

                table = Table(table_data, colWidths=col_widths)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#ADD8E6')), # Header background (Light Blue)
                    ('TEXTCOLOR', (0,0), (-1,0), colors.black),
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0,0), (-1,0), 12),
                    ('BACKGROUND', (0,1), (-1,-1), colors.white),
                    ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#D3D3D3')), # Light grey grid
                    ('LEFTPADDING', (0,0), (-1,-1), 6),
                    ('RIGHTPADDING', (0,0), (-1,-1), 6),
                    ('TOPPADDING', (0,0), (-1,-1), 6),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ]))
                story.append(table)
                story.append(Spacer(1, 24)) # Space between division tables
            else:
                story.append(Paragraph("No specifications found for this division.", styles["Normal"]))
    else:
        story.append(Paragraph("No analysis results available.", styles["Normal"]))

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

if st.session_state.report_df is not None and not st.session_state.report_df.empty:
    if uploaded_file: # Ensure uploaded_file is available for naming the PDF
        pdf_bytes = generate_pdf_with_table(st.session_state.report_df, st.session_state.timings, uploaded_file.name.replace(".pdf", ""))
        st.download_button(
            label="Download Report as PDF",
            data=pdf_bytes,
            file_name=f"{uploaded_file.name.replace('.pdf', '')}_analysis_report.pdf",
            mime="application/pdf"
        )