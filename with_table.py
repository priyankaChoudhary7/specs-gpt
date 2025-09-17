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
import pandas as pd # Import pandas for DataFrame

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Specification Analyzer",
    page_icon="",
    layout="wide"
)

# --- Configuration ---
DB_DIRECTORY = "chroma_db"
EMBEDDING_MODEL = "mxbai-embed-large"
GENERATIVE_MODEL = "llama3:8b"

# --- Keywords and Specifications Structure ---
KEYWORDS_AND_SPECS = {
    "DIVISION 06/09 (FINISHES)": {
        "06 41 00 - ARCHITECTURAL WOOD CASEWORK": {
            "keywords": "ARCHITECTURAL WOOD CASEWORK, PLASTIC LAMINATE, HIGH PRESSURE LAMINATE, THERMOFOIL, SOLID SURFACE, WILSONART, OMNOVA, FORMICA",
            "specifications": ["Type of unit finish", "Color and manufacturer of the finish", "3form material requirements", "Any details related to headwalls in this section"]
        }
    },
    "DIVISION 10/11 (SPECIALTIES)": {
        "10 25 13 - PATIENT BED SERVICE WALLS": {
            "keywords": "PATIENT BED SERVICE WALLS (PBSW), PRE-FABRICATION, HEADWALL, BED LOCATOR, MODULAR, AMICO",
            "specifications": ["Type of Headwall", "Preferred Manufacturer", "Headwall Dimensions", "Headwall Finish details", "List of specified Headwall Services", "Headwall Accessories and scope"]
        }
    },
    "DIVISION 22 (PLUMBING)": {
        "22 60 00 / 22 61 00 / 22 62 00 - GAS AND VACUUM SYSTEMS": {
            "keywords": "OXYGEN, MEDICAL AIR, VACUUM, GAS OUTLET, BEACON, D.I.S.S., COPPER PIPE, DISS",
            "specifications": ["Gas connection type", "Mention of 'BEACON' brand", "Pipe sizes for Oxygen, Med Air, Vacuum", "Copper pipe type (K or L)"]
        }
    },
    "DIVISION 26 (ELECTRICAL)": {
        "26 05 33 - RACEWAYS AND BOXES": {
            "keywords": "RACEWAYS, BACKBOX, CONDUIT, STAINLESS STEEL",
            "specifications": ["Conduit size requirements", "Backbox size and type requirements"]
        },
        "26 27 26 - WIRING DEVICES": {
            "keywords": "WIRING DEVICES, TAMPERPROOF, GFCI, HOSPITAL GRADE, RECEPTACLE, SWITCH, HUBBELL, LEVITON",
            "specifications": ["Receptacle brand", "Receptacle type (Tamperproof, GFCI, etc.)", "Faceplate color and material", "Switch types"]
        }
    },
    "DIVISION 27 (COMMUNICATIONS)": {
        "27 09 04 / 27 52 23 - LOW VOLTAGE & NURSE CALL": {
            "keywords": "LOW VOLTAGE, NURSE CALL, CODE BLUE, CONDUIT, BACKBOX",
            "specifications": ["Low voltage conduit size", "Nurse Call system brand", "Code Blue system brand and requirements"]
        }
    }
}

# --- Utility Function to Clear Chroma DB ---
def clear_chroma_db(directory):
    if os.path.exists(directory):
        st.info(f"Clearing existing Chroma DB at '{directory}' for a fresh analysis...")
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

# --- Caching for Performance ---
@st.cache_resource
def get_llm_and_embeddings():
    llm = OllamaLLM(model=GENERATIVE_MODEL, temperature=0.0)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return llm, embeddings

def initialize_and_clear_vector_db_directory(directory):
    if os.path.exists(directory):
        st.info(f"Clearing existing Chroma DB at '{directory}' for a fresh analysis...")
        # Give some time for file handles to be released, although not guaranteed
        # You might need a more robust solution for persistent file locks if this doesn't work.
        try:
            shutil.rmtree(directory)
        except PermissionError as e:
            st.error(f"Could not clear Chroma DB directory: {e}. Please ensure no other process is using it.")
            # Optionally, exit or prevent further execution if the DB can't be cleared.
            st.stop()
    os.makedirs(directory, exist_ok=True)

def get_new_vector_store(_embeddings):
    clear_chroma_db(DB_DIRECTORY)
    return Chroma(persist_directory=DB_DIRECTORY, embedding_function=_embeddings)

# --- Core Functions ---
def extract_text_from_pdf(pdf_bytes):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "".join(page.get_text() for page in doc)

def format_docs(docs):
    docs = sorted(docs, key=lambda d: d.metadata.get("source", "") + str(d.metadata.get("page", "")))
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# Modified to return a list of dictionaries for table display
def run_analysis_for_section(rag_chain, division_title, section_title, specifications):
    results_for_section = []
    # For each individual specification (question)
    for spec_question in specifications:
        # Construct a more focused query for each spec_question
        search_query = f"{spec_question} in {section_title} ({division_title})"
        input_data = {"query": search_query, "questions": spec_question} # Pass single question
        try:
            analysis_result = rag_chain.invoke(input_data)
            results_for_section.append({
                "Division": division_title,
                "Section": section_title,
                "Specification": spec_question,
                "Result": analysis_result
            })
        except Exception as e:
            st.error(f"Error during analysis for '{spec_question}' in {section_title}: {str(e)}")
            results_for_section.append({
                "Division": division_title,
                "Section": section_title,
                "Specification": spec_question,
                "Result": f"Error during analysis: {str(e)}"
            })
    return results_for_section


def process_and_analyze_pdf(pdf_file, vector_store, llm, embeddings):
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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300) 
        chunks = text_splitter.split_text(full_text)
        documents = [Document(page_content=chunk, metadata={"source": pdf_filename}) for chunk in chunks]
        s2_end = time.perf_counter()
        duration2 = s2_end - s2_start
        timings.append(("Step 2: Document Chunking", duration2))
        status.update(label=f"Step 2/4: Document chunking complete in {duration2:.2f}s", state="complete")

    with st.status("Step 3/4: Creating semantic embeddings and storing in vector database...", expanded=True) as status:
        s3_start = time.perf_counter()
        vector_store.add_documents(documents)
        s3_end = time.perf_counter()
        duration3 = s3_end - s3_start
        timings.append(("Step 3: Embedding Generation & DB Storage", duration3))
        status.update(label=f"Step 3/4: Embeddings created and stored in {duration3:.2f}s", state="complete")

    with st.status("Step 4/4: Analyzing all document sections in parallel...", expanded=True) as status:
        s4_start = time.perf_counter()
        template = """Based ONLY on the context provided below, answer the following question. If the information is not in the context, explicitly state "Information not found." Be concise and directly answer the question.
        CONTEXT: {context}
        QUESTION: {questions}
        YOUR ANSWER:"""
        prompt = PromptTemplate.from_template(template)
        
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 10, 'filter': {'source': pdf_filename}})
        
        setup_and_retrieval = RunnableParallel(
            context=(lambda x: x["query"]) | retriever | format_docs,
            questions=(lambda x: x["questions"])
        )
        rag_chain = setup_and_retrieval | prompt | llm | StrOutputParser()

        all_results_flat = [] # To store a flat list of dicts for the table
        
        tasks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for division, sections in KEYWORDS_AND_SPECS.items():
                for section_title, details in sections.items():
                    # Submit a task for each specification within a section
                    for spec_question in details['specifications']:
                        retrieval_query_for_this_spec = f"{details['keywords']} {spec_question} in {section_title}"
                        tasks.append(executor.submit(
                            run_analysis_for_section_single_spec, # New function for single spec analysis
                            rag_chain, 
                            division, 
                            section_title, 
                            spec_question,
                            retrieval_query_for_this_spec # Pass the specific search query for this spec
                        ))
            
            for future in concurrent.futures.as_completed(tasks):
                result_item = future.result()
                all_results_flat.append(result_item) # Append single result dict
        
        s4_end = time.perf_counter()
        duration4 = s4_end - s4_start
        timings.append(("Step 4: Parallel Analysis (RAG)", duration4))
        status.update(label=f"Step 4/4: Analysis complete in {duration4:.2f}s", state="complete")
    
    return all_results_flat, timings # Return the flat list of results

# New function to process a single specification question
def run_analysis_for_section_single_spec(rag_chain, division_title, section_title, spec_question, search_query_context):
    input_data = {"query": search_query_context, "questions": spec_question}
    try:
        analysis_result = rag_chain.invoke(input_data)
        return {
            "Division": division_title,
            "Section": section_title,
            "Specification": spec_question,
            "Result": analysis_result
        }
    except Exception as e:
        st.error(f"Error during analysis for '{spec_question}' in {section_title}: {str(e)}")
        return {
            "Division": division_title,
            "Section": section_title,
            "Specification": spec_question,
            "Result": f"Error during analysis: {str(e)}"
        }

# --- Streamlit UI ---
st.title("Specification Analyzer")

llm, embeddings = get_llm_and_embeddings()

if 'report_df' not in st.session_state: # Change to store DataFrame
    st.session_state.report_df = None
if 'timings' not in st.session_state:
    st.session_state.timings = None

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if st.button(f"Analyze {uploaded_file.name}", help="Click to start the analysis. This will clear previous data for consistency."):
        
        clear_chroma_db(DB_DIRECTORY)

        st.session_state.report_df = None
        st.session_state.timings = None
        
        initialize_and_clear_vector_db_directory(DB_DIRECTORY)
        
        vector_store = get_new_vector_store(embeddings) 
        
        # process_and_analyze_pdf now returns a flat list of dicts
        all_results_flat, timings_data = process_and_analyze_pdf(uploaded_file, vector_store, llm, embeddings)
        
        if all_results_flat:
            st.session_state.report_df = pd.DataFrame(all_results_flat)
        else:
            st.session_state.report_df = pd.DataFrame(columns=["Division", "Section", "Specification", "Result"]) # Empty DataFrame
        
        st.session_state.timings = timings_data
        del vector_store 
        import gc
        gc.collect() 
        
        st.rerun()

# --- MODIFIED DISPLAY SECTION ---
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
                    return ['background-color: #ffe0e0' if 'Information not found.' in v else '' for v in s]
                

                display_df = section_df.copy()
                display_df['Result'] = display_df['Result'].apply(
                    lambda x: f"<span style='color:red;'>{x}</span>" if "Information not found." in x else x
                )

                styled_df = section_df.style.apply(
                    lambda x: ['background-color: #ffe0e0' if 'Information not found.' in str(v) else '' for v in x],
                    subset=['Result'],
                    axis=0
                )
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                st.markdown("---") # Separator between sections

elif st.session_state.report_df is not None and st.session_state.report_df.empty:
    st.info("No analysis results to display. Please upload a PDF and run the analysis.")

# --- DOWNLOAD PDF BUTTON ---
# This part also needs significant changes to render a table in ReportLab
def generate_pdf_with_table(report_df, timings_data, filename="specification_analysis_report.pdf"):
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
            story.append(Paragraph(division, styles["Heading3"]))
            story.append(Spacer(1, 6))

            # Prepare data for table
            table_data = [["Section", "Specification", "Result"]]
            
            # Sort sections for consistent PDF output
            for section, section_df in division_df.groupby("Section"):
                # Add a row for the section title (optional, can be a merged cell)
                # For simplicity, let's just list the specifications under each section.
                # section_df = section_df.sort_values(by="Specification") # Sort within section
                for index, row in section_df.iterrows():
                    result_text = row["Result"].replace("\n", "<br/>") # Replace newlines for ReportLab
                    
                    # Apply color for "Information not found." in PDF table
                    if "Information not found." in row["Result"]:
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
            
            if len(table_data) > 1: # If there are actual results
                # Create the ReportLab table
                table = Table(table_data, colWidths=[1.5*inch, 2.5*inch, 3.5*inch]) # Adjust column widths as needed
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

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

if st.session_state.report_df is not None:
    pdf_bytes = generate_pdf_with_table(st.session_state.report_df, st.session_state.timings, uploaded_file.name.replace(".pdf", "_report.pdf"))
    st.download_button(
        label="Download Report as PDF",
        data=pdf_bytes,
        file_name=f"{uploaded_file.name.replace('.pdf', '')}_analysis_report.pdf",
        mime="application/pdf"
    )