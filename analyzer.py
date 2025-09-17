import streamlit as st
import fitz
import concurrent.futures
import time
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
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
# --- Caching for Performance ---
@st.cache_resource
def get_llm_and_embeddings():
    llm = OllamaLLM(model=GENERATIVE_MODEL, temperature=0.0)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return llm, embeddings

@st.cache_resource
def get_vector_store(_embeddings):
    return Chroma(persist_directory=DB_DIRECTORY, embedding_function=_embeddings)
    
# --- Core Functions ---
def extract_text_from_pdf(pdf_bytes):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "".join(page.get_text() for page in doc)

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def run_analysis_for_section(rag_chain, section_title, details):
    search_query = f"{section_title} {details['keywords']}"
    questions_text = "\n".join([f"- {spec}" for spec in details['specifications']])
    input_data = {"query": search_query, "questions": questions_text}
    try:
        return section_title, rag_chain.invoke(input_data)
    except Exception as e:
        return section_title, f"Error during analysis: {str(e)}"

# --- MODIFIED FUNCTION WITH TIMERS ---
def process_and_analyze_pdf(pdf_file, vector_store, llm):
    timings = []
    pdf_filename = pdf_file.name
    
    # Step 1: Text Extraction
    with st.status(f"Step 1/3: Extracting text from '{pdf_filename}'...", expanded=True) as status:
        s1_start = time.perf_counter()
        pdf_bytes = pdf_file.getvalue()
        full_text = extract_text_from_pdf(pdf_bytes)
        s1_end = time.perf_counter()
        duration1 = s1_end - s1_start
        timings.append(("Step 1: Text Extraction", duration1))
        
        if not full_text:
            status.update(label="Text extraction failed.", state="error")
            return None, timings
        status.update(label=f"Step 1/3: Text Extraction complete in {duration1:.2f}s", state="complete")

    # Step 2: Embedding Generation
    with st.status("Step 2/3: Creating semantic embeddings...", expanded=True) as status:
        s2_start = time.perf_counter()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)
        chunks = text_splitter.split_text(full_text)
        documents = [Document(page_content=chunk, metadata={"source": pdf_filename}) for chunk in chunks]
        vector_store.add_documents(documents)
        s2_end = time.perf_counter()
        duration2 = s2_end - s2_start
        timings.append(("Step 2: Embedding Generation", duration2))
        status.update(label=f"Step 2/3: Embeddings created in {duration2:.2f}s", state="complete")

    # Step 3: Parallel Analysis (RAG)
    with st.status("Step 3/3: Analyzing all document sections in parallel...", expanded=True) as status:
        s3_start = time.perf_counter()
        template = """Based ONLY on the context provided below, answer the user's questions. If the information is not in the context, explicitly state "Information not found." Be concise.
        CONTEXT: {context}
        QUESTIONS: {questions}
        YOUR STRUCTURED ANSWERS:"""
        prompt = PromptTemplate.from_template(template)
        retriever = vector_store.as_retriever(search_kwargs={'k': 5, 'filter': {'source': pdf_filename}})
        
        setup_and_retrieval = RunnableParallel(
            context=(lambda x: x["query"]) | retriever | format_docs,
            questions=(lambda x: x["questions"])
        )
        rag_chain = setup_and_retrieval | prompt | llm | StrOutputParser()

        report = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            tasks = [executor.submit(run_analysis_for_section, rag_chain, section_title, details)
                     for division, specs in KEYWORDS_AND_SPECS.items()
                     for section_title, details in specs.items()]
            
            for future in concurrent.futures.as_completed(tasks):
                section_title, analysis_result = future.result()
                for division, sections in KEYWORDS_AND_SPECS.items():
                    if section_title in sections:
                        if division not in report: report[division] = {}
                        report[division][section_title] = analysis_result
                        break
        
        s3_end = time.perf_counter()
        duration3 = s3_end - s3_start
        timings.append(("Step 3: Parallel Analysis (RAG)", duration3))
        status.update(label=f"Step 3/3: Analysis complete in {duration3:.2f}s", state="complete")
    
    return report, timings

# --- Streamlit UI ---
st.title("Specification Analyzer")
st.markdown("Upload a PDF to automatically perform a parallel semantic analysis and view a performance breakdown of each step.")

llm, embeddings = get_llm_and_embeddings()
vector_store = get_vector_store(embeddings)

if 'report' not in st.session_state:
    st.session_state.report = None
if 'timings' not in st.session_state:
    st.session_state.timings = None

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if st.button(f"Analyze {uploaded_file.name}"):
        st.session_state.report = None
        st.session_state.timings = None
        report_data, timings_data = process_and_analyze_pdf(uploaded_file, vector_store, llm)
        st.session_state.report = report_data
        st.session_state.timings = timings_data
        st.rerun()

# --- MODIFIED DISPLAY SECTION ---
if st.session_state.report:
    st.header("Performance Breakdown")
    total_time = 0
    with st.container(border=True):
        for label, duration in st.session_state.timings:
            st.markdown(f"**{label}:** `{duration:.2f}` seconds")
            total_time += duration
        st.divider()
        st.metric(label="**Total Processing Time**", value=f"{total_time:.2f} seconds")

    st.header("Analysis Report")
    # Sort divisions for consistent report ordering
    sorted_divisions = sorted(st.session_state.report.items())
    for division, sections in sorted_divisions:
        with st.expander(f"**{division}**", expanded=True):
            sorted_sections = sorted(sections.items())
            for section_title, analysis in sorted_sections:
                st.subheader(section_title)
                st.markdown(analysis)
                st.divider()

     # --- DOWNLOAD PDF BUTTON ---
    def generate_pdf(report_data, timings_data):
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
        for division, sections in sorted(report_data.items()):
            story.append(Paragraph(division, styles["Heading3"]))
            for section_title, analysis in sorted(sections.items()):
                story.append(Paragraph(f"<b>{section_title}</b>", styles["Heading4"]))
                story.append(Paragraph(analysis.replace("\n", "<br/>"), styles["Normal"]))
                story.append(Spacer(1, 12))

        doc.build(story)
        pdf = buffer.getvalue()
        buffer.close()
        return pdf

    pdf_bytes = generate_pdf(st.session_state.report, st.session_state.timings)
    st.download_button(
        label="Download Report as PDF",
        data=pdf_bytes,
        file_name="specification_analysis_report.pdf",
        mime="application/pdf"
    )
