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
import gc

# --- For Re-ranking ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

# --- Fix for Pydantic v2 compatibility with FlashrankRerank ---
FlashrankRerank.model_rebuild()

# --- Page Configuration ---
st.set_page_config(
    page_title="Document Analyzer",
    page_icon="🏗️",
    layout="wide"
)

# --- GLOBAL CONFIGURATION ---
PARENT_DB_DIRECTORY = "chroma_dbs_temp"
EMBEDDING_MODEL = "mxbai-embed-large"
GENERATIVE_MODEL = "llama3:8b"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150
RETRIEVER_K = 30
RERANK_TOP_N = 5

# --- DATA STRUCTURES (Unchanged) ---
# KEYWORDS_AND_SPECS and PLANS_QUESTIONS dictionaries remain the same.
# (They are omitted here for brevity but should be included in your final script)

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
                {"question": "Conduit size requirements", "related_terms": "conduit diameter, size of conduit, pipe size for electrical, raceway size, trade size, electrical piping, conduit dimensions"},
                {"question": "Backbox size and type requirements", "related_terms": "backbox dimensions, backbox material, gang box, electrical box, mounting box, flush, surface, device box, outlet box, backbox type"}
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
PLANS_QUESTIONS = {
    "ARCHITECTURAL DRAWING SECTION": {
        "ELEVATION": [
            {"question": "Identify the Unit/Product Type mentioned (e.g., NULOOK, REGALS, MAJESTICS, CONSOLE, BED LOCATOR, BED BUMPERS, ARTWALL)."},
            {"question": "List the dimensions (width and height) for the units shown."},
            {"question": "List all required services and their quantities (e.g., Receptacles, Medgas outlets, Vacuum slides, Switches, Data/Aux/Telephone, Nurse Calls, Monitor channels)."},
            {"question": "What are the specified service locations or heights?"},
            {"question": "Identify any out-of-scope services mentioned, such as Vents, Grilles, Dialysis Box, or Over-Bed Lights (OBL)."}
        ],
        "FLOOR PLAN": [
            {"question": "List the Room Numbers and the total quantity of rooms."},
            {"question": "What are the Room Types or Names (e.g., Patient Room, ISO, Bariatric, ICU, PACU, LDR)?"},
            {"question": "Determine the unit mounting type (Surface Mount or Recessed)."},
            {"question": "Describe the headwall location and note any potential obstructions like windows, columns, or sinks."},
            {"question": "What is the room orientation (As Shown or Mirrored)?"}
        ],
        "PARTITION PLAN/TYPE": [
            {"question": "What is the partition type, including stud size, wall thickness, and fire rating?"},
            {"question": "If the partition is fire-rated, is the headwall specified as surface mounted?"}
        ],
        "CEILING PLAN": [
            {"question": "What is the ceiling height? Note any changes in height within a room."},
            {"question": "Are there any obstructions above the unit, such as bulkheads or soffits?"}
        ],
        "FINISH PLAN AND SCHEDULE": [
            {"question": "What is the specified laminate for the units (e.g., Thermofoil or P-LAM)?"},
            {"question": "Is there any information on resin panels or backlighting, such as 3form details?"}
        ],
        "DETAILED SECTION": [
            {"question": "What is the specified height of the headwall from the ceiling or floor?"},
            {"question": "Is there a detailed top plate specification?"},
            {"question": "Are there details for reveals or rails, like Fry Reglet?"}
        ]
    },
    "ENGINEERING DRAWING SECTION": {
        "PLUMBING PLAN": [
            {"question": "What are the medical gas requirements (e.g., Oxygen, Med Air, Vacuum)?"},
            {"question": "What are the specified pipe sizes for medgas? (Standard is ½” OXY, ½” MED AIR, ¾” VACUUM)."},
            {"question": "Identify the headwall location from the plumbing plan."},
            {"question": "Is the system a single Point of Connection (POC) or multiple?"}
        ],
        "ELECTRICAL PLAN - POWER": [
            {"question": "List the types and quantities of receptacles (e.g., Standard, Tamperproof, GFCI, USB, Simplex, Quadplex)."},
            {"question": "What are the receptacle power types (Normal, Emergency, UPS)?"},
            {"question": "Describe the circuitry details, including panel source, circuit number, and receptacle groupings."}
        ],
        "ELECTRICAL PLAN - LIGHTING": [
            {"question": "List the types and quantities of switches (e.g., Momentary, Single Pole, Three Way, Dimmers)."},
            {"question": "Describe the lighting connections from switches."},
            {"question": "What is the specified lighting voltage (120V or 277V)?"}
        ],
        "LOW VOLTAGE / TECHNOLOGY PLAN": [
            {"question": "List all low voltage devices and their quantities (e.g., Nurse Calls, Data, Aux, Telephone)."}
        ]
    }
}
# --- Caching for Global Resources ---
@st.cache_resource
def get_llm_and_embeddings():
    """Initializes and caches the LLM and Embedding models for the entire app."""
    st.info("Loading AI models... This happens once per application start.")
    llm = OllamaLLM(model=GENERATIVE_MODEL, temperature=0.0)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    # Ensure the parent directory for all session DBs exists
    os.makedirs(PARENT_DB_DIRECTORY, exist_ok=True)
    return llm, embeddings

# --- Document Processing Class for Session State ---
class DocumentProcessor:
    def __init__(self, pdf_bytes, pdf_name, session_id):
        self.pdf_bytes = pdf_bytes
        self.pdf_name = pdf_name
        self.session_id = session_id
        # CRITICAL: Each user gets a unique, session-specific directory
        self.db_directory = os.path.join(PARENT_DB_DIRECTORY, f"session_{self.session_id}")
        self.vector_store = None
        self.timings = []

        # Clean up any old directories from this specific session first
        if os.path.exists(self.db_directory):
            shutil.rmtree(self.db_directory)
        os.makedirs(self.db_directory)

    def _extract_text(self):
        with fitz.open(stream=self.pdf_bytes, filetype="pdf") as doc:
            return "".join(page.get_text() for page in doc)

    def _format_docs(self, docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def analyze(self, llm, embeddings, questions_dict, analysis_type):
        """Main analysis pipeline for the document."""
        with st.status(f"Step 1/4: Extracting text from '{self.pdf_name}'...", expanded=True) as status:
            s_time = time.perf_counter()
            full_text = self._extract_text()
            self.timings.append(("Step 1: Text Extraction", time.perf_counter() - s_time))
            if not full_text:
                status.update(label="Text extraction failed.", state="error")
                return None
            status.update(label=f"Step 1 complete in {self.timings[-1][1]:.2f}s", state="complete")

        with st.status("Step 2/4: Chunking document...", expanded=True) as status:
            s_time = time.perf_counter()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunks = text_splitter.split_text(full_text)
            documents = [Document(page_content=chunk, metadata={"source": self.pdf_name}) for chunk in chunks]
            self.timings.append(("Step 2: Document Chunking", time.perf_counter() - s_time))
            status.update(label=f"Step 2 complete ({len(chunks)} chunks) in {self.timings[-1][1]:.2f}s", state="complete")

        with st.status("Step 3/4: Creating embeddings...", expanded=True) as status:
            s_time = time.perf_counter()
            self.vector_store = Chroma.from_documents(documents, embeddings, persist_directory=self.db_directory)
            self.timings.append(("Step 3: Embedding & DB Storage", time.perf_counter() - s_time))
            status.update(label=f"Step 3 complete in {self.timings[-1][1]:.2f}s", state="complete")

        with st.status("Step 4/4: Analyzing sections in parallel...", expanded=True) as status:
            s_time = time.perf_counter()
            template = """You are an expert at extracting technical information from construction documents.
            Answer the QUESTION based **ONLY** on the provided CONTEXT. If the information is not found after a thorough check, state: "Information not found."
            Be concise and direct.
            CONTEXT: {context}
            QUESTION: {questions}
            YOUR ANSWER:"""
            prompt = PromptTemplate.from_template(template)
            base_retriever = self.vector_store.as_retriever(search_kwargs={'k': RETRIEVER_K})
            compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2", top_n=RERANK_TOP_N)
            retriever_with_rerank = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

            setup_and_retrieval = RunnableParallel(
                context=(lambda x: x["query"]) | retriever_with_rerank | self._format_docs,
                questions=(lambda x: x["questions"])
            )
            rag_chain = setup_and_retrieval | prompt | llm | StrOutputParser()

            all_results_flat = self._run_parallel_analysis(rag_chain, questions_dict, analysis_type)
            self.timings.append(("Step 4: Parallel Analysis", time.perf_counter() - s_time))
            status.update(label=f"Step 4 complete in {self.timings[-1][1]:.2f}s", state="complete")

        return pd.DataFrame(all_results_flat)

    def _run_parallel_analysis(self, rag_chain, questions_dict, analysis_type):
        """Helper to run analysis tasks concurrently."""
        tasks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for category, sub_categories in questions_dict.items():
                for sub_category_title, details in sub_categories.items():
                    if analysis_type == "Specifications":
                        for spec_item in details['specifications']:
                            question, terms = spec_item['question'], spec_item['related_terms']
                            query = f"In section {sub_category_title}: {question} {terms}"
                            tasks.append(executor.submit(self._run_single_item, rag_chain, category, sub_category_title, question, query))
                    elif analysis_type == "Plans":
                        for question_item in details:
                            question = question_item['question']
                            query = f"From the {sub_category_title} drawings, find: {question}"
                            tasks.append(executor.submit(self._run_single_item, rag_chain, category, sub_category_title, question, query))
        return [future.result() for future in concurrent.futures.as_completed(tasks)]

    def _run_single_item(self, rag_chain, category, sub_category, question, retrieval_query):
        """Executes a single RAG chain invocation."""
        try:
            result = rag_chain.invoke({"query": retrieval_query, "questions": question})
            return {"Category": category, "SubCategory": sub_category, "Question": question, "Result": result}
        except Exception as e:
            return {"Category": category, "SubCategory": sub_category, "Question": question, "Result": f"Error: {e}"}

    def cleanup(self):
        """Removes the session-specific ChromaDB directory."""
        st.info(f"Cleaning up resources for session {self.session_id}...")
        if hasattr(self, "vector_store"):
            del self.vector_store
        # del self.vector_store
        gc.collect() # Force garbage collection
        if os.path.exists(self.db_directory):
            try:
                shutil.rmtree(self.db_directory)
                st.success(f"Successfully cleaned up temporary database for session.")
            except Exception as e:
                st.warning(f"Could not remove temporary DB for session: {e}")

# --- PDF Generation (Unchanged) ---
def generate_pdf_with_table(report_df, timings_data, analysis_type):
    # This function is omitted for brevity but should be included in your final script.
    # The original implementation is correct and doesn't need changes.
    """Generates a PDF report from the analysis DataFrame and timings data."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    story = []

    report_title = f"{analysis_type} Analysis Report"
    story.append(Paragraph(report_title, styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Performance Breakdown", styles["Heading2"]))
    total_time = sum(d for _, d in timings_data)
    for label, duration in timings_data:
        story.append(Paragraph(f"{label}: {duration:.2f} seconds", styles["Normal"]))
    story.append(Paragraph(f"<b>Total Processing Time:</b> {total_time:.2f} seconds", styles["Normal"]))
    story.append(Spacer(1, 24))

    story.append(Paragraph("Detailed Analysis", styles["Heading2"]))
    story.append(Spacer(1, 12))

    if not report_df.empty:
        # Define column names based on analysis type
        if analysis_type == "Specifications":
            col_names = ["Division", "Section", "Specification", "Result"]
            report_df.columns = ["Category", "SubCategory", "Question", "Result"] # Use generic names for processing
        else: # Plans
            col_names = ["Drawing Section", "Plan Type", "Information Requested", "Result"]
            report_df.columns = ["Category", "SubCategory", "Question", "Result"]

        # Group by the primary category for sections in PDF
        for category, category_df in report_df.groupby("Category"):
            story.append(Paragraph(f"<b>{category}</b>", styles["Heading3"]))
            story.append(Spacer(1, 6))

            table_data = [[col_names[1], col_names[2], col_names[3]]] # Header row
            category_df_sorted = category_df.sort_values(by=["SubCategory", "Question"])

            for _, row in category_df_sorted.iterrows():
                result_text = str(row["Result"]).replace("\n", "<br/>")
                row_data = [
                    Paragraph(row["SubCategory"], styles["Normal"]),
                    Paragraph(row["Question"], styles["Normal"]),
                    Paragraph(f"<font color='red'>{result_text}</font>" if "Information not found" in row["Result"] else result_text, styles["Normal"])
                ]
                table_data.append(row_data)

            if len(table_data) > 1:
                col_widths = [1.5*inch, 2.0*inch, 3.5*inch] # Adjust as needed
                table = Table(table_data, colWidths=col_widths)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#ADD8E6')),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.black),
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0,0), (-1,0), 12),
                    ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#D3D3D3')),
                ]))
                story.append(table)
                story.append(Spacer(1, 24))

    else:
        story.append(Paragraph("No analysis results available.", styles["Normal"]))

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
# --- Streamlit UI ---
st.title("📄 Document Analyzer")
st.markdown("Upload a PDF to extract key information. This system supports multiple concurrent users.")

# Load models once for the entire application
llm, embeddings = get_llm_and_embeddings()

# Get the unique session ID for this user
session_id = st.runtime.scriptrunner.get_script_run_ctx().session_id

# --- Main App Logic ---
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type="pdf",
    key=f"file_uploader_{session_id}" # Keying widget to session
)

if uploaded_file:
    # --- Sidebar for options ---
    st.sidebar.header("Analysis Options")
    analysis_type = st.sidebar.radio(
        "Select document type:",
        ("Specifications", "Plans"),
        key=f"analysis_type_{session_id}"
    )

    if st.button(f"Analyze '{uploaded_file.name}'", key=f"analyze_button_{session_id}"):
        # If there's an old processor in the session, clean it up first
        if 'processor' in st.session_state:
            st.session_state.processor.cleanup()

        # Create a new processor for the current file and store it in the session state
        st.session_state.processor = DocumentProcessor(
            pdf_bytes=uploaded_file.getvalue(),
            pdf_name=uploaded_file.name,
            session_id=session_id
        )

        # Run the analysis
        questions_to_ask = KEYWORDS_AND_SPECS if analysis_type == "Specifications" else PLANS_QUESTIONS
        report_df = st.session_state.processor.analyze(llm, embeddings, questions_to_ask, analysis_type)

        # Store results in session state
        st.session_state.report_df = report_df
        st.session_state.timings = st.session_state.processor.timings
        st.session_state.analysis_type = analysis_type
        st.session_state.uploaded_filename = uploaded_file.name

        # Trigger a rerun to display the results
        st.rerun()

# --- DISPLAY RESULTS ---
if 'report_df' in st.session_state and st.session_state.report_df is not None:
    st.header("Performance Breakdown")
    total_time = sum(duration for _, duration in st.session_state.timings)
    with st.container(border=True):
        for label, duration in st.session_state.timings:
            st.markdown(f"**{label}:** `{duration:.2f}` seconds")
        st.divider()
        st.metric(label="**Total Processing Time**", value=f"{total_time:.2f} seconds")

    st.header("Analysis Report")
    report_df = st.session_state.report_df
    analysis_type = st.session_state.analysis_type

    if not report_df.empty:
        if analysis_type == "Specifications":
            display_df = report_df.rename(columns={"Category": "Division", "SubCategory": "Section", "Question": "Specification"})
        else:
            display_df = report_df.rename(columns={"Category": "Drawing Section", "SubCategory": "Plan Type", "Question": "Information Requested"})

        for category, group in display_df.groupby(display_df.columns[0]):
            with st.expander(f"**{category}**", expanded=True):
                for sub_category, sub_group in group.groupby(display_df.columns[1]):
                    st.subheader(sub_category)
                    st.dataframe(sub_group.drop(columns=display_df.columns[:2]), hide_index=True, use_container_width=True)

        pdf_bytes = generate_pdf_with_table(report_df.copy(), st.session_state.timings, analysis_type)
        st.download_button(
            label="Download Report as PDF",
            data=pdf_bytes,
            file_name=f"{st.session_state.uploaded_filename.replace('.pdf', '')}_report.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("Analysis completed, but no results were generated.")
else:
    st.info("Upload a PDF file and click 'Analyze' to start.")