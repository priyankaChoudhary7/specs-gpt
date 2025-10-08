import streamlit as st
import fitz
import base64
import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_ollama.chat_models import ChatOllama
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Local Drawing Analyzer v2",
    page_icon="",
    layout="wide"
)

# --- Core Functions ---

@st.cache_data(show_spinner=False)
def convert_pdf_page_to_image_bytes(_pdf_bytes, page_number):
    """Converts a specific page of a PDF into high-resolution PNG image bytes."""
    try:
        doc = fitz.open(stream=_pdf_bytes, filetype="pdf")
        if page_number < 0 or page_number >= doc.page_count:
            return None
        page = doc.load_page(page_number)
        # Use a higher DPI for better clarity in complex drawings
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        return img_bytes
    except Exception as e:
        st.error(f"Error converting PDF page to image: {e}")
        return None

def image_to_base64_str(image_bytes):
    """Converts image bytes to a base64 encoded string."""
    return base64.b64encode(image_bytes).decode('utf-8')

def analyze_drawing_with_specific_question(image_bytes, question):
    """
    Analyzes a drawing image with a single, focused question using the local LLaVA model.
    """
    try:
        # Initialize the connection to the local Ollama model
        llm = ChatOllama(model="llava:13b", temperature=0.0)
        base64_image = image_to_base64_str(image_bytes)

        # A more focused prompt template
        prompt_template = f"""
        You are an expert AI assistant analyzing a construction drawing.
        Based ONLY on the provided image, answer the following specific question.
        
        Question: '{question}'
        
        Provide a direct, concise answer. If the information is not present, you MUST state "Information not found."
        """

        msg = llm.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt_template},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{base64_image}",
                        },
                    ]
                )
            ]
        )
        # Clean the response
        return msg.content.strip().replace('"', '')
    except Exception as e:
        st.error(f"Failed to connect to Ollama. Is it running with the 'llava:13b' model? Error: {e}")
        return "Error connecting to model"


# --- Structured Questions ---
# This dictionary drives the iterative analysis process.
QUESTIONS_TO_ASK = {
    "Architectural - Elevation": [
        "UNIT/PRODUCT TYPE (e.g., NULOOK, REGALS, Patient Lift)",
        "SIZES – DIMENSIONS (WIDTH AND HEIGHT)",
        "SERVICES REQUIRED (RECEPTACLES, MEDGAS, SWITCHES, DATA)",
        "SERVICE LOCATION (SERVICE HEIGHTS)",
    ],
    "Architectural - Floor Plan": [
        "ROOM NUMBERS AND QUANTITY",
        "ROOM TYPE/NAME (e.g., PATIENT ROOM, ICU, WASHROOM)",
        "UNIT TYPE (SURFACE MOUNT OR RECESSED)",
        "HEADWALL LOCATION AND OBSTRUCTIONS (WINDOWS, COLUMNS, SINKS)",
    ],
    "Architectural - Partition Plan": [
        "PARTITION TYPE (STUD SIZE, WALL THICKNESS, FIRE RATING)",
    ],
    "Engineering - Plumbing Plan": [
        "MEDGAS REQUIREMENTS (OXYGEN, MED AIR, VACUUM)",
        "PIPE SIZES for medical gas",
    ],
    "Engineering - Electrical Plan": [
        "RECEPTACLE TYPES (STANDARD, TAMPERPROOF, GFCI)",
        "RECEPTACLE POWER TYPE (NORMAL, EMERGENCY, UPS)",
        "CIRCUITRY DETAILS (PANEL SOURCE, CIRCUIT NUMBER)",
        "SWITCH TYPES (MOMENTARY, SINGLE POLE, DIMMERS)",
        "LIGHTING VOLTAGE (120V OR 277V)",
    ],
    "Engineering - Low Voltage Plan": [
        "LOW VOLTAGE DEVICES (NURSE CALLS, DATA, AUX)",
    ]
}


# --- Streamlit User Interface ---
def main():
    st.title("Local Construction Drawing Analyzer v2")
    st.info(
        "**How it works:** Upload a drawing PDF. The app will analyze each page, asking a series of specific questions to improve accuracy. "
        "Results will appear in real-time below. \n\n"
        "**Important:** Ensure the Ollama application is running and you have downloaded the `llava:13b` model."
    )

    uploaded_file = st.file_uploader(
        "Choose a drawing PDF...", type="pdf"
    )

    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' uploaded successfully.")
        
        if st.button("Analyze Drawing", type="primary"):
            pdf_bytes = uploaded_file.getvalue()
            
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                num_pages = doc.page_count
                st.write(f"Found {num_pages} page(s) in the document. Starting analysis...")

                all_results = []

                for i in range(num_pages):
                    page_number = i + 1
                    with st.expander(f"**Analyzing Page {page_number} of {num_pages}**", expanded=True):
                        
                        image_bytes = convert_pdf_page_to_image_bytes(pdf_bytes, i)
                        if not image_bytes:
                            st.warning(f"Could not process page {page_number}.")
                            continue
                        
                        st.image(image_bytes, caption=f"Drawing - Page {page_number}", use_column_width=True)
                        
                        st.subheader("Extracted Information:")
                        placeholder = st.empty()
                        
                        page_results = []

                        # Iteratively ask each question for the current page
                        for category, questions in QUESTIONS_TO_ASK.items():
                            for question in questions:
                                with st.spinner(f"Page {page_number}: Asking about '{question}'..."):
                                    # Analyze the image with one specific question
                                    answer = analyze_drawing_with_specific_question(image_bytes, question)
                                    
                                    # Add result to the list
                                    result_entry = {
                                        "Page": page_number,
                                        "Category": category,
                                        "Information Requested": question,
                                        "Result": answer
                                    }
                                    page_results.append(result_entry)
                                    
                                    # Update the displayed dataframe in real-time
                                    df = pd.DataFrame(page_results)
                                    placeholder.dataframe(df, use_container_width=True, hide_index=True)
                                    
                                    # Small delay to prevent overwhelming the server
                                    time.sleep(0.5)
                        
                        all_results.extend(page_results)

                st.balloons()
                st.success("Analysis Complete!")
                st.subheader("Final Combined Report")
                final_df = pd.DataFrame(all_results)
                st.dataframe(final_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"An unexpected error occurred while processing the PDF: {e}")

if __name__ == "__main__":
    main()