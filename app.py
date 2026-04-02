import streamlit as st
import os
import tempfile
import pandas as pd
import json
from PIL import Image

from ocr_engine import PaddleOCREngine
from qwen_engine import QwenExtractor
from matcher import highlight_and_save_pdf, highlight_single_field

try:
    from pdf2image import convert_from_path
except ImportError:
    pass

# UI Configuration
st.set_page_config(page_title="Document AI Extractor", layout="wide", page_icon="📄")

st.title("📄 Intelligent Document Extraction Pipeline")
st.markdown("Upload a Document (PDF/Image) to instantly extract structured semantic fields, match them precisely to OCR coordinates, and generate a highlighted verification PDF.")

# Cache the AI models so they don't reload every time the user clicks a button!
@st.cache_resource(show_spinner=False)
def load_ai_models():
    with st.spinner("Loading Vision-Language Model and OCR Engines (First run only)..."):
        qwen = QwenExtractor()
        
        # NOTE: If your local machine doesn't have an Nvidia GPU installed, you may need to set use_gpu=False
        try:
            ocr = PaddleOCREngine(use_gpu=True) 
        except Exception:
            ocr = PaddleOCREngine(use_gpu=False)
            
        return qwen, ocr


def load_document_images(file_path):
    """Load the original document pages as PIL images."""
    if file_path.lower().endswith(".pdf"):
        return convert_from_path(file_path)
    else:
        return [Image.open(file_path).convert("RGB")]


uploaded_file = st.file_uploader("Upload an Invoice, Claim, or Form", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    
    # Save the uploaded file temporarily so the backend engines can read it from a path
    file_bytes = uploaded_file.read()
    file_extension = os.path.splitext(uploaded_file.name)[1]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
        
    st.success(f"**{uploaded_file.name}** uploaded safely to memory!")
    
    if st.button("🚀 Run AI Extraction Pipeline", use_container_width=True, type="primary"):
        try:
            # 1. Load Models
            qwen_extractor, ocr_engine = load_ai_models()
            
            # 2. Extract Logic
            st.markdown("### Pipeline Execution Steps:")
            
            with st.spinner("🧠 Step 1/3: Running Qwen 2.5 Vision-Language Model..."):
                qwen_data = qwen_extractor.extract_data(temp_path)
                st.success("✅ Step 1: Semantic understanding complete!")
                
            with st.spinner("🔍 Step 2/3: Running PaddleOCR engine across all pages..."):
                ocr_data = ocr_engine.extract_text_with_confidence(temp_path)
                st.success("✅ Step 2: Pixel-level word extraction complete!")
                
            with st.spinner("🔗 Step 3/3: Running Anchor & Spatial Matching to link Qwen with OCR..."):
                output_pdf = temp_path + "_highlighted.pdf"
                output_csv = output_pdf.replace(".pdf", ".csv").replace(".jpg", ".csv")
                
                # Run the matcher which draws the boxes, saves the PDF, and outputs the CSV
                all_matched = highlight_and_save_pdf(temp_path, qwen_data, ocr_data, output_pdf)
                st.success("✅ Step 3: Visual highlighting and alignment CSV generated!")
                
            # --- Store results in session state for interactive highlighting ---
            original_images = load_document_images(temp_path)
            st.session_state['matched_results'] = all_matched if all_matched else []
            st.session_state['original_images'] = original_images
            st.session_state['qwen_data'] = qwen_data
            st.session_state['output_pdf'] = output_pdf
            st.session_state['output_csv'] = output_csv
            st.session_state['uploaded_name'] = uploaded_file.name
            st.session_state['selected_field_idx'] = None  # Reset selection
            st.session_state['pipeline_done'] = True
                
        except Exception as e:
            st.error(f"Pipeline crashed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# =========================================================================
# RESULTS DASHBOARD — Only shows after pipeline has run
# =========================================================================
if st.session_state.get('pipeline_done'):
    
    matched_results = st.session_state.get('matched_results', [])
    original_images = st.session_state.get('original_images', [])
    qwen_data = st.session_state.get('qwen_data', {})
    output_pdf = st.session_state.get('output_pdf', '')
    output_csv = st.session_state.get('output_csv', '')
    uploaded_name = st.session_state.get('uploaded_name', 'document')

    st.divider()
    st.header("Results Dashboard")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Raw LLM Extraction (JSON)")
        st.json(qwen_data)
        
    with col2:
        st.subheader("Final Matched Entities")
        if os.path.exists(output_csv):
            df = pd.read_csv(output_csv)
            # Color formatting trick for highlighting missed values in red
            def highlight_missed(row):
                if row["OCR_Matched_Text"] == "NO MATCH":
                    return ['background-color: #ffcccc'] * len(row)
                return [''] * len(row)
                    
            st.dataframe(df.style.apply(highlight_missed, axis=1), use_container_width=True, height=500)
        else:
            st.warning("CSV Data Not Found.")

    # =====================================================================
    # INTERACTIVE FIELD HIGHLIGHTING — Click a field to see it on the image
    # =====================================================================
    st.divider()
    st.header("🔍 Interactive Field Highlighter")
    st.markdown("Click on any matched field below to highlight **only that value** on the original document image.")

    # Build list of matched fields that have bounding boxes
    matched_with_bbox = [
        (i, res) for i, res in enumerate(matched_results) if res.get('bbox')
    ]
    unmatched = [
        (i, res) for i, res in enumerate(matched_results) if not res.get('bbox')
    ]

    if matched_with_bbox:
        # Create a grid of buttons for each matched field
        st.markdown("#### ✅ Matched Fields (click to highlight)")
        
        # Display as rows of 4 buttons
        cols_per_row = 4
        for row_start in range(0, len(matched_with_bbox), cols_per_row):
            row_items = matched_with_bbox[row_start:row_start + cols_per_row]
            cols = st.columns(len(row_items))
            
            for col, (idx, res) in zip(cols, row_items):
                with col:
                    field_name = res['field']
                    qwen_val = res['qwen_value']
                    # Truncate long values for the button label
                    display_val = qwen_val if len(qwen_val) <= 25 else qwen_val[:22] + "..."
                    
                    btn_label = f"🔍 {field_name}\n\"{display_val}\""
                    if st.button(btn_label, key=f"highlight_btn_{idx}", use_container_width=True):
                        st.session_state['selected_field_idx'] = idx
    
    if unmatched:
        with st.expander(f"❌ Unmatched Fields ({len(unmatched)})", expanded=False):
            for idx, res in unmatched:
                st.markdown(f"- **{res['field']}**: `{res['qwen_value']}`")

    # --- Show the highlighted image when a field is selected ---
    selected_idx = st.session_state.get('selected_field_idx')
    
    if selected_idx is not None and selected_idx < len(matched_results):
        selected = matched_results[selected_idx]
        page_num = selected.get('page', 1)
        
        st.divider()
        st.subheader(f"📌 Highlighting: **{selected['field']}**")
        
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.metric("Field", selected['field'])
            st.metric("Qwen Value", selected['qwen_value'])
        with info_col2:
            st.metric("OCR Matched Text", selected.get('matched_ocr_text', 'N/A'))
            st.metric("Page", page_num)
        
        if selected.get('bbox') and page_num <= len(original_images):
            # Get the clean original image for that page
            clean_image = original_images[page_num - 1]
            
            # Highlight only the selected field
            highlighted_img = highlight_single_field(clean_image, selected)
            
            # Display the highlighted image inline
            st.image(
                highlighted_img,
                caption=f"Page {page_num} — Field: {selected['field']} = \"{selected['qwen_value']}\"",
                use_container_width=True
            )
        else:
            st.warning("No bounding box available for this field — it was not matched to any OCR text.")
    
    # --- Download Buttons (unchanged) ---
    st.divider()
    st.subheader("📥 Download Generated Artifacts")
    d_col1, d_col2 = st.columns(2)
    
    with d_col1:
        if os.path.exists(output_pdf):
            with open(output_pdf, "rb") as f:
                # Ensures the downloaded file format makes sense (pdf if available, otherwise original format)
                dl_name = f"Verified_{uploaded_name}"
                if not dl_name.lower().endswith('.pdf'):
                    dl_name = dl_name.rsplit('.', 1)[0] + '.pdf'
                    
                st.download_button(
                    label="Download Highlighted Document",
                    data=f,
                    file_name=dl_name,
                    mime="application/pdf" if dl_name.endswith('.pdf') else "image/jpeg",
                    type="primary"
                )
                    
    with d_col2:
        if os.path.exists(output_csv):
            with open(output_csv, "rb") as f:
                st.download_button(
                    label="Download Data Table (CSV)",
                    data=f,
                    file_name=f"Data_{uploaded_name}.csv",
                    mime="text/csv",
                    type="primary"
                )
