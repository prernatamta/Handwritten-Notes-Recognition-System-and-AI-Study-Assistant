
import os

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import io
import cv2
import numpy as np
import streamlit as st
import torch

from PIL import Image

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.styles import getSampleStyleSheet

from study_assistant import render_study_assistant


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Scriptify",
    page_icon="📘",
    layout="wide"
)


# =====================================================
# SESSION STATE
# =====================================================

defaults = {
    "dark_mode": False,
    "detected_lines": [],
    "lines_ready": False,
    "extracted_text": "",
    "study_notes": "",
    "digitize_image_name": ""
}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# =====================================================
# THEME
# =====================================================

def apply_theme(dark):

    if dark:

        background = "#101827"
        sidebar = "#111D31"
        card = "#1B2A41"
        text = "#F1F5F9"
        muted = "#B5C2D3"
        border = "#34445C"

    else:

        background = "#F4F7FB"
        sidebar = "#172B49"
        card = "#FFFFFF"
        text = "#172B49"
        muted = "#64748B"
        border = "#DFE6EF"

    st.markdown(f"""
    <style>

    .stApp {{
        background-color: {background};
        color: {text};
    }}

    [data-testid="stSidebar"] {{
        background-color: {sidebar};
    }}

    [data-testid="stSidebar"] * {{
        color: white !important;
    }}

    .main-title {{
        font-size: 36px;
        font-weight: 800;
        color: {text};
        margin-bottom: 5px;
    }}

    .subtitle {{
        color: {muted};
        font-size: 15px;
        margin-bottom: 25px;
    }}

    .info-card {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 15px;
    }}

    .info-card h3 {{
        color: {text};
    }}

    .info-card p {{
        color: {muted};
        font-size: 14px;
    }}

    .stTextArea textarea,
    .stTextInput input {{
        background-color: {card} !important;
        color: {text} !important;
        border: 1px solid {border} !important;
    }}

    [data-testid="stMetric"] {{
        background-color: {card};
        padding: 15px;
        border-radius: 12px;
        border: 1px solid {border};
    }}

    div.stButton > button {{
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }}

    div.stButton > button:hover {{
        background-color: #1D4ED8;
        color: white;
    }}

    hr {{
        border-color: {border};
    }}

    </style>
    """, unsafe_allow_html=True)


# =====================================================
# PAGE HEADER
# =====================================================

def page_header(title, subtitle):

    st.markdown(
        f'<div class="main-title">{title}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )


# =====================================================
# TROCR MODEL
# =====================================================

@st.cache_resource
def load_trocr():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )

    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )

    model.to(device)
    model.eval()

    return processor, model, device


# =====================================================
# IMAGE PREPROCESSING
# =====================================================

def preprocess_image(pil_image):

    image = np.array(
        pil_image.convert("RGB")
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        10,
        7,
        21
    )

    return denoised


# =====================================================
# LINE SEGMENTATION
# =====================================================

def segment_lines(processed_image):

    height, width = processed_image.shape

    binary = cv2.adaptiveThreshold(
        processed_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(40, width // 4), 1)
    )

    horizontal_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    cleaned = cv2.subtract(
        binary,
        horizontal_lines
    )

    small_kernel = np.ones(
        (2, 2),
        np.uint8
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        small_kernel
    )

    join_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(15, width // 80), 3)
    )

    connected = cv2.dilate(
        cleaned,
        join_kernel,
        iterations=1
    )

    contours, hierarchy = cv2.findContours(
        connected,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        if w < width * 0.08:
            continue

        if h < 8 or h > height * 0.25:
            continue

        boxes.append(
            (x, y, x + w, y + h)
        )

    boxes.sort(
        key=lambda box: box[1]
    )

    merged = []

    for box in boxes:

        if not merged:

            merged.append(box)
            continue

        previous = merged[-1]

        vertical_gap = box[1] - previous[3]

        overlap = min(
            box[3],
            previous[3]
        ) - max(
            box[1],
            previous[1]
        )

        if overlap > 0 or vertical_gap < 12:

            merged[-1] = (
                min(previous[0], box[0]),
                min(previous[1], box[1]),
                max(previous[2], box[2]),
                max(previous[3], box[3])
            )

        else:

            merged.append(box)

    line_images = []

    for x1, y1, x2, y2 in merged:

        x1 = max(0, x1 - 12)
        y1 = max(0, y1 - 10)

        x2 = min(width, x2 + 12)
        y2 = min(height, y2 + 10)

        crop = processed_image[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:
            continue

        if crop.shape[1] < 30:
            continue

        line_images.append(crop)

    return line_images, cleaned


# =====================================================
# RECOGNISE SELECTED LINES
# =====================================================

def recognise_selected_lines(selected_lines):

    processor, model, device = load_trocr()

    results = []

    progress = st.progress(0)
    status = st.empty()

    total = len(selected_lines)

    for index, line in enumerate(selected_lines):

        status.write(
            f"Recognising line {index + 1} of {total}"
        )

        rgb = cv2.cvtColor(
            line,
            cv2.COLOR_GRAY2RGB
        )

        pil_line = Image.fromarray(rgb)

        pixel_values = processor(
            images=pil_line,
            return_tensors="pt"
        ).pixel_values

        pixel_values = pixel_values.to(device)

        with torch.no_grad():

            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=128
            )

        text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        results.append(text.strip())

        progress.progress(
            (index + 1) / total
        )

    status.success(
        "TrOCR recognition completed."
    )

    progress.empty()

    return "\n".join(
        text for text in results if text
    )


# =====================================================
# PDF GENERATION
# =====================================================

def create_pdf(text):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    story = [
        Paragraph(
            "Scriptify - Digitized Notes",
            styles["Title"]
        ),
        Spacer(1, 20)
    ]

    for paragraph in text.split("\n"):

        safe_text = (
            paragraph
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        story.append(
            Paragraph(
                safe_text if safe_text.strip() else " ",
                styles["Normal"]
            )
        )

        story.append(
            Spacer(1, 8)
        )

    document.build(story)

    buffer.seek(0)

    return buffer


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("# Scriptify")

    st.caption("Your personal learning workspace")

    st.divider()

    dark_mode = st.toggle(
        "Dark mode",
        value=st.session_state.dark_mode
    )

    if dark_mode != st.session_state.dark_mode:

        st.session_state.dark_mode = dark_mode
        st.rerun()

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Home",
            "Digitize Notes",
            "Study Assistant"
        ]
    )

    st.divider()

    st.caption("OCR: Microsoft TrOCR")
    st.caption("Preprocessing: OpenCV")
    st.caption("AI: Groq")


apply_theme(
    st.session_state.dark_mode
)


# =====================================================
# HOME PAGE
# =====================================================

if page == "Home":

    page_header(
        "Welcome to Scriptify",
        "Transform handwritten notes into an intelligent digital study experience."
    )

    st.markdown("""
    <div class="info-card">
        <h3>Learn smarter. Revise better.</h3>
        <p>
        Digitize handwritten notes, edit the extracted content
        and use AI tools to prepare for examinations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        <div class="info-card">
            <h3>01. Digitize</h3>
            <p>Detect, inspect and select individual handwritten lines.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="info-card">
            <h3>02. Understand</h3>
            <p>Generate summaries and identify important concepts.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="info-card">
            <h3>03. Practise</h3>
            <p>Create quizzes and topic-wise revision material.</p>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Detected lines",
            len(st.session_state.detected_lines)
        )

    with col2:

        st.metric(
            "Extracted characters",
            len(st.session_state.extracted_text)
        )


# =====================================================
# DIGITIZE NOTES PAGE
# =====================================================

elif page == "Digitize Notes":

    page_header(
        "Digitize Your Notes",
        "Upload, inspect, select and recognise handwritten lines."
    )

    uploaded_file = st.file_uploader(
        "Upload handwritten notes",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        pil_image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.markdown("### Original image")

        st.image(
            pil_image,
            use_container_width=True
        )

        if st.button(
            "Detect Handwritten Lines",
            use_container_width=True
        ):

            with st.spinner(
                "Preprocessing and detecting lines..."
            ):

                processed = preprocess_image(
                    pil_image
                )

                lines, cleaned = segment_lines(
                    processed
                )

            st.session_state.detected_lines = lines

            st.session_state.processed_preview = processed

            st.session_state.segmentation_preview = cleaned

            st.session_state.lines_ready = True

            st.session_state.extracted_text = ""

            st.session_state.digitize_image_name = uploaded_file.name

            if not lines:

                st.warning(
                    "No lines detected. Try another image."
                )

            else:

                st.success(
                    f"{len(lines)} lines detected."
                )

    if st.session_state.lines_ready:

        st.divider()

        st.markdown("## Preprocessing results")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("**Enhanced image**")

            st.image(
                st.session_state.processed_preview,
                use_container_width=True
            )

        with col2:

            st.markdown("**Segmentation mask**")

            st.image(
                st.session_state.segmentation_preview,
                use_container_width=True
            )

        st.divider()

        st.markdown("## Select the correct lines")

        lines = st.session_state.detected_lines

        total = len(lines)

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "Select All",
                use_container_width=True
            ):

                for i in range(total):

                    st.session_state[
                        f"line_select_{i}"
                    ] = True

                st.rerun()

        with col2:

            if st.button(
                "Deselect All",
                use_container_width=True
            ):

                for i in range(total):

                    st.session_state[
                        f"line_select_{i}"
                    ] = False

                st.rerun()

        selected_indices = []

        for start in range(0, total, 2):

            columns = st.columns(2)

            for offset in range(2):

                index = start + offset

                if index >= total:
                    continue

                with columns[offset]:

                    with st.container(border=True):

                        st.markdown(
                            f"**Line {index + 1}**"
                        )

                        st.image(
                            lines[index],
                            use_container_width=True
                        )

                        checked = st.checkbox(
                            "Select this line",
                            value=True,
                            key=f"line_select_{index}"
                        )

                        if checked:

                            selected_indices.append(index)

        st.info(
            f"{len(selected_indices)} of {total} lines selected."
        )

        if st.button(
            "Recognise Selected Lines with TrOCR",
            use_container_width=True
        ):

            if not selected_indices:

                st.warning(
                    "Select at least one line."
                )

            else:

                selected_lines = [
                    lines[i] for i in selected_indices
                ]

                try:

                    extracted = recognise_selected_lines(
                        selected_lines
                    )

                    st.session_state.extracted_text = extracted

                    st.session_state.study_notes = extracted

                    st.success(
                        "Selected lines recognised successfully."
                    )

                except Exception as error:

                    st.error(
                        "TrOCR Error: " + str(error)
                    )

        if st.session_state.extracted_text:

            st.divider()

            st.markdown("## Editable extracted text")

            edited_text = st.text_area(
                "Review and edit your notes",
                value=st.session_state.extracted_text,
                height=300,
                key="digitized_text_editor"
            )

            st.session_state.extracted_text = edited_text

            st.session_state.study_notes = edited_text

            col1, col2 = st.columns(2)

            with col1:

                st.download_button(
                    "Download TXT",
                    data=edited_text,
                    file_name="digitized_notes.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            with col2:

                st.download_button(
                    "Download PDF",
                    data=create_pdf(edited_text),
                    file_name="digitized_notes.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )


# =====================================================
# STUDY ASSISTANT PAGE
# =====================================================

elif page == "Study Assistant":

    page_header(
        "AI Study Assistant",
        "Your personal AI-powered learning and revision workspace."
    )

    notes = st.text_area(
        "Study material",
        value=st.session_state.study_notes or st.session_state.extracted_text,
        height=220,
        key="study_material_editor"
    )

    st.session_state.study_notes = notes

    render_study_assistant(notes)


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Scriptify | Handwritten Notes Digitization and AI Study Assistant"
)
