
import os
os.environ["USE_TF"] = "0"

import streamlit as st
import cv2
import numpy as np
import torch

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


st.set_page_config(
    page_title="Handwritten Notes Digitizer",
    page_icon="📖",
    layout="wide"
)

st.title("Handwritten Notes Digitizer")

st.subheader(
    "Handwritten Paragraph Recognition using OpenCV and TrOCR"
)

st.write(
    "Upload handwritten notes. The system detects individual "
    "lines and recognises them using TrOCR."
)


# Load TrOCR model

@st.cache_resource
def load_trocr():

    model_name = "microsoft/trocr-base-handwritten"

    processor = TrOCRProcessor.from_pretrained(model_name)

    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return processor, model, device


# Separate handwritten lines

def separate_lines(image):

    image_array = np.array(
        image.convert("RGB")
    )

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    height, width = gray.shape

    # Improve image quality

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Convert image into binary format

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15
    )

    # Detect long horizontal notebook lines

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(60, width // 2), 1)
    )

    horizontal_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    # Remove notebook lines only from segmentation mask

    text_mask = cv2.subtract(
        binary,
        horizontal_lines
    )

    # Small morphological closing to connect nearby character strokes

    connect_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, 3)
    )

    text_mask = cv2.morphologyEx(
        text_mask,
        cv2.MORPH_CLOSE,
        connect_kernel
    )

    # Calculate horizontal pixel projection

    projection = np.sum(
        text_mask > 0,
        axis=1
    ).astype(np.float32)

    # Smooth projection

    smooth_projection = np.convolve(
        projection,
        np.ones(5, dtype=np.float32) / 5,
        mode="same"
    )

    maximum = np.max(smooth_projection)

    if maximum == 0:

        return [], text_mask

    # Dynamic threshold

    threshold = max(
        3,
        maximum * 0.08
    )

    active_rows = np.where(
        smooth_projection > threshold
    )[0]

    if len(active_rows) == 0:

        return [], text_mask

    # Group active rows into handwritten lines

    groups = []

    start = active_rows[0]
    previous = active_rows[0]

    merge_gap = max(
        2,
        int(height * 0.003)
    )

    for row in active_rows[1:]:

        if row - previous <= merge_gap:

            previous = row

        else:

            groups.append(
                (start, previous)
            )

            start = row
            previous = row

    groups.append(
        (start, previous)
    )

    # Create individual line images

    line_images = []

    padding_y = max(
        10,
        int(height * 0.015)
    )

    padding_x = 15

    for y1, y2 in groups:

        if y2 - y1 < 8:
            continue

        line_region = text_mask[
            y1:y2 + 1,
            :
        ]

        columns = np.where(
            np.sum(line_region > 0, axis=0) > 0
        )[0]

        if len(columns) == 0:
            continue

        x1 = max(
            0,
            int(columns[0]) - padding_x
        )

        x2 = min(
            width,
            int(columns[-1]) + padding_x + 1
        )

        top = max(
            0,
            int(y1) - padding_y
        )

        bottom = min(
            height,
            int(y2) + padding_y + 1
        )

        # Crop from original image

        line_image = image_array[
            top:bottom,
            x1:x2
        ]

        if line_image.size > 0:

            line_images.append(
                Image.fromarray(line_image)
            )

    return line_images, text_mask


# Recognise individual line

def recognize_line(
    line_image,
    processor,
    model,
    device
):

    pixel_values = processor(
        images=line_image.convert("RGB"),
        return_tensors="pt"
    ).pixel_values.to(device)

    with torch.no_grad():

        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=4
        )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return text.strip()


# Main application

uploaded_file = st.file_uploader(
    "Upload handwritten notes",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Original Handwritten Image")

    st.image(
        image,
        use_container_width=True
    )

    # Line segmentation

    if st.button("Separate Handwritten Lines"):

        with st.spinner("Detecting handwritten lines..."):

            lines, text_mask = separate_lines(image)

            st.session_state["lines"] = lines

            st.session_state["recognized_text"] = ""

            st.session_state["debug_mask"] = text_mask

    # Display detected lines

    if "lines" in st.session_state:

        lines = st.session_state["lines"]

        st.subheader("Line Detection Result")

        if len(lines) == 0:

            st.error(
                "No handwritten lines detected. "
                "Try uploading a clearer image."
            )

        else:

            st.success(
                str(len(lines)) + " handwritten lines detected."
            )

            # Display cleaned detection mask

            with st.expander("View processed detection mask"):

                st.image(
                    st.session_state["debug_mask"],
                    caption="White pixels represent detected text",
                    clamp=True,
                    use_container_width=True
                )

            # Display individual lines

            st.subheader("Detected Handwritten Lines")

            for index, line in enumerate(lines):

                st.write(
                    "Line " + str(index + 1)
                )

                st.image(
                    line,
                    width=650
                )

            st.divider()

            # OCR recognition

            if st.button("Recognise All Lines"):

                try:

                    with st.spinner(
                        "Loading TrOCR and recognising handwriting..."
                    ):

                        processor, model, device = load_trocr()

                        recognized_lines = []

                        progress = st.progress(0)

                        for index, line in enumerate(lines):

                            text = recognize_line(
                                line,
                                processor,
                                model,
                                device
                            )

                            recognized_lines.append(text)

                            progress.progress(
                                (index + 1) / len(lines)
                            )

                        final_text = "\n".join(
                            recognized_lines
                        )

                        st.session_state["recognized_text"] = final_text

                except Exception as error:

                    st.error(
                        "Recognition error: " + str(error)
                    )

            # Display editable text

            if st.session_state.get("recognized_text"):

                st.subheader("Recognised Paragraph")

                edited_text = st.text_area(
                    "Edit extracted text:",
                    value=st.session_state["recognized_text"],
                    height=300
                )

                st.download_button(
                    label="Download Text File",
                    data=edited_text,
                    file_name="handwritten_notes.txt",
                    mime="text/plain"
                )


st.divider()

st.caption(
    "Technologies: Python | OpenCV | NumPy | PyTorch | TrOCR | Streamlit"
)
