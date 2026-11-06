
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
st.subheader("Line Segmentation + TrOCR Handwriting Recognition")

st.write(
    "Upload a handwritten page. The system will separate "
    "individual lines and recognise them using TrOCR."
)


@st.cache_resource
def load_trocr():

    model_name = "microsoft/trocr-base-handwritten"

    processor = TrOCRProcessor.from_pretrained(model_name)

    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    return processor, model, device


def separate_lines(image):

    image_array = np.array(image.convert("RGB"))

    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15
    )

    height, width = binary.shape

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(25, width // 30), 1)
    )

    horizontal_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    text_mask = cv2.subtract(binary, horizontal_lines)

    projection = np.sum(text_mask > 0, axis=1)

    threshold = max(2, int(width * 0.0015))

    active_rows = np.where(projection > threshold)[0]

    if len(active_rows) == 0:
        return [], text_mask

    groups = []

    start = active_rows[0]
    previous = active_rows[0]

    merge_gap = max(8, int(height * 0.008))

    for row in active_rows[1:]:

        if row - previous <= merge_gap:
            previous = row

        else:
            groups.append((start, previous))
            start = row
            previous = row

    groups.append((start, previous))

    line_images = []

    padding = 12

    for y1, y2 in groups:

        if y2 - y1 < 8:
            continue

        line_region = text_mask[y1:y2 + 1, :]

        columns = np.where(np.sum(line_region > 0, axis=0) > 0)[0]

        if len(columns) == 0:
            continue

        x1 = max(0, columns[0] - padding)
        x2 = min(width, columns[-1] + padding)

        top = max(0, y1 - padding)
        bottom = min(height, y2 + padding)

        line_image = image_array[top:bottom, x1:x2]

        if line_image.size > 0:
            line_images.append(Image.fromarray(line_image))

    return line_images, text_mask


def recognize_line(line_image, processor, model, device):

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


uploaded_file = st.file_uploader(
    "Upload handwritten notes",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")

    st.image(image, use_container_width=True)

    if st.button("Separate Handwritten Lines"):

        lines, text_mask = separate_lines(image)

        st.session_state["lines"] = lines
        st.session_state["recognized_text"] = ""

    if "lines" in st.session_state:

        lines = st.session_state["lines"]

        st.subheader("Detected Lines")

        if len(lines) == 0:

            st.warning(
                "No handwritten lines detected. Try another image "
                "with clearer handwriting."
            )

        else:

            st.success(f"{len(lines)} lines detected.")

            for index, line in enumerate(lines):

                st.write(f"Line {index + 1}")

                st.image(
                    line,
                    use_container_width=False,
                    width=650
                )

            st.divider()

            if st.button("Recognise All Lines"):

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

                    final_text = "\n".join(recognized_lines)

                    st.session_state["recognized_text"] = final_text

            if st.session_state.get("recognized_text"):

                st.subheader("Recognised Paragraph")

                edited_text = st.text_area(
                    "Edit the extracted text:",
                    value=st.session_state["recognized_text"],
                    height=300
                )

                st.download_button(
                    "Download Text File",
                    data=edited_text,
                    file_name="handwritten_notes.txt",
                    mime="text/plain"
                )


st.divider()

st.caption(
    "Technologies: Python | OpenCV | PyTorch | TrOCR | Streamlit"
)
