import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image




st.set_page_config(
    page_title="Handwritten Notes Digitizer",
    page_icon="📝",
    layout="wide"
)

st.title("Handwritten Notes Digitizer")
st.write(
    "Phase 2 Prototype: Image Preprocessing + OCR + Editable Text"
)




@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"], gpu=False)




def preprocess_image(image):
    """
    Detects blue handwriting, crops the relevant region,
    and prepares the image for OCR.
    """

    # Convert PIL image to NumPy array
    img = np.array(image)

    # Convert RGB to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)



    lower_blue = np.array([90, 40, 30])
    upper_blue = np.array([140, 255, 255])

    blue_mask = cv2.inRange(
        hsv,
        lower_blue,
        upper_blue
    )



    kernel = np.ones((3, 3), np.uint8)

    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_CLOSE,
        kernel
    )



    contours, _ = cv2.findContours(
        blue_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        # Ignore very small noise
        if w > 5 and h > 5:
            boxes.append((x, y, w, h))


    if boxes:

        x1 = min(x for x, y, w, h in boxes)
        y1 = min(y for x, y, w, h in boxes)

        x2 = max(
            x + w
            for x, y, w, h in boxes
        )

        y2 = max(
            y + h
            for x, y, w, h in boxes
        )

        # Add padding
        padding = 30

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)

        x2 = min(
            img.shape[1],
            x2 + padding
        )

        y2 = min(
            img.shape[0],
            y2 + padding
        )

        cropped = img[y1:y2, x1:x2]

    else:
        # If blue handwriting is not detected,
        # use the complete image
        cropped = img



    height, width = cropped.shape[:2]

    scale = 2

    resized = cv2.resize(
        cropped,
        (width * scale, height * scale),
        interpolation=cv2.INTER_CUBIC
    )



    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_RGB2GRAY
    )



    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    

    denoised = cv2.GaussianBlur(
        enhanced,
        (3, 3),
        0
    )

    

    thresholded = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        5
    )

    return cropped, resized, thresholded




uploaded_file = st.file_uploader(
    "Upload an image of your handwritten notes",
    type=["jpg", "jpeg", "png"]
)




if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")



    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")

        st.image(
            image,
            use_container_width=True
        )


    cropped_image, enhanced_image, processed_image = (
        preprocess_image(image)
    )

    with col2:

        st.subheader("Detected Text Region")

        st.image(
            cropped_image,
            use_container_width=True
        )



    st.subheader("Processed Image")

    st.image(
        processed_image,
        use_container_width=True
    )


    if st.button(
        "Extract Text",
        type="primary"
    ):

        with st.spinner(
            "Recognizing handwritten text..."
        ):

            reader = load_ocr()



            results_color = reader.readtext(
                enhanced_image,
                detail=1,
                paragraph=False,
                mag_ratio=2
            )



            results_processed = reader.readtext(
                processed_image,
                detail=1,
                paragraph=False,
                mag_ratio=2
            )


            text_color = "\n".join(
                [
                    text
                    for _, text, confidence in results_color
                    if confidence > 0.20
                ]
            )

            text_processed = "\n".join(
                [
                    text
                    for _, text, confidence in results_processed
                    if confidence > 0.20
                ]
            )

           
            if text_color.strip():

                extracted_text = text_color

            elif text_processed.strip():

                extracted_text = text_processed

            else:

                extracted_text = ""

   

        if extracted_text.strip():

            st.session_state[
                "extracted_text"
            ] = extracted_text

            st.success(
                "Text extracted successfully."
            )

        else:

            st.warning(
                "No text was detected. "
                "Try a clearer image with good lighting."
            )



if "extracted_text" in st.session_state:

    st.subheader("Extracted Text")

    edited_text = st.text_area(
        "You can correct the OCR result here:",
        value=st.session_state["extracted_text"],
        height=300
    )

    # Save edited text
    st.session_state[
        "extracted_text"
    ] = edited_text



    download_text = edited_text.encode(
        "utf-8"
    )

    st.download_button(
        label="Download Text File",
        data=download_text,
        file_name="digitized_notes.txt",
        mime="text/plain"
    )




st.divider()

st.caption(
    "Handwritten Notes Digitization using "
    "OpenCV and EasyOCR"
)