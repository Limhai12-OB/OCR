"""Streamlit interface for extracting text from images with EasyOCR."""

import hashlib
from time import perf_counter

import streamlit as st

from ocr.engine import OCRResult, SUPPORTED_LANGUAGES, extract_document
from llm_service import parse_receipt
from receipt.validator import validate_receipt
from utils.file_utils import validate_image


st.set_page_config(page_title="Receipt OCR and Extraction", page_icon="🧾", layout="centered")


def uploaded_image_id(image_bytes: bytes) -> str:
    """Return a stable identifier so results cannot be mistaken for another image."""
    return hashlib.sha256(image_bytes).hexdigest()


def render_result(result: OCRResult, mode: str, receipt=None, validation=None, total_seconds=0.0) -> None:
    """Show extracted text with a plain-text download action."""
    text = result.structured_text if mode == "Structured Text" else result.plain_text
    st.subheader("Extracted text")
    st.caption(
        f"Average OCR confidence: {result.average_confidence:.0%} · "
        f"Preprocessing selected: {result.selected_preprocessing} · "
        f"{len(result.detections)} text regions"
    )
    if result.uncertain_count:
        st.warning(
            f"{result.uncertain_count} text region(s) have low OCR confidence. "
            "Review numbers and other important values against the image."
        )
    st.text_area(
        "OCR raw text",
        value=text,
        height=260,
        key="extracted_text_area",
    )
    st.download_button(
        "Download Text",
        data=text.encode("utf-8"),
        file_name="extracted_text.txt",
        mime="text/plain; charset=utf-8",
        disabled=not text,
    )
    if receipt is not None:
        st.subheader("Structured receipt result")
        if validation["valid"]:
            st.success("Validation passed")
        else:
            st.error("Validation needs review")
        data = receipt.to_dict()
        st.write({
            "Merchant": data["merchant"], "Date": data["date"],
            "Amount": data["amount"], "Currency": data["currency"],
            "Category": data["category"], "Description": data["description"],
            "Items": data["items"], "OCR confidence": data["ocr_confidence"],
            "Extraction confidence": data["extraction_confidence"],
            "Needs review": validation["needs_review"],
        })
        st.caption(
            f"Preprocessing: {result.preprocessing_seconds:.2f}s · OCR: {result.inference_seconds:.2f}s · "
            f"LLM: {receipt.llm_seconds:.2f}s · Total: {total_seconds:.2f}s"
        )
        if receipt.llm_error:
            st.warning(f"LLM unavailable; showing rule-based extraction: {receipt.llm_error}")
        with st.expander("Raw OCR detections"):
            st.json(result.raw_detections)
        with st.expander("OCR preprocessing pass scores"):
            st.json(result.candidate_scores)
        with st.expander("LLM raw JSON"):
            st.code(receipt.llm_raw_json or "No valid LLM JSON returned", language="json")
        with st.expander("Validation errors"):
            st.json(validation["errors"])


def main() -> None:
    st.title("Receipt OCR and Extraction")
    st.write("Inspect OCR detections and extract structured receipt data locally.")

    st.subheader("Image upload")
    uploaded_file = st.file_uploader(
        "Choose a PNG, JPG, or JPEG image",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("No image has been uploaded yet. Choose an image to get started.")
        return

    image_bytes = uploaded_file.getvalue()
    try:
        image = validate_image(image_bytes)
    except (ValueError, OSError) as error:
        st.error(f"This file could not be opened as a valid image: {error}")
        return

    st.image(image, caption=uploaded_file.name, width="stretch")

    image_id = uploaded_image_id(image_bytes)
    language_names = list(SUPPORTED_LANGUAGES)
    language_name = st.selectbox("OCR language", language_names, index=0)
    language_codes = SUPPORTED_LANGUAGES[language_name]
    language_key = "+".join(language_codes)
    ocr_mode = st.radio("Output format", ["Plain Text", "Structured Text"], horizontal=True)
    show_processed = st.checkbox("Show selected processed image", value=False)

    if st.button("Extract Receipt", type="primary"):
        # Hide any older result until this extraction succeeds.
        st.session_state.pop("ocr_image_id", None)
        try:
            started = perf_counter()
            with st.spinner("Comparing receipt OCR passes and parsing receipt fields…"):
                result = extract_document(image, language_codes)
                receipt = parse_receipt(result.structured_text, ocr_confidence=result.average_confidence)
                validation = validate_receipt(receipt)
                total_seconds = perf_counter() - started
            # Reset the widget's previous value so the newly extracted result is shown.
            st.session_state.pop("extracted_text_area", None)
            st.session_state["ocr_result"] = result
            st.session_state["receipt_result"] = receipt
            st.session_state["receipt_validation"] = validation
            st.session_state["receipt_total_seconds"] = total_seconds
            st.session_state["ocr_image_id"] = image_id
            st.session_state["ocr_language"] = language_key
            selected_text = result.structured_text if ocr_mode == "Structured Text" else result.plain_text
            if selected_text.strip():
                st.success("Receipt extraction is complete.")
            else:
                st.warning("No text was detected. Try a clearer or higher-resolution image.")
        except Exception as error:  # OCR models can raise several backend-specific errors.
            st.error(f"OCR could not be completed: {error}")

    result_is_current = (
        st.session_state.get("ocr_image_id") == image_id
        and st.session_state.get("ocr_language") == language_key
    )
    if result_is_current:
        result = st.session_state["ocr_result"]
        if show_processed:
            st.image(result.processed_preview, caption=f"Selected preprocessing: {result.selected_preprocessing}", width="stretch")
        render_result(result, ocr_mode,
                      st.session_state.get("receipt_result"),
                      st.session_state.get("receipt_validation"),
                      st.session_state.get("receipt_total_seconds", 0.0))


if __name__ == "__main__":
    main()
