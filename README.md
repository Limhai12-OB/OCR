# OCR Image to Text and Receipt Extraction Prototype

A local Streamlit application that uses EasyOCR to recognize text in an uploaded image. The recognized text appears in an editable text area and can be downloaded as a UTF-8 `.txt` file. Choose plain text or structured text, which uses OCR box coordinates to retain reading order, line grouping, and horizontal column gaps.

This repository also includes a small command-line receipt extraction prototype. It runs EasyOCR, prints the raw text, asks a local Ollama model to return receipt JSON, applies USD/KHR currency rules, validates fields, and marks low-confidence fields for review. It is a prototype, not the LedgerVision application.

## Features

- Upload PNG, JPG, and JPEG images and preview them in the app.
- Compare OCR results from the original, enhanced grayscale, and thresholded image.
- Extract text with EasyOCR bounding boxes and confidence values.
- Choose plain text or structured text with approximate row and column spacing.
- Review the average OCR confidence and the number of low-confidence text regions.
- Choose the OCR language (English is included by default; more languages can be added in `ocr/engine.py`).
- Review, copy, and download the extracted text.
- Validate uploaded images and show readable errors when loading or OCR fails.
- Run entirely on your local machine. EasyOCR downloads its model files the first time a language is used.

## Requirements

- Python 3.9 or newer.
- Internet access on first OCR use so EasyOCR can download model files.
- A supported local environment for PyTorch, which EasyOCR installs as a dependency. CPU mode is used by default.

## Installation

From the project directory, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the application

```bash
streamlit run app.py
```

Streamlit prints a local address (usually `http://localhost:8501`) to open in your browser.

## Run receipt extraction

Install and start [Ollama](https://ollama.com), then download the default model:

```bash
ollama pull llama3.2:3b
```

Pass a receipt image to the CLI:

```bash
python main.py receipt.jpg
```

The command prints OCR raw text, structured AI output with per-field confidence and `needs_review`, validation errors, and the final object. You can select another installed Ollama model with `--model MODEL_NAME`.

Three built-in OCR text examples are available to try the model and validation path without running image OCR:

```bash
python main.py --demo usd
python main.py --demo khr
python main.py --demo low-confidence
```

Currency is assigned from OCR evidence only: `$`, USD, and US Dollar indicate USD; `៛`, KHR, and Riel indicate KHR. Other currencies are unsupported, and no conversion is performed. Categories are restricted to Ingredients, Rent, Utilities, Salary, Marketing, Equipment, Software, Transportation, and Other. Ollama must be reachable at `http://localhost:11434`.

## Example usage

1. Open the local Streamlit page.
2. Choose a PNG, JPG, or JPEG image containing printed or handwritten text.
3. Select a language and click **Extract Text**.
4. Review or copy the result in the text area, then click **Download Text** to save it as `extracted_text.txt`.

## How the OCR flow works

The uploader provides image bytes to the app. The app validates and previews the image, then passes it and the selected language code to `ocr/engine.py`. `ocr/preprocessing.py` resizes only when needed and prepares original, enhanced grayscale, and adaptive-threshold versions. EasyOCR runs on each version and returns each text region's coordinates and confidence. The engine compares the passes and selects one using confidence, text coverage, and the share of low-confidence regions. `ocr/layout.py` sorts regions into rows and uses measured horizontal gaps to preserve approximate columns. `ocr/postprocessing.py` normalizes whitespace without rewriting recognized digits or currency. Streamlit displays the selected text and provides a UTF-8 download.

Structured output is an approximation based on the positions EasyOCR detects. When one detection combines several table columns into a single box, the original column boundaries may not be recoverable from OCR coordinates alone. Always review important amounts and identifiers against the image.

## Add another language

Add a display name and EasyOCR code in `SUPPORTED_LANGUAGES` in `ocr/engine.py`. For example, many EasyOCR configurations support French using `"fr"`. Some languages require compatible language combinations; see the EasyOCR documentation before combining codes.

## Troubleshooting

- **First extraction takes a while:** EasyOCR may be downloading its detection and recognition models. Later runs reuse the local model cache.
- **Model download fails:** Check your internet connection, then retry the extraction.
- **PyTorch installation fails:** Use a supported Python version and consult the official PyTorch installation guidance for your operating system. EasyOCR uses CPU mode by default in this app.
- **No text is detected:** Try a sharper, brighter, higher-resolution image with text that is not heavily rotated or obscured.
- **Image cannot be opened:** Confirm the file is a valid PNG, JPG, or JPEG rather than a renamed file with a different format.
# OCR
