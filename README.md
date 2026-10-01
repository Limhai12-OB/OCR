# Receipt OCR and Extraction Prototype

A local Streamlit application that uses EasyOCR to recognize text in an uploaded image. The recognized text appears in an editable text area and can be downloaded as a UTF-8 `.txt` file. Choose plain text or structured text, which uses OCR box coordinates to retain reading order, line grouping, and horizontal column gaps.

This prototype keeps OCR separate from receipt understanding. EasyOCR returns text, boxes, and OCR confidence; deterministic receipt rules identify explicit totals, dates, and USD/KHR markers; Ollama interprets merchant/category/items; validation reports missing or unsupported financial data. Rule based output remains available when Ollama is offline.

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

The command prints OCR detections and raw text, structured receipt output, validation results, and timings. It continues with deterministic extraction if Ollama is unavailable. Configure the local model and URL with `OLLAMA_MODEL` and `OLLAMA_URL` environment variables.

Three built-in OCR text examples are available to try the model and validation path without running image OCR:

```bash
python main.py --demo usd
python main.py --demo khr
python main.py --demo low-confidence
python main.py --demo subtotal
python main.py --demo bad-ocr
```

Currency is assigned from OCR evidence only: `$`, USD, and Dollar indicate USD; `៛`, KHR, and Riel indicate KHR. Other currencies are unsupported, and no conversion is performed. Totals are taken from explicit payable/total lines; unlabeled numbers are not promoted to totals. Categories are restricted to Ingredients, Rent, Utilities, Salary, Marketing, Equipment, Software, Transportation, and Other. `OCR_ENGINE` currently accepts `easyocr`.

## Test

Run the text fixture tests with the project environment:

```bash
python -m unittest discover -s tests -v
```

These cover USD/KHR markers, total versus subtotal, date normalization, missing dates, invalid amounts, validation, and Ollama failure fallback. OCR image accuracy still needs a representative image benchmark because the repository does not include receipt image fixtures.

To compare OCR on a folder of sample receipts without involving Ollama:

```bash
python benchmark.py ./sample_receipts
```

## Example usage

1. Open the local Streamlit page.
2. Choose a PNG, JPG, or JPEG image containing printed or handwritten text.
3. Select a language and click **Extract Text**.
4. Review or copy the result in the text area, then click **Download Text** to save it as `extracted_text.txt`.

## How the OCR flow works

The uploader validates and previews the image, then `ocr/engine.py` runs cached EasyOCR over original, grayscale, contrast enhanced, adaptive threshold, Otsu, and sharpened variants. A modest skew correction runs before grayscale variants. Receipt-aware scoring gives total, date, amount, and currency evidence weight alongside OCR confidence and meaningful lines. Detections retain boxes and confidence; layout code reconstructs reading order and postprocessing normalizes whitespace. The UI shows raw detections, OCR text, candidate scores, structured fields, validation, and timing.

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
