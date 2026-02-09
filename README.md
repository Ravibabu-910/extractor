# Extractor MVP

This MVP extracts chart data, tables, text, and image captions from PDFs, images, and DOCX files into structured JSON and XLSX outputs.

## Features
- Chart digitization using the free **google/deplot** model (image-to-text) for graph value extraction.
- Image captioning using **Salesforce/blip-image-captioning-base**.
- OCR for text extraction using **easyocr** (no Tesseract).
- Table extraction from PDFs via **camelot** with multi-page table merging.
- Table/text extraction from DOCX via **python-docx**.

## Usage
```bash
python -m extractor_app.main /path/to/file.pdf --output-dir output
```

### Output
- `output/extraction.json` with structured data.
- `output/extraction.xlsx` with separate sheets for charts, tables, text, and images.

## Notes
- For best results, provide high-resolution images and vector PDFs.
- The MVP focuses on extraction. You can extend post-processing to map axes, detect units, or normalize dates.
