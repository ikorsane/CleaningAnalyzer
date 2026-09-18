# Cleaning Analyzer — hosted web app

This is the browser version of Cleaning Analyzer v4.

## Deploy on Streamlit Community Cloud
1. Create a private GitHub repository and upload `app.py`, `analysis_backend.py`, and `requirements.txt`.
2. In Streamlit Community Cloud choose **Create app**, select the repository, and set the main file to `app.py`.
3. Deploy. The app will receive a normal HTTPS URL.

Important: only use a third-party host for Wilhelmsen test photographs/data if company policy permits it. For confidential data, deploy the same files to an approved internal host instead.

## Workflow
- Upload one or more replicate plate photographs together.
- Set the four plate corners for each replicate.
- Mark dirty and clean control rectangles.
- Mark one rough rectangle for each product track.
- Review each automatically proposed footprint; accept it or replace it with a manual polygon.
- Review combined mean ± sample SD across replicate plates.
- Download the report-ready Excel workbook and a ZIP containing CSV, masks, heatmaps, rectified plates and diagnostics.

The bottom 5% of the rectified plate is excluded from cleaning scoring to remove the rack/product-pooling artifact. Set `BOTTOM_MARGIN_FRAC = 0.0` in `analysis_backend.py` if that zone is cropped out during plate selection.
