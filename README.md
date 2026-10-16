# NEU VISION · Streamlit

Streamlit app for defect-mask labeling, model training, and single-image or full-folder
classification/segmentation. Streamlit Community Cloud users upload ZIP datasets; local
users can instead enter folder paths accessible to the computer running Streamlit.

## Run locally

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy from this folder on Streamlit Community Cloud

1. Push the contents of this `Deep` folder to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set **Main file path** to `streamlit_app.py`.
4. Select Python 3.11 if the deployment settings ask for a Python version.

Keep these trained model files in the repository:

- `outputs/segmentation_classifier.pt` — manual-mask segmentation
- `outputs/full_image_segmentation.pt` — full-image-mask segmentation
- `outputs/image_classifier.pt` — classification only

The `.gitignore` excludes generated masks, prediction files, local datasets, and virtual
environments, but does not exclude the trained `.pt` model files. The app can also use a
locally trained `.pt`/`.pth` checkpoint uploaded in the sidebar (up to 20 MB); it is
loaded with PyTorch's restricted `weights_only` mode and is not saved to the app's
filesystem.

The Streamlit navigation has four workflows:

- **AI inspection:** upload individual images and optionally matching binary grayscale
  ground-truth masks. A mask must match the image dimensions and filename stem, such as
  `part-01.jpg` with `part-01.mask.png`. Segmentation results include IoU/Dice
  comparisons; results can be downloaded as CSV and PNG/ZIP masks.
- **Defect labeling:** upload a training ZIP or enter a local folder path. The training
  layout is `train/class_name/image.jpg`. Draw or erase defect pixels image by image.
  Manual masks are kept separately from full-image masks. The full-image labeling action
  marks every pixel as defective; use it only when that is the intended label. The image
  table reports training inclusion separately for manual-mask segmentation, full-image
  segmentation, and classification, and detects changed images or masks.
- **Model training:** train manual-mask segmentation, full-image-mask segmentation, or
  classification-only models separately. Each class needs at least two images for
  training/validation splitting. Trained checkpoints are available to download and use
  in the current session.
- **Folder analysis:** upload a ZIP or enter a local test folder path. All supported
  images in the folder and subfolders are analyzed, with CSV results and predicted
  segmentation masks available as downloads.

ZIP uploads are limited to 500 MB and 1 GB uncompressed. Uploaded datasets and generated
masks are kept in an ignored `outputs/streamlit_sessions/` workspace; they are not
committed to the repository. Streamlit Cloud cannot browse folders on the user's
computer, so use ZIP uploads there. Cloud session data and checkpoints should be
downloaded before the app restarts.
ZIPs may contain up to 20,000 files, and a folder analysis can process up to 20,000
images per run.

The original local Gradio program remains available in `neu_training_gui.py`.

Classification scores are not calibrated probabilities or accuracy guarantees. Predicted
defect masks are model predictions and are not guaranteed to match ground truth.
