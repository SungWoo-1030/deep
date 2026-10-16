# NEU VISION · Streamlit

Streamlit app for running the saved image-classification and defect-segmentation models.
The app accepts image uploads and does not depend on the original Windows dataset paths.

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

Upload inspection images in the main panel. Optionally upload binary grayscale ground
truth masks as PNG, BMP, or TIFF to compare segmentation results. A mask must have the
same dimensions as its image and a matching filename stem, for example `part-01.jpg`
with `part-01.png` or `part-01.mask.png`. The app reports per-image and average IoU/Dice
scores and displays the reference mask next to the prediction. Classification-only
checkpoints cannot be compared against masks. Image, checkpoint, and mask uploads are
processed in memory and are not saved to the app's filesystem. Results can be downloaded
as CSV, and predicted segmentation masks as PNG or a ZIP archive.

## Model training

The existing Gradio training program remains in `neu_training_gui.py`. The Streamlit app
is an inference front end for the saved model files; it does not train models on Streamlit
Community Cloud. Run the Gradio training app locally when you need to create or update
the model files, then commit the desired checkpoint files in `outputs/`.

Classification scores are not calibrated probabilities or accuracy guarantees. Predicted
defect masks are model predictions and are not guaranteed to match ground truth.
