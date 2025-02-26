from PIL import Image, UnidentifiedImageError
import numpy as np
from fashion_mnist_project.models.model import load_classifier, predict_class
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from starlette.responses import RedirectResponse
import streamlit as st


class PredictedClass(BaseModel):
    prediction: str


app = FastAPI(
    title="Fashion MNIST API",
    summary="An API endpoint for predicting clothing types from images, trained on the Fashion MNIST dataset.",
    description="""\
## Fashion MNIST Classifier API Endpoint
Access a specialized classifier trained on the Fashion MNIST dataset for identifying single clothing items in images.

### Model Usage
- **Training Data**: 28x28 grayscale images of individual clothing items.
- **Image Formats**: Accepts formats supported by the Pillow library.
- **Note**: The model is designed to predict only one item per image.

Refer to the [Pillow library documentation](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html) for details on supported image formats.""",
    version="0.1",
)



@app.get("/", description="Root endpoint that redirects to the docs.")
async def root():
    return RedirectResponse(url="/docs")


def process_image(file):
    image = Image.open(file.file).convert('L').resize((28, 28))
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image / np.max(image)
    return image


@app.post("/predict", description="Endpoint for predicting the clothing type of a fashion item from an image. "
                                  "Image should be an image of a single clothing item. "
                                  "Returns a JSON object with the predicted class.",
          response_model=PredictedClass,
          response_description="Predict clothing article class from image.",
          responses={
              200: {
                  "description": "Prediction successful.",
                  "content": {
                      "application/json": {
                          "example": {
                              "prediction": "T-shirt/top"
                          }
                      }
                  }
              },
              400: {
                  "description": "No file uploaded."
              },
              422: {
                  "description": "Invalid file type. Please upload an image."
              }
          })
async def predict_image(file: UploadFile = None):
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    try:
        image = process_image(file)
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="Invalid file type. Please upload an image.")
    model = load_classifier()
    prediction = PredictedClass(prediction=predict_class(model, image))
    return prediction


if __name__ == "__main__":

    st.title("Fashion MNIST Classifier")

    st.write("""\
    Access a specialized classifier trained on the Fashion MNIST dataset for identifying single clothing items in images.\n
    ### Model Usage
    Upload an image using the file uploader below. The model is designed to predict only one item per image.\n
    """)

    uploaded_file = st.file_uploader("Choose an image...")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L').resize((28, 28))
        image = np.array(image)
        image = image.reshape(1, 28, 28, 1)
        image = image / np.max(image)
        model = load_classifier()
        prediction = predict_class(model, image)
        st.write("""\
        ### Input Image
        """)
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("""\
        ### Prediction
        {}""".format(prediction))

