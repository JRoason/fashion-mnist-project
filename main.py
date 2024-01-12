from PIL import Image
import numpy as np
from fashion_mnist_project.models.mvp import load_model, predict_class
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from starlette.responses import RedirectResponse

app = FastAPI(
    title="Fashion MNIST API",
    summary="An API endpoint for predicting the clothing type of a fashion item from an image. Trained on the Fashion "
            "MNIST dataset.",
    description="""
    # An API endpoint to access a classifier trained on the Fashion MNIST dataset.
    # Model Usage
    The model is trained on 28x28 grayscale images of single clothing items.
    The model can not predict multiple items in an image.
    """,
    version="0.1",
)


@app.get("/", description="Root endpoint that redirects to the docs.")
async def root():
    return RedirectResponse(url="/docs")


def process_image(file):
    image = Image.open(file.file).convert('L').resize((28, 28))
    image = np.array(image)
    image = image.reshape((1, 784))
    image = image / np.max(image)
    return image


@app.post("/predict", description="Endpoint for predicting the clothing type of a fashion item from an image. "
                                  "Image should be an image of a single clothing item. "
                                  "Returns a JSON object with the predicted class.")
async def predict_image(file: UploadFile = None):
    if file is None:
        return {"error": "No file uploaded."}
    image = process_image(file)
    model = load_model()
    prediction = predict_class(model, image)
    return {"prediction": prediction}
