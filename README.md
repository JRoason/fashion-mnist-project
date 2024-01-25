# Fashion-MNIST Clothing Classifier

Welcome to our project!

This is our repository for our final project for the course Machine Learning Practical, taken during our BSc in Artificial Intelligence at the University of Groningen.

The repository hosts the code for our project, as well as the trained models.

## Usage

### Pre-requisites

- **Pipenv**: We use Pipenv to manage our dependencies. If you don't have it installed, you can do so by running:
  ```bash
  $ pip install pipenv
  ```
  Once installed, activate the virtual environment by running:
  ```bash
  $ pipenv shell
  ```
  
  And install the dependencies by running:
  ```bash
  $ pipenv install
  ```

### Running the code

We have integrated our CNN model into a simple Streamlit GUI, which can be run by running:

```bash
$ python -m streamlit main.py
```

Additionally, we have integrated our model into a FastAPI server, which can be started by running:

```bash
$ uvicorn main:app --reload
```

If you wish to use the model directly, you must first load it and then use it to predict the class of an image:

```python
from model import predict, load_classifier
image = ...
prediction = predict(load_classifier(cnn=True), image)
```

The `load_classifier` function loads the model from the `models` folder, and takes an optional argument `cnn`, which is set to True by default. If set to True, it will load the CNN model, otherwise it will load the Logistic Regression model.

The `predict` function returns the predicted class of the image, as a string.
