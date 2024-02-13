import tensorflow as tf
from tensorflow import keras
import numpy as np
from omnixai.data.image import Image
from omnixai.explainers.vision import ShapImage

model = keras.models.load_model("models/basic")
image = np.random.rand(48, 64)

img = Image(image)
preprocess_func = lambda x: np.expand_dims(x.to_numpy() / 255, axis=-1)

explainer = ShapImage(
        model=model,
        preprocess_function=preprocess_func,
    )

explanations = explainer.explain(img)
scores = explanations.get_explanations()[0]['scores']
print(scores.shape)