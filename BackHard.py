import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

#Read a file as image
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

#Read the mode from our disk
Model = tf.keras.models.load_model("Models-Pepper&Potato&TomatoVillage/1")
# CLASS_NAMES = ["Potato Early Blight", "Potato Late Blight", "Potato Healthy"]

# CLASS_NAMES = ['Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy']

CLASS_NAMES = ["Pepper__bell___Bacterial_spot","Pepper__bell___healthy","Potato___Early_blight","Potato___Late_blight",
"Potato___healthy","Tomato_Early_blight","Tomato_Late_blight","Tomato_healthy"]