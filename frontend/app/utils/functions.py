import base64
from PIL import Image
import io
import pandas as pd
import matplotlib.pyplot as plt



def decode_fig_from_base64(base64_string):
    base64_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(base64_bytes))
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    return fig