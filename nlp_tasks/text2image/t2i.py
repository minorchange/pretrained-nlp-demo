from diffusers import StableDiffusionPipeline
from os.path import exists
import re
import hashlib
import os


def hash_text(text):
    m = hashlib.md5()
    m.update(b"Nobody inspects")
    m.update(text.encode('utf8')) 
    h= m.hexdigest()
    return(h[:5])


def generate_filename_png_from_text(text):
    text_single_space = re.sub('\s+',' ',text)
    text_underscore = text_single_space.replace(" ","_")
    cleaned_text = ''.join(c for c in text_underscore if c.isalpha() or c == "_")
    filename = cleaned_text.lower()[:50]
    h = hash_text(text)
    filename = f'{filename}_{h}.png'
    return filename


def generate_image(text, img_path):

    if exists(img_path):
        print(f"Filename {img_path} already exists. Will not create another one..")
    else:
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        pipe = pipe.to("cpu")
        image = pipe(text).images[0]
        image.save(img_path)
    return(img_path)

