from diffusers import StableDiffusionPipeline
from os.path import exists


def generate_image(text, img_filename):
    if exists(img_filename):
        print("Filename already exists. Will not create another one..")
    else:
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        pipe = pipe.to("cpu")
        image = pipe(text).images[0]
        image.save(img_filename)
