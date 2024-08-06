# ==============================================================================
# Ignore comfy.utils functions -> There are none
# ==============================================================================

import io
import base64
from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo
import torch
import folder_paths
import numpy
import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import re
import urllib.parse
import urllib.request
import os
import json
import random
import folder_paths as comfy_paths
from PIL.PngImagePlugin import PngInfo
from datetime import datetime
import time
from openai import OpenAI
from anthropic import Anthropic
from collections import defaultdict
import hashlib
import node_helpers

dirPath = os.path.dirname(os.path.realpath(__file__))
ALLOWED_EXT = ('jpeg', 'jpg', 'png', 'tiff', 'gif', 'bmp', 'webp')


class Base64Decode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base64_img": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render_image"
    CATEGORY = "t4ggno/image"
    OUTPUT_NODE = True

    def render_image(self, base64_img):

        print("=============================")
        print("== Base64 Decode")

        # Convert base64 image to PIL image
        image = Image.open(io.BytesIO(base64.b64decode(base64_img)))
        # Convert PIL image to torch tensor
        image_torch = torch.from_numpy(numpy.array(image).astype(numpy.float32) / 255.0).unsqueeze(
            0
        )
        return (image_torch,)


class LayoutSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "size1": ("INT", {"default": 1024}),
                "size2": ("INT", {"default": 1024}),
                "layout": (["Landscape", "Portrait", "Square"], {"default": "Landscape"}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "render_resolution"
    CATEGORY = "t4ggno/utils"
    OUTPUT_NODE = False

    def render_resolution(self, size1, size2, layout):  # size1 = width, size2 = height

        print("=============================")
        print("== Layout Switch")

        if layout == "Landscape":
            print("Output: " + str(size1) + "x" + str(size2))
            if size1 > size2:
                return (size1, size2)
            else:
                return (size2, size1)
        elif layout == "Portrait":
            print("Output: " + str(size2) + "x" + str(size1))
            if size1 > size2:
                return (size2, size1)
            else:
                return (size1, size2)
        elif layout == "Square":
            print("Output: " + str(size1) + "x" + str(size1))
            size = (size1 + size2) / 2
            return (size, size)


class PredefinedResolutions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dimension": (["1.5", "SDXL", "Famous"], {"default": "SDXL"}),
                "layout": (["Square", "Landscape", "Landscape (Ultra Wide)", "Portrait", "Portrait (Ultra Tall)", "Random"], {"default": "Square"}),
                "at_random_enable_square": (["true", "false"], {"default": "true"}),
                "at_random_enable_landscape": (["true", "false"], {"default": "true"}),
                "at_random_enable_landscape_ultra_wide": (["true", "false"], {"default": "true"}),
                "at_random_enable_portrait": (["true", "false"], {"default": "true"}),
                "at_random_enable_portrait_ultra_tall": (["true", "false"], {"default": "true"}),
                "scale": ("FLOAT", {"default": 1, "min": 1, "max": 100, "step": 0.1}),
            },
            "hidden": {
                "control_after_generate": (["fixed", "random", "increment"], {"default": "increment"}),
                "value": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "render_resolution"
    CATEGORY = "t4ggno/utils"
    OUTPUT_NODE = False

    def render_resolution(self, dimension, layout, at_random_enable_square, at_random_enable_landscape, at_random_enable_landscape_ultra_wide, at_random_enable_portrait, at_random_enable_portrait_ultra_tall, scale):

        print("=============================")
        print("== Predefined Resolutions")

        if layout == "Random":
            print("Random layout...")
            allowed_layouts = []
            if at_random_enable_square == "true":
                allowed_layouts.append("Square")
            if at_random_enable_landscape == "true":
                allowed_layouts.append("Landscape")
            if at_random_enable_landscape_ultra_wide == "true":
                allowed_layouts.append("Landscape (Ultra Wide)")
            if at_random_enable_portrait == "true":
                allowed_layouts.append("Portrait")
            if at_random_enable_portrait_ultra_tall == "true":
                allowed_layouts.append("Portrait (Ultra Tall)")
            layout = random.choice(allowed_layouts)
        print("Layout: " + layout)

        resolution = None
        if (dimension == "1.5"):
            print("Dimension: 1.5")
            if layout == "Square":
                resolution = (512, 512)
            elif layout == "Landscape":
                resolution = (576, 448)
            elif layout == "Landscape (Ultra Wide)":
                resolution = (768, 320)
            elif layout == "Portrait":
                resolution = (448, 576)
            elif layout == "Portrait (Ultra Tall)":
                resolution = (320, 768)
            else:
                resolution = (512, 512)
        elif (dimension == "SDXL"):
            print("Dimension: SDXL")
            if layout == "Square":
                resolution = (1024, 1024)
            elif layout == "Landscape":
                resolution = (1216, 832)
            elif layout == "Landscape (Ultra Wide)":
                resolution = (1536, 640)
            elif layout == "Portrait":
                resolution = (832, 1216)
            elif layout == "Portrait (Ultra Tall)":
                resolution = (640, 1536)
            else:
                resolution = (1024, 1024)
        elif (dimension == "Famous"):
            print("Dimension: Famous")
            if layout == "Square":
                resolution = (1024, 1024)
            elif layout == "Landscape":
                resolution = (1366, 768)
            elif layout == "Landscape (Ultra Wide)":
                resolution = (1596, 684)
            elif layout == "Portrait":
                resolution = (768, 1366)
            elif layout == "Portrait (Ultra Tall)":
                resolution = (684, 1596)
            else:
                resolution = (1024, 1024)

        # Scale and floor
        print("Scale: " + str(scale) + "x -> " + str(int(resolution[0] * scale)) + "x" + str(int(resolution[1] * scale)))
        return (int(resolution[0] * scale), int(resolution[1] * scale))

    @classmethod
    def IS_CHANGED(self, **kwargs):
        return float("nan")


class ResolutionSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "active": (["Resolution 1", "Resolution 2"], {"default": "Resolution 1"}),
                "width1": ("INT", {"default": 1024}),
                "height1": ("INT", {"default": 1024}),
                "width2": ("INT", {"default": 1024}),
                "height2": ("INT", {"default": 1024}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_resolution"
    CATEGORY = "t4ggno/utils"
    OUTPUT_NODE = False

    def get_resolution(self, active, width1, height1, width2, height2):

        print("=============================")
        print("== Resolution Switch")

        if active == "Resolution 1":
            print("Output: " + str(width1) + "x" + str(height1))
            return (width1, height1)
        elif active == "Resolution 2":
            print("Output: " + str(width2) + "x" + str(height2))
            return (width2, height2)


class TextSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "active": (["Text 1", "Text 2"], {"default": "Text 1"}),
                "text1": ("STRING", {"default": ""}),
                "text2": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "get_text"
    CATEGORY = "t4ggno/utils"
    OUTPUT_NODE = False

    def get_text(self, active, text1, text2):

        print("=============================")
        print("== Text Switch")

        if active == "Text 1":
            print("Output: " + text1)
            return (text1,)
        elif active == "Text 2":
            print("Output: " + text2)
            return (text2,)

def get_prompt_data_from_image(image: Image, fallback_positive_prompt: str, fallback_negative_prompt: str):
    return_positive_prompt = fallback_positive_prompt
    return_negative_prompt = fallback_negative_prompt
    return_positive_prompt_original = fallback_positive_prompt
    return_negative_prompt_original = fallback_negative_prompt
    return_additional_prompt = ""
    return_checkpoint = ""
    return_checkpoint_without_extension = ""
    # Get metadata
    metadata = image.info
    # Convert metadata to JSON
    metadata_json = json.dumps(metadata)
    # Check if metadata contains PositiveText, NegativeText and AdditionalText
    if "PositiveText" in metadata_json:
        positive_text = metadata["PositiveText"].strip()
        if positive_text != "":
            return_positive_prompt = positive_text
    if "PositiveTextOriginal" in metadata_json:
        positive_text_original = metadata["PositiveTextOriginal"].strip()
        if positive_text_original.strip() != "":
            return_positive_prompt_original = positive_text_original
    if "NegativeText" in metadata_json:
        negative_text = metadata["NegativeText"].strip()
        if negative_text != "":
            return_negative_prompt += negative_text
    if "NegativeTextOriginal" in metadata_json:
        negative_text_original = metadata["NegativeTextOriginal"].strip()
        if negative_text_original != "":
            return_negative_prompt_original = negative_text_original
    if "AdditionalText" in metadata_json:
        additional_text = metadata["AdditionalText"].strip()
        if additional_text != "":
            return_additional_prompt = additional_text
    if "Checkpoint" in metadata_json:
        checkpoint = metadata["Checkpoint"].strip()
        if checkpoint != "":
            return_checkpoint = checkpoint
    if "CheckpointWithoutExtension" in metadata_json:
        checkpoint_without_extension = metadata["CheckpointWithoutExtension"].strip()
        if checkpoint_without_extension != "":
            return_checkpoint_without_extension = checkpoint_without_extension
    return (return_positive_prompt, return_negative_prompt, return_positive_prompt_original, return_negative_prompt_original, return_additional_prompt, return_checkpoint, return_checkpoint_without_extension)

class ImageMetadataExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "fallback_positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "fallback_negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "positive_prompt_original", "negative_prompt_original", "checkpoint", "checkpoint_without_extension", "additional_prompt")
    FUNCTION = "load_image"
    CATEGORY = "t4ggno/image"
    OUTPUT_NODE = False

    def load_image(self: object, image: torch.Tensor, fallback_positive_prompt: str, fallback_negative_prompt: str):

        print("=============================")
        print("== Image Metadata Extractor")
        
        # Read positive and negative prompt from metadata
        positive_prompt, negative_prompt, positive_prompt_original, negative_prompt_original, additional_prompt, checkpoint, checkpoint_without_extension = get_prompt_data_from_image(image, fallback_positive_prompt, fallback_negative_prompt)
        print("====================================")
        print("Positive Prompt: " + positive_prompt)
        print("Positive Prompt Original: " + positive_prompt_original)
        print("Negative Prompt: " + negative_prompt)
        print("Negative Prompt Original: " + negative_prompt_original)
        print("Additional Prompt: " + additional_prompt)
        print("Checkpoint: " + checkpoint)
        print("Checkpoint Without Extension: " + checkpoint_without_extension)
        print("====================================")
        # Return
        return (positive_prompt, negative_prompt, positive_prompt_original, negative_prompt_original, checkpoint, checkpoint_without_extension, additional_prompt)

    @classmethod
    def IS_CHANGED(self, **kwargs):
        return float("nan")


class AutoLoadImageForUpscaler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": ""}),
                "max_width": ("INT", {"default": 8192, "min": 1, "max": 100000, "step": 1}),
                "max_height": ("INT", {"default": 8192, "min": 1, "max": 100000, "step": 1}),
                "scale": ("FLOAT", {"default": 2, "min": 1, "max": 100, "step": 0.1}),
                "max_scale": ("FLOAT", {"default": 100000, "min": 1, "max": 100000, "step": 0.1}),
                "fallback_positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "fallback_negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {
                "control_after_generate": (["fixed", "random", "increment"], {"default": "increment"}),
                "value": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT", "STRING", "STRING", "FLOAT", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "filename_without_scale", "current_scale", "positive_prompt", "negative_prompt", "next_scale", "positive_prompt_original", "negative_prompt_original", "checkpoint", "checkpoint_without_extension", "additional_prompt")
    FUNCTION = "load_image"
    CATEGORY = "t4ggno/image"
    OUTPUT_NODE = False

    def load_image(self: object, folder: str, max_width: int, max_height: int, scale: float, max_scale: float, fallback_positive_prompt: str, fallback_negative_prompt: str):

        print("=============================")
        print("== Auto Load Image For Upscaler")

        while True:
            # Load files from folder
            files = os.listdir(folder)
            # Filter only for png images
            files = [file for file in files if file.endswith(".png")]
            # Find image with lowest available scale -> _1.0x, _2.0x, _3.5x, ...
            # Create a collection of images. Remove all lower scales than current scale and also if max_width or max_height is exceeded
            images = []
            for file in files:
                # Get filename and scale using regex
                filename_regex = re.search(r"(.+?)(?:_(\d(?:\.\d+)?)x)\.png", file)
                if filename_regex != None:
                    filename_regex_name = filename_regex.group(1)
                    filename_regex_scale = float(filename_regex.group(2))

                    # Check if same filename but with higher scale is in images list -> Continue if yes, else add to images list
                    higher_scale_exists = False
                    for image in images:
                        if image["filename"] == filename_regex_name and image["scale"] > scale:
                            higher_scale_exists = True
                            break
                    if higher_scale_exists:
                        print("Higher scale exists")
                        continue
                    # Remove lower scales of same filename if in images list
                    images = [image for image in images if not (image["filename"] == filename_regex_name)]
                    # Check if image is lower than max_width and max_height
                    images.append({"filename": filename_regex_name, "scale": filename_regex_scale, "file": file})
            # Remove images with scale * 2 > max_scale
            images = [image for image in images if not (image["scale"] * 2 > max_scale)]
            # Remove images with size higher than max_width or max_height
            for image in images:
                image = Image.open(folder + "/" + file)
                if image.width > max_width or image.height > max_height:
                    images.remove(image)
            # Check if list empty
            if len(images) > 0:
                # Get image with lowest scale
                lower_scale = None
                for image in images:
                    if lower_scale == None or image["scale"] < lower_scale["scale"]:
                        lower_scale = image
                # Use that image
                loaded_image = Image.open(folder + "/" + lower_scale["file"])
                # Calculate if scale is possible, else calculate optimal scale
                """ Disabled because not needed currently
                width_after_scale = loaded_image.width * scale
                height_after_scale = loaded_image.height * scale
                if width_after_scale > max_width or height_after_scale > max_height:
                    use_scale = min(max_width / loaded_image.width, max_height / loaded_image.height)
                else:
                    use_scale = scale"""
                use_scale = scale
                # Read positive and negative prompt from metadata
                positive_prompt, negative_prompt, positive_prompt_original, negative_prompt_original, additional_prompt, checkpoint, checkpoint_without_extension = get_prompt_data_from_image(loaded_image, fallback_positive_prompt, fallback_negative_prompt)
                print("Image: " + lower_scale["file"])
                print("Current scale: " + str(lower_scale["scale"]) + (lower_scale["scale"] == 1 and " (Base)" or ""))
                print("Next scale: " + str(use_scale))
                print("Positive Prompt: " + positive_prompt)
                print("Positive Prompt Original: " + positive_prompt_original)
                print("Negative Prompt: " + negative_prompt)
                print("Negative Prompt Original: " + negative_prompt_original)
                print("Additional Prompt: " + additional_prompt)
                print("Checkpoint: " + checkpoint)
                print("Checkpoint Without Extension: " + checkpoint_without_extension)
                # Return
                return (torch.from_numpy(numpy.array(loaded_image).astype(numpy.float32) / 255.0).unsqueeze(0), lower_scale["filename"], image["scale"], positive_prompt, negative_prompt, use_scale, positive_prompt_original, negative_prompt_original, checkpoint, checkpoint_without_extension, additional_prompt)

            # Wait 10 seconds and try again
            time.sleep(10)

    @classmethod
    def IS_CHANGED(self, **kwargs):
        return float("nan")
    
class LoadImageWithMetadata:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "filename", "positive_prompt", "negative_prompt", "positive_prompt_original", "negative_prompt_original", "checkpoint", "checkpoint_without_extension", "additional_prompt")
    FUNCTION = "load_image"
    CATEGORY = "t4ggno/image"
    OUTPUT_NODE = False

    def load_image(self: object, image: str):

        print("=============================")
        print("== Auto Load Image For Upscaler")

        # Use that image
        loaded_image = Image.open(folder_paths.get_annotated_filepath(image))
        # Read positive and negative prompt from metadata
        positive_prompt, negative_prompt, positive_prompt_original, negative_prompt_original, additional_prompt, checkpoint, checkpoint_without_extension = get_prompt_data_from_image(loaded_image, "", "")
        print("====================================")
        print("Positive Prompt: " + positive_prompt)
        print("Positive Prompt Original: " + positive_prompt_original)
        print("Negative Prompt: " + negative_prompt)
        print("Negative Prompt Original: " + negative_prompt_original)
        print("Additional Prompt: " + additional_prompt)
        print("Checkpoint: " + checkpoint)
        print("Checkpoint Without Extension: " + checkpoint_without_extension)
        print("====================================")

        image_path = folder_paths.get_annotated_filepath(image)
        
        img = node_helpers.pillow(Image.open, image_path)
        
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']
        
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            
            image = numpy.array(image).astype(numpy.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = numpy.array(i.getchannel('A')).astype(numpy.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        # Return
        return (output_image, output_mask, image, positive_prompt, negative_prompt, positive_prompt_original, negative_prompt_original, checkpoint, checkpoint_without_extension, additional_prompt)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True


class ImageSave:
    def __init__(self):
        self.output_dir = comfy_paths.output_directory
        self.type = os.path.basename(self.output_dir)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename": ("STRING", {"default": ""}),
                "output_path": ("STRING", {"default": ""}),
                "extension": (['png', 'jpeg', 'gif', 'tiff', 'webp'], ),
                "quality": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1}),
                "lossless_webp": (["false", "true"],),
                "show_previews": (["true", "false"],),
                "positive_text": ("STRING", {"default": "", }), # Positive prompt with loras
                "negative_text": ("STRING", {"default": "", }), # Negative prompt with loras
                "additional_text": ("STRING", {"default": "", }), # Additional information
                "model": ("STRING", {"default": ""}), # Checkpoint used
                "positive_text_original": ("STRING", {"default": "", }), # Text before replacement
                "negative_text_original": ("STRING", {"default": "", }), # Text before replacement
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "t4ggno/image"

    def save_images(self, images, filename="", output_path="", extension='png', quality=100, lossless_webp="false", show_previews="true", positive_text="", negative_text="", additional_text="", model="", positive_text_original="", negative_text_original=""):

        print("=============================")
        print("== Image Save")

        lossless_webp = (lossless_webp == "true")

        # Remove category from model. For example: "Anime\Counterfeit-V2.5.safetensors" -> "Counterfeit-V2.5.safetensors" or  "Anime/Counterfeit-V2.5.safetensors" -> "Counterfeit-V2.5.safetensors"
        model = model.split("/")[-1] # For Linux
        model = model.split("\\")[-1] # For Windows

        # Setup output path
        now = datetime.now()
        if output_path.strip() == '':
            output_path = os.path.join(self.output_dir, now.strftime("%Y-%m-%d"))

        # Check output destination
        if not os.path.exists(output_path.strip()):
            print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
            os.makedirs(output_path, exist_ok=True)

        # Set Extension
        if extension not in ALLOWED_EXT:
            print(f"The extension `{extension}` is not valid. The valid formats are: {', '.join(sorted(ALLOWED_EXT))}")
            extension = "png"

        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(numpy.clip(i, 0, 255).astype(numpy.uint8))

            # Create JSON with positive, negative and additional text
            metadata = PngInfo()
            metadata.add_text("PositiveText", positive_text)
            metadata.add_text("PositiveTextOriginal", positive_text_original)
            metadata.add_text("NegativeText", negative_text)
            metadata.add_text("NegativeTextOriginal", negative_text_original)
            metadata.add_text("AdditionalText", additional_text)
            metadata.add_text("Checkpoint", model)
            metadata.add_text("CheckpointWithoutExtension", os.path.splitext(model)[0])

            # If filename is empty, use "yyyy-mm-dd_hh-mm-ss" as filename
            if filename.strip() == '':
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"{timestamp}.{extension}"
            else:
                filename = f"{filename}.{extension}"
            try:
                output_file = os.path.abspath(os.path.join(output_path, filename))
                print(f"Saving image to: {output_file}")
                if extension == 'png':
                    img.save(output_file, pnginfo=metadata, optimize=True)
                elif extension == 'webp':
                    img.save(output_file, quality=quality, lossless=lossless_webp)
                elif extension == 'jpeg':
                    img.save(output_file, quality=quality, optimize=True)
                elif extension == 'tiff':
                    img.save(output_file, quality=quality, optimize=True)

                print(f"Image file saved to: {output_file}")
                print(f"Positive Text: {positive_text}")
                print(f"Positive Text Original: {positive_text_original}")
                print(f"Negative Text: {negative_text}")
                print(f"Negative Text Original: {negative_text_original}")
                print(f"Additional Text: {additional_text}")
                print(f"Checkpoint: {model}")
                print(f"Checkpoint Without Extension: {model.split('.')[0]}")
                results.append({"filename": filename, "path": output_file})

            except OSError as e:
                print(f'Unable to save file to: {output_file}')
                print(e)
            except Exception as e:
                print('Unable to save file due to the following error:')
                print(e)

        if show_previews == 'true':
            return (json.dumps({"ui": {"images": results}}),)
        else:
            return (json.dumps({"ui": {"images": []}}),)

    def get_subfolder_path(self, image_path, output_path):
        output_parts = output_path.strip(os.sep).split(os.sep)
        image_parts = image_path.strip(os.sep).split(os.sep)
        common_parts = os.path.commonprefix([output_parts, image_parts])
        subfolder_parts = image_parts[len(common_parts):]
        subfolder_path = os.sep.join(subfolder_parts[:-1])
        return subfolder_path

class TextReplacer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": ""}),
            },
            "hidden": {
                "control_after_generate": (["fixed", "random", "increment"], {"default": "increment"}),
                "value": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "replace_text"
    CATEGORY = "t4ggno/utils"
    OUTPUT_NODE = False

    def replace_text(self, text):

        print("=============================")
        print("== Text Replacer")

        # Remove empty lines and trim
        text = "\n".join([line.strip() for line in text.split("\n") if line.strip() != ""])
        # Remve multiple "," in a row -> "a,,b" -> "a,b
        text = re.sub(r"\,+?", ",", text)
        # Remove comments - Either // or /* */, but not if /* escaped lik
        # text = re.sub(r"(?<!\\)\/\/.*", "", text)
        # text = re.sub(r"(?<!\\)\/\*(?:.|\n)*?\*\/", "", text)
        # Remove comments - Either # or ''' '''
        text = re.sub(r"#.*", "", text)  # Fallbck for now
        text = re.sub(r"'''(?:.|\n)*?'''", "", text)

        # Remove empty lines and trim
        text = "\n".join([line.strip() for line in text.split("\n") if line.strip() != ""])
        print("Text (After removing comments and empty lines): " + text)

        # Load global whitelist and blacklist - Schema: [Blacklist:TriggerWord:Category,Category,...] and / or [Whitelist:TriggerWord:Category,Category,...]
        globalWhitelist = []
        globalBlacklist = []
        # Find in text
        matches = re.findall(r"(\[(Blacklist|Whitelist)\:(\w+)\:([\w\,]+?)\])", text)
        for match in matches:
            completeString = match[0]
            type = match[1]
            triggerWord = match[2]
            categories = match[3].split(",")
            # Add to globalWhitelist or globalBlacklist
            if type == "Whitelist":
                # Check if triggerWord already in globalWhitelist
                found = False
                for globalWhitelistEntry in globalWhitelist:
                    if globalWhitelistEntry["triggerWord"] == triggerWord:
                        globalWhitelistEntry["categories"] = list(set(globalWhitelistEntry["categories"] + categories))
                        found = True
                        break
                if not found:
                    print("Add to Whitelist: " + str({"triggerWord": triggerWord, "categories": categories}))
                    globalWhitelist.append({"triggerWord": triggerWord, "categories": categories})
            elif type == "Blacklist":
                # Check if triggerWord already in globalBlacklist
                found = False
                for globalBlacklistEntry in globalBlacklist:
                    if globalBlacklistEntry["triggerWord"] == triggerWord:
                        globalBlacklistEntry["categories"] = list(set(globalBlacklistEntry["categories"] + categories))
                        found = True
                        break
                if not found:
                    print("Add to Blacklist: " + str({"triggerWord": triggerWord, "categories": categories}))
                    globalBlacklist.append({"triggerWord": triggerWord, "categories": categories})
            # Remove from text
            text = text.replace(completeString, "")
        print("Global whitelist: " + str(globalWhitelist))
        print("Global blacklist: " + str(globalBlacklist))

        # Replace [PickOne:Option1,Option2,...] with random option - We need to do this before the other replacements, because they might contain [PickOne:Option1,Option2,...]
        while True:
            somethingReplaced = False
            pickOneMatches = re.findall(r"(\[PickOne:(\[(.+?)\]|.+?)\])", text)
            for match in pickOneMatches:
                completeString = match[0]
                options = match[1].split(",")
                randomIndex = numpy.random.randint(0, len(options))
                print("Replace " + completeString + " with " + options[randomIndex])
                indexOf = text.find(completeString)
                text = text[:indexOf] + options[randomIndex] + text[indexOf + len(completeString):]
                somethingReplaced = True

            if not somethingReplaced:
                break

        # Remove empty lines and trim
        text = "\n".join([line.strip() for line in text.split("\n") if line.strip() != ""])
        print("Text (After PickOne replacement): " + text)

        # Now replace all other things
        while True:
            somethingReplaced = False

            # Inline function
            def replaceUsingJson(triggerWord, jsonFileName, quantity=1, shifting=False, atStepStart=0.1, atStepEnd=0.9):
                nonlocal somethingReplaced
                nonlocal text
                nonlocal globalWhitelist
                nonlocal globalBlacklist
                if re.search(r"(\[" + triggerWord + "(?:\:([\w\,]+?))?\])", text) != None:
                    with open(dirPath + "/" + jsonFileName, "r") as infile:
                        jsonData = json.load(infile)
                        jsonDataCategories = list(jsonData.keys())
                        matches = re.findall(r"(\[" + triggerWord + "(?:\:([\w\,]+?))?\])", text)
                        for match in matches:
                            completeString = match[0]
                            category = match[1]
                            # Create list of categories
                            allowedCategories = []
                            if category == "":
                                allowedCategories = jsonDataCategories
                            else:
                                # Split category by comma
                                categorySplit = category.split(",")
                                # Check if in jsonDataCategories -> Add to allowedCategories
                                for category in categorySplit:
                                    if category in jsonDataCategories:
                                        allowedCategories.append(category)
                            # If triggerWord in globalWhitelist -> Remove all categories from allowedCategories that are not in globalWhitelist
                            for globalWhitelistEntry in globalWhitelist:
                                if globalWhitelistEntry["triggerWord"] == triggerWord:
                                    allowedCategories = list(set(allowedCategories).intersection(
                                        globalWhitelistEntry["categories"]))
                            # If triggerWord in globalBlacklist -> Remove all categories from allowedCategories that are in globalBlacklist
                            for globalBlacklistEntry in globalBlacklist:
                                if globalBlacklistEntry["triggerWord"] == triggerWord:
                                    allowedCategories = list(set(allowedCategories).difference(
                                        globalBlacklistEntry["categories"]))
                            # Skip if no allowed categories
                            if len(allowedCategories) == 0:
                                indexOf = text.find(completeString)
                                text = text[:indexOf] + text[indexOf + len(completeString):]
                                continue

                            # Get random value
                            randomValue = ""
                            if shifting:
                                for i in range(quantity):
                                    # Get random category 1
                                    randomCategory1 = allowedCategories[numpy.random.randint(
                                        0, len(allowedCategories))]
                                    # Get random category 2
                                    randomCategory2 = allowedCategories[numpy.random.randint(
                                        0, len(allowedCategories))]
                                    # Get random names
                                    randomValue += "[" + jsonData[randomCategory1][numpy.random.randint(0, len(jsonData[randomCategory1]))] + ":" + jsonData[randomCategory2][numpy.random.randint(
                                        0, len(jsonData[randomCategory2]))] + ":" + str(round(random.uniform(atStepStart, atStepEnd))) + "]"
                            else:
                                for i in range(quantity):
                                    # Get random category
                                    randomCategory = allowedCategories[numpy.random.randint(
                                        0, len(allowedCategories))]
                                    # Get random value
                                    randomValue += jsonData[randomCategory][numpy.random.randint(
                                        0, len(jsonData[randomCategory]))]

                            # Replace completeString with randomValue
                            print("Replace " + completeString +
                                  " with " + randomValue)
                            indexOf = text.find(completeString)
                            text = text[:indexOf] + randomValue + text[indexOf + len(completeString):]
                            somethingReplaced = True

            def replaceUsingArray(triggerWord, array, quantity=1, shifting=False, atStepStart=0.1, atStepEnd=0.9):
                nonlocal somethingReplaced
                nonlocal text
                nonlocal globalWhitelist
                nonlocal globalBlacklist
                # Remove elements from array if in globalBlacklist
                for globalBlacklistEntry in globalBlacklist:
                    if globalBlacklistEntry["triggerWord"] == triggerWord:
                        array = list(set(array).difference(globalBlacklistEntry["categories"]))
                # Remove elements from array if not in globalWhitelist
                for globalWhitelistEntry in globalWhitelist:
                    if globalWhitelistEntry["triggerWord"] == triggerWord:
                        array = list(set(array).intersection(globalWhitelistEntry["categories"]))
                # Skip if no allowed categories
                if len(array) == 0:
                    # Remove triggerWord from text
                    text = text.replace("[" + triggerWord + "]", "")
                    return
                matches = re.findall(r"(\[" + triggerWord + "\])", text)
                for match in matches:
                    completeString = match
                    # Get random value
                    randomValue = ""
                    if shifting:
                        for i in range(quantity):
                            # Get random value 1
                            randomValue1 = array[numpy.random.randint(0, len(array))]
                            # Get random value 2
                            randomValue2 = array[numpy.random.randint(0, len(array))]
                            # Get random names
                            randomValue += "[" + randomValue1 + ":" + randomValue2 + ":" + str(round(random.uniform(atStepStart, atStepEnd))) + "]"
                    else:
                        for i in range(quantity):
                            # Get random value
                            randomValue += array[numpy.random.randint(0, len(array))]
                    # Replace completeString with randomValue
                    print("Replace " + completeString + " with " + randomValue)
                    indexOf = text.find(completeString)
                    text = text[:indexOf] + randomValue + text[indexOf + len(completeString):]
                    somethingReplaced = True

            # Replace [TimeOfDay] (if exists) with random time of day
            timeOfDay = ["morning", "afternoon", "evening", "night", "sunset", "sunrise"]
            replaceUsingArray("TimeOfDay", timeOfDay)
            # Replace [Weather] (if exists) with random weather
            weather = ["sunny", "cloudy", "rainy", "snowy"]
            replaceUsingArray("Weather", weather)
            # Replace [Nudity] (if exists) with random location
            replaceUsingJson("Nudity", "nudity.json")
            # Replace [FacialExpression] (if exists) with random facial expression
            facialExpression = ["happy", "sad", "angry", "surprised", "disgusted", "scared", "cry", "laugh", "smile", "light smile"]
            replaceUsingArray("FacialExpression", facialExpression)
            # Replace [Location] (if exists) with random location
            replaceUsingJson("Location", "locations.json")
            # Replace [Scenario] (if exists) with random location
            replaceUsingJson("Scenario", "scenarios.json")
            # Replace [HairColor] (if exists) with random hair color
            replaceUsingJson("HairColor", "hair_colors.json")

            # Replace [AttireEyewear] (if exists) with random eyewear
            replaceUsingJson("AttireEyewear", "attire_eyewear.json")
            # Replace [AttireSleeve] (if exists) with random sleeve
            replaceUsingJson("AttireSleeve", "attire_sleeves.json")
            # Replace [AttireJewelryAndAccessories] (if exists) with random jewelry and accessories
            replaceUsingJson("AttireJewelryAndAccessories", "attire_jewelry_and_accessories.json")
            # Replace [AttireStylesAndPatterns] (if exists) with random styles and patterns
            replaceUsingJson("AttireStylesAndPatterns", "attire_styles_and_patterns.json")

            # Replace [AttireFemaleBodysuit] (if exists) with random bodysuit
            replaceUsingJson("AttireFemaleBodysuit", "attire_female_bodysuits.json")
            # Replace [AttireFemaleBottomwear] (if exists) with random bottomwear
            replaceUsingJson("AttireFemaleBottomwear", "attire_female_bottomwear.json")
            # Replace [AttireFemaleBra] (if exists) with random bra
            replaceUsingJson("AttireFemaleBra", "attire_female_bra.json")
            # Replace [AttireFemaleFootwear] (if exists) with random footwear
            replaceUsingJson("AttireFemaleFootwear", "attire_female_footwear.json")
            # Replace [AttireFemaleHeadwear] (if exists) with random headwear
            replaceUsingJson("AttireFemaleHeadwear", "attire_female_headwear.json")
            # Replace [AttireFemaleLegwear] (if exists) with random legwear
            replaceUsingJson("AttireFemaleLegwear", "attire_female_legwear.json")
            # Replace [AttireFemalePanties] (if exists) with random panties
            replaceUsingJson("AttireFemalePanties", "attire_female_panties.json")
            # Replace [AttireFemaleSwimsuit] (if exists) with random swimsuit
            replaceUsingJson("AttireFemaleSwimsuit", "attire_female_swimsuit.json")
            # Replace [AttireFemaleTopwear] (if exists) with random topwear
            replaceUsingJson("AttireFemaleTopwear", "attire_female_topwear.json")
            # Replace [AttireFemaleTraditionalClothing] (if exists) with random traditional clothing
            replaceUsingJson("AttireFemaleTraditionalClothing", "attire_female_traditional_clothing.json")
            # Replace [AttireFemaleUniformsAndCostumes] (if exists) with random uniforms and costumes
            replaceUsingJson("AttireFemaleUniformsAndCostumes", "attire_female_uniforms_and_costumes.json")

            # Replace [AttireMaleBodysuit] (if exists) with random bodysuit
            replaceUsingJson("AttireMaleBodysuit", "attire_male_bodysuits.json")
            # Replace [AttireMaleBottomwear] (if exists) with random bottomwear
            replaceUsingJson("AttireMaleBottomwear", "attire_male_bottomwear.json")
            # Replace [AttireMaleFootwear] (if exists) with random footwear
            replaceUsingJson("AttireMaleFootwear", "attire_male_footwear.json")
            # Replace [AttireMaleHeadwear] (if exists) with random headwear
            replaceUsingJson("AttireMaleHeadwear", "attire_male_headwear.json")
            # Replace [AttireMaleLegwear] (if exists) with random legwear
            replaceUsingJson("AttireMaleLegwear", "attire_male_legwear.json")
            # Replace [AttireMaleSwimsuit] (if exists) with random swimsuit
            replaceUsingJson("AttireMaleSwimsuit", "attire_male_swimsuit.json")
            # Replace [AttireMaleTopwear] (if exists) with random topwear
            replaceUsingJson("AttireMaleTopwear", "attire_male_topwear.json")
            # Replace [AttireMaleTraditionalClothing] (if exists) with random traditional clothing
            replaceUsingJson("AttireMaleTraditionalClothing", "attire_male_traditional_clothing.json")
            # Replace [AttireMaleUniformsAndCostumes] (if exists) with random uniforms and costumes
            replaceUsingJson("AttireMaleUniformsAndCostumes", "attire_male_uniforms_and_costumes.json")

            # Replace [Fetishes] (if exists) with random stuff
            replaceUsingJson("Fetishes", "fetishes.json")
            # Replace [SexToys] (if exists) with random stuff
            replaceUsingJson("SexToys", "sex_toys.json")
            # Replace [SexToysBondage] (if exists) with random stuff
            replaceUsingJson("SexToysBondage", "sex_toys_bondage.json")
            # Replace [SexPositions] (if exists) with random stuff
            replaceUsingJson("SexPositions", "sex_positions.json")

            # Replace [SexualActs] (if exists) with random stuff
            replaceUsingJson("SexualActs", "sexual_acts.json")
            # Replace [SexualAttireBDSM] (if exists) with random stuff
            replaceUsingJson("SexualAttireBDSM", "sexual_attire_bdsm.json")
            # Replace [SexualAttireExposure] (if exists) with random stuff
            replaceUsingJson("SexualAttireExposure", "sexual_attire_exposure.json")
            # Replace [SexualAttireLingerine] (if exists) with random stuff
            replaceUsingJson("SexualAttireLingerine", "sexual_attire_lingerine.json")
            # Replace [SexualAttireMiscellaneous] (if exists) with random stuff
            replaceUsingJson("SexualAttireMiscellaneous", "sexual_attire_miscellaneous.json")

            # Replace [PostureArm] (if exists) with random stuff
            replaceUsingJson("PostureArm", "posture_arm.json")
            # Replace [PostureBasic] (if exists) with random stuff
            replaceUsingJson("PostureBasic", "posture_basic.json")
            # Replace [PostureCarrying] (if exists) with random stuff
            replaceUsingJson("PostureCarrying", "posture_carrying.json")
            # Replace [PostureHead] (if exists) with random stuff
            replaceUsingJson("PostureHead", "posture_head.json")
            # Replace [PostureHips] (if exists) with random stuff
            replaceUsingJson("PostureHips", "posture_hips.json")
            # Replace [PostureHug] (if exists) with random stuff
            replaceUsingJson("PostureHug", "posture_hug.json")
            # Replace [PostureLeg] (if exists) with random stuff
            replaceUsingJson("PostureLeg", "posture_leg.json")
            # Replace [PostureMovement] (if exists) with random stuff
            replaceUsingJson("PostureMovement", "posture_movement.json")
            # Replace [PostureMultipleCharacter] (if exists) with random stuff
            replaceUsingJson("PostureMultipleCharacter", "posture_multiple_character.json")
            # Replace [PostureOther] (if exists) with random stuff
            replaceUsingJson("PostureOther", "posture_other.json")
            # Replace [PosturePoses] (if exists) with random stuff
            replaceUsingJson("PosturePoses", "posture_poses.json")
            # Replace [PostureTorso] (if exists) with random stuff
            replaceUsingJson("PostureTorso", "posture_torso.json")

            # Replace [Color] (if exists) with random color
            colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray", "grey", "gold", "silver", "bronze", "copper"]
            replaceUsingArray("Color", colors)
            # Replace [PickMultiple,Option1,Option2,...] (if exists) with random option
            pickMultipleMatches = re.findall(r"(\[PickMultiple(?:\:([\w\,]+?))?\])", text)
            for match in pickMultipleMatches:
                completeString = match[0]
                options = match[1].split(",")
                quantity = numpy.random.randint(1, len(options) + 1)
                randomIndexes = numpy.random.choice(len(options), quantity, replace=False)
                randomOptions = []
                for randomIndex in randomIndexes:
                    randomOptions.append(options[randomIndex])
                print("Replace " + completeString + " with " + ", ".join(randomOptions))
                indexOf = text.find(completeString)
                text = text[:indexOf] + ", ".join(randomOptions) + text[indexOf + len(completeString):]
                somethingReplaced = True
            # 14. Replace [NationalityByContinent] (if exists) with random option
            replaceUsingJson("NationalityByContinent", "nationality_by_continent.json")
            # 15. Replace [NationalityByEthnic] (if exists) with random option
            replaceUsingJson("NationalityByEthnic", "nationality_by_ethnic.json")
            # Replace [Random] (if exists) with random option
            if re.search(r"(\[Random(?:\:[\w]+){0,}\])", text) != None:
                randomMatches = re.findall(r"(\[Random(?:\:[\w]+){0,}\])", text)
                for match in randomMatches:
                    completeString = match
                    newText = ""
                    # Has a person, yes or no
                    hasPerson = (numpy.random.randint(0, 3) == 0 or re.search(r"\[Random.*\:HasPerson.*\]", text) != None) and re.search(r"\[Random.*\:HasNoPerson.*\]", text) == None
                    if hasPerson:
                        # Nationality by ethnic
                        newText += "[NationalityByEthnic] "
                        isFemale = (numpy.random.randint(0, 2) == 0 or re.search(r"\[Random.*\:IsFemale.*\]", text) != None) and re.search(r"\[Random.*\:IsMale.*\]", text) == None
                        if isFemale:
                            newText += "woman,[Name],[HairFemale],"
                            # Sexy or not
                            if numpy.random.randint(0, 2) == 0 or re.search(r"\[Random.*\:SexyOnly.*\]", text) != None:
                                newText += "[SexyThings],"
                                # Nudity
                                if numpy.random.randint(0, 2) == 0:
                                    newText += "[Nudity],"
                                # Dress or full body
                                if numpy.random.randint(0, 2) == 0:
                                    # Upper body
                                    if numpy.random.randint(0, 2) == 0:
                                        newText += "[ClothesFemaleSexy:UpperBody],"
                                    # Lower body
                                    if numpy.random.randint(0, 2) == 0:
                                        newText += "[ClothesFemaleSexy:LowerBody],"
                                else:
                                    newText += "[ClothesFemaleSexy:Dress],"
                                # Head
                                if numpy.random.randint(0, 2) == 0:
                                    newText += "[ClothesFemaleSexy:Head],"
                                # Shoes
                                if numpy.random.randint(0, 2) == 0:
                                    newText += "[ClothesFemaleSexy:Footwear],"
                            else:
                                # Dress or full body
                                if numpy.random.randint(0, 2) == 0:
                                    # Upper body
                                    if numpy.random.randint(0, 2) == 0:
                                        newText += "[ClothesFemale:UpperBody],"
                                    # Lower body
                                    if numpy.random.randint(0, 2) == 0:
                                        newText += "[ClothesFemale:LowerBody],"
                                else:
                                    newText += "[ClothesFemale:Dress],"
                                # Head
                                if numpy.random.randint(0, 2) == 0:
                                    newText += "[ClothesFemale:Head],"
                                # Shoes
                                if numpy.random.randint(0, 2) == 0:
                                    newText += "[ClothesFemale:Footwear],"
                        else:
                            newText = "man,[Name],[HairMale],"
                            # Sexy or not
                            if numpy.random.randint(0, 2) == 0 or re.search(r"\[Random.*\:SexyOnly.*\]", text) != None:
                                newText += "[SexyThings],"
                                # Nudity
                                if numpy.random.randint(0, 2) == 0:
                                    newText += "[Nudity],"
                                # Dress or full body
                                if numpy.random.randint(0, 2) == 0:
                                    # Upper body
                                    if numpy.random.randint(0, 2) == 0:
                                        newText += "[ClothesMaleSexy:UpperBody],"
                                    # Lower body
                                    if numpy.random.randint(0, 2) == 0:
                                        newText += "[ClothesMaleSexy:LowerBody],"
                                else:
                                    newText += "[ClothesMaleSexy:Dress],"
                                # Head
                                if numpy.random.randint(0, 2) == 0:
                                    newText += "[ClothesMaleSexy:Head],"
                                # Shoes
                                if numpy.random.randint(0, 2) == 0:
                                    newText += "[ClothesMaleSexy:Footwear],"
                            else:
                                # Dress or full body
                                if numpy.random.randint(0, 2) == 0:
                                    # Upper body
                                    if numpy.random.randint(0, 2) == 0:
                                        newText += "[ClothesMale:UpperBody],"
                                    # Lower body
                                    if numpy.random.randint(0, 2) == 0:
                                        newText += "[ClothesMale:LowerBody],"
                                else:
                                    newText += "[ClothesMale:Dress]"
                                # Head
                                if numpy.random.randint(0, 2) == 0:
                                    newText += "[ClothesMale:Head],"
                                # Shoes
                                if numpy.random.randint(0, 2) == 0:
                                    newText += "[ClothesMale:Footwear],"
                        # Facial expression
                        if numpy.random.randint(0, 2) == 0:
                            newText += "[FacialExpression],"
                        # Hair color
                        if numpy.random.randint(0, 2) == 0:
                            newText += "[HairColor] hair,"
                        # Pose
                        if numpy.random.randint(0, 2) == 0:
                            newText += "[Poses],"
                        # Eye color
                        if numpy.random.randint(0, 2) == 0:
                            newText += "[Color] eyes,"
                        # Imperfections
                        newText += "(imperfections:0.3),(freckles:0.3),"
                    # Time of day
                    if numpy.random.randint(0, 2) == 0:
                        newText += "[TimeOfDay],"
                    # Location - No shifting
                    locationQuantity = numpy.random.randint(1, 5)
                    for i in range(locationQuantity):
                        if numpy.random.randint(0, 2) == 0:
                            newText += "[Location],"
                    # Location - Shifting
                    """if numpy.random.randint(0, 2) == 0:
                        locationQuantity = numpy.random.randint(1, 3)
                        for i in range(locationQuantity):
                            # Float betwen 0 and 1
                            atStep = round(random.uniform(0.1, 0.9), 2)
                            newText += "[[Location]:[Location]:{0}],".format(
                                atStep)"""
                    # Scenario - No shifting
                    scenarioQuantity = numpy.random.randint(1, 5)
                    for i in range(scenarioQuantity):
                        if numpy.random.randint(0, 2) == 0:
                            newText += "[Scenario],"
                    # Scenario - Shifting
                    """if numpy.random.randint(0, 2) == 0:
                        scenarioQuantity = numpy.random.randint(1, 3)
                        for i in range(scenarioQuantity):
                            # Float betwen 0 and 1
                            atStep = round(random.uniform(0.1, 0.9), 2)
                            newText += "[[Scenario]:[Scenario]:{0}],".format(
                                atStep)"""
                    # Weather
                    if numpy.random.randint(0, 2) == 0:
                        newText += "[Weather],"
                    # Style
                    newText += "[Style],"
                    # Interesting ideas
                    if numpy.random.randint(0, 2) == 0:
                        newText += "[InterestingIdeas],"

                    indexOf = text.find(completeString)
                    text = text[:indexOf] + newText + text[indexOf + len(completeString):]
                    somethingReplaced = True
            # Replace [Name] (if exists) with random name
            if re.search(r"(\[Name(?:\:([\w\,]+?))?\])", text) != None:
                isMale = re.search(r"(?:^|\ |,)(male|boy|man)", text) != None
                isFemale = re.search(r"(?:^|\ |,)(female|girl|woman)", text) != None
                print("Is Male: " + str(isMale) + "; Is Female: " + str(isFemale))
                fileName = ""
                if isMale:
                    fileName = "names_male.json"
                elif isFemale:
                    fileName = "names_female.json"
                # Load names from json list
                if fileName != "":
                    replaceUsingJson("Name", fileName)
            # Hair female
            replaceUsingJson("HairFemale", "hair_female.json")
            # Hair male
            replaceUsingJson("HairMale", "hair_male.json")
            # Sexy things
            replaceUsingJson("SexyThings", "sexy_things.json")
            # Styles
            replaceUsingJson("Style", "styles.json")
            # Interresting ideas
            replaceUsingJson("InterestingIdeas", "interesting_ideas.json")
            # Materials
            replaceUsingJson("Material", "materials.json")

            # Last: "If" -> If:Contains:TriggerWord:Add:Value -> Allow "[Statement]" in Value via escaping. Example: [If:Contains:woman:Add:[PickOne:22,80] years old]
            if re.search(r"(\[\[If:(Contains):([\w\ ]+?):(Add):([\w\ \[\]\:\,]+?)\]\])", text) != None:
                matches = re.findall(r"(\[\[If:(Contains):([\w\ ]+?):(Add):([\w\ \[\]\:\,]+?)\]\])", text)
                for match in matches:
                    completeString = match[0]
                    contains = match[2]
                    value = match[4]
                    if contains in text:
                        indexOf = text.find(completeString)
                        text = text[:indexOf] + value + text[indexOf + len(completeString):]
                        somethingReplaced = True
                    else:
                        continue

            if not somethingReplaced:
                break

        # Remove empty lines and trim
        text = "\n".join([line.strip() for line in text.split("\n") if line.strip() != ""])
        print("Text (After all replacements): " + text)

        # Remove unmatched "If" statements
        while True:
            somethingRemoved = False
            if re.search(r"(\[\[If:.*?\]\])", text) != None:
                matches = re.findall(r"(\[\[If:.*?\]\])", text)
                for match in matches:
                    completeString = match
                    indexOf = text.find(completeString)
                    text = text[:indexOf] + text[indexOf + len(completeString):]
                    somethingRemoved = True
            if not somethingRemoved:
                break

        # Remove empty lines and trim
        text = "\n".join([line.strip() for line in text.split("\n") if line.strip() != ""])
        print("Text (After removing unmatched If statements): " + text)

        # Replace regex lora with random lora
        all_loras_regex = re.findall(r"<RE\:(.+?)(?:\:(-?[0-9](?:\.[0-9]*)?)|(?:\:(-?[0-9]+(?:\.[0-9]*)?|))(?:\:(-?[0-9]+(?:\.[0-9]*)?|)))?>", text)
        if len(all_loras_regex) > 0:
            avaialbe_loras = folder_paths.get_filename_list("loras")
            print("All loras: " + str(avaialbe_loras))
            for lora in all_loras_regex:
                print("Lora regex: " + lora[0])
                # Convert regex to regex object
                regex = re.compile(lora[0])
                # Create list with all avaialbe loras that match regex
                avaialbe_loras_regex = list(filter(lambda x: regex.match(x) != None, avaialbe_loras))
                # print("Avaialbe loras regex: " + str(avaialbe_loras_regex))
                # If no avaialbe loras match regex, skip
                if len(avaialbe_loras_regex) == 0:
                    print("No avaialbe loras match regex")
                    # Remove lora from text
                    text = text.replace("<RE:" + lora[0] + ">", "")
                    continue
                # Get random lora from avaialbe_loras_regex
                random_lora = avaialbe_loras_regex[numpy.random.randint(0, len(avaialbe_loras_regex))]
                # Remove extension name
                random_lora = re.sub(r"\.[a-zA-Z0-9]+$", "", random_lora)
                print("Random lora: " + random_lora)
                # Replace lora with random_lora
                text = text.replace("<RE:" + lora[0], "<" + random_lora)

        # Remove empty lines and trim
        text = "\n".join([line.strip() for line in text.split("\n") if line.strip() != ""])
        print("Text (After replacing regex lora with random lora): " + text)

        # Cleanup - Remove empty lines, trim, remove multiple spaces, remove multiple commas, etc
        text = "\n".join([line.strip() for line in text.split("\n") if line.strip() != ""])
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\,+?", ",", text)

        print("Final text: " + text)
        return (text,)

    def IS_CHANGED(cls, **kwargs):
        return True


def get_next_prompt(cls, append_prefix, append_suffix, images_per_batch):
    # Read prompt_from_ai.txt
    try:
        with open("prompt_from_ai.txt", "r") as infile:
            prompt_from_ai = infile.read()
    except:
        print("prompt_from_ai.txt not found")
        return None
    # Split prompt_from_ai.txt by empty line (could contain whitespace) using regex
    prompt_from_ai = re.split(r"^\s*$", prompt_from_ai, flags=re.MULTILINE)
    # In first element is index and image count
    index = int(prompt_from_ai[0].split("\n")[0].split(":")[1])
    image_count = int(prompt_from_ai[0].split("\n")[1].split(":")[1])
    # The rest are prompts
    prompts = prompt_from_ai[1:]
    # Return None if no prompts
    if len(prompts) == 0:
        print("No prompts found")
        return None
    # Remove empty prompts (could contain whitespace and newlines)
    prompts = list(filter(lambda x: x != "", prompts))
    # If image_count is higher or equal than images_per_batch, reset image_count and increase index by 1
    if image_count >= images_per_batch:
        print("Image count higher or equal than images per batch -> Reset image count and increase index by 1")
        image_count = 0
        index += 1
    # If index is higher than len(prompts), return None
    if index >= len(prompts):
        print("Index higher than len(prompts)")
        return None
    # Increase image_count by 1
    image_count += 1
    # Output index and image_count
    print("Index: " + str(index))
    print("Image count: " + str(image_count))
    # Get prompt
    prompt = prompts[index]
    # Remove empty lines in prompt
    prompt = re.sub(r"^\s*$", "", prompt, flags=re.MULTILINE)
    # Write prompt_from_ai.txt
    with open("prompt_from_ai.txt", "w") as outfile:
        outfile.write("index:" + str(index) + "\nimage:" + str(image_count) + "\n\n" + "\n".join(prompts))
    # Return prompt
    prompt_splitted = prompt.split("\n")
    # Remove empty lines in prompt
    prompt_splitted = list(filter(lambda x: x != "", prompt_splitted))
    prompt = prompt_splitted[0]
    # Append prefix and suffix
    prompt = append_prefix + " " + prompt + " " + append_suffix
    print("Prompt: " + prompt)
    return (prompt,)


class PromptFromAIOpenAI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "gpt": (["gpt-3.5-turbo", "gpt-3.5-turbo-16K", "gpt-4", "gpt-4-turbo", "gpt-4o", "Custom"], {"default": "gpt-4o"}),
                "gpt_custom": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 1.1}),
                "frequency_penalty": ("FLOAT", {"default": 0.2}),
                "presence_penalty": ("FLOAT", {"default": 0.2}),
                "details": ("STRING", {"multiline": True}),
                "append_prefix": ("STRING",  {"multiline": True}),
                "append_suffix": ("STRING",  {"multiline": True}),
                "batch_quantity": ("INT", {"default": 1}),
                "images_per_batch": ("INT", {"default": 1}),
            },
            "hidden": {
                "control_after_generate": (["fixed", "random", "increment"], {"default": "increment"}),
                "value": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "get_prompt"
    CATEGORY = "t4ggno/utils"
    OUTPUT_NODE = False

    def get_prompt(cls, api_key, gpt, gpt_custom, temperature, frequency_penalty, presence_penalty, details, append_prefix, append_suffix, batch_quantity, images_per_batch):
        print("=============================")
        print("== Get prompt from OpenAI")

        next_prompt = get_next_prompt(cls, append_prefix, append_suffix, images_per_batch)
        if next_prompt:
            return next_prompt

        # Get all available loras
        # Folder structure will be: category\[sub_category]?\[name].safetensors
        available_loras = folder_paths.get_filename_list("loras")
        # Convert lora list to a json. For example: {"character": [ "JenniferLopez-V1.0", "Alita-V1.0" ], "effect": [ "GlowingEyes-V1.0" ]}
        converted_loras = {}
        for lora in available_loras:
            lora_split = lora.split("\\")
            category = lora_split[0]
            if category not in converted_loras:
                converted_loras[category] = []
            # Name is always the last element
            converted_loras[category].append(lora_split[-1])
        available_loras = json.dumps(converted_loras)
        # print("Available loras: " + available_loras)

        # Get prompt from ChatGPT for GPT-4
        client = OpenAI(
            api_key=api_key
        )

        # Load keywoar list from file if exists
        keywoard_list = ""
        try:
            with open("keywoard_list.txt", "r") as infile:
                keywoard_list = infile.read()
        except:
            print("keywoard_list.txt not found")

        # Get new prompts from ChatGPT
        gpt_assistant_prompt = "You are a Stable Diffusion prompt generator. The prompt should be detailed. Use Keyword sentences. The scene should be interesting and engaging. The prompt should be creative and unique. Start directly with the prompt and dont use a caption. If you have to create multiple prompts, add an empty line between them. Don't number them! \nYou can also use loras in the following format: <loraname:strength>. \nYou can strengthen parts of the prompt using bradcks. For example: a white (cute cat) is playing with a (red ball:1.2)."
        gpt_assistant_prompt += "\n---------------------------------"
        gpt_assistant_prompt += """\nExample Output:
            Bosstyle,Silhouette of a Woman fighting a giant DARK mononoke Monster Boss resembling a ethereal fox with seven tails: Capture the essence of nostalgia and color in this vintage photograph. Dominating the scene is a meticulously detailed,translucent dark-red Blood filled firefox,adorned with vibrant patterns,blending seamlessly with its surroundings. Its long tail,composed of intricately woven ribbons in a spectrum of colors,trails behind,adding to the spectacle of the moment. Amidst the quietude of the night,its presence exudes a sense of ancestral strength and resilience. Behind it,the low-toned hues of the Milky Way cast a mesmerizing backdrop,a cosmic tapestry weaving tales of both past and future. In this moment,the convergence of tradition and technology,of ancient wisdom and industrial progress,hangs palpably in the air.,atmospheric haze,Film grain,cinematic film still,shallow depth of field,highly detailed,high budget,cinemascope,moody,epic,OverallDetail,2000s vintage RAW photo,photorealistic,candid camera,color graded cinematic,eye catchlights,atmospheric lighting,imperfections,natural,shallow dof

            strong and powerful dungeons and dragons epic movie poster barbarian woman with cape charging into battle violent roar riding a vicious ice [wolf|tiger] beast leather and fur boots warriors and red banners (windy dust debris storm:1.1) volumetric lighting fog depth mist pass z pass great stone castle very bright morning sunlight from side

            enchanting scene,a steaming cup of coffee,the liquid within the cup forms a swirling cosmic galaxy,The coffee surface is a mesmerizing mix of deep purples and blues and pinks dotted with stars and nebulae,fit the vastness of space inside the cup,The cup rests on a saucer that reflects the celestial patterns,surrounded by tiny star-shaped cookies and cosmic-themed decorations,The atmosphere is magical and dreamy,<EpicF4nta5yXL:0.7>

            an anthropomorphic turtle with a large decorated shell,walking through a dense misty forest,the turtle is wearing a long flowing robe adorned with intricate patterns and holding a wooden staff,The shell is filled with various items including flowers and pouches,wandering sage,towering trees background,moss-covered rocks,soft ethereal light filtering through the forest canopy,a magical and serene atmosphere,<detailed_notrigger:0.6>,<zavy-cntrst-sdxl:0.7>

            Assasin,A closeup of fantastical image of a assasin false glory,darkness of night fall,Their filled with power and determination,clad in flowing,flowing scarf,wielding an scarf ornate,scarf ornate,as they wield the scarf n a fluid,sword carried behind the body,squatting on the roof of the clock tower building like an assassin,sitting squatting pose,hands touch the ground,squatting all four,action pose,clock tower,from behind view,cyberpunk city,neon city,skindentation,glowing,shiny,scifi,neon lights,intricate artistic color,cinematic light,radiant,vibrant,perfect symmetry,Grayscale,Dramatic spotlight,highly color focused,dynamic motion,very dark

            A captivating paranormal photograph,(oddly tilted fisheyelens) that showcases intense dynamic composition of a (partiallyvaporous terrifying vampire that is part of the shadows) (emerging in a dark bedroom at night),(floating above) (an innocent,adorable,victim),((high angle photograph,highly dramatic cinematic atmosphere)),((the very detailed background is imposing,with angular intimidating,artdeco style architectural designs)),the (terrifying shadow vampire looming over its adorable victim is part of dark and smokey shadows at nighttime in the (girly bedroom of the victim)),cinematic spiritual horror,dynamic,((double exposures)),very detailed background,ominous ethereal background,epiCRealism,epiCNegative,Photographer Martin Kimbell Style,ral-dissolve

            fanciful,translucent vintage camera with metallic copper and silver details,filled with an array of soft pastel flowers including hues of pink and white and green inside it,floating golden sparkles,magical touch to the composition,<sdxl_glass:0.7>,<tranzp:0.7>,<add-detail-xl:1>

            linquivera,Ink illustration,(anime:0.5),brown tones,aged black red paper,inkpunk,moonlight,surreal,a ghost standing on a boat in swampy wetlands,(at a distance),Will-o'-the-wisp,moonlit,lonely,solitude,windy,tall trees,willows,willowy,OverallDetail,extremely detailed,UHD,(long exposure ,dystopian but extremely beautiful:1.4),<Cute_Collectible:1>,<XL_boss_battle:0.6> <add-detail-xl:1>,<MJ52:1>
        """
        gpt_user_prompt = "\nDetails for the prompt: " + details
        gpt_user_prompt += "\n---------------------------------"
        gpt_user_prompt += "\nQuantity of prompts: " + str(batch_quantity)
        gpt_user_prompt += "\n---------------------------------"
        gpt_user_prompt += "\nAvailable loras: " + available_loras

        if keywoard_list != "":
            gpt_user_prompt += "\n---------------------------------"
            gpt_user_prompt += "\nAvoid prompts similar to the following keywoards: " + keywoard_list

        response = client.chat.completions.create(
            model=gpt if gpt_custom == "" else gpt_custom,
            messages=[{"role": "assistant", "content": re.sub(r"^\s+", "", gpt_assistant_prompt, flags=re.MULTILINE)}, {"role": "user", "content": re.sub(r"^\s+", "", gpt_user_prompt, flags=re.MULTILINE)}],
            temperature=temperature,
            max_tokens=4095,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        # Check for errors or empty responses
        if response.choices[0].message.content == "" or response.choices[0].message.content == " ":  # or response.choices[0].message.content == "No prompts found":
            print("No prompts found")
            return cls.get_prompt(api_key, gpt, gpt_custom, temperature, frequency_penalty, presence_penalty, details, append_prefix, append_suffix, batch_quantity, images_per_batch)
        prompt = response.choices[0].message.content

        # Get a new overview of the prompts (Keywoards like "castle", "underwater sear world", ...)
        gpt_assistant_prompt = """
            You will receive an overview of prompts. Create a short keywoard list of prompts I can use, to prevent furhter generations of the same or similiar prompts later. Dont be too detailed.
            Start directly with the keywoards! Seperate the keywoards with a comma. If already a keywoard list is provided, attach it to the end of the list.
        """
        if keywoard_list != "":
            gpt_user_prompt = "Prompts:\n\n" + response.content[0].text
            gpt_user_prompt += "\n---------------------------------"
            gpt_user_prompt += "Keywoard list: " + keywoard_list
        else:
            gpt_user_prompt = "Prompts:\n\n" + response.content[0].text

        responseKeywoarList = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "assistant", "content": re.sub(r"^\s+", "", gpt_assistant_prompt, flags=re.MULTILINE)}, {"role": "user", "content": re.sub(r"^\s+", "", gpt_user_prompt, flags=re.MULTILINE)}],
            temperature=0.8,
            max_tokens=4095,
            frequency_penalty=0.2,
            presence_penalty=0.2,
        )

        print("Response prompt: " + response.choices[0].message.content)
        print("Response keywoard list: " + responseKeywoarList.choices[0].message.content)

        # Remove numbering from prompts - Via regex
        prompt = re.sub(r"^\d+\.", "", prompt, flags=re.MULTILINE)

        # Combines lines where none empty lines are between them
        prompt = re.sub(r"([^\n])\n([^\n])", r"\1 \2", prompt)

        # Fix invalid loras -> For example <Character:Gamora-V1.0.safetensors> to <lora:Gamora-V1.0> or <NSFW:NoBra-V1.0.safetensors> to <lora:NoBra-V1.0>
        # Remove the part before the colon and probably the .safetenors part
        prompt = re.sub(r"<(?:[\w\-. \\]+?)\:((?:[\w-]+?)(?:\-V\d+\.\d+))?(?:\.safetensors?)?>", r"<lora:\1>", prompt)
        # Change <loraname: to <lora:
        prompt = re.sub(r"<loraname:", "<lora:", prompt)

        # Remove lines with only whitespace or "-----" or similar but not empty lines
        prompt = re.sub(r"^\s*[\-]+\s*$", "", prompt, flags=re.MULTILINE)

        # Write prompt to prompt_from_ai.txt
        with open("prompt_from_ai.txt", "w") as outfile:
            outfile.write("index:0\nimage:0\n\n" + prompt)
        # Write keywoard list to keywoard_list.txt
        with open("keywoard_list.txt", "w") as outfile:
            outfile.write(responseKeywoarList.choices[0].message.content)
        # Rerun get_prompt to get the first prompt or generate new if something went wrong
        return cls.get_prompt(api_key, gpt, gpt_custom, temperature, frequency_penalty, presence_penalty, details, append_prefix, append_suffix, batch_quantity, images_per_batch)

    @classmethod
    def IS_CHANGED(self, **kwargs):
        return float("nan")


class PromptFromAIAnthropic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "gpt": (["claude-3.5-sonnet", "claude-3-opus"], {"default": "claude-3.5-sonnet"}),
                "details": ("STRING", {"multiline": True}),
                "append_prefix": ("STRING",  {"multiline": True}),
                "append_suffix": ("STRING",  {"multiline": True}),
                "batch_quantity": ("INT", {"default": 1}),
                "images_per_batch": ("INT", {"default": 1}),
                "lora_whitelist_regex": ("STRING", {"default": "", "multiline": True}),
                "lora_blacklist_regex": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {
                "control_after_generate": (["fixed", "random", "increment"], {"default": "increment"}),
                "value": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "get_prompt"
    CATEGORY = "t4ggno/utils"
    OUTPUT_NODE = False

    def get_prompt(cls, api_key, gpt, details, append_prefix, append_suffix, batch_quantity, images_per_batch, lora_whitelist_regex, lora_blacklist_regex):

        print("=============================")
        print("== Get prompt from Anthropic")

        next_prompt = get_next_prompt(cls, append_prefix, append_suffix, images_per_batch)
        if next_prompt:
            return next_prompt

        # Get all available loras
        # Folder structure will be: category\[sub_category]?\[name].safetensors
        lora_paths = folder_paths.get_folder_paths("loras")
        if len(lora_paths) == 0:
            raise Exception("No lora paths found")
        # Output path list
        available_loras = folder_paths.get_filename_list("loras")
        # Remove loras that are not in the whitelist (partial match)
        # Process whitelist and blacklist regex
        if lora_whitelist_regex != "":
            whitelist_patterns = [re.compile(pattern.strip()) for pattern in lora_whitelist_regex.split('\n') if pattern.strip()]
            available_loras = [lora for lora in available_loras if any(pattern.match(lora) for pattern in whitelist_patterns)]
        if lora_blacklist_regex != "":
            blacklist_patterns = [re.compile(pattern.strip()) for pattern in lora_blacklist_regex.split('\n') if pattern.strip()]
            available_loras = [lora for lora in available_loras if not any(pattern.match(lora) for pattern in blacklist_patterns)]
        # Convert lora list to a json. For example: 
        # {
        #   "character":[
        #       {
        #           name:"JenniferLopez-V1.0",
        #       },
        #       { 
        #           name:"Alita-V1.0",
        #           description:"Female cyborg with big eyes and a lot of attitude. Movie character from Alita: Battle Angel."
        #       }
        #   ],
        #   "effect":[
        #       {
        #           name:"GlowingEyes-V1.0",
        #           description:"A lora that adds glowing eyes to a character."
        #       }
        #   ]
        # }
        # We will extract the description from a "description.txt" file in the same folder as the lora (if exists)
        converted_loras = {}
        for lora in available_loras:
            lora_split = lora.split("\\")
            # If there is no sub category, use "general"
            if len(lora_split) == 1:
                category = "General"
            else:
                category = lora_split[0]
            if category not in converted_loras:
                converted_loras[category] = []
            # Name is always the last element
            lora_name = lora_split[-1]
            # Check if description file exists
            description_file = os.path.join(lora_paths[0], lora.replace(".safetensors", ".txt"))
            description = None
            if os.path.isfile(description_file):
                with open(description_file, "r") as infile:
                    description = infile.read()
            if description is None or description.strip() == "":
                converted_loras[category].append({"name": lora_name.replace(".safetensors", "")})
            else:
                converted_loras[category].append({"name": lora_name.replace(".safetensors", ""), "description": description})
        available_loras = json.dumps(converted_loras)

        # Get prompt from ChatGPT for GPT-4
        client = Anthropic(
            # This is the default and can be omitted
            api_key=api_key,
        )

        # Load keywoar list from file if exists
        keywoard_list = ""
        try:
            with open("keywoard_list.txt", "r") as infile:
                keywoard_list = infile.read()
        except:
            print("keywoard_list.txt not found")

        # Get new prompts from ChatGPT
        gpt_assistant_prompt = "You are a Stable Diffusion prompt generator. The prompt should be detailed. Use Keyword sentences. The scene should be interesting and engaging. The prompt should be creative and unique. Start directly with the prompt and dont use a caption. If you have to create multiple prompts, add an empty line between them. Don't number them or title them! \nYou can also use loras in the following format: <loraname:strength>. Loras are plugins to create specific effects or characters. But only use available loras! \nYou can strengthen parts of the prompt using bradcks. For example: a white (cute cat) is playing with a (red ball:1.2)."
        gpt_assistant_prompt += "\n---------------------------------"
        gpt_assistant_prompt += """\nExample Output:
            Bosstyle,Silhouette of a Woman fighting a giant DARK mononoke Monster Boss resembling a ethereal fox with seven tails: Capture the essence of nostalgia and color in this vintage photograph. Dominating the scene is a meticulously detailed,translucent dark-red Blood filled firefox,adorned with vibrant patterns,blending seamlessly with its surroundings. Its long tail,composed of intricately woven ribbons in a spectrum of colors,trails behind,adding to the spectacle of the moment. Amidst the quietude of the night,its presence exudes a sense of ancestral strength and resilience. Behind it,the low-toned hues of the Milky Way cast a mesmerizing backdrop,a cosmic tapestry weaving tales of both past and future. In this moment,the convergence of tradition and technology,of ancient wisdom and industrial progress,hangs palpably in the air.,atmospheric haze,Film grain,cinematic film still,shallow depth of field,highly detailed,high budget,cinemascope,moody,epic,OverallDetail,2000s vintage RAW photo,photorealistic,candid camera,color graded cinematic,eye catchlights,atmospheric lighting,imperfections,natural,shallow dof

            strong and powerful dungeons and dragons epic movie poster, barbarian woman with cape charging into battle violent roar riding a vicious ice [wolf|tiger] beast leather and fur boots warriors and red banners (windy dust debris storm:1.1) volumetric lighting fog depth mist pass z pass great stone castle very bright morning sunlight from side

            enchanting scene,a steaming cup of coffee,the liquid within the cup forms a swirling cosmic galaxy,The coffee surface is a mesmerizing mix of deep purples and blues and pinks dotted with stars and nebulae,fit the vastness of space inside the cup,The cup rests on a saucer that reflects the celestial patterns,surrounded by tiny star-shaped cookies and cosmic-themed decorations,The atmosphere is magical and dreamy,<EpicF4nta5yXL:0.7>

            An anthropomorphic turtle with a large decorated shell,walking through a dense misty forest,the turtle is wearing a long flowing robe adorned with intricate patterns and holding a wooden staff,The shell is filled with various items including flowers and pouches,wandering sage,towering trees background,moss-covered rocks,soft ethereal light filtering through the forest canopy,a magical and serene atmosphere,<detailed_notrigger:0.6>,<zavy-cntrst-sdxl:0.7>

            Assasin,A closeup of fantastical image of a assasin false glory,darkness of night fall,Their filled with power and determination,clad in flowing,flowing scarf,wielding an scarf ornate,scarf ornate,as they wield the scarf n a fluid,sword carried behind the body,squatting on the roof of the clock tower building like an assassin,sitting squatting pose,hands touch the ground,squatting all four,action pose,clock tower,from behind view,cyberpunk city,neon city,skindentation,glowing,shiny,scifi,neon lights,intricate artistic color,cinematic light,radiant,vibrant,perfect symmetry,Grayscale,Dramatic spotlight,highly color focused,dynamic motion,very dark

            A captivating paranormal photograph,(oddly tilted fisheyelens) that showcases intense dynamic composition of a (partiallyvaporous terrifying vampire that is part of the shadows) (emerging in a dark bedroom at night),(floating above) (an innocent,adorable,victim),((high angle photograph,highly dramatic cinematic atmosphere)),((the very detailed background is imposing,with angular intimidating,artdeco style architectural designs)),the (terrifying shadow vampire looming over its adorable victim is part of dark and smokey shadows at nighttime in the (girly bedroom of the victim)),cinematic spiritual horror,dynamic,((double exposures)),very detailed background,ominous ethereal background,epiCRealism,epiCNegative,Photographer Martin Kimbell Style,ral-dissolve

            fanciful,translucent vintage camera with metallic copper and silver details,filled with an array of soft pastel flowers including hues of pink and white and green inside it,floating golden sparkles,magical touch to the composition,<sdxl_glass:0.7>,<tranzp:0.7>,<add-detail-xl:1>

            linquivera,Ink illustration,(anime:0.5),brown tones,aged black red paper,inkpunk,moonlight,surreal,a ghost standing on a boat in swampy wetlands,(at a distance),Will-o'-the-wisp,moonlit,lonely,solitude,windy,tall trees,willows,willowy,OverallDetail,extremely detailed,UHD,(long exposure ,dystopian but extremely beautiful:1.4),<Cute_Collectible:1>,<XL_boss_battle:0.6> <add-detail-xl:1>,<MJ52:1>
        """
        gpt_user_prompt = "\nDetails for the prompt: " + details
        gpt_user_prompt += "\n---------------------------------"
        gpt_user_prompt += "\nQuantity of prompts: " + str(batch_quantity)
        gpt_user_prompt += "\n---------------------------------"
        gpt_user_prompt += "\nAvailable loras: " + available_loras

        if keywoard_list != "":
            gpt_user_prompt += "\n---------------------------------"
            gpt_user_prompt += "\DONT use any of the following keywoards or similiar: " + keywoard_list

        model = "claude-3-5-sonnet-20240620" if gpt == "claude-3.5-sonnet" else "claude-3-opus-20240229"

        # Output request to console
        print("Request prompt from AI:")
        print("API key: " + api_key)
        print("GPT: " + gpt)
        print("System prompt: " + gpt_assistant_prompt)
        print("User prompt: " + gpt_user_prompt)

        response = client.messages.create(
            max_tokens=1024,
            system=re.sub(r"^\s+", "", gpt_assistant_prompt, flags=re.MULTILINE),  # Remove leading whitespace
            messages=[{"role": "user", "content": re.sub(r"^\s+", "", gpt_user_prompt, flags=re.MULTILINE)}],
            model=model,
        )

        # Check for errors or empty responses
        if len(response.content) == 0 or response.content[0].text == "":
            print("No prompts found")
            return cls.get_prompt(api_key, gpt, details, append_prefix, append_suffix, batch_quantity, images_per_batch, lora_whitelist_regex, lora_blacklist_regex)
        else:
            print("Response prompt: " + response.content[0].text)

        prompt = response.content[0].text

        # Get a new overview of the prompts (Keywoards like "castle", "underwater sear world", ...)
        gpt_assistant_prompt = """
            You will receive an overview of prompts. Create a short keywoard list of prompts I can use, to prevent furhter generations of the same or similiar prompts later. Dont be too detailed.
            Start directly with the keywoards! Seperate the keywoards with a comma. If already a keywoard list is provided, attach it to the end of the list.
        """
        if keywoard_list != "":
            gpt_user_prompt = "Prompts:\n\n" + response.content[0].text
            gpt_user_prompt += "\n---------------------------------"
            gpt_user_prompt += "Keywoard list: " + keywoard_list
        else:
            gpt_user_prompt = "Prompts:\n\n" + response.content[0].text

        responseKeywoarList = client.messages.create(
            max_tokens=1024,
            system=re.sub(r"^\s+", "", gpt_assistant_prompt, flags=re.MULTILINE),
            messages=[{"role": "user", "content": re.sub(r"^\s+", "", gpt_user_prompt, flags=re.MULTILINE)}],
            model=model,
        )

        # Save prompt and keywoard list to files if response is not empty
        if len(responseKeywoarList.content) > 0 and responseKeywoarList.content[0].text != "":

            print("Response keywoard list: " + responseKeywoarList.content[0].text)

            # Remove numbering from prompts - Via regex
            prompt = re.sub(r"^\d+\.", "", prompt, flags=re.MULTILINE)

            # Combines lines where none empty lines are between them
            prompt = re.sub(r"([^\n])\n([^\n])", r"\1 \2", prompt)

            # Fix invalid loras -> For example <Character:Gamora-V1.0.safetensors> to <lora:Gamora-V1.0> or <NSFW:NoBra-V1.0.safetensors> to <lora:NoBra-V1.0>
            # Remove the part before the colon and probably the .safetenors part
            prompt = re.sub(r"<(?:[\w\-. \\]+?)\:((?:[\w-]+?)(?:\-V\d+\.\d+))?(?:\.safetensors?)?>", r"<lora:\1>", prompt)

            # Change <loraname: to <lora:
            prompt = re.sub(r"<loraname:", "<lora:", prompt)

            # Remove lines with only whitespace or "-----" or similar but not empty lines
            prompt = re.sub(r"^\s*[\-]+\s*$", "", prompt, flags=re.MULTILINE)

            # Write prompt to prompt_from_ai.txt
            with open("prompt_from_ai.txt", "w") as outfile:
                outfile.write("index:0\nimage:0\n\n" + prompt)
            # Write keywoard list to keywoard_list.txt
            with open("keywoard_list.txt", "w") as outfile:
                outfile.write(responseKeywoarList.content[0].text)
        else:
            print("Failed to get keywoard list")

        # Rerun get_prompt to get the first prompt or generate new if something went wrong
        return cls.get_prompt(api_key, gpt, details, append_prefix, append_suffix, batch_quantity, images_per_batch, lora_whitelist_regex, lora_blacklist_regex)

    @classmethod
    def IS_CHANGED(self, **kwargs):
        return float("nan")

class LoraLoaderFromPrompt:
    def __init__(self):
        self.loaded_loras = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP", ),
                "prompt": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    FUNCTION = "load_loras"
    CATEGORY = "t4ggno/loaders"

    def load_loras(self, model, clip, prompt):
        print("=============================")
        print("== Load Loras from Prompt ==")

        avaialbe_loras = folder_paths.get_filename_list("loras")
        # print("Avaialbe loras: " + str(avaialbe_loras))
        # Extract lora from _prompt: <name:model_stremgth:clip_strength> or <name:model_stremgth> or <name::clip_strength> or <name> or <name::>
        all_loras = re.findall(r"<(?:lora:?)?([\w\-. \\]+?)(?:\:(-?[0-9](?:\.[0-9]*)?)|(?:\:(-?[0-9]+(?:\.[0-9]*)?|))(?:\:(-?[0-9]+(?:\.[0-9]*)?|)))?>", prompt)
        # Remove loras from prompt
        prompt = re.sub(r"<(?:lora:?)?([\w\-. \\]+?)(?:\:(-?[0-9](?:\.[0-9]*)?)|(?:\:(-?[0-9]+(?:\.[0-9]*)?|))(?:\:(-?[0-9]+(?:\.[0-9]*)?|)))?>", "", prompt)
        # Map lora to object with name, model_strength and clip_strength
        all_loras = list(map(lambda x: {"name": x[0], "model_strength": x[1], "clip_strength": x[2]}, all_loras))

        # Convert model_strength and clip_strength to float
        # If only model_strength is set, use model_strength for clip_strength
        # If no model_strength or clip_strength is set, use 1.0
        for lora in all_loras:
            lora["model_strength"] = float(lora["model_strength"]) if lora["model_strength"] != "" else 1.0
            lora["clip_strength"] = float(lora["clip_strength"]) if lora["clip_strength"] != "" else float(lora["model_strength"]) if lora["model_strength"] != "" else 1.0

        # Go trough every lora and fix name. For exmaple: Happy -> Emotions\\Happy-V1.0.safetensors or Happy-V1.0 -> Emotions\\Happy-V1.0.safetensors
        for lora in all_loras:
            current_name = lora["name"]
            found_exact = False
            for avaialbe_lora in avaialbe_loras:
                if avaialbe_lora == current_name:
                    found_exact = True
                    break
            if not found_exact:
                possible_loras = []
                for avaialbe_lora in avaialbe_loras:
                    # Check if avaialbe_lora contains current_name
                    if avaialbe_lora.find(current_name) != -1:
                        print("Found lora as: " + avaialbe_lora)
                        possible_loras.append(avaialbe_lora)
                        break
                if len(possible_loras) > 0:
                    # Pick one random lora
                    lora["name"] = possible_loras[numpy.random.randint(0, len(possible_loras))]

        # Go thorugh all loras and warn if lora not available
        for lora in all_loras:
            if lora["name"] not in avaialbe_loras:
                print("Lora not available: " + lora["name"])

        # Go trough every lora and check for duplicates. If duplicate found, calculate average of model_strength and clip_strength
        # If model_strength or clip_strength is not set, use 1.0
        all_loras_tmp = []
        for lora in all_loras:
            found = False
            for lora_tmp in all_loras_tmp:
                if lora_tmp["name"] == lora["name"]:
                    found = True
                    lora_tmp["model_strength"] = (float(lora_tmp["model_strength"]) + float(lora["model_strength"]))
                    lora_tmp["clip_strength"] = (float(lora_tmp["clip_strength"]) + float(lora["clip_strength"]))
                    lora_tmp["quantity"] += 1
                    break
            if not found:
                lora["quantity"] = 1
                all_loras_tmp.append(lora)
        all_loras = all_loras_tmp
        # Calculate average
        for lora in all_loras:
            lora["model_strength"] = lora["model_strength"] / lora["quantity"]
            lora["clip_strength"] = lora["clip_strength"] / lora["quantity"]
            del lora["quantity"]

        # Filter out loras that are not available
        all_loras = list(filter(lambda x: x["name"] in avaialbe_loras, all_loras))
        if len(all_loras) == 0:
            return (model, clip, prompt)
        else:
            for lora in all_loras:
                lora_path = folder_paths.get_full_path("loras", lora["name"])
                loaded_lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                # Strenght
                model_strength = float(lora["model_strength"]) if lora["model_strength"] != "" else 1.0
                clip_strength = float(lora["clip_strength"]) if lora["clip_strength"] != "" else float(lora["model_strength"]) if lora["model_strength"] != "" else 1.0
                model, clip = comfy.sd.load_lora_for_models(model, clip, loaded_lora, model_strength, clip_strength)  # Overwrite model and clip with new model and clip
                print("Loaded lora: " + lora_path + " with model strength: " + str(model_strength) + " and clip strength: " + str(clip_strength))

                # Load metadata from prompt and check if trigger word exists in prompt. The metadata is in file as string as JSON.
                try:
                    # Extract metadata from file. Source is bytes, but need it as ANSI string.
                    f = io.open(lora_path, mode="r", encoding="ansi")
                    # Extract metadata as string. We should go trough character by character in first line until we find the first "{" and then go trough character by character until we find the last "}".
                    metadataAsString = ""
                    startCurlyBraceCounter = 0
                    endCurlyBraceCounter = 0
                    stop = False
                    for line in f:
                        if stop:
                            break
                        for character in line:
                            if character == "{":
                                startCurlyBraceCounter += 1
                            elif character == "}":
                                endCurlyBraceCounter += 1
                            if startCurlyBraceCounter > 0:
                                metadataAsString += character
                            if startCurlyBraceCounter > 0 and startCurlyBraceCounter == endCurlyBraceCounter:
                                stop = True
                                break
                    if metadataAsString is not None and metadataAsString != "":
                        # Convert string to json
                        extractedMetadata = json.loads(metadataAsString)
                        # Check if metadata has ss_dataset_dirs
                        if "__metadata__" in extractedMetadata and "ss_dataset_dirs" in extractedMetadata["__metadata__"]:
                            # Get ss_dataset_dirs
                            ssDatasetDirs = extractedMetadata["__metadata__"]["ss_dataset_dirs"]
                            # Convert ss_dataset_dirs to json
                            ssDatasetDirs = json.loads(ssDatasetDirs)
                            ssDatasetDirsKeys = ssDatasetDirs.keys()
                            # Filter trigger word. Format should be "NUMBER_WORD PHRASE"
                            ssDatasetDirsKeys = list(filter(lambda x: re.match(r"^\d+_.+$", x) != None, ssDatasetDirsKeys))
                            if len(ssDatasetDirsKeys) > 0:
                                # Check if at least one trigger word is in prompt
                                triggerWordFound = False
                                for ssDatasetDirsKey in ssDatasetDirsKeys:
                                    triggerWord = ssDatasetDirsKey.split("_")[1]
                                    if triggerWord in prompt:
                                        triggerWordFound = True
                                        break
                                if not triggerWordFound:
                                    print("Trigger word not found in prompt. Available trigger words: " + str(ssDatasetDirsKeys))
                                    # Add random trigger word to prompt
                                    randomTriggerWord = ssDatasetDirsKeys[numpy.random.randint(0, len(ssDatasetDirsKeys))]
                                    prompt += " " + randomTriggerWord.split("_")[1]
                                    print("Added trigger word: '" + randomTriggerWord.split("_")[1] + "' to prompt")
                except:
                    print("Error while extracting metadata from file: " + lora_path)

            print("Text prompt after loading loras: " + prompt)

            return (model, clip, prompt)


class CurrentDateTime:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "hidden": {
                "control_after_generate": (["fixed", "random", "increment"], {"default": "increment"}),
                "value": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "get_current_date_time"
    CATEGORY = "t4ggno/utils"
    OUTPUT_NODE = False
    DESCRIPTION = """
Get the current date and time as a string in the format "YYYY-MM-DD_HH-MM-SS". The date and time are obtained using the Python `datetime` module. 
"""

    def get_current_date_time(self, **kwargs):
        print("=============================")
        print("== Getting current date time")

        # Get current date time using python
        now = datetime.now()
        # Alert time for debugging
        print("Current date and time: ", now.strftime("%Y-%m-%d %H:%M:%S"))
        # Convert to "YYYY-MM-DD_HH-MM-SS"
        current_date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        return (current_date_time,)

    @classmethod
    def IS_CHANGED(self, **kwargs):
        return float("nan")

class CheckpointLoaderByName:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "type": (["Detect (Manual)","Detect (Random)"], {"default": "Detect (Manual)"}),
                "checkpoint_name": ("STRING", {"default": ""}),
                "fallback": (comfy_paths.get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "NAME_STRING")
    FUNCTION = "load_checkpoint"
    CATEGORY = "t4ggno/utils"
    OUTPUT_NODE = False

    def load_checkpoint(cls, type, checkpoint_name, fallback):
        print("=============================")
        print("== Load Checkpoint by Name ==")
        print("Type: " + type)

        avaiable_checkpoints:list[str] = comfy_paths.get_filename_list("checkpoints")

        if checkpoint_name != "":
            # checkpoint is for example "BetterThanWords-V3.0.safetensors"
            # Step 1: Try to find exact match: BetterThanWords-V3.0.safetensors
            # Step 2: Try to find checkpoint without extension: BetterThanWords-V3.0
            # Step 3: Try to find checkpoint without version and extension: BetterThanWords

            # Step 1
            checkpoint_path = comfy_paths.get_full_path("checkpoints", checkpoint_name)
            if checkpoint_path is not None and os.path.isfile(checkpoint_path):
                print("Load checkpoint [Exact Match]: " + checkpoint_name + " | " + os.path.basename(checkpoint_path))
                out = comfy.sd.load_checkpoint_guess_config(checkpoint_path, output_vae=True, output_clip=True, embedding_directory=comfy_paths.get_folder_paths("embeddings"))
                return (out[0], out[1], out[2], os.path.splitext(checkpoint_name)[0])
            # Step 2
            checkpoint_name_without_extension = os.path.splitext(checkpoint_name)[0]
            avaiable_checkpoint_filtered = []
            for avaiable_checkpoint in avaiable_checkpoints:
                if avaiable_checkpoint.find(checkpoint_name_without_extension) != -1:
                    avaiable_checkpoint_filtered = avaiable_checkpoint_filtered + [avaiable_checkpoint]
            if len(avaiable_checkpoint_filtered) > 0:
                checkpoint_path = comfy_paths.get_full_path("checkpoints", avaiable_checkpoint_filtered[0])
                print("Load checkpoint [Without Extension]: " + checkpoint_name_without_extension + " | " + os.path.basename(checkpoint_path))
                out = comfy.sd.load_checkpoint_guess_config(checkpoint_path, output_vae=True, output_clip=True, embedding_directory=comfy_paths.get_folder_paths("embeddings"))
                return (out[0], out[1], out[2], os.path.splitext(avaiable_checkpoint_filtered[0])[0])
            # Step 3
            checkpoint_name_without_extension_and_version = re.sub(r"-V\d+\.\d+", "", checkpoint_name_without_extension)
            avaiable_checkpoint_filtered = []
            for avaiable_checkpoint in avaiable_checkpoints:
                if avaiable_checkpoint.find(checkpoint_name_without_extension_and_version) != -1:
                    avaiable_checkpoint_filtered = avaiable_checkpoint_filtered + [avaiable_checkpoint]
            if len(avaiable_checkpoint_filtered) > 0:
                checkpoint_path = comfy_paths.get_full_path("checkpoints", avaiable_checkpoint_filtered[0])
                print("Load checkpoint [Without Version and Extension]: " + checkpoint_name_without_extension_and_version + " | " + os.path.basename(checkpoint_path))
                out = comfy.sd.load_checkpoint_guess_config(checkpoint_path, output_vae=True, output_clip=True, embedding_directory=comfy_paths.get_folder_paths("embeddings"))
                return (out[0], out[1], out[2], os.path.splitext(avaiable_checkpoint_filtered[0])[0])
        else:
            print("Missing checkpoint name")
        
        # Fallback
        if type == "Detect (Random)":
            checkpoint_path = comfy_paths.get_full_path("checkpoints", numpy.random.choice(avaiable_checkpoints))
            print("Load checkpoint [Random]: " + os.path.basename(checkpoint_path))
            out = comfy.sd.load_checkpoint_guess_config(checkpoint_path, output_vae=True, output_clip=True, embedding_directory=comfy_paths.get_folder_paths("embeddings"))
            return (out[0], out[1], out[2], os.path.splitext(os.path.basename(checkpoint_path))[0])
        else:
            checkpoint_path = comfy_paths.get_full_path("checkpoints", fallback)
            out = comfy.sd.load_checkpoint_guess_config(checkpoint_path, output_vae=True, output_clip=True, embedding_directory=comfy_paths.get_folder_paths("embeddings"))
            print("Load checkpoint [Fallback]: " + os.path.basename(checkpoint_path))
            return (out[0], out[1], out[2], os.path.splitext(os.path.basename(checkpoint_path))[0])

    @classmethod
    def IS_CHANGED(self, **kwargs):
        return float("nan")

class RandomCheckpointLoader:
    def __init__(self):
        self.checkpoint_usage = defaultdict(int)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "whitelist_regex": ("STRING", {"default": "", "multiline": True}),
                "blacklist_regex": ("STRING", {"default": "", "multiline": True}),
                "usage_limit": ("INT", {"default": 1, "min": 1, "max": 100}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "checkpoint_name")
    FUNCTION = "load_random_checkpoint"
    CATEGORY = "t4ggno/loaders"
    DESCRIPTION = """
Load a random checkpoint from the available checkpoints. The checkpoint is selected randomly from the list of available checkpoints. 
The checkpoint is loaded only if it has not reached the usage limit. The usage limit is the maximum number of times a checkpoint can be loaded before it is reset. 
The checkpoint is selected based on the whitelist and blacklist regex patterns. The whitelist regex is used to filter the checkpoints that match the pattern. 
The blacklist regex is used to exclude the checkpoints that match the pattern. 
If no checkpoints are available after applying the whitelist and blacklist regex, an error is raised. 
If all checkpoints have reached the usage limit, the usage count is reset, and the process is repeated until an eligible checkpoint is found.
"""

    def load_random_checkpoint(self, whitelist_regex, blacklist_regex, usage_limit):
        print("=============================")
        print("== Loading random checkpoint")

        # Get all available checkpoints
        all_checkpoints = comfy_paths.get_filename_list("checkpoints")
        print(f"Total checkpoints found: {len(all_checkpoints)}")

        # Remove None values and empty strings
        all_checkpoints = [cp for cp in all_checkpoints if cp]
        print(f"Checkpoints after removing None and empty values: {len(all_checkpoints)}")

        # Process whitelist and blacklist regex
        whitelist_patterns = [re.compile(pattern.strip()) for pattern in whitelist_regex.split('\n') if pattern.strip()]
        blacklist_patterns = [re.compile(pattern.strip()) for pattern in blacklist_regex.split('\n') if pattern.strip()]

        # Filter checkpoints using regex
        if whitelist_patterns:
            checkpoints = [cp for cp in all_checkpoints if any(pattern.search(cp) for pattern in whitelist_patterns)]
            print(f"Checkpoints after whitelist: {len(checkpoints)}")
        else:
            checkpoints = all_checkpoints

        checkpoints = [cp for cp in checkpoints if not any(pattern.search(cp) for pattern in blacklist_patterns)]
        print(f"Checkpoints after blacklist: {len(checkpoints)}")

        if not checkpoints:
            raise ValueError("No checkpoints available after applying whitelist and blacklist regex")

        # Find an eligible checkpoint
        eligible_checkpoints = [cp for cp in checkpoints if self.checkpoint_usage[cp] < usage_limit]
        print(f"Eligible checkpoints: {len(eligible_checkpoints)}")

        if not eligible_checkpoints:
            # Reset usage count if all checkpoints have reached the limit
            self.checkpoint_usage = defaultdict(int)
            eligible_checkpoints = checkpoints

        # Shuffle the list of eligible checkpoints
        numpy.random.shuffle(eligible_checkpoints)

        for random_checkpoint in eligible_checkpoints:
            print(f"Attempting to load checkpoint: {random_checkpoint}")

            # Increment usage count
            self.checkpoint_usage[random_checkpoint] += 1

            # Load checkpoint
            checkpoint_path = comfy_paths.get_full_path("checkpoints", random_checkpoint)
            print(f"Loading checkpoint from: {checkpoint_path}")

            try:
                result = comfy.sd.load_checkpoint_guess_config(checkpoint_path, output_vae=True, output_clip=True, embedding_directory=comfy_paths.get_folder_paths("embeddings"))

                if len(result) >= 3:
                    model, clip, vae, *_ = result
                else:
                    raise ValueError(f"Unexpected number of return values from load_checkpoint_guess_config: {len(result)}")

                if model is None or clip is None or vae is None:
                    raise ValueError(f"Failed to load checkpoint: {random_checkpoint}")

                print(f"Successfully loaded checkpoint: {random_checkpoint}")
                return (model, clip, vae, random_checkpoint)
            except Exception as e:
                print(f"Error loading checkpoint {random_checkpoint}: {e}")
                print("Attempting to load next checkpoint...")

        raise ValueError("Unable to load any checkpoints. All attempts failed.")

    @classmethod
    @classmethod
    def IS_CHANGED(self, **kwargs):
        return float("nan")


class ColorMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
                "method": ([
                    'mkl',
                    'hm',
                    'reinhard',
                    'mvgd',
                    'hm-mvgd-hm',
                    'hm-mkl-hm',
                ], {"default": 'mkl'}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "colormatch"

    CATEGORY = "t4ggno/utils"

    DESCRIPTION = """
MODIFIED VERSION OF: KJNodes/ColorMatch

color-matcher enables color transfer across images which comes in handy for automatic  
color-grading of photographs, paintings and film sequences as well as light-field  
and stopmotion corrections.  

The methods behind the mappings are based on the approach from Reinhard et al.,  
the Monge-Kantorovich Linearization (MKL) as proposed by Pitie et al. and our analytical solution  
to a Multi-Variate Gaussian Distribution (MVGD) transfer in conjunction with classical histogram   
matching. As shown below our HM-MVGD-HM compound outperforms existing methods.   
https://github.com/hahnec/color-matcher/

"""

    def colormatch(self, image_ref, image_target, method):
        try:
            from color_matcher import ColorMatcher
        except:
            raise Exception("Can't import color-matcher, did you install requirements.txt? Manual install: pip install color-matcher")

        cm = ColorMatcher()
        image_ref = image_ref.cpu()
        image_target = image_target.cpu()
        batch_size = image_target.size(0)
        out = []
        images_target = image_target.squeeze()
        images_ref = image_ref.squeeze()

        image_ref_np = images_ref.numpy()
        images_target_np = images_target.numpy()

        if image_ref.size(0) > 1 and image_ref.size(0) != batch_size:
            raise ValueError("ColorMatch: Use either single reference image or a matching batch of reference images.")

        for i in range(batch_size):
            image_target_np = images_target_np if batch_size == 1 else images_target[i].numpy()
            image_ref_np_i = image_ref_np if image_ref.size(0) == 1 else images_ref[i].numpy()
            try:
                image_result = cm.transfer(src=image_target_np, ref=image_ref_np_i, method=method)
                out.append(torch.from_numpy(image_result))
            except BaseException as e:
                print(f"Error occurred during transfer for image {i}: {e}")
                # Append the original image instead of the processed one
                out.append(torch.from_numpy(image_target_np))

        if not out:
            print("ColorMatch: No images were successfully processed. Returning original target images.")
            return (image_target,)

        out = torch.stack(out, dim=0).to(torch.float32)
        out.clamp_(0, 1)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "Base64Decode": Base64Decode,
    "LayoutSwitch": LayoutSwitch,
    "PredefinedResolutions": PredefinedResolutions,
    "ResolutionSwitch": ResolutionSwitch,
    "LoraLoaderFromPrompt": LoraLoaderFromPrompt,
    "PromptFromAI": PromptFromAIOpenAI,
    "PromptFromAIOpenAI": PromptFromAIOpenAI,
    "PromptFromAIAnthropic": PromptFromAIAnthropic,
    "CurrentDateTime": CurrentDateTime,
    "TextSwitch": TextSwitch,
    "TextReplacer": TextReplacer,
    "ImageSave": ImageSave,
    "AutoLoadImageForUpscaler": AutoLoadImageForUpscaler,
    "RandomCheckpointLoader": RandomCheckpointLoader,
    "ColorMatch": ColorMatch,
    "CheckpointLoaderByName": CheckpointLoaderByName,
    "ImageMetadataExtractor": ImageMetadataExtractor,
    "LoadImageWithMetadata": LoadImageWithMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Base64Decode": "Load Image (Base64)",
    "LayoutSwitch": "Switch Layout",
    "PredefinedResolutions": "Predefined Resolutions",
    "ResolutionSwitch": "Switch Resolution",
    "LoraLoaderFromPrompt": "Load Lora From Prompt",
    "PromptFromAI": "Prompt From AI (Deprecated - OpenAI)",
    "PromptFromAIOpenAI": "Prompt From AI (OpenAI)",
    "PromptFromAIAnthropic": "Prompt From AI (Anthropic)",
    "CurrentDateTime": "Current Date Time",
    "TextSwitch": "Switch Text",
    "TextReplacer": "Replace Text",
    "ImageSave": "Save Image",
    "AutoLoadImageForUpscaler": "Auto Load Image For Upscaler",
    "RandomCheckpointLoader": "Random Checkpoint Loader",
    "ColorMatch": "Color Match",
    "CheckpointLoaderByName": "Checkpoint Loader By Name",
    "ImageMetadataExtractor": "Image Metadata Extractor",
    "LoadImageWithMetadata": "Load Image With Metadata",
}
