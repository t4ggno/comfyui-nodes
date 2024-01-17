# ==============================================================================
# Ignore comfy.utils functions -> There are none
# ==============================================================================

import io
import base64
from PIL import Image
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
        if layout == "Landscape":
            if size1 > size2:
                return (size1, size2)
            else:
                return (size2, size1)
        elif layout == "Portrait":
            if size1 > size2:
                return (size2, size1)
            else:
                return (size1, size2)
        elif layout == "Square":
            size = (size1 + size2) / 2
            return (size, size)


"""
NOTE: For SDXL, it is recommended to use trained values listed below:
 - 1024 x 1024
 - 1152 x 896
 - 896  x 1152
 - 1216 x 832
 - 832  x 1216
 - 1344 x 768
 - 768  x 1344
 - 1536 x 640
 - 640  x 1536
"""
class PredefinedResolutions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "engine": (["1.5", "SDXL"], {"default": "SDXL"}),
                "layout": (["Square", "Landscape", "Landscape (Ultra Wide)", "Portrait", "Portrait (Ultra Tall)", "Random"], {"default": "Square"}),
                "at_random_enable_square": (["true", "false"], {"default": "true"}),
                "at_random_enable_landscape": (["true", "false"], {"default": "true"}),
                "at_random_enable_landscape_ultra_wide": (["true", "false"], {"default": "true"}),
                "at_random_enable_portrait": (["true", "false"], {"default": "true"}),
                "at_random_enable_portrait_ultra_tall": (["true", "false"], {"default": "true"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "scale": ("FLOAT", {"default": 1, "min": 1, "max": 100, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "render_resolution"
    CATEGORY = "t4ggno/utils"
    OUTPUT_NODE = False

    def render_resolution(self, engine, layout, at_random_enable_square, at_random_enable_landscape, at_random_enable_landscape_ultra_wide, at_random_enable_portrait, at_random_enable_portrait_ultra_tall, seed, scale):
        if layout == "Random":
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

        resolution = None
        if (engine == "1.5"):
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
        elif (engine == "SDXL"):
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
        # Scale and floor
        return (int(resolution[0] * scale), int(resolution[1] * scale))


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
        if active == "Resolution 1":
            return (width1, height1)
        elif active == "Resolution 2":
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
        if active == "Text 1":
            return (text1,)
        elif active == "Text 2":
            return (text2,)


class AutoLoadImageForUpscaler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": ""}),
                "max_width": ("INT", {"default": 8192, "min": 1, "max": 100000, "step": 1}),
                "max_height": ("INT", {"default": 8192, "min": 1, "max": 100000, "step": 1}),
                "scale": ("FLOAT", {"default": 2, "min": 1, "max": 100, "step": 1}),
                "max_scale": ("FLOAT", {"default": 100000, "min": 1, "max": 100000, "step": 1}),
                "fallback_positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "fallback_negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("image", "filename_without_scale", "current_scale", "positive_prompt", "negative_prompt", "next_scale")
    FUNCTION = "load_image"
    CATEGORY = "t4ggno/image"
    OUTPUT_NODE = False

    def get_prompt_data_from_image(self, image, positive_prompt, negative_prompt):
        return_positive_prompt = positive_prompt
        return_negative_prompt = negative_prompt
        return_additional_prompt = ""
        # Get metadata
        metadata = image.info
        # Convert metadata to JSON
        metadata_json = json.dumps(metadata)
        # Check if metadata contains PositiveText, NegativeText and AdditionalText
        if "PositiveText" in metadata_json:
            # Get PositiveText
            positive_text = metadata["PositiveText"]
            # Check if PositiveText is not empty
            if positive_text.strip() != "":
                # Add to positive_prompt
                return_positive_prompt += positive_text
        if "NegativeText" in metadata_json:
            # Get NegativeText
            negative_text = metadata["NegativeText"]
            # Check if NegativeText is not empty
            if negative_text.strip() != "":
                # Add to negative_prompt
                return_negative_prompt += negative_text
        if "AdditionalText" in metadata_json:
            # Get AdditionalText
            additional_text = metadata["AdditionalText"]
            # Check if AdditionalText is not empty
            if additional_text.strip() != "":
                # Add to additional_prompt
                return_additional_prompt += additional_text
        return (return_positive_prompt, return_negative_prompt, return_additional_prompt)

    def load_image(self, folder, max_width, max_height, scale, max_scale, fallback_positive_prompt, fallback_negative_prompt, seed):
        while True:
            # Load files from folder
            files = os.listdir(folder)
            # Filter only for png images
            files = [file for file in files if file.endswith(".png")]
            # Find image with lowest avaiable scale -> _1.0x, _2.0x, _3.5x, ...
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
                positive_prompt, negative_prompt, additional_prompt = self.get_prompt_data_from_image(loaded_image, fallback_positive_prompt, fallback_negative_prompt)
                print("====================================")
                print("Image: " + lower_scale["file"])
                print("Current scale: " + str(lower_scale["scale"]) + (lower_scale["scale"] == 1 and " (Base)" or ""))
                print("Next scale: " + str(use_scale))
                print("Positive prompt: " + positive_prompt)
                print("Negative prompt: " + negative_prompt)
                print("====================================")
                # Return
                return (torch.from_numpy(numpy.array(loaded_image).astype(numpy.float32) / 255.0).unsqueeze(0), lower_scale["filename"], image["scale"], positive_prompt, negative_prompt, use_scale)

            # Wait 10 seconds and try again
            time.sleep(10)


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
                "positive_text": ("STRING", {"default": "", }),
                "negative_text": ("STRING", {"default": "", }),
                "additional_text": ("STRING", {"default": "", }),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "t4ggno/image"

    def save_images(self, images, filename="", output_path="", extension='png', quality=100, lossless_webp="false", show_previews="true", positive_text="", negative_text="", additional_text=""):

        lossless_webp = (lossless_webp == "true")

        # Setup output path
        now = datetime.now()
        if output_path.strip() == '':
            output_path = os.path.join(self.output_dir, now.strftime("%Y-%m-%d"))

        # Check output destination
        if not os.path.exists(output_path.strip()):
            print('The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
            os.makedirs(output_path, exist_ok=True)

        # Set Extension
        if extension not in ALLOWED_EXT:
            print(f"The extension `{extension}` is not valid. The valid formats are: {', '.join(sorted(ALLOWED_EXT))}")
            extension = "png"

        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(numpy.clip(i, 0, 255).astype(numpy.uint8))

            # Crate JSON with positive, negative and additional text
            metadata = PngInfo()
            metadata.add_text("PositiveText", positive_text)
            metadata.add_text("NegativeText", negative_text)
            metadata.add_text("AdditionalText", additional_text)

            # If filename is empty, use "yyyy-mm-dd_hh-mm-ss" as filename
            if filename.strip() == '':
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"{timestamp}.{extension}"
            else:
                filename = f"{filename}.{extension}"
            try:
                output_file = os.path.abspath(os.path.join(output_path, filename))
                print(f"Saving image to: {output_file}")
                if extension == 'png':
                    img.save(output_file, pnginfo=metadata, optimize=True)
                elif extension == 'webp':
                    img.save(output_file, quality=quality)
                elif extension == 'jpeg':
                    img.save(output_file, quality=quality, optimize=True)
                elif extension == 'tiff':
                    img.save(output_file,
                             quality=quality, optimize=True)
                elif extension == 'webp':
                    img.save(output_file, quality=quality,
                             lossless=lossless_webp, exif=metadata)

                print("Image file saved to: {output_file}")

            except OSError as e:
                print('Unable to save file to: {output_file}')
                print(e)
            except Exception as e:
                print('Unable to save file due to the to the following error:')
                print(e)

        if show_previews == 'true':
            return {"ui": {"images": results}}
        else:
            return {"ui": {"images": []}}

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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "replace_text"
    CATEGORY = "t4ggno/utils"
    OUTPUT_NODE = False

    def replace_text(self, text, seed):

        # Remove empty lines and trim
        text = "\n".join([line.strip() for line in text.split("\n") if line.strip() != ""])
        # Remve multiple "," in a row -> "a,,b" -> "a,b
        text = re.sub(r"\,+?", ",", text)
        # Remove comments - Either // or /* */, but not if /* escaped lik
        # text = re.sub(r"(?<!\\)\/\/.*", "", text)
        # text = re.sub(r"(?<!\\)\/\*(?:.|\n)*?\*\/", "", text)
        text = re.sub(r"#.*", "", text)  # Fallbck for now
        text = re.sub(r"'''(?:.|\n)*?'''", "", text)
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
                        globalWhitelistEntry["categories"] = list(
                            set(globalWhitelistEntry["categories"] + categories))
                        found = True
                        break
                if not found:
                    globalWhitelist.append(
                        {"triggerWord": triggerWord, "categories": categories})
            elif type == "Blacklist":
                # Check if triggerWord already in globalBlacklist
                found = False
                for globalBlacklistEntry in globalBlacklist:
                    if globalBlacklistEntry["triggerWord"] == triggerWord:
                        globalBlacklistEntry["categories"] = list(
                            set(globalBlacklistEntry["categories"] + categories))
                        found = True
                        break
                if not found:
                    globalBlacklist.append(
                        {"triggerWord": triggerWord, "categories": categories})
            # Remove from text
            text = text.replace(completeString, "")
        print("Global whitelist: " + str(globalWhitelist))
        print("Global blacklist: " + str(globalBlacklist))

        # Replace [PickOne:Option1,Option2,...] with random option - We need to do this before the other replacements, because they might contain [PickOne:Option1,Option2,...]
        while True:
            somethingReplaced = False

            pickOneMatches = re.findall(r"(\[PickOne(?:\:(.+?))?\])", text)
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

            # 1. Replace [TimeOfDay] (if exists) with random time of day
            timeOfDay = ["morning", "afternoon", "evening", "night", "sunset", "sunrise"]
            replaceUsingArray("TimeOfDay", timeOfDay)
            # 2. Replace [Weather] (if exists) with random weather
            weather = ["sunny", "cloudy", "rainy", "snowy"]
            replaceUsingArray("Weather", weather)
            # 3. Replace [Nudity] (if exists) with random nudity
            nudity = ["nude", "topless", "nipple", "bottomless", "naked", "nude", "bikini", "lingerie", "underwear", "swimsuit", "swimwear"]
            replaceUsingArray("Nudity", nudity)
            # 4. Replace [FacialExpression] (if exists) with random facial expression
            facialExpression = ["happy", "sad", "angry", "surprised", "disgusted", "scared", "cry", "laugh", "smile", "light smile"]
            replaceUsingArray("FacialExpression", facialExpression)
            # 5. Replace [Location] (if exists) with random location
            replaceUsingJson("Location", "locations.json")
            # 6. Replace [ClothesFemale] (if exists) with random location
            replaceUsingJson("ClothesFemale", "clothes_female.json")
            # 7. Replace [ClothesFemaleSexy] (if exists) with random location
            replaceUsingJson("ClothesFemaleSexy", "clothes_female_sexy.json")
            # 8. Replace [ClothesMale] (if exists) with random location
            replaceUsingJson("ClothesMale", "clothes_male.json")
            # 9. Replace [ClothesMaleSexy] (if exists) with random location
            replaceUsingJson("ClothesMaleSexy", "clothes_male_sexy.json")
            # 10. Replace [Scenario] (if exists) with random location
            replaceUsingJson("Scenario", "scenarios.json")
            # 11. Replace [HairColor] (if exists) with random hair color
            replaceUsingJson("HairColor", "hair_colors.json")
            # 12. Replace [Color] (if exists) with random color
            colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray", "grey", "gold", "silver", "bronze", "copper"]
            replaceUsingArray("Color", colors, 2, True)
            # 13. Replace [PickMultiple,Option1,Option2,...] (if exists) with random option
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
            # 16. Replace [Poses] (if exists) with random option
            replaceUsingJson("Poses", "poses.json")
            # 17. Replace [Random] (if exists) with random option
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
            # 18. Replace [Name] (if exists) with random name
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
            # 19. Replace [Weather] (if exists) with random weather
            weather = ["sunny", "cloudy", "rainy", "snowy"]
            replaceUsingArray("Weather", weather)
            # 20. Hair female
            replaceUsingJson("HairFemale", "hair_female.json")
            # 21. Hair male
            replaceUsingJson("HairMale", "hair_male.json")
            # 22. Sexy things
            replaceUsingJson("SexyThings", "sexy_things.json")
            # 23. Styles
            replaceUsingJson("Style", "styles.json")
            # 24. Interresting ideas
            replaceUsingJson("InterestingIdeas", "interesting_ideas.json")

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

        print("Text: " + text)
        return (text,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return True


class PromptFromAI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "token": ("STRING", {"default": ""}),
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

    RETURN_TYPES = ("STRING", "INT", "INT",)
    RETURN_NAMES = ("prompt", "width", "height",)
    FUNCTION = "get_prompt"
    CATEGORY = "t4ggno/loaders"
    OUTPUT_NODE = False

    def get_prompt(cls, token, details, append_prefix, append_suffix, batch_quantity, images_per_batch):
        """
            [EXAMPLE_START]
            product shot of ultra realistic juicy [cheeseburger:cake:0.2] against a (dark background:1.2), two tone lighting, (advertisement, octane:1.1)
            landscape

            sensual silhouette of a (woman:1.5), draped in silk, [candlelight:moonlight:0.3], surreal elements like floating bubbles with galaxies inside them, neon butterflies, crystal raindrops, (chocolate river:1.2), glowing flowers, ethereal mist, vibrant colors
            portrait

            seductive scene set in an alien landscape, bioluminescent plants and creatures, erotic statues made of glass and gold, waterfall of sparkling wine, sky filled with three moons and shooting stars, velvet roses growing on trees, sensual shadows playing on the ground
            landscape
            [EXAMPLE_END]
        """
        print("Get next prompt")
        next_prompt = cls.get_next_prompt(
            token, details, append_prefix, append_suffix, batch_quantity, images_per_batch)
        if next_prompt:
            return next_prompt

        # Get all avaiable loras
        # Folder structure will be: category\[sub_category]?\[name].safetensors
        avaiable_loras = folder_paths.get_filename_list("loras")
        # Convert lora list to a json. For example: {"character": [ "JenniferLopez-V1.0", "Alita-V1.0" ], "effect": [ "GlowingEyes-V1.0" ]}
        converted_loras = {}
        for lora in avaiable_loras:
            lora_split = lora.split("\\")
            category = lora_split[0]
            if category not in converted_loras:
                converted_loras[category] = []
            # Name is always the last element
            converted_loras[category].append(lora_split[-1])
        avaiable_loras = json.dumps(converted_loras)
        # print("Avaiable loras: " + avaiable_loras)

        # Call "http://localhost:3000/api/stable-concept" with details and batch_quantity as body
        url = 'http://localhost:3000/api/stable-concept'
        values = {
            'details': details,
            'engine': 'gpt-4.0-8k',
            'quantity': batch_quantity,
            'loras': avaiable_loras,
        }
        headers = {
            'token': token
        }
        data = urllib.parse.urlencode(values)
        full_url = url + "?" + data
        req = urllib.request.Request(full_url, headers=headers)
        timeout = 120  # 2 minutes

        with urllib.request.urlopen(req, None, timeout) as response:
            result = response.read().decode("utf-8")
            with open("prompt_from_ai.txt", "w") as outfile:
                outfile.write("index:0\nimage:0\n\n" + result)
        next_prompt = cls.get_next_prompt(
            token, details, append_prefix, append_suffix, batch_quantity, images_per_batch)
        if next_prompt:
            return next_prompt
        else:
            return (None, None, None)

    def get_next_prompt(cls, token, details, append_prefix, append_suffix, batch_quantity, images_per_batch):
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
        # Remove empty prompts (could contain whitespace and newlines)
        prompts = list(filter(lambda x: x != "", prompts))
        # If image_count is higher or equal than images_per_batch, reset image_count and increase index by 1
        if image_count >= images_per_batch:
            print(
                "Image count higher or equal than images per batch -> Reset image count and increase index by 1")
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
            outfile.write("index:" + str(index) + "\nimage:" +
                          str(image_count) + "\n\n" + "\n".join(prompts))
        # Return prompt and layout (first line prompt, second line layout) -> If layout = landscape, width = 1216, height = 832; If layout = portrait, width = 832, height = 1216; If layout = square, width = 1024, height = 1024
        prompt_splitted = prompt.split("\n")
        # Remove empty lines in prompt
        prompt_splitted = list(filter(lambda x: x != "", prompt_splitted))
        prompt = prompt_splitted[0]
        layout = prompt_splitted[1]
        # Append prefix and suffix
        prompt = append_prefix + " " + prompt + " " + append_suffix
        if layout == "landscape":
            return (prompt, 1216, 832)
        elif layout == "portrait":
            return (prompt, 832, 1216)
        elif layout == "square":
            return (prompt, 1024, 1024)

    """@classmethod
    def IS_CHANGED(cls, token, details, append_prefix, append_suffix, batch_quantity, images_per_batch):
        # Clear config file
        if os.path.isfile("prompt_from_ai.txt"):
            os.remove("prompt_from_ai.txt")
        return True"""

    def IS_CHANGED(cls, token, details, append_prefix, append_suffix, batch_quantity, images_per_batch):
        print("PARTY HARD")


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
        avaialbe_loras = folder_paths.get_filename_list("loras")
        # print("Avaialbe loras: " + str(avaialbe_loras))
        # Extract lora from _prompt: <name:model_stremgth:clip_strength> or <name:model_stremgth> or <name::clip_strength> or <name> or <name::>
        all_loras = re.findall(r"<([\w\-. ]+?)(?:\:([0-9](?:\.[0-9]*)?)|(?:\:([0-9]+(?:\.[0-9]*)?|))(?:\:([0-9]+(?:\.[0-9]*)?|)))?>", prompt)
        all_loras_regex = re.findall(r"<RE\:(.+?)(?:\:([0-9](?:\.[0-9]*)?)|(?:\:([0-9]+(?:\.[0-9]*)?|))(?:\:([0-9]+(?:\.[0-9]*)?|)))?>", prompt)
        # Remove loras from prompt
        prompt = re.sub(r"<([\w\-. ]+?)(?:\:([0-9](?:\.[0-9]*)?)|(?:\:([0-9]+(?:\.[0-9]*)?|))(?:\:([0-9]+(?:\.[0-9]*)?|)))?>", "", prompt)
        prompt = re.sub(r"<RE\:(.+?)(?:\:([0-9](?:\.[0-9]*)?)|(?:\:([0-9]+(?:\.[0-9]*)?|))(?:\:([0-9]+(?:\.[0-9]*)?|)))?>", "", prompt)
        # Map lora to object with name, model_strength and clip_strength
        all_loras = list(map(lambda x: {
                         "name": x[0], "model_strength": x[1], "clip_strength": x[2]}, all_loras))

        # Go trough every lora and fix name. For exmaple: FineNude -> NSFW\\FineNude-V0.2.safetensors or FineNude-V0.2 -> NSFW\\FineNude-V0.2.safetensors
        for lora in all_loras:
            current_name = lora["name"]
            found_exact = False
            for avaialbe_lora in avaialbe_loras:
                if avaialbe_lora == current_name:
                    found_exact = True
                    break
            if not found_exact:
                for avaialbe_lora in avaialbe_loras:
                    # Check if avaialbe_lora contains current_name
                    if avaialbe_lora.find(current_name) != -1:
                        print("Found lora as: " + avaialbe_lora)
                        lora["name"] = avaialbe_lora
                        break

        # In regex search for avaiable loras using that regex
        for lora in all_loras_regex:
            print("Lora regex: " + lora[0])
            # Convert regex to regex object
            regex = re.compile(lora[0])
            # Create list with all avaialbe loras that match regex
            avaialbe_loras_regex = list(filter(lambda x: regex.match(x) != None, avaialbe_loras))
            # print("Avaialbe loras regex: " + str(avaialbe_loras_regex))
            # If no avaialbe loras match regex, skip
            if len(avaialbe_loras_regex) == 0:
                continue
            # Get random lora from avaialbe_loras_regex
            random_lora = avaialbe_loras_regex[numpy.random.randint(0, len(avaialbe_loras_regex))]
            # Replace lora with random_lora
            prompt = prompt.replace("<RE:" + lora[0] + ">", random_lora)
            # Add random_lora to all_loras
            all_loras.append({"name": random_lora, "model_strength": lora[1], "clip_strength": lora[2]})

        # Go thorugh all loras and warn if lora not available
        for lora in all_loras:
            if lora["name"] not in avaialbe_loras:
                print("Lora not available: " + lora["name"])

        # Filter out loras that are not available
        all_loras = list(
            filter(lambda x: x["name"] in avaialbe_loras, all_loras))
        print(all_loras)
        if len(all_loras) == 0:
            return (model, clip, prompt)
        else:
            for lora in all_loras:
                lora_path = folder_paths.get_full_path("loras", lora["name"])
                loaded_lora = comfy.utils.load_torch_file(
                    lora_path, safe_load=True)
                # Strenght
                model_strength = float(lora["model_strength"]) if lora["model_strength"] != "" else 1.0
                clip_strength = float(lora["clip_strength"]) if lora["clip_strength"] != "" else float(lora["model_strength"]) if lora["model_strength"] != "" else 1.0
                model, clip = comfy.sd.load_lora_for_models(model, clip, loaded_lora, model_strength, clip_strength)
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
                                    # Add random trigger word to prompt
                                    randomTriggerWord = ssDatasetDirsKeys[numpy.random.randint(0, len(ssDatasetDirsKeys))]
                                    prompt += " " + randomTriggerWord.split("_")[1]
                                    print("Added trigger word: '" + randomTriggerWord.split("_")[1] + "' to prompt")
                except:
                    print("Error while extracting metadata from file: " + lora_path)

            return (model, clip, prompt)


class CurrentDateTime:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "get_current_date_time"
    CATEGORY = "t4ggno/utils"
    OUTPUT_NODE = False

    def get_current_date_time(self, **kwargs):
        # Get current date time using python
        now = datetime.now()
        # Alert time for debugging
        print("Current date and time : ")
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
        # Convert to "YYYY-MM-DD_HH-MM-SS"
        current_date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        return (current_date_time,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return True


NODE_CLASS_MAPPINGS = {
    "Base64Decode": Base64Decode,
    "LayoutSwitch": LayoutSwitch,
    "PredefinedResolutions": PredefinedResolutions,
    "ResolutionSwitch": ResolutionSwitch,
    "LoraLoaderFromPrompt": LoraLoaderFromPrompt,
    "PromptFromAI": PromptFromAI,
    "CurrentDateTime": CurrentDateTime,
    "TextSwitch": TextSwitch,
    "TextReplacer": TextReplacer,
    "ImageSave": ImageSave,
    "AutoLoadImageForUpscaler": AutoLoadImageForUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Base64Decode": "Load Image (Base64)",
    "LayoutSwitch": "Switch Layout",
    "PredefinedResolutions": "Predefined Resolutions",
    "ResolutionSwitch": "Switch Resolution",
    "LoraLoaderFromPrompt": "Load Lora From Prompt",
    "PromptFromAI": "Prompt From AI",
    "CurrentDateTime": "Current Date Time",
    "TextSwitch": "Switch Text",
    "TextReplacer": "Replace Text",
    "ImageSave": "Save Image",
    "AutoLoadImageForUpscaler": "Auto Load Image For Upscaler",
}
