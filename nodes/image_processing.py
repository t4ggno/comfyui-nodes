from .base_imports import *
from typing import Tuple, Optional, Dict, Any, List, Union
from dataclasses import dataclass

@dataclass
class ImageMetadata:
    positive_prompt: str
    negative_prompt: str
    positive_prompt_original: str
    negative_prompt_original: str
    additional_prompt: str
    checkpoint: str
    checkpoint_without_extension: str

def extract_metadata_from_image(image: Image, fallback_positive_prompt: str, fallback_negative_prompt: str) -> ImageMetadata:
    """Extract metadata from image with fallback values."""
    metadata = image.info
    
    def get_metadata_value(key: str, fallback: str = "") -> str:
        """Safely extract and clean metadata value."""
        if key not in metadata:
            return fallback
        value = metadata[key]
        return value.strip() if isinstance(value, str) else fallback
    
    return ImageMetadata(
        positive_prompt=get_metadata_value("PositiveText", fallback_positive_prompt),
        negative_prompt=get_metadata_value("NegativeText", fallback_negative_prompt),
        positive_prompt_original=get_metadata_value("PositiveTextOriginal", fallback_positive_prompt),
        negative_prompt_original=get_metadata_value("NegativeTextOriginal", fallback_negative_prompt),
        additional_prompt=get_metadata_value("AdditionalText"),
        checkpoint=get_metadata_value("Checkpoint"),
        checkpoint_without_extension=get_metadata_value("CheckpointWithoutExtension")
    )

class Base64ImageDecoder:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "base64_img": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode_base64_image"
    CATEGORY = "t4ggno/image"
    OUTPUT_NODE = True

    def decode_base64_image(self, base64_img: str) -> Tuple[torch.Tensor]:
        """Convert base64 string to ComfyUI image tensor."""
        print("Base64 Image Decoder - Processing...")
        
        try:
            image_bytes = base64.b64decode(base64_img)
            pil_image = Image.open(io.BytesIO(image_bytes))
            image_array = numpy.array(pil_image).astype(numpy.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).unsqueeze(0)
            
            print(f"Successfully decoded image: {pil_image.size}")
            return (image_tensor,)
        except Exception as e:
            print(f"Error decoding base64 image: {e}")
            raise ValueError(f"Failed to decode base64 image: {e}")

class ImageMetadataExtractor:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "fallback_positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "fallback_negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "positive_prompt_original", "negative_prompt_original", "checkpoint", "checkpoint_without_extension", "additional_prompt")
    FUNCTION = "extract_metadata"
    CATEGORY = "t4ggno/image"
    OUTPUT_NODE = False

    def extract_metadata(self, image: torch.Tensor, fallback_positive_prompt: str, fallback_negative_prompt: str) -> Tuple[str, ...]:
        """Extract metadata from image tensor."""
        print("Image Metadata Extractor - Processing...")
        
        try:
            # Convert tensor to PIL image to access metadata
            if hasattr(image, 'info'):
                pil_image = image
            else:
                # If it's a tensor, we need to handle it differently
                # For now, return fallback values as tensors don't have metadata
                return (fallback_positive_prompt, fallback_negative_prompt, fallback_positive_prompt, fallback_negative_prompt, "", "", "")
            
            metadata = extract_metadata_from_image(pil_image, fallback_positive_prompt, fallback_negative_prompt)
            
            print(f"Extracted metadata successfully")
            return (
                metadata.positive_prompt,
                metadata.negative_prompt,
                metadata.positive_prompt_original,
                metadata.negative_prompt_original,
                metadata.checkpoint,
                metadata.checkpoint_without_extension,
                metadata.additional_prompt
            )
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return (fallback_positive_prompt, fallback_negative_prompt, fallback_positive_prompt, fallback_negative_prompt, "", "", "")

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        return float("nan")

@dataclass
class ImageInfo:
    filename: str
    scale: float
    file_path: str

class AutoLoadImageForUpscaler:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
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
    FUNCTION = "load_image_for_upscaling"
    CATEGORY = "t4ggno/image"
    OUTPUT_NODE = False

    def _parse_scaled_filename(self, filename: str) -> Optional[Tuple[str, float]]:
        """Parse filename to extract base name and scale factor."""
        pattern = r"(.+?)(?:_(\d(?:\.\d+)?)x)\.png"
        match = re.search(pattern, filename)
        if match:
            return match.group(1), float(match.group(2))
        return None

    def _get_image_candidates(self, folder: str, scale: float, max_scale: float) -> List[ImageInfo]:
        """Get list of valid image candidates for upscaling."""
        if not os.path.exists(folder):
            raise ValueError(f"Folder does not exist: {folder}")
        
        files = [f for f in os.listdir(folder) if f.endswith(".png")]
        image_candidates = {}
        
        for filename in files:
            parsed = self._parse_scaled_filename(filename)
            if not parsed:
                continue
                
            base_name, file_scale = parsed
            
            # Skip if scale is too high
            if file_scale * 2 > max_scale:
                continue
                
            # Keep only the highest scale version of each base image
            if base_name not in image_candidates or file_scale > image_candidates[base_name].scale:
                image_candidates[base_name] = ImageInfo(
                    filename=base_name,
                    scale=file_scale,
                    file_path=os.path.join(folder, filename)
                )
        
        return list(image_candidates.values())

    def _filter_by_dimensions(self, candidates: List[ImageInfo], max_width: int, max_height: int) -> List[ImageInfo]:
        """Filter candidates by maximum dimensions."""
        valid_candidates = []
        
        for candidate in candidates:
            try:
                with Image.open(candidate.file_path) as img:
                    if img.width <= max_width and img.height <= max_height:
                        valid_candidates.append(candidate)
            except Exception as e:
                print(f"Error checking dimensions for {candidate.file_path}: {e}")
        
        return valid_candidates

    def load_image_for_upscaling(self, folder: str, max_width: int, max_height: int, scale: float, max_scale: float, fallback_positive_prompt: str, fallback_negative_prompt: str) -> Tuple[torch.Tensor, str, float, str, str, float, str, str, str, str, str]:
        """Load image for upscaling with automatic candidate selection."""
        print("Auto Load Image For Upscaler - Processing...")

        while True:
            try:
                candidates = self._get_image_candidates(folder, scale, max_scale)
                filtered_candidates = self._filter_by_dimensions(candidates, max_width, max_height)
                
                if not filtered_candidates:
                    print("No valid candidates found, waiting 10 seconds...")
                    time.sleep(10)
                    continue
                
                # Select candidate with lowest scale
                selected_candidate = min(filtered_candidates, key=lambda x: x.scale)
                
                # Load the selected image
                loaded_image = Image.open(selected_candidate.file_path)
                metadata = extract_metadata_from_image(loaded_image, fallback_positive_prompt, fallback_negative_prompt)
                
                # Convert to tensor
                image_array = numpy.array(loaded_image).astype(numpy.float32) / 255.0
                image_tensor = torch.from_numpy(image_array).unsqueeze(0)
                
                print(f"Selected image: {selected_candidate.file_path}")
                print(f"Current scale: {selected_candidate.scale}")
                print(f"Next scale: {scale}")
                
                return (
                    image_tensor,
                    selected_candidate.filename,
                    selected_candidate.scale,
                    metadata.positive_prompt,
                    metadata.negative_prompt,
                    scale,
                    metadata.positive_prompt_original,
                    metadata.negative_prompt_original,
                    metadata.checkpoint,
                    metadata.checkpoint_without_extension,
                    metadata.additional_prompt
                )
                
            except Exception as e:
                print(f"Error in load_image_for_upscaling: {e}")
                time.sleep(10)

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        return float("nan")

class LoadImageWithMetadata:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "filename", "positive_prompt", "negative_prompt", "positive_prompt_original", "negative_prompt_original", "checkpoint", "checkpoint_without_extension", "additional_prompt")
    FUNCTION = "load_image_with_metadata"
    CATEGORY = "t4ggno/image"
    OUTPUT_NODE = False

    def load_image_with_metadata(self, image: str) -> Tuple[torch.Tensor, torch.Tensor, str, str, str, str, str, str, str, str]:
        """Load image with metadata extraction."""
        print("Load Image With Metadata - Processing...")

        try:
            image_path = folder_paths.get_annotated_filepath(image)
            pil_image = Image.open(image_path)
            
            # Extract metadata
            metadata = extract_metadata_from_image(pil_image, "", "")
            
            # Process image sequence
            output_images = []
            output_masks = []
            width, height = None, None
            excluded_formats = ['MPO']
            
            for frame in ImageSequence.Iterator(pil_image):
                frame = node_helpers.pillow(ImageOps.exif_transpose, frame)
                
                if frame.mode == 'I':
                    frame = frame.point(lambda i: i * (1 / 255))
                
                rgb_image = frame.convert("RGB")
                
                if len(output_images) == 0:
                    width, height = rgb_image.size
                
                if rgb_image.size != (width, height):
                    continue
                
                # Convert to tensor
                image_array = numpy.array(rgb_image).astype(numpy.float32) / 255.0
                image_tensor = torch.from_numpy(image_array)[None,]
                
                # Handle alpha channel for mask
                if 'A' in frame.getbands():
                    mask_array = numpy.array(frame.getchannel('A')).astype(numpy.float32) / 255.0
                    mask_tensor = 1.0 - torch.from_numpy(mask_array)
                else:
                    mask_tensor = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
                
                output_images.append(image_tensor)
                output_masks.append(mask_tensor.unsqueeze(0))
            
            # Combine frames
            if len(output_images) > 1 and pil_image.format not in excluded_formats:
                final_image = torch.cat(output_images, dim=0)
                final_mask = torch.cat(output_masks, dim=0)
            else:
                final_image = output_images[0]
                final_mask = output_masks[0]
            
            print(f"Successfully loaded image: {image}")
            
            return (
                final_image,
                final_mask,
                image,
                metadata.positive_prompt,
                metadata.negative_prompt,
                metadata.positive_prompt_original,
                metadata.negative_prompt_original,
                metadata.checkpoint,
                metadata.checkpoint_without_extension,
                metadata.additional_prompt
            )
            
        except Exception as e:
            print(f"Error loading image with metadata: {e}")
            raise ValueError(f"Failed to load image: {e}")

    @classmethod
    def IS_CHANGED(cls, image: str) -> str:
        image_path = folder_paths.get_annotated_filepath(image)
        hash_obj = hashlib.sha256()
        with open(image_path, 'rb') as f:
            hash_obj.update(f.read())
        return hash_obj.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image: str) -> Union[bool, str]:
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True

class ImageSaver:
    def __init__(self):
        self.output_dir = comfy_paths.output_directory
        self.type = os.path.basename(self.output_dir)

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
                "filename": ("STRING", {"default": ""}),
                "output_path": ("STRING", {"default": ""}),
                "extension": (['png', 'jpeg', 'gif', 'tiff', 'webp'],),
                "quality": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1}),
                "lossless_webp": (["false", "true"],),
                "show_previews": (["true", "false"],),
                "positive_text": ("STRING", {"default": ""}),
                "negative_text": ("STRING", {"default": ""}),
                "additional_text": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": ""}),
                "positive_text_original": ("STRING", {"default": ""}),
                "negative_text_original": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "t4ggno/image"

    def _extract_model_name(self, model_path: str) -> str:
        """Extract model name from path, removing directory structure."""
        return os.path.basename(model_path.replace("\\", "/"))

    def _setup_output_path(self, output_path: str) -> str:
        """Setup and validate output path."""
        if not output_path.strip():
            now = datetime.now()
            output_path = os.path.join(self.output_dir, now.strftime("%Y-%m-%d"))
        
        output_path = output_path.strip()
        if not os.path.exists(output_path):
            print(f'Creating directory: {output_path}')
            os.makedirs(output_path, exist_ok=True)
        
        return output_path

    def _create_metadata(self, positive_text: str, negative_text: str, additional_text: str, model: str, positive_text_original: str, negative_text_original: str) -> PngInfo:
        """Create PNG metadata from text inputs."""
        metadata = PngInfo()
        metadata.add_text("PositiveText", positive_text)
        metadata.add_text("PositiveTextOriginal", positive_text_original)
        metadata.add_text("NegativeText", negative_text)
        metadata.add_text("NegativeTextOriginal", negative_text_original)
        metadata.add_text("AdditionalText", additional_text)
        metadata.add_text("Checkpoint", model)
        metadata.add_text("CheckpointWithoutExtension", os.path.splitext(model)[0])
        return metadata

    def _generate_filename(self, base_filename: str, extension: str) -> str:
        """Generate filename with timestamp if base is empty."""
        if not base_filename.strip():
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            return f"{timestamp}.{extension}"
        return f"{base_filename}.{extension}"

    def _save_single_image(self, image: torch.Tensor, output_path: str, filename: str, extension: str, quality: int, lossless_webp: bool, metadata: PngInfo) -> Optional[Dict[str, str]]:
        """Save a single image with error handling."""
        try:
            # Convert tensor to PIL image
            image_array = 255.0 * image.cpu().numpy()
            pil_image = Image.fromarray(numpy.clip(image_array, 0, 255).astype(numpy.uint8))
            
            output_file = os.path.abspath(os.path.join(output_path, filename))
            
            # Save based on extension
            if extension == 'png':
                pil_image.save(output_file, pnginfo=metadata, optimize=True)
            elif extension == 'webp':
                pil_image.save(output_file, quality=quality, lossless=lossless_webp)
            elif extension == 'jpeg':
                pil_image.save(output_file, quality=quality, optimize=True)
            elif extension == 'tiff':
                pil_image.save(output_file, quality=quality, optimize=True)
            elif extension == 'gif':
                pil_image.save(output_file, optimize=True)
            
            print(f"Successfully saved: {output_file}")
            return {"filename": filename, "path": output_file}
            
        except Exception as e:
            print(f"Error saving image {filename}: {e}")
            return None

    def save_images(self, images: torch.Tensor, filename: str = "", output_path: str = "", extension: str = 'png', quality: int = 100, lossless_webp: str = "false", show_previews: str = "true", positive_text: str = "", negative_text: str = "", additional_text: str = "", model: str = "", positive_text_original: str = "", negative_text_original: str = "") -> Tuple[str]:
        """Save images with metadata."""
        print("Image Saver - Processing...")

        # Validate extension
        if extension not in ALLOWED_EXT:
            print(f"Invalid extension '{extension}'. Using 'png' instead.")
            extension = "png"

        # Setup paths and metadata
        model_name = self._extract_model_name(model)
        output_path = self._setup_output_path(output_path)
        metadata = self._create_metadata(positive_text, negative_text, additional_text, model_name, positive_text_original, negative_text_original)
        
        lossless_webp_bool = lossless_webp == "true"
        results = []
        
        # Save each image
        for i, image in enumerate(images):
            current_filename = self._generate_filename(filename, extension)
            if len(images) > 1:
                name_part, ext_part = os.path.splitext(current_filename)
                current_filename = f"{name_part}_{i:03d}{ext_part}"
            
            result = self._save_single_image(image, output_path, current_filename, extension, quality, lossless_webp_bool, metadata)
            if result:
                results.append(result)

        # Return results
        if show_previews == 'true':
            return (json.dumps({"ui": {"images": results}}),)
        else:
            return (json.dumps({"ui": {"images": []}}),)

class ColorMatcher:
    """Color matching node with improved error handling and type safety."""
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
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
    FUNCTION = "match_colors"
    CATEGORY = "t4ggno/utils"

    DESCRIPTION = """
    Color matching node based on color-matcher library.
    
    Enables color transfer across images for automatic color-grading of photographs,
    paintings and film sequences as well as light-field and stopmotion corrections.
    
    Methods available:
    - mkl: Monge-Kantorovich Linearization
    - hm: Histogram Matching
    - reinhard: Reinhard et al. approach
    - mvgd: Multi-Variate Gaussian Distribution
    - hm-mvgd-hm: Compound method
    - hm-mkl-hm: Compound method
    
    Based on: https://github.com/hahnec/color-matcher/
    """

    def _validate_inputs(self, image_ref: torch.Tensor, image_target: torch.Tensor) -> None:
        """Validate input tensor dimensions and compatibility."""
        if image_ref.size(0) > 1 and image_ref.size(0) != image_target.size(0):
            raise ValueError("Use either single reference image or a matching batch of reference images.")

    def _process_single_image(self, color_matcher, target_image: numpy.ndarray, ref_image: numpy.ndarray, method: str) -> numpy.ndarray:
        """Process a single image with error handling."""
        try:
            return color_matcher.transfer(src=target_image, ref=ref_image, method=method)
        except Exception as e:
            print(f"Error during color transfer: {e}")
            return target_image

    def match_colors(self, image_ref: torch.Tensor, image_target: torch.Tensor, method: str) -> Tuple[torch.Tensor]:
        """Match colors between reference and target images."""
        print(f"Color Matcher - Processing with method: {method}")
        
        try:
            from color_matcher import ColorMatcher
        except ImportError:
            raise ImportError("color-matcher library not found. Install with: pip install color-matcher")

        try:
            self._validate_inputs(image_ref, image_target)
            
            color_matcher = ColorMatcher()
            image_ref_cpu = image_ref.cpu()
            image_target_cpu = image_target.cpu()
            batch_size = image_target_cpu.size(0)
            
            # Prepare numpy arrays
            images_target = image_target_cpu.squeeze()
            images_ref = image_ref_cpu.squeeze()
            
            ref_np = images_ref.numpy()
            target_np = images_target.numpy()
            
            processed_images = []
            
            # Process each image in the batch
            for i in range(batch_size):
                current_target = target_np if batch_size == 1 else images_target[i].numpy()
                current_ref = ref_np if image_ref_cpu.size(0) == 1 else images_ref[i].numpy()
                
                processed_image = self._process_single_image(color_matcher, current_target, current_ref, method)
                processed_images.append(torch.from_numpy(processed_image))
            
            if not processed_images:
                print("No images were successfully processed. Returning original target images.")
                return (image_target,)
            
            # Stack and clamp results
            result = torch.stack(processed_images, dim=0).to(torch.float32)
            result.clamp_(0, 1)
            
            print(f"Successfully processed {len(processed_images)} images")
            return (result,)
            
        except Exception as e:
            print(f"Error in color matching: {e}")
            return (image_target,)

NODE_CLASS_MAPPINGS = {
    "Base64Decode": Base64ImageDecoder,
    "ImageMetadataExtractor": ImageMetadataExtractor,
    "AutoLoadImageForUpscaler": AutoLoadImageForUpscaler,
    "LoadImageWithMetadata": LoadImageWithMetadata,
    "ImageSave": ImageSaver,
    "ColorMatch": ColorMatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Base64Decode": "Load Image (Base64)",
    "ImageMetadataExtractor": "Image Metadata Extractor",
    "AutoLoadImageForUpscaler": "Auto Load Image For Upscaler",
    "LoadImageWithMetadata": "Load Image With Metadata",
    "ImageSave": "Save Image",
    "ColorMatch": "Color Match",
}
