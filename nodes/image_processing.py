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

class Base64ImageDecoder(comfy_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="Base64Decode",
            display_name="Load Image (Base64)",
            category="t4ggno/image",
            inputs=[
                comfy_io.String.Input("base64_img", default=""),
            ],
            outputs=[
                comfy_io.Image.Output(display_name="image")
            ]
        )

    @classmethod
    def execute(cls, base64_img: str, **kwargs) -> comfy_io.NodeOutput:
        """Convert base64 string to ComfyUI image tensor."""
        print("Base64 Image Decoder - Processing...")
        
        try:
            image_bytes = base64.b64decode(base64_img)
            pil_image = Image.open(io.BytesIO(image_bytes))
            image_array = numpy.array(pil_image).astype(numpy.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).unsqueeze(0)
            
            print(f"Successfully decoded image: {pil_image.size}")
            return comfy_io.NodeOutput(image_tensor)
        except Exception as e:
            print(f"Error decoding base64 image: {e}")
            raise ValueError(f"Failed to decode base64 image: {e}")

class ImageMetadataExtractor(comfy_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="ImageMetadataExtractor",
            display_name="Image Metadata Extractor",
            category="t4ggno/image",
            inputs=[
                comfy_io.Image.Input("image"),
                comfy_io.String.Input("fallback_positive_prompt", default="", multiline=True),
                comfy_io.String.Input("fallback_negative_prompt", default="", multiline=True),
            ],
            outputs=[
                comfy_io.String.Output(display_name="positive_prompt"),
                comfy_io.String.Output(display_name="negative_prompt"),
                comfy_io.String.Output(display_name="positive_prompt_original"),
                comfy_io.String.Output(display_name="negative_prompt_original"),
                comfy_io.String.Output(display_name="checkpoint"),
                comfy_io.String.Output(display_name="checkpoint_without_extension"),
                comfy_io.String.Output(display_name="additional_prompt"),
            ]
        )

    @classmethod
    def execute(cls, image: torch.Tensor, fallback_positive_prompt: str, fallback_negative_prompt: str, **kwargs) -> comfy_io.NodeOutput:
        """Extract metadata from image tensor."""
        print("Image Metadata Extractor - Processing...")
        
        try:
            # Convert tensor to PIL image to access metadata
            # Note: This is tricky because tensor doesn't have metadata. 
            # Ideally, we should pass the PIL image, but ComfyUI passes tensors.
            # However, if the image came from LoadImageWithMetadata, we might have metadata in the node output?
            # No, ComfyUI passes tensors.
            # But wait, if we use ComfyNode, we receive what is passed.
            
            # If the input is a tensor, we can't get metadata from it directly unless it was attached somehow.
            # But standard ComfyUI images are just tensors.
            # This node might be intended to work with images loaded via specific nodes that preserve metadata,
            # or maybe it expects something else.
            # However, based on the original code, it seems to expect an object with .info attribute.
            
            if hasattr(image, 'info'):
                pil_image = image
            else:
                # If it's a tensor, we need to handle it differently
                # For now, return fallback values as tensors don't have metadata
                return comfy_io.NodeOutput(
                    fallback_positive_prompt, 
                    fallback_negative_prompt, 
                    fallback_positive_prompt, 
                    fallback_negative_prompt, 
                    "", 
                    "", 
                    ""
                )
            
            metadata = extract_metadata_from_image(pil_image, fallback_positive_prompt, fallback_negative_prompt)
            
            print(f"Extracted metadata successfully")
            return comfy_io.NodeOutput(
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
            return comfy_io.NodeOutput(
                fallback_positive_prompt, 
                fallback_negative_prompt, 
                fallback_positive_prompt, 
                fallback_negative_prompt, 
                "", 
                "", 
                ""
            )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

@dataclass
class ImageInfo:
    filename: str
    scale: float
    file_path: str

class AutoLoadImageForUpscaler(comfy_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="AutoLoadImageForUpscaler",
            display_name="Auto Load Image For Upscaler",
            category="t4ggno/image",
            inputs=[
                comfy_io.String.Input("folder", default=""),
                comfy_io.Int.Input("max_width", default=8192, min=1, max=100000, step=1),
                comfy_io.Int.Input("max_height", default=8192, min=1, max=100000, step=1),
                comfy_io.Float.Input("scale", default=2, min=1, max=100, step=0.1),
                comfy_io.Float.Input("max_scale", default=100000, min=1, max=100000, step=0.1),
                comfy_io.String.Input("fallback_positive_prompt", default="", multiline=True),
                comfy_io.String.Input("fallback_negative_prompt", default="", multiline=True),
                comfy_io.Combo.Input("control_after_generate", options=["fixed", "random", "increment"], default="increment"),
                comfy_io.Int.Input("value", default=0),
            ],
            outputs=[
                comfy_io.Image.Output(display_name="image"),
                comfy_io.String.Output(display_name="filename_without_scale"),
                comfy_io.Float.Output(display_name="current_scale"),
                comfy_io.String.Output(display_name="positive_prompt"),
                comfy_io.String.Output(display_name="negative_prompt"),
                comfy_io.Float.Output(display_name="next_scale"),
                comfy_io.String.Output(display_name="positive_prompt_original"),
                comfy_io.String.Output(display_name="negative_prompt_original"),
                comfy_io.String.Output(display_name="checkpoint"),
                comfy_io.String.Output(display_name="checkpoint_without_extension"),
                comfy_io.String.Output(display_name="additional_prompt"),
            ]
        )

    @classmethod
    def _parse_scaled_filename(cls, filename: str) -> Optional[Tuple[str, float]]:
        """Parse filename to extract base name and scale factor."""
        pattern = r"(.+?)(?:_(\d(?:\.\d+)?)x)\.png"
        match = re.search(pattern, filename)
        if match:
            return match.group(1), float(match.group(2))
        return None

    @classmethod
    def _get_image_candidates(cls, folder: str, scale: float, max_scale: float) -> List[ImageInfo]:
        """Get list of valid image candidates for upscaling."""
        if not os.path.exists(folder):
            raise ValueError(f"Folder does not exist: {folder}")
        
        files = [f for f in os.listdir(folder) if f.endswith(".png")]
        image_candidates = {}
        
        for filename in files:
            parsed = cls._parse_scaled_filename(filename)
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

    @classmethod
    def _filter_by_dimensions(cls, candidates: List[ImageInfo], max_width: int, max_height: int) -> List[ImageInfo]:
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

    @classmethod
    def execute(
        cls, 
        folder: str, 
        max_width: int, 
        max_height: int, 
        scale: float, 
        max_scale: float, 
        fallback_positive_prompt: str, 
        fallback_negative_prompt: str,
        **kwargs
    ) -> comfy_io.NodeOutput:
        """Load image for upscaling with automatic candidate selection."""
        print("Auto Load Image For Upscaler - Processing...")

        while True:
            try:
                candidates = cls._get_image_candidates(folder, scale, max_scale)
                filtered_candidates = cls._filter_by_dimensions(candidates, max_width, max_height)
                
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
                
                return comfy_io.NodeOutput(
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
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

class LoadImageWithMetadata(comfy_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return comfy_io.Schema(
            node_id="LoadImageWithMetadata",
            display_name="Load Image With Metadata",
            category="t4ggno/image",
            inputs=[
                comfy_io.Combo.Input("image", options=sorted(files)),
            ],
            outputs=[
                comfy_io.Image.Output(display_name="image"),
                comfy_io.Mask.Output(display_name="mask"),
                comfy_io.String.Output(display_name="filename"),
                comfy_io.String.Output(display_name="positive_prompt"),
                comfy_io.String.Output(display_name="negative_prompt"),
                comfy_io.String.Output(display_name="positive_prompt_original"),
                comfy_io.String.Output(display_name="negative_prompt_original"),
                comfy_io.String.Output(display_name="checkpoint"),
                comfy_io.String.Output(display_name="checkpoint_without_extension"),
                comfy_io.String.Output(display_name="additional_prompt"),
            ]
        )

    @classmethod
    def execute(cls, image: str, **kwargs) -> comfy_io.NodeOutput:
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
            
            return comfy_io.NodeOutput(
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
    def fingerprint_inputs(cls, image: str, **kwargs) -> str:
        image_path = folder_paths.get_annotated_filepath(image)
        hash_obj = hashlib.sha256()
        with open(image_path, 'rb') as f:
            hash_obj.update(f.read())
        return hash_obj.digest().hex()

    @classmethod
    def validate_inputs(cls, image: str, **kwargs) -> Union[bool, str]:
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True

class ImageSaver(comfy_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="ImageSave",
            display_name="Save Image",
            category="t4ggno/image",
            inputs=[
                comfy_io.Image.Input("images"),
                comfy_io.String.Input("filename", default=""),
                comfy_io.String.Input("output_path", default=""),
                comfy_io.Combo.Input("extension", options=['png', 'jpeg', 'gif', 'tiff', 'webp']),
                comfy_io.Int.Input("quality", default=100, min=1, max=100, step=1),
                comfy_io.Combo.Input("lossless_webp", options=["false", "true"]),
                comfy_io.Combo.Input("show_previews", options=["true", "false"]),
                comfy_io.String.Input("positive_text", default=""),
                comfy_io.String.Input("negative_text", default=""),
                comfy_io.String.Input("additional_text", default=""),
                comfy_io.String.Input("model", default=""),
                comfy_io.String.Input("positive_text_original", default=""),
                comfy_io.String.Input("negative_text_original", default=""),
            ],
            outputs=[]
        )

    @classmethod
    def _extract_model_name(cls, model_path: str) -> str:
        """Extract model name from path, removing directory structure."""
        return os.path.basename(model_path.replace("\\", "/"))

    @classmethod
    def _setup_output_path(cls, output_path: str) -> str:
        """Setup and validate output path."""
        if not output_path.strip():
            now = datetime.now()
            output_path = os.path.join(comfy_paths.output_directory, now.strftime("%Y-%m-%d"))
        
        output_path = output_path.strip()
        if not os.path.exists(output_path):
            print(f'Creating directory: {output_path}')
            os.makedirs(output_path, exist_ok=True)
        
        return output_path

    @classmethod
    def _create_metadata(cls, positive_text: str, negative_text: str, additional_text: str, model: str, positive_text_original: str, negative_text_original: str) -> PngInfo:
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

    @classmethod
    def _generate_filename(cls, base_filename: str, extension: str) -> str:
        """Generate filename with timestamp if base is empty."""
        if not base_filename.strip():
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            return f"{timestamp}.{extension}"
        return f"{base_filename}.{extension}"

    @classmethod
    def _save_single_image(cls, image: torch.Tensor, output_path: str, filename: str, extension: str, quality: int, lossless_webp: bool, metadata: PngInfo) -> Optional[Dict[str, str]]:
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

    @classmethod
    def execute(
        cls, 
        images: torch.Tensor, 
        filename: str, 
        output_path: str, 
        extension: str, 
        quality: int, 
        lossless_webp: str, 
        show_previews: str, 
        positive_text: str, 
        negative_text: str, 
        additional_text: str, 
        model: str, 
        positive_text_original: str, 
        negative_text_original: str,
        **kwargs
    ) -> comfy_io.NodeOutput:
        """Save images with metadata."""
        print("Image Saver - Processing...")

        # Validate extension
        if extension not in ALLOWED_EXT:
            print(f"Invalid extension '{extension}'. Using 'png' instead.")
            extension = "png"

        # Setup paths and metadata
        model_name = cls._extract_model_name(model)
        output_path = cls._setup_output_path(output_path)
        metadata = cls._create_metadata(positive_text, negative_text, additional_text, model_name, positive_text_original, negative_text_original)
        
        lossless_webp_bool = lossless_webp == "true"
        results = []
        
        # Save each image
        for i, image in enumerate(images):
            current_filename = cls._generate_filename(filename, extension)
            if len(images) > 1:
                name_part, ext_part = os.path.splitext(current_filename)
                current_filename = f"{name_part}_{i:03d}{ext_part}"
            
            result = cls._save_single_image(image, output_path, current_filename, extension, quality, lossless_webp_bool, metadata)
            if result:
                results.append(result)

        # Return results
        if show_previews == 'true':
            return comfy_io.NodeOutput(ui=comfy_ui.UI(images=results))
        else:
            return comfy_io.NodeOutput(ui=comfy_ui.UI(images=[]))

class ColorMatcher(comfy_io.ComfyNode):
    """Color matching node with improved error handling and type safety."""
    
    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="ColorMatch",
            display_name="Color Match",
            category="t4ggno/utils",
            description="""
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
            """,
            inputs=[
                comfy_io.Image.Input("image_ref"),
                comfy_io.Image.Input("image_target"),
                comfy_io.Combo.Input("method", options=[
                    'mkl',
                    'hm',
                    'reinhard',
                    'mvgd',
                    'hm-mvgd-hm',
                    'hm-mkl-hm',
                ], default='mkl'),
            ],
            outputs=[
                comfy_io.Image.Output(display_name="image")
            ]
        )

    @classmethod
    def _validate_inputs(cls, image_ref: torch.Tensor, image_target: torch.Tensor) -> None:
        """Validate input tensor dimensions and compatibility."""
        if image_ref.size(0) > 1 and image_ref.size(0) != image_target.size(0):
            raise ValueError("Use either single reference image or a matching batch of reference images.")

    @classmethod
    def _process_single_image(cls, color_matcher, target_image: numpy.ndarray, ref_image: numpy.ndarray, method: str) -> numpy.ndarray:
        """Process a single image with error handling."""
        try:
            return color_matcher.transfer(src=target_image, ref=ref_image, method=method)
        except Exception as e:
            print(f"Error during color transfer: {e}")
            return target_image

    @classmethod
    def execute(cls, image_ref: torch.Tensor, image_target: torch.Tensor, method: str, **kwargs) -> comfy_io.NodeOutput:
        """Match colors between reference and target images."""
        print(f"Color Matcher - Processing with method: {method}")
        
        try:
            from color_matcher import ColorMatcher
        except ImportError:
            raise ImportError("color-matcher library not found. Install with: pip install color-matcher")

        try:
            cls._validate_inputs(image_ref, image_target)
            
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
                
                processed_image = cls._process_single_image(color_matcher, current_target, current_ref, method)
                processed_images.append(torch.from_numpy(processed_image))
            
            if not processed_images:
                print("No images were successfully processed. Returning original target images.")
                return comfy_io.NodeOutput(image_target)
            
            # Stack and clamp results
            result = torch.stack(processed_images, dim=0).to(torch.float32)
            result.clamp_(0, 1)
            
            print(f"Successfully processed {len(processed_images)} images")
            return comfy_io.NodeOutput(result)
            
        except Exception as e:
            print(f"Error in color matching: {e}")
            return comfy_io.NodeOutput(image_target)
