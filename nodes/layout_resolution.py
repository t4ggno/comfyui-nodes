from .base_imports import *
from typing import Tuple

LAYOUT_LANDSCAPE = "Landscape"
LAYOUT_PORTRAIT = "Portrait"
LAYOUT_SQUARE = "Square"

class LayoutSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024}),
                "height": ("INT", {"default": 1024}),
                "layout": ([LAYOUT_LANDSCAPE, LAYOUT_PORTRAIT, LAYOUT_SQUARE], {"default": LAYOUT_LANDSCAPE}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "render_resolution"
    CATEGORY = "t4ggno/utils"
    OUTPUT_NODE = False

    def render_resolution(self, width: int, height: int, layout: str) -> Tuple[int, int]:
        print(f"Layout Switch: {layout}")

        if layout == LAYOUT_LANDSCAPE:
            result_width, result_height = max(width, height), min(width, height)
        elif layout == LAYOUT_PORTRAIT:
            result_width, result_height = min(width, height), max(width, height)
        elif layout == LAYOUT_SQUARE:
            average_size = int((width + height) / 2)
            result_width, result_height = average_size, average_size
        else:
            result_width, result_height = width, height

        print(f"Output: {result_width}x{result_height}")
        return (result_width, result_height)

DIMENSION_1_5 = "1.5"
DIMENSION_SDXL = "SDXL"
DIMENSION_FAMOUS = "Famous"

LAYOUT_ULTRA_WIDE = "Landscape (Ultra Wide)"
LAYOUT_ULTRA_TALL = "Portrait (Ultra Tall)"
LAYOUT_RANDOM = "Random"

RESOLUTION_MAPPINGS = {
    DIMENSION_1_5: {
        LAYOUT_SQUARE: (512, 512),
        LAYOUT_LANDSCAPE: (576, 448),
        LAYOUT_ULTRA_WIDE: (768, 320),
        LAYOUT_PORTRAIT: (448, 576),
        LAYOUT_ULTRA_TALL: (320, 768),
    },
    DIMENSION_SDXL: {
        LAYOUT_SQUARE: (1024, 1024),
        LAYOUT_LANDSCAPE: (1216, 832),
        LAYOUT_ULTRA_WIDE: (1536, 640),
        LAYOUT_PORTRAIT: (832, 1216),
        LAYOUT_ULTRA_TALL: (640, 1536),
    },
    DIMENSION_FAMOUS: {
        LAYOUT_SQUARE: (1024, 1024),
        LAYOUT_LANDSCAPE: (1366, 768),
        LAYOUT_ULTRA_WIDE: (1596, 684),
        LAYOUT_PORTRAIT: (768, 1366),
        LAYOUT_ULTRA_TALL: (684, 1596),
    }
}

class PredefinedResolutions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dimension": ([DIMENSION_1_5, DIMENSION_SDXL, DIMENSION_FAMOUS], {"default": DIMENSION_SDXL}),
                "layout": ([LAYOUT_SQUARE, LAYOUT_LANDSCAPE, LAYOUT_ULTRA_WIDE, LAYOUT_PORTRAIT, LAYOUT_ULTRA_TALL, LAYOUT_RANDOM], {"default": LAYOUT_SQUARE}),
                "enable_square_random": ("BOOLEAN", {"default": True, "lazy": True}),
                "enable_landscape_random": ("BOOLEAN", {"default": True, "lazy": True}),
                "enable_ultra_wide_random": ("BOOLEAN", {"default": True, "lazy": True}),
                "enable_portrait_random": ("BOOLEAN", {"default": True, "lazy": True}),
                "enable_ultra_tall_random": ("BOOLEAN", {"default": True, "lazy": True}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
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

    def check_lazy_status(self, dimension, layout, enable_square_random, enable_landscape_random, enable_ultra_wide_random, enable_portrait_random, enable_ultra_tall_random, scale):
        needed = []
        if layout == LAYOUT_RANDOM:
            if enable_square_random is None: needed.append("enable_square_random")
            if enable_landscape_random is None: needed.append("enable_landscape_random")
            if enable_ultra_wide_random is None: needed.append("enable_ultra_wide_random")
            if enable_portrait_random is None: needed.append("enable_portrait_random")
            if enable_ultra_tall_random is None: needed.append("enable_ultra_tall_random")
        return needed

    def _get_enabled_layouts(self, enable_square_random: bool, enable_landscape_random: bool, 
                           enable_ultra_wide_random: bool, enable_portrait_random: bool, 
                           enable_ultra_tall_random: bool) -> list:
        layout_mappings = [
            (enable_square_random, LAYOUT_SQUARE),
            (enable_landscape_random, LAYOUT_LANDSCAPE),
            (enable_ultra_wide_random, LAYOUT_ULTRA_WIDE),
            (enable_portrait_random, LAYOUT_PORTRAIT),
            (enable_ultra_tall_random, LAYOUT_ULTRA_TALL),
        ]
        
        return [layout for enabled, layout in layout_mappings if enabled]

    def _get_resolution(self, dimension: str, layout: str) -> Tuple[int, int]:
        resolution_map = RESOLUTION_MAPPINGS.get(dimension, RESOLUTION_MAPPINGS[DIMENSION_SDXL])
        return resolution_map.get(layout, resolution_map[LAYOUT_SQUARE])

    def render_resolution(self, dimension: str, layout: str, enable_square_random: bool, 
                         enable_landscape_random: bool, enable_ultra_wide_random: bool, 
                         enable_portrait_random: bool, enable_ultra_tall_random: bool, 
                         scale: float) -> Tuple[int, int]:
        print(f"Predefined Resolutions: {dimension}")

        selected_layout = layout
        if layout == LAYOUT_RANDOM:
            enabled_layouts = self._get_enabled_layouts(
                enable_square_random, enable_landscape_random, enable_ultra_wide_random,
                enable_portrait_random, enable_ultra_tall_random
            )
            
            if not enabled_layouts:
                enabled_layouts = [LAYOUT_SQUARE]
            
            selected_layout = random.choice(enabled_layouts)
            print(f"Random layout selected: {selected_layout}")

        resolution = self._get_resolution(dimension, selected_layout)
        scaled_width = int(resolution[0] * scale)
        scaled_height = int(resolution[1] * scale)
        
        print(f"Output: {scaled_width}x{scaled_height} (scale: {scale}x)")
        return (scaled_width, scaled_height)

    @classmethod
    def IS_CHANGED(self, **kwargs):
        return float("nan")

class ResolutionSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "active": (["Resolution 1", "Resolution 2"], {"default": "Resolution 1"}),
                "width1": ("INT", {"default": 1024, "lazy": True}),
                "height1": ("INT", {"default": 1024, "lazy": True}),
                "width2": ("INT", {"default": 1024, "lazy": True}),
                "height2": ("INT", {"default": 1024, "lazy": True}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_resolution"
    CATEGORY = "t4ggno/utils"
    OUTPUT_NODE = False

    def check_lazy_status(self, active, width1, height1, width2, height2):
        needed = []
        if active == "Resolution 1":
            if width1 is None: needed.append("width1")
            if height1 is None: needed.append("height1")
        else:
            if width2 is None: needed.append("width2")
            if height2 is None: needed.append("height2")
        return needed

    def get_resolution(self, active: str, width1: int, height1: int, 
                      width2: int, height2: int) -> Tuple[int, int]:
        print(f"Resolution Switch: {active}")

        if active == "Resolution 1":
            result_width, result_height = width1, height1
        else:
            result_width, result_height = width2, height2
        
        print(f"Output: {result_width}x{result_height}")
        return (result_width, result_height)

NODE_CLASS_MAPPINGS = {
    "LayoutSwitch": LayoutSwitch,
    "PredefinedResolutions": PredefinedResolutions,
    "ResolutionSwitch": ResolutionSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayoutSwitch": "Switch Layout",
    "PredefinedResolutions": "Predefined Resolutions",
    "ResolutionSwitch": "Switch Resolution",
}
