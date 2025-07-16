from .base_imports import *
import random
import json
import os
from typing import Dict, List, Tuple, Any, Optional, Union

class RandomJSONSelector:
    """Select random items from JSON files in the workspace."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_file": ([
                    "attire_eyewear.json",
                    "attire_female_bodysuits.json",
                    "attire_female_bottomwear.json",
                    "attire_female_bra.json",
                    "attire_female_footwear.json",
                    "attire_female_headwear.json",
                    "attire_female_legwear.json",
                    "attire_female_panties.json",
                    "attire_female_swimsuits.json",
                    "attire_female_topwear.json",
                    "attire_female_traditional_clothing.json",
                    "attire_female_uniforms_and_costumes.json",
                    "attire_jewelry_and_accessories.json",
                    "attire_male_bodysuits.json",
                    "attire_male_bottomwear.json",
                    "attire_male_footwear.json",
                    "attire_male_headwear.json",
                    "attire_male_legwear.json",
                    "attire_male_swimsuits.json",
                    "attire_male_topwear.json",
                    "attire_male_traditional_clothing.json",
                    "attire_male_uniforms_and_costumes.json",
                    "attire_sleeves.json",
                    "attire_styles_and_patterns.json",
                    "events_and_scenes.json",
                    "facial_expressions.json",
                    "hair_colors.json",
                    "hair_female.json",
                    "hair_male.json",
                    "interesting_ideas.json",
                    "locations.json",
                    "materials.json",
                    "names_female.json",
                    "names_male.json",
                    "nationality_by_continent.json",
                    "nationality_by_ethnic.json",
                    "poses.json",
                    "posture_arm.json",
                    "posture_basic.json",
                    "posture_carrying.json",
                    "posture_head.json",
                    "posture_hips.json",
                    "posture_hug.json",
                    "posture_leg.json",
                    "posture_movement.json",
                    "posture_multiple_character.json",
                    "posture_other.json",
                    "posture_poses.json",
                    "posture_torso.json",
                    "scenarios.json",
                    "styles.json"
                ], {"default": "styles.json"}),
                "category": ("STRING", {"default": "", "placeholder": "Leave empty for random category"}),
                "count": ("INT", {"default": 1, "min": 1, "max": 10}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "separator": ("STRING", {"default": ", "}),
                "prefix": ("STRING", {"default": "", "multiline": True, "placeholder": "Elements to add at the beginning"}),
                "suffix": ("STRING", {"default": "", "multiline": True, "placeholder": "Elements to add at the end"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("selected_items", "category_used")
    FUNCTION = "select_random_items"
    CATEGORY = "t4ggno/prompt_generation"
    OUTPUT_NODE = False

    def select_random_items(self, json_file: str, category: str, count: int, seed: int, separator: str = ", ", prefix: str = "", suffix: str = "") -> Tuple[str, str]:
        """Select random items from a JSON file."""
        
        # Set random seed if provided
        if seed != -1:
            random.seed(seed)
        
        # Get the base path (parent directory of nodes)
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_path, json_file)
        
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError:
            print(f"Warning: JSON file not found: {json_file}")
            return ("", "")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file {json_file}: {e}")
            return ("", "")
        
        # If category is specified, use it; otherwise pick random category
        if category and category in data:
            selected_category = category
            items = data[category]
        elif category:
            print(f"Warning: Category '{category}' not found in {json_file}")
            return ("", "")
        else:
            # Pick random category
            categories = list(data.keys())
            selected_category = random.choice(categories)
            items = data[selected_category]
        
        # Select random items
        if not items:
            return ("", selected_category)
        
        selected_items = random.sample(items, min(count, len(items)))
        result = separator.join(selected_items)
        
        # Add prefix and suffix if provided
        if prefix and suffix:
            result = f"{prefix.strip()}, {result}, {suffix.strip()}"
        elif prefix:
            result = f"{prefix.strip()}, {result}"
        elif suffix:
            result = f"{result}, {suffix.strip()}"
        
        print(f"Selected from {json_file}[{selected_category}]: {result}")
        return (result, selected_category)

class SmartPromptBuilder:
    """Build intelligent prompts by combining multiple elements."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_type": ([
                    "Portrait",
                    "Fashion",
                    "Artistic",
                    "Scenic",
                    "Character",
                    "Fantasy",
                    "Modern",
                    "Traditional",
                    "Sensual",
                    "Erotic",
                    "Intimate",
                    "Fetish",
                    "Custom",
                    "Random"
                ], {"default": "Portrait"}),
                "gender": (["Any", "Female", "Male"], {"default": "Any"}),
                "complexity": (["Simple", "Medium", "Complex"], {"default": "Medium"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "custom_elements": ("STRING", {"default": "", "multiline": True, "placeholder": "Additional elements to include"}),
                "style_preference": ("STRING", {"default": "", "placeholder": "Style preference (e.g., 'realistic', 'anime', 'artistic')"}),
                "exclude_categories": ("STRING", {"default": "", "placeholder": "Categories to exclude (comma-separated)"}),
                "prefix": ("STRING", {"default": "", "multiline": True, "placeholder": "Elements to add at the beginning"}),
                "suffix": ("STRING", {"default": "", "multiline": True, "placeholder": "Elements to add at the end"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "description")
    FUNCTION = "build_prompt"
    CATEGORY = "t4ggno/prompt_generation"
    OUTPUT_NODE = False

    def build_prompt(self, prompt_type: str, gender: str, complexity: str, seed: int, 
                    custom_elements: str = "", style_preference: str = "", exclude_categories: str = "", prefix: str = "", suffix: str = "") -> Tuple[str, str]:
        """Build an intelligent prompt based on the specified parameters."""
        
        if seed != -1:
            random.seed(seed)
        
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        excluded_cats = [cat.strip().lower() for cat in exclude_categories.split(",") if cat.strip()]
        
        # Handle random prompt type selection
        if prompt_type == "Random":
            available_types = ["Portrait", "Fashion", "Artistic", "Scenic", "Character", "Fantasy", "Modern", "Traditional", "Sensual", "Erotic", "Intimate", "Fetish"]
            prompt_type = random.choice(available_types)
        
        # Define spicy themes
        spicy_themes = ["Sensual", "Erotic", "Intimate", "Fetish"]
        
        # Define prompt templates based on type
        prompt_components = []
        description_parts = []
        
        # Character/Subject
        if prompt_type in ["Portrait", "Character", "Fashion"]:
            # Add name
            if gender == "Female":
                name = self._get_random_from_file(base_path, "names_female.json")
            elif gender == "Male":
                name = self._get_random_from_file(base_path, "names_male.json")
            else:
                name = self._get_random_from_file(base_path, random.choice(["names_female.json", "names_male.json"]))
            
            if name:
                prompt_components.append(name)
                description_parts.append(f"Name: {name}")
            
            # Add nationality/ethnicity
            if "nationality" not in excluded_cats:
                nationality = self._get_random_from_file(base_path, "nationality_by_continent.json")
                if nationality:
                    prompt_components.append(f"{nationality} person")
                    description_parts.append(f"Nationality: {nationality}")
        
        # Appearance
        if prompt_type in ["Portrait", "Character", "Fashion"]:
            # Hair
            if "hair" not in excluded_cats:
                hair_color = self._get_random_from_file(base_path, "hair_colors.json")
                if gender == "Female":
                    hair_style = self._get_random_from_file(base_path, "hair_female.json")
                elif gender == "Male":
                    hair_style = self._get_random_from_file(base_path, "hair_male.json")
                else:
                    hair_style = self._get_random_from_file(base_path, random.choice(["hair_female.json", "hair_male.json"]))
                
                if hair_color and hair_style:
                    prompt_components.append(f"{hair_color} {hair_style}")
                    description_parts.append(f"Hair: {hair_color} {hair_style}")
            
            # Facial expression
            if "expression" not in excluded_cats:
                expression = self._get_random_from_file(base_path, "facial_expressions.json")
                if expression:
                    prompt_components.append(expression)
                    description_parts.append(f"Expression: {expression}")
        
        # Clothing
        if prompt_type in ["Portrait", "Character", "Fashion"] or prompt_type in spicy_themes:
            clothing_items = []
            
            # Determine clothing files based on theme and gender
            if prompt_type in spicy_themes:
                # Use spicy attire for spicy themes
                if prompt_type == "Sensual":
                    clothing_files = ["sexual_attire_lingerine.json", "sexy_things.json"]
                elif prompt_type == "Fetish":
                    clothing_files = ["sexual_attire_bdsm.json", "sexual_attire_miscellaneous.json"]
                elif prompt_type in ["Intimate", "Erotic"]:
                    clothing_files = ["sexual_attire_exposure.json", "nudity.json", "sexy_things.json"]
            else:
                # Regular clothing selection
                if gender == "Female":
                    clothing_files = ["attire_female_topwear.json", "attire_female_bottomwear.json"]
                elif gender == "Male":
                    clothing_files = ["attire_male_topwear.json", "attire_male_bottomwear.json"]
                else:
                    clothing_files = [random.choice(["attire_female_topwear.json", "attire_male_topwear.json"]),
                                    random.choice(["attire_female_bottomwear.json", "attire_male_bottomwear.json"])]
            
            if "clothing" not in excluded_cats:
                for clothing_file in clothing_files:
                    item = self._get_random_from_file(base_path, clothing_file)
                    if item:
                        clothing_items.append(item)
                
                if clothing_items:
                    prompt_components.extend(clothing_items)
                    description_parts.append(f"Clothing: {', '.join(clothing_items)}")
            
            # Accessories
            if "accessories" not in excluded_cats and complexity in ["Medium", "Complex"]:
                accessory = self._get_random_from_file(base_path, "attire_jewelry_and_accessories.json")
                if accessory:
                    prompt_components.append(accessory)
                    description_parts.append(f"Accessories: {accessory}")
        
        # Pose/Posture
        if (prompt_type in ["Portrait", "Character", "Fashion"] or prompt_type in spicy_themes) and "pose" not in excluded_cats:
            if prompt_type in spicy_themes:
                # Use more suggestive poses for spicy themes
                pose_files = ["poses.json", "posture_basic.json", "posture_hug.json"]
            else:
                pose_files = ["poses.json", "posture_basic.json", "posture_arm.json"]
            
            pose_file = random.choice(pose_files)
            pose = self._get_random_from_file(base_path, pose_file)
            if pose:
                prompt_components.append(pose)
                description_parts.append(f"Pose: {pose}")
        
        # Location/Background
        if prompt_type in ["Scenic", "Fantasy", "Character"] and "location" not in excluded_cats:
            location = self._get_random_from_file(base_path, "locations.json")
            if location:
                prompt_components.append(f"in {location}")
                description_parts.append(f"Location: {location}")
        
        # Style
        if style_preference:
            prompt_components.append(style_preference)
            description_parts.append(f"Style: {style_preference}")
        elif "style" not in excluded_cats:
            style = self._get_random_from_file(base_path, "styles.json")
            if style:
                prompt_components.append(style)
                description_parts.append(f"Style: {style}")
        
        # Add complexity-based elements
        if complexity == "Complex":
            # Add patterns/materials
            if "patterns" not in excluded_cats:
                pattern = self._get_random_from_file(base_path, "attire_styles_and_patterns.json")
                if pattern:
                    prompt_components.append(pattern)
                    description_parts.append(f"Pattern: {pattern}")
            
            # Add interesting ideas
            if "ideas" not in excluded_cats:
                idea = self._get_random_from_file(base_path, "interesting_ideas.json")
                if idea:
                    prompt_components.append(idea)
                    description_parts.append(f"Creative element: {idea}")
        
        # Add custom elements
        if custom_elements:
            prompt_components.append(custom_elements)
            description_parts.append(f"Custom: {custom_elements}")
        
        # Build final prompt
        final_prompt = ", ".join(prompt_components)
        
        # Add prefix and suffix if provided
        if prefix and suffix:
            final_prompt = f"{prefix.strip()}, {final_prompt}, {suffix.strip()}"
        elif prefix:
            final_prompt = f"{prefix.strip()}, {final_prompt}"
        elif suffix:
            final_prompt = f"{final_prompt}, {suffix.strip()}"
        
        description = " | ".join(description_parts)
        
        print(f"Generated {prompt_type} prompt: {final_prompt}...")
        return (final_prompt, description)
    
    def _get_random_from_file(self, base_path: str, filename: str) -> Optional[str]:
        """Get a random item from a JSON file."""
        filepath = os.path.join(base_path, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)
            
            # Get all items from all categories
            all_items = []
            for category, items in data.items():
                if isinstance(items, list):
                    all_items.extend(items)
            
            return random.choice(all_items) if all_items else None
        except (FileNotFoundError, json.JSONDecodeError):
            return None

class PromptTemplateManager:
    """Manage and use pre-built prompt templates."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": ([
                    "Random",
                    "Elegant Portrait",
                    "Casual Fashion",
                    "Fantasy Character",
                    "Modern Art",
                    "Traditional Clothing",
                    "Action Pose",
                    "Romantic Scene",
                    "Professional Look",
                    "Vintage Style",
                    "Futuristic Design",
                    "Artistic Expression",
                    "Dynamic Movement",
                    "Serene Moment",
                    "Dramatic Lighting",
                    "Minimalist Style",
                    "Sensual Lingerie",
                    "Seductive Pose",
                    "Intimate Scene",
                    "Fetish Fashion",
                    "Erotic Art",
                    "Spicy Romance"
                ], {"default": "Random"}),
                "gender": (["Any", "Female", "Male"], {"default": "Any"}),
                "variation_level": (["Low", "Medium", "High"], {"default": "Medium"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "custom_additions": ("STRING", {"default": "", "multiline": True, "placeholder": "Additional elements to include"}),
                "prefix": ("STRING", {"default": "", "multiline": True, "placeholder": "Elements to add at the beginning"}),
                "suffix": ("STRING", {"default": "", "multiline": True, "placeholder": "Elements to add at the end"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "template_info")
    FUNCTION = "generate_from_template"
    CATEGORY = "t4ggno/prompt_generation"
    OUTPUT_NODE = False

    def generate_from_template(self, template: str, gender: str, variation_level: str, seed: int, custom_additions: str = "", prefix: str = "", suffix: str = "") -> Tuple[str, str]:
        """Generate a prompt from a predefined template."""
        
        if seed != -1:
            random.seed(seed)
        
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Handle random template selection
        if template == "Random":
            available_templates = [
                "Elegant Portrait", "Casual Fashion", "Fantasy Character", "Modern Art",
                "Traditional Clothing", "Action Pose", "Romantic Scene", "Professional Look",
                "Vintage Style", "Futuristic Design", "Artistic Expression", "Dynamic Movement",
                "Serene Moment", "Dramatic Lighting", "Minimalist Style", "Sensual Lingerie",
                "Seductive Pose", "Intimate Scene", "Fetish Fashion", "Erotic Art", "Spicy Romance"
            ]
            template = random.choice(available_templates)
        
        # Define templates
        templates = {
            "Elegant Portrait": {
                "base": "elegant portrait",
                "elements": ["hair_colors.json", "facial_expressions.json", "attire_jewelry_and_accessories.json"],
                "styles": ["Photorealistic", "High-definition", "Cinematic"],
                "poses": ["posture_basic.json"],
                "required": ["sophisticated", "refined"]
            },
            "Casual Fashion": {
                "base": "casual fashion",
                "elements": ["hair_colors.json", "attire_styles_and_patterns.json"],
                "styles": ["Digital Painting", "Illustration"],
                "poses": ["poses.json"],
                "required": ["relaxed", "contemporary"]
            },
            "Fantasy Character": {
                "base": "fantasy character",
                "elements": ["hair_colors.json", "locations.json", "interesting_ideas.json"],
                "styles": ["Fantasy", "Digital Art"],
                "poses": ["posture_poses.json"],
                "required": ["magical", "ethereal"]
            },
            "Modern Art": {
                "base": "modern artistic portrait",
                "elements": ["styles.json", "attire_styles_and_patterns.json"],
                "styles": ["Abstract", "Contemporary"],
                "poses": ["posture_arm.json"],
                "required": ["artistic", "creative"]
            },
            "Traditional Clothing": {
                "base": "traditional attire",
                "elements": ["nationality_by_continent.json", "hair_colors.json"],
                "styles": ["Cultural", "Traditional"],
                "poses": ["posture_basic.json"],
                "required": ["authentic", "cultural"]
            },
            "Action Pose": {
                "base": "dynamic action",
                "elements": ["posture_movement.json", "facial_expressions.json"],
                "styles": ["Dynamic", "High-energy"],
                "poses": ["posture_movement.json"],
                "required": ["energetic", "powerful"]
            },
            "Romantic Scene": {
                "base": "romantic scene",
                "elements": ["facial_expressions.json", "locations.json"],
                "styles": ["Soft", "Romantic"],
                "poses": ["posture_hug.json"],
                "required": ["tender", "emotional"]
            },
            "Professional Look": {
                "base": "professional appearance",
                "elements": ["facial_expressions.json", "attire_jewelry_and_accessories.json"],
                "styles": ["Clean", "Professional"],
                "poses": ["posture_basic.json"],
                "required": ["confident", "polished"]
            },
            "Vintage Style": {
                "base": "vintage style",
                "elements": ["hair_colors.json", "attire_styles_and_patterns.json"],
                "styles": ["Vintage", "Retro"],
                "poses": ["poses.json"],
                "required": ["nostalgic", "classic"]
            },
            "Futuristic Design": {
                "base": "futuristic design",
                "elements": ["hair_colors.json", "interesting_ideas.json"],
                "styles": ["Futuristic", "Sci-fi"],
                "poses": ["posture_poses.json"],
                "required": ["advanced", "technological"]
            },
            "Artistic Expression": {
                "base": "artistic expression",
                "elements": ["facial_expressions.json", "styles.json"],
                "styles": ["Artistic", "Expressive"],
                "poses": ["posture_arm.json"],
                "required": ["creative", "expressive"]
            },
            "Dynamic Movement": {
                "base": "dynamic movement",
                "elements": ["posture_movement.json", "facial_expressions.json"],
                "styles": ["Dynamic", "Motion"],
                "poses": ["posture_movement.json"],
                "required": ["fluid", "energetic"]
            },
            "Serene Moment": {
                "base": "serene moment",
                "elements": ["facial_expressions.json", "locations.json"],
                "styles": ["Peaceful", "Calm"],
                "poses": ["posture_basic.json"],
                "required": ["tranquil", "peaceful"]
            },
            "Dramatic Lighting": {
                "base": "dramatic lighting",
                "elements": ["facial_expressions.json", "styles.json"],
                "styles": ["Dramatic", "Cinematic"],
                "poses": ["poses.json"],
                "required": ["dramatic", "moody"]
            },
            "Minimalist Style": {
                "base": "minimalist style",
                "elements": ["styles.json"],
                "styles": ["Minimalist", "Clean"],
                "poses": ["posture_basic.json"],
                "required": ["simple", "clean"]
            },
            "Sensual Lingerie": {
                "base": "sensual lingerie",
                "elements": ["sexual_attire_lingerine.json", "hair_colors.json", "facial_expressions.json"],
                "styles": ["Sensual", "Elegant", "Intimate"],
                "poses": ["posture_basic.json", "poses.json"],
                "required": ["alluring", "sophisticated"]
            },
            "Seductive Pose": {
                "base": "seductive pose",
                "elements": ["sexy_things.json", "facial_expressions.json", "hair_colors.json"],
                "styles": ["Seductive", "Artistic", "Photorealistic"],
                "poses": ["poses.json", "posture_basic.json"],
                "required": ["captivating", "enticing"]
            },
            "Intimate Scene": {
                "base": "intimate scene",
                "elements": ["sexy_things.json", "locations.json", "facial_expressions.json"],
                "styles": ["Romantic", "Intimate", "Soft"],
                "poses": ["posture_hug.json", "poses.json"],
                "required": ["passionate", "tender"]
            },
            "Fetish Fashion": {
                "base": "fetish fashion",
                "elements": ["sexual_attire_bdsm.json", "sexual_attire_miscellaneous.json", "hair_colors.json"],
                "styles": ["Bold", "Dramatic", "Artistic"],
                "poses": ["posture_poses.json", "poses.json"],
                "required": ["striking", "provocative"]
            },
            "Erotic Art": {
                "base": "erotic art",
                "elements": ["nudity.json", "facial_expressions.json", "styles.json"],
                "styles": ["Artistic", "Sensual", "Fine Art"],
                "poses": ["poses.json", "posture_basic.json"],
                "required": ["artistic", "tasteful"]
            },
            "Spicy Romance": {
                "base": "spicy romance",
                "elements": ["sexy_things.json", "facial_expressions.json", "locations.json"],
                "styles": ["Romantic", "Passionate", "Cinematic"],
                "poses": ["posture_hug.json", "poses.json"],
                "required": ["passionate", "romantic"]
            }
        }
        
        template_config = templates.get(template, templates["Elegant Portrait"])
        prompt_parts = [template_config["base"]]
        
        # Add gender-appropriate clothing
        spicy_templates = ["Sensual Lingerie", "Seductive Pose", "Intimate Scene", "Fetish Fashion", "Erotic Art", "Spicy Romance"]
        
        if template in spicy_templates:
            # Use spicy attire for spicy themes
            if template == "Sensual Lingerie":
                clothing_files = ["sexual_attire_lingerine.json"]
            elif template == "Fetish Fashion":
                clothing_files = ["sexual_attire_bdsm.json", "sexual_attire_miscellaneous.json"]
            elif template in ["Intimate Scene", "Erotic Art"]:
                clothing_files = ["sexual_attire_exposure.json", "nudity.json"]
            else:  # Seductive Pose, Spicy Romance
                clothing_files = ["sexual_attire_lingerine.json", "sexy_things.json"]
        else:
            # Regular clothing selection
            if gender == "Female":
                clothing_files = ["attire_female_topwear.json", "attire_female_bottomwear.json"]
            elif gender == "Male":
                clothing_files = ["attire_male_topwear.json", "attire_male_bottomwear.json"]
            else:
                clothing_files = [random.choice(["attire_female_topwear.json", "attire_male_topwear.json"])]
        
        for clothing_file in clothing_files:
            item = self._get_random_from_file(base_path, clothing_file)
            if item:
                prompt_parts.append(item)
        
        # Add template-specific elements
        variation_count = {"Low": 1, "Medium": 2, "High": 3}[variation_level]
        
        for element_file in template_config["elements"][:variation_count]:
            item = self._get_random_from_file(base_path, element_file)
            if item:
                prompt_parts.append(item)
        
        # Add required elements
        prompt_parts.extend(template_config["required"])
        
        # Add style
        if template_config["styles"]:
            style = random.choice(template_config["styles"])
            prompt_parts.append(style)
        
        # Add custom additions
        if custom_additions:
            prompt_parts.append(custom_additions)
        
        final_prompt = ", ".join(prompt_parts)
        
        # Add prefix and suffix if provided
        if prefix and suffix:
            final_prompt = f"{prefix.strip()}, {final_prompt}, {suffix.strip()}"
        elif prefix:
            final_prompt = f"{prefix.strip()}, {final_prompt}"
        elif suffix:
            final_prompt = f"{final_prompt}, {suffix.strip()}"
        
        template_info = f"Template: {template} | Gender: {gender} | Variation: {variation_level}"
        
        print(f"Generated template prompt: {final_prompt}...")
        return (final_prompt, template_info)
    
    def _get_random_from_file(self, base_path: str, filename: str) -> Optional[str]:
        """Get a random item from a JSON file."""
        filepath = os.path.join(base_path, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)
            
            # Get all items from all categories
            all_items = []
            for category, items in data.items():
                if isinstance(items, list):
                    all_items.extend(items)
            
            return random.choice(all_items) if all_items else None
        except (FileNotFoundError, json.JSONDecodeError):
            return None

class PromptEnhancer:
    """Enhance existing prompts with additional elements."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "placeholder": "Enter your base prompt"}),
                "enhancement_type": ([
                    "Random",
                    "Artistic Style",
                    "Lighting Effects",
                    "Mood Enhancement",
                    "Detail Boost",
                    "Composition",
                    "Color Palette",
                    "Technical Quality",
                    "Creative Flair",
                    "Atmospheric",
                    "Professional",
                    "Sensual Enhancement",
                    "Seductive Mood"
                ], {"default": "Random"}),
                "intensity": (["Subtle", "Moderate", "Strong"], {"default": "Moderate"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "custom_enhancements": ("STRING", {"default": "", "multiline": True, "placeholder": "Custom enhancement terms"}),
                "prefix": ("STRING", {"default": "", "multiline": True, "placeholder": "Elements to add at the beginning"}),
                "suffix": ("STRING", {"default": "", "multiline": True, "placeholder": "Elements to add at the end"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "enhancements_applied")
    FUNCTION = "enhance_prompt"
    CATEGORY = "t4ggno/prompt_generation"
    OUTPUT_NODE = False

    def enhance_prompt(self, prompt: str, enhancement_type: str, intensity: str, seed: int, custom_enhancements: str = "", prefix: str = "", suffix: str = "") -> Tuple[str, str]:
        """Enhance a prompt with additional elements."""
        
        if seed != -1:
            random.seed(seed)
        
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Handle random enhancement type selection
        if enhancement_type == "Random":
            available_types = [
                "Artistic Style", "Lighting Effects", "Mood Enhancement", "Detail Boost",
                "Composition", "Color Palette", "Technical Quality", "Creative Flair",
                "Atmospheric", "Professional", "Sensual Enhancement", "Seductive Mood"
            ]
            enhancement_type = random.choice(available_types)
        
        # Define enhancement categories
        enhancements = {
            "Artistic Style": {
                "subtle": ["painterly", "artistic", "stylized"],
                "moderate": ["masterpiece", "award-winning", "professional artwork", "gallery quality"],
                "strong": ["breathtaking masterpiece", "museum quality", "legendary artwork", "visually stunning"]
            },
            "Lighting Effects": {
                "subtle": ["soft lighting", "natural light", "well-lit"],
                "moderate": ["dramatic lighting", "golden hour", "rim lighting", "studio lighting"],
                "strong": ["cinematic lighting", "god rays", "volumetric lighting", "ethereal glow"]
            },
            "Mood Enhancement": {
                "subtle": ["peaceful", "serene", "calm"],
                "moderate": ["emotional", "expressive", "captivating", "engaging"],
                "strong": ["deeply emotional", "soul-stirring", "profoundly moving", "transcendent"]
            },
            "Detail Boost": {
                "subtle": ["detailed", "clear", "sharp"],
                "moderate": ["highly detailed", "intricate", "ultra-detailed", "fine details"],
                "strong": ["incredibly detailed", "hyper-detailed", "meticulously crafted", "extraordinary detail"]
            },
            "Composition": {
                "subtle": ["well-composed", "balanced", "harmonious"],
                "moderate": ["perfect composition", "rule of thirds", "dynamic composition", "visually appealing"],
                "strong": ["flawless composition", "compositional masterpiece", "perfectly balanced", "visually striking"]
            },
            "Color Palette": {
                "subtle": ["vibrant colors", "rich colors", "harmonious palette"],
                "moderate": ["stunning color palette", "perfect color harmony", "vivid and rich", "beautiful colors"],
                "strong": ["extraordinary color palette", "mesmerizing colors", "color perfection", "sublime hues"]
            },
            "Technical Quality": {
                "subtle": ["high quality", "professional", "polished"],
                "moderate": ["ultra-high quality", "8K resolution", "crystal clear", "pristine quality"],
                "strong": ["unparalleled quality", "technical perfection", "flawless execution", "supreme craftsmanship"]
            },
            "Creative Flair": {
                "subtle": ["creative", "unique", "original"],
                "moderate": ["imaginative", "innovative", "artistic vision", "creative genius"],
                "strong": ["revolutionary creativity", "groundbreaking artistry", "visionary masterpiece", "creative transcendence"]
            },
            "Atmospheric": {
                "subtle": ["atmospheric", "ambient", "immersive"],
                "moderate": ["deeply atmospheric", "evocative atmosphere", "immersive environment", "rich atmosphere"],
                "strong": ["breathtaking atmosphere", "otherworldly ambiance", "transcendent environment", "magical atmosphere"]
            },
            "Professional": {
                "subtle": ["professional", "polished", "refined"],
                "moderate": ["studio quality", "commercial grade", "professional photography", "expert craftsmanship"],
                "strong": ["industry standard", "world-class quality", "professional excellence", "masterful execution"]
            },
            "Sensual Enhancement": {
                "subtle": ["alluring", "captivating", "enchanting"],
                "moderate": ["sensual", "seductive", "mesmerizing", "enticing"],
                "strong": ["irresistibly sensual", "breathtakingly seductive", "hypnotically alluring", "utterly captivating"]
            },
            "Seductive Mood": {
                "subtle": ["inviting", "mysterious", "intriguing"],
                "moderate": ["seductive", "provocative", "tantalizing", "bewitching"],
                "strong": ["devastatingly seductive", "irresistibly provocative", "spellbindingly alluring", "intensely magnetic"]
            }
        }
        
        # Get enhancement level
        level = intensity.lower()
        enhancement_list = enhancements.get(enhancement_type, enhancements["Artistic Style"])
        
        # Select enhancements based on intensity
        count = {"subtle": 1, "moderate": 2, "strong": 3}[level]
        selected_enhancements = random.sample(enhancement_list[level], min(count, len(enhancement_list[level])))
        
        # Also add some style elements
        if enhancement_type == "Artistic Style":
            style_item = self._get_random_from_file(base_path, "styles.json")
            if style_item:
                selected_enhancements.append(style_item)
        
        # Add custom enhancements
        if custom_enhancements:
            selected_enhancements.extend([e.strip() for e in custom_enhancements.split(",") if e.strip()])
        
        # Combine with original prompt
        enhanced_prompt = f"{prompt}, {', '.join(selected_enhancements)}"
        
        # Add prefix and suffix if provided
        if prefix and suffix:
            enhanced_prompt = f"{prefix.strip()}, {enhanced_prompt}, {suffix.strip()}"
        elif prefix:
            enhanced_prompt = f"{prefix.strip()}, {enhanced_prompt}"
        elif suffix:
            enhanced_prompt = f"{enhanced_prompt}, {suffix.strip()}"
        
        enhancements_applied = f"Applied {enhancement_type} ({intensity}): {', '.join(selected_enhancements)}"
        
        print(f"Enhanced prompt with {enhancement_type}: {enhanced_prompt}...")
        return (enhanced_prompt, enhancements_applied)
    
    def _get_random_from_file(self, base_path: str, filename: str) -> Optional[str]:
        """Get a random item from a JSON file."""
        filepath = os.path.join(base_path, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)
            
            # Get all items from all categories
            all_items = []
            for category, items in data.items():
                if isinstance(items, list):
                    all_items.extend(items)
            
            return random.choice(all_items) if all_items else None
        except (FileNotFoundError, json.JSONDecodeError):
            return None

class QuickPromptGenerator:
    """Generate complete prompts with one click."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_style": ([
                    "Random",
                    "Amazing Portrait",
                    "Stunning Fashion",
                    "Epic Fantasy",
                    "Artistic Masterpiece",
                    "Cinematic Scene",
                    "Dreamy Atmosphere",
                    "Professional Photo",
                    "Creative Art",
                    "Elegant Beauty",
                    "Dynamic Action",
                    "Serene Moment",
                    "Dramatic Portrait",
                    "Stylish Modern",
                    "Vintage Classic",
                    "Futuristic Vision",
                    "Sensual Beauty",
                    "Seductive Portrait",
                    "Intimate Mood",
                    "Erotic Art"
                ], {"default": "Random"}),
                "subject_type": (["Person", "Character", "Model", "Artist", "Professional"], {"default": "Person"}),
                "gender": (["Any", "Female", "Male"], {"default": "Any"}),
                "quality_level": (["Good", "Great", "Amazing", "Legendary"], {"default": "Amazing"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "prefix": ("STRING", {"default": "", "multiline": True, "placeholder": "Elements to add at the beginning"}),
                "suffix": ("STRING", {"default": "", "multiline": True, "placeholder": "Elements to add at the end"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("complete_prompt", "prompt_breakdown", "settings_used")
    FUNCTION = "generate_complete_prompt"
    CATEGORY = "t4ggno/prompt_generation"
    OUTPUT_NODE = False

    def generate_complete_prompt(self, prompt_style: str, subject_type: str, gender: str, quality_level: str, seed: int, prefix: str = "", suffix: str = "") -> Tuple[str, str, str]:
        """Generate a complete, amazing prompt with one click."""
        
        if seed != -1:
            random.seed(seed)
        
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Handle random prompt style selection
        if prompt_style == "Random":
            available_styles = [
                "Amazing Portrait", "Stunning Fashion", "Epic Fantasy", "Artistic Masterpiece",
                "Cinematic Scene", "Dreamy Atmosphere", "Professional Photo", "Creative Art",
                "Elegant Beauty", "Dynamic Action", "Serene Moment", "Dramatic Portrait",
                "Stylish Modern", "Vintage Classic", "Futuristic Vision", "Sensual Beauty",
                "Seductive Portrait", "Intimate Mood", "Erotic Art"
            ]
            prompt_style = random.choice(available_styles)
        
        # Quality prefixes based on level
        quality_prefixes = {
            "Good": ["high quality", "professional"],
            "Great": ["stunning", "beautiful", "high quality", "professional"],
            "Amazing": ["breathtaking", "masterpiece", "award-winning", "stunning", "incredible"],
            "Legendary": ["legendary masterpiece", "godlike", "transcendent", "otherworldly", "divine perfection"]
        }
        
        # Style-specific elements
        style_configs = {
            "Amazing Portrait": {
                "base": "portrait",
                "quality": ["photorealistic", "hyper-detailed", "8K resolution"],
                "style": ["cinematic lighting", "professional photography"]
            },
            "Stunning Fashion": {
                "base": "fashion photography",
                "quality": ["high fashion", "editorial", "magazine quality"],
                "style": ["dramatic lighting", "studio lighting"]
            },
            "Epic Fantasy": {
                "base": "fantasy character",
                "quality": ["epic", "mythical", "legendary"],
                "style": ["fantasy art", "magical atmosphere"]
            },
            "Artistic Masterpiece": {
                "base": "artistic portrait",
                "quality": ["masterpiece", "museum quality", "fine art"],
                "style": ["painterly", "artistic excellence"]
            },
            "Cinematic Scene": {
                "base": "cinematic portrait",
                "quality": ["movie quality", "hollywood", "cinematic"],
                "style": ["dramatic lighting", "film photography"]
            },
            "Dreamy Atmosphere": {
                "base": "dreamy portrait",
                "quality": ["ethereal", "dreamlike", "surreal"],
                "style": ["soft lighting", "atmospheric"]
            },
            "Professional Photo": {
                "base": "professional portrait",
                "quality": ["studio quality", "commercial", "expert"],
                "style": ["professional lighting", "clean composition"]
            },
            "Creative Art": {
                "base": "creative portrait",
                "quality": ["innovative", "artistic", "unique"],
                "style": ["creative lighting", "artistic vision"]
            },
            "Elegant Beauty": {
                "base": "elegant portrait",
                "quality": ["sophisticated", "refined", "graceful"],
                "style": ["soft lighting", "classical beauty"]
            },
            "Dynamic Action": {
                "base": "dynamic portrait",
                "quality": ["energetic", "powerful", "intense"],
                "style": ["dynamic lighting", "action photography"]
            },
            "Serene Moment": {
                "base": "serene portrait",
                "quality": ["peaceful", "calm", "tranquil"],
                "style": ["natural lighting", "gentle atmosphere"]
            },
            "Dramatic Portrait": {
                "base": "dramatic portrait",
                "quality": ["intense", "powerful", "emotional"],
                "style": ["dramatic lighting", "high contrast"]
            },
            "Stylish Modern": {
                "base": "modern portrait",
                "quality": ["contemporary", "stylish", "trendy"],
                "style": ["modern lighting", "clean aesthetics"]
            },
            "Vintage Classic": {
                "base": "vintage portrait",
                "quality": ["timeless", "classic", "nostalgic"],
                "style": ["vintage lighting", "retro aesthetics"]
            },
            "Futuristic Vision": {
                "base": "futuristic portrait",
                "quality": ["advanced", "cutting-edge", "visionary"],
                "style": ["sci-fi lighting", "futuristic aesthetics"]
            },
            "Sensual Beauty": {
                "base": "sensual portrait",
                "quality": ["alluring", "captivating", "seductive"],
                "style": ["intimate lighting", "sensual atmosphere"]
            },
            "Seductive Portrait": {
                "base": "seductive portrait",
                "quality": ["enticing", "provocative", "mesmerizing"],
                "style": ["dramatic lighting", "seductive mood"]
            },
            "Intimate Mood": {
                "base": "intimate portrait",
                "quality": ["passionate", "romantic", "tender"],
                "style": ["soft lighting", "intimate atmosphere"]
            },
            "Erotic Art": {
                "base": "erotic art portrait",
                "quality": ["artistic", "sensual", "tasteful"],
                "style": ["artistic lighting", "fine art aesthetics"]
            }
        }
        
        config = style_configs.get(prompt_style, style_configs["Amazing Portrait"])
        prompt_parts = []
        breakdown_parts = []
        
        # Add prefix if provided
        if prefix:
            # Clean prefix and ensure it doesn't end with comma
            clean_prefix = prefix.strip()
            if clean_prefix.endswith(','):
                clean_prefix = clean_prefix[:-1].strip()
            prompt_parts.append(clean_prefix)
            breakdown_parts.append(f"Prefix: {clean_prefix}")
        
        # Add quality prefix
        quality_terms = quality_prefixes[quality_level]
        selected_quality = random.choice(quality_terms)
        prompt_parts.append(selected_quality)
        breakdown_parts.append(f"Quality: {selected_quality}")
        
        # Add base style
        prompt_parts.append(config["base"])
        breakdown_parts.append(f"Base: {config['base']}")
        
        # Add subject
        if subject_type != "Person":
            prompt_parts.append(f"of a {subject_type.lower()}")
            breakdown_parts.append(f"Subject: {subject_type}")
        
        # Add character details
        character_parts = []
        
        # Name
        if gender == "Female":
            name = self._get_random_from_file(base_path, "names_female.json")
        elif gender == "Male":
            name = self._get_random_from_file(base_path, "names_male.json")
        else:
            name = self._get_random_from_file(base_path, random.choice(["names_female.json", "names_male.json"]))
        
        if name:
            character_parts.append(name)
            breakdown_parts.append(f"Name: {name}")
        
        # Nationality
        nationality = self._get_random_from_file(base_path, "nationality_by_continent.json")
        if nationality:
            character_parts.append(f"{nationality} person")
            breakdown_parts.append(f"Nationality: {nationality}")
        
        # Hair
        hair_color = self._get_random_from_file(base_path, "hair_colors.json")
        if gender == "Female":
            hair_style = self._get_random_from_file(base_path, "hair_female.json")
        elif gender == "Male":
            hair_style = self._get_random_from_file(base_path, "hair_male.json")
        else:
            hair_style = self._get_random_from_file(base_path, random.choice(["hair_female.json", "hair_male.json"]))
        
        if hair_color and hair_style:
            character_parts.append(f"{hair_color} {hair_style}")
            breakdown_parts.append(f"Hair: {hair_color} {hair_style}")
        
        # Expression
        expression = self._get_random_from_file(base_path, "facial_expressions.json")
        if expression:
            character_parts.append(expression)
            breakdown_parts.append(f"Expression: {expression}")
        
        # Clothing
        spicy_styles = ["Sensual Beauty", "Seductive Portrait", "Intimate Mood", "Erotic Art"]
        
        if prompt_style in spicy_styles:
            # Use spicy attire for spicy styles
            if prompt_style == "Sensual Beauty":
                clothing_files = ["sexual_attire_lingerine.json"]
            elif prompt_style == "Seductive Portrait":
                clothing_files = ["sexual_attire_lingerine.json", "sexy_things.json"]
            elif prompt_style == "Intimate Mood":
                clothing_files = ["sexual_attire_exposure.json", "sexy_things.json"]
            else:  # Erotic Art
                clothing_files = ["nudity.json", "sexual_attire_lingerine.json"]
        else:
            # Regular clothing selection
            if gender == "Female":
                clothing_files = ["attire_female_topwear.json", "attire_female_bottomwear.json"]
            elif gender == "Male":
                clothing_files = ["attire_male_topwear.json", "attire_male_bottomwear.json"]
            else:
                clothing_files = [random.choice(["attire_female_topwear.json", "attire_male_topwear.json"])]
        
        clothing_items = []
        for clothing_file in clothing_files:
            item = self._get_random_from_file(base_path, clothing_file)
            if item:
                clothing_items.append(item)
        
        if clothing_items:
            character_parts.extend(clothing_items)
            breakdown_parts.append(f"Clothing: {', '.join(clothing_items)}")
        
        # Accessories
        accessory = self._get_random_from_file(base_path, "attire_jewelry_and_accessories.json")
        if accessory:
            character_parts.append(accessory)
            breakdown_parts.append(f"Accessory: {accessory}")
        
        # Pose
        pose = self._get_random_from_file(base_path, "poses.json")
        if pose:
            character_parts.append(pose)
            breakdown_parts.append(f"Pose: {pose}")
        
        # Add character parts to prompt
        if character_parts:
            prompt_parts.extend(character_parts)
        
        # Add style-specific quality and style elements
        prompt_parts.extend(config["quality"])
        prompt_parts.extend(config["style"])
        
        # Add artistic style
        artistic_style = self._get_random_from_file(base_path, "styles.json")
        if artistic_style:
            prompt_parts.append(artistic_style)
            breakdown_parts.append(f"Art Style: {artistic_style}")
        
        # Add suffix if provided
        if suffix:
            # Clean suffix and ensure it doesn't start with comma
            clean_suffix = suffix.strip()
            if clean_suffix.startswith(','):
                clean_suffix = clean_suffix[1:].strip()
            prompt_parts.append(clean_suffix)
            breakdown_parts.append(f"Suffix: {clean_suffix}")
        
        # Final assembly
        complete_prompt = ", ".join(prompt_parts)
        prompt_breakdown = " | ".join(breakdown_parts)
        settings_used = f"Style: {prompt_style} | Subject: {subject_type} | Gender: {gender} | Quality: {quality_level}"
        
        print(f"Generated complete prompt: {complete_prompt}...")
        return (complete_prompt, prompt_breakdown, settings_used)
    
    def _get_random_from_file(self, base_path: str, filename: str) -> Optional[str]:
        """Get a random item from a JSON file."""
        filepath = os.path.join(base_path, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)
            
            # Get all items from all categories
            all_items = []
            for category, items in data.items():
                if isinstance(items, list):
                    all_items.extend(items)
            
            return random.choice(all_items) if all_items else None
        except (FileNotFoundError, json.JSONDecodeError):
            return None

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "RandomJSONSelector": RandomJSONSelector,
    "SmartPromptBuilder": SmartPromptBuilder,
    "PromptTemplateManager": PromptTemplateManager,
    "PromptEnhancer": PromptEnhancer,
    "QuickPromptGenerator": QuickPromptGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomJSONSelector": "Random JSON Selector",
    "SmartPromptBuilder": "Smart Prompt Builder",
    "PromptTemplateManager": "Prompt Template Manager",
    "PromptEnhancer": "Prompt Enhancer",
    "QuickPromptGenerator": "Quick Prompt Generator",
}
