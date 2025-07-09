from .base_imports import *
from typing import Dict, List, Tuple, Any, Optional

class TextSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "active": (["Text 1", "Text 2", "Text 3", "Text 4"], {"default": "Text 1"}),
                "text1": ("STRING", {"default": "", "multiline": True}),
                "text2": ("STRING", {"default": "", "multiline": True}),
                "text3": ("STRING", {"default": "", "multiline": True}),
                "text4": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "get_text"
    CATEGORY = "t4ggno/utils"
    OUTPUT_NODE = False

    def get_text(self, active: str, text1: str, text2: str, text3: str, text4: str) -> Tuple[str]:
        """Switch between multiple text inputs based on the active selection."""
        text_map = {
            "Text 1": text1,
            "Text 2": text2,
            "Text 3": text3,
            "Text 4": text4
        }
        
        selected_text = text_map.get(active, text1)
        print(f"Text Switch: Selected '{active}' -> '{selected_text[:50]}{'...' if len(selected_text) > 50 else ''}'")
        
        return (selected_text,)

class TextReplacementProcessor:
    """Helper class to handle text replacement operations."""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.global_whitelist: List[Dict[str, Any]] = []
        self.global_blacklist: List[Dict[str, Any]] = []
    
    def load_json_data(self, filename: str) -> Dict[str, Any]:
        """Load JSON data from file with error handling."""
        try:
            filepath = os.path.join(self.base_path, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Warning: JSON file not found: {filename}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file {filename}: {e}")
            return {}
    
    def process_global_filters(self, text: str) -> str:
        """Process global whitelist and blacklist filters."""
        matches = re.findall(r"(\[(Blacklist|Whitelist)\:(\w+)\:([\w\,]+?)\])", text)
        
        for match in matches:
            complete_string, filter_type, trigger_word, categories = match
            categories_list = categories.split(",")
            
            target_list = self.global_whitelist if filter_type == "Whitelist" else self.global_blacklist
            
            # Check if trigger word already exists
            existing_entry = next((entry for entry in target_list if entry["triggerWord"] == trigger_word), None)
            
            if existing_entry:
                existing_entry["categories"] = list(set(existing_entry["categories"] + categories_list))
            else:
                new_entry = {"triggerWord": trigger_word, "categories": categories_list}
                target_list.append(new_entry)
                print(f"Added to {filter_type}: {new_entry}")
            
            text = text.replace(complete_string, "")
        
        return text
    
    def apply_filters(self, trigger_word: str, allowed_categories: List[str]) -> List[str]:
        """Apply global whitelist and blacklist filters to categories."""
        # Apply whitelist filters
        for whitelist_entry in self.global_whitelist:
            if whitelist_entry["triggerWord"] == trigger_word:
                allowed_categories = list(set(allowed_categories).intersection(whitelist_entry["categories"]))
        
        # Apply blacklist filters
        for blacklist_entry in self.global_blacklist:
            if blacklist_entry["triggerWord"] == trigger_word:
                allowed_categories = list(set(allowed_categories).difference(blacklist_entry["categories"]))
        
        return allowed_categories
    
    def replace_using_json(self, text: str, trigger_word: str, json_filename: str, quantity: int = 1) -> str:
        """Replace trigger words using JSON data."""
        pattern = rf"(\[{re.escape(trigger_word)}(?:\:([\w\,]+?))?\])"
        
        if not re.search(pattern, text):
            return text
            
        json_data = self.load_json_data(json_filename)
        if not json_data:
            return text
            
        json_categories = list(json_data.keys())
        matches = re.findall(pattern, text)
        
        for match in matches:
            complete_string, category = match
            
            # Determine allowed categories
            if category:
                allowed_categories = [cat for cat in category.split(",") if cat in json_categories]
            else:
                allowed_categories = json_categories
            
            # Apply filters
            allowed_categories = self.apply_filters(trigger_word, allowed_categories)
            
            if not allowed_categories:
                text = text.replace(complete_string, "")
                continue
            
            # Generate replacement values
            replacement_values = []
            for _ in range(quantity):
                random_category = allowed_categories[numpy.random.randint(0, len(allowed_categories))]
                category_items = json_data[random_category]
                random_value = category_items[numpy.random.randint(0, len(category_items))]
                replacement_values.append(random_value)
            
            replacement_text = ", ".join(replacement_values) if len(replacement_values) > 1 else replacement_values[0]
            print(f"Replacing '{complete_string}' with '{replacement_text}'")
            
            text = text.replace(complete_string, replacement_text, 1)
        
        return text
    
    def replace_using_array(self, text: str, trigger_word: str, array: List[str], quantity: int = 1) -> str:
        """Replace trigger words using array data."""
        pattern = rf"(\[{re.escape(trigger_word)}\])"
        
        if not re.search(pattern, text):
            return text
        
        # Apply filters
        filtered_array = self.apply_filters(trigger_word, array)
        
        if not filtered_array:
            return text.replace(f"[{trigger_word}]", "")
        
        matches = re.findall(pattern, text)
        
        for match in matches:
            replacement_values = []
            for _ in range(quantity):
                random_value = filtered_array[numpy.random.randint(0, len(filtered_array))]
                replacement_values.append(random_value)
            
            replacement_text = ", ".join(replacement_values) if len(replacement_values) > 1 else replacement_values[0]
            print(f"Replacing '{match}' with '{replacement_text}'")
            
            text = text.replace(match, replacement_text, 1)
        
        return text

class TextReplacer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
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

    def __init__(self):
        self.processor = TextReplacementProcessor(dirPath)

    def clean_text(self, text: str) -> str:
        """Clean text by removing empty lines, comments, and normalizing whitespace."""
        # Remove empty lines and trim
        text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])
        
        # Remove multiple commas in a row
        text = re.sub(r",+", ",", text)
        
        # Remove comments (# and ''' ''')
        text = re.sub(r"#.*", "", text)
        text = re.sub(r"'''(?:.|\n)*?'''", "", text)
        
        # Remove empty lines again after comment removal
        text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])
        
        return text

    def process_pick_one(self, text: str) -> str:
        """Process [PickOne:Option1,Option2,...] replacements."""
        while True:
            pick_one_matches = re.findall(r"(\[PickOne:([^\]]+)\])", text)
            if not pick_one_matches:
                break
            
            for match in pick_one_matches:
                complete_string, options_str = match
                options = [opt.strip() for opt in options_str.split(",")]
                
                if options:
                    random_choice = options[numpy.random.randint(0, len(options))]
                    print(f"PickOne: Replacing '{complete_string}' with '{random_choice}'")
                    text = text.replace(complete_string, random_choice, 1)
        
        return text

    def process_pick_multiple(self, text: str) -> str:
        """Process [PickMultiple:Option1,Option2,...] replacements."""
        pick_multiple_matches = re.findall(r"(\[PickMultiple:([^\]]+)\])", text)
        
        for match in pick_multiple_matches:
            complete_string, options_str = match
            options = [opt.strip() for opt in options_str.split(",")]
            
            if options:
                quantity = numpy.random.randint(1, len(options) + 1)
                selected_indices = numpy.random.choice(len(options), quantity, replace=False)
                selected_options = [options[i] for i in selected_indices]
                
                replacement = ", ".join(selected_options)
                print(f"PickMultiple: Replacing '{complete_string}' with '{replacement}'")
                text = text.replace(complete_string, replacement, 1)
        
        return text

    def process_basic_replacements(self, text: str) -> str:
        """Process basic array-based replacements."""
        replacements = {
            "TimeOfDay": ["morning", "afternoon", "evening", "night", "sunset", "sunrise"],
            "Weather": ["sunny", "cloudy", "rainy", "snowy"],
            "FacialExpression": ["happy", "sad", "angry", "surprised", "disgusted", "scared", "cry", "laugh", "smile", "light smile"],
            "Color": ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray", "grey", "gold", "silver", "bronze", "copper"]
        }
        
        for trigger_word, options in replacements.items():
            text = self.processor.replace_using_array(text, trigger_word, options)
        
        return text

    def process_json_replacements(self, text: str) -> str:
        """Process JSON-based replacements."""
        json_replacements = {
            "Nudity": "nudity.json",
            "Location": "locations.json",
            "Scenario": "scenarios.json",
            "HairColor": "hair_colors.json",
            "AttireEyewear": "attire_eyewear.json",
            "AttireSleeve": "attire_sleeves.json",
            "AttireJewelryAndAccessories": "attire_jewelry_and_accessories.json",
            "AttireStylesAndPatterns": "attire_styles_and_patterns.json",
            "AttireFemaleBodysuit": "attire_female_bodysuits.json",
            "AttireFemaleBottomwear": "attire_female_bottomwear.json",
            "AttireFemaleBra": "attire_female_bra.json",
            "AttireFemaleFootwear": "attire_female_footwear.json",
            "AttireFemaleHeadwear": "attire_female_headwear.json",
            "AttireFemaleLegwear": "attire_female_legwear.json",
            "AttireFemalePanties": "attire_female_panties.json",
            "AttireFemaleSwimsuit": "attire_female_swimsuits.json",
            "AttireFemaleTopwear": "attire_female_topwear.json",
            "AttireFemaleTraditionalClothing": "attire_female_traditional_clothing.json",
            "AttireFemaleUniformsAndCostumes": "attire_female_uniforms_and_costumes.json",
            "AttireMaleBodysuit": "attire_male_bodysuits.json",
            "AttireMaleBottomwear": "attire_male_bottomwear.json",
            "AttireMaleFootwear": "attire_male_footwear.json",
            "AttireMaleHeadwear": "attire_male_headwear.json",
            "AttireMaleLegwear": "attire_male_legwear.json",
            "AttireMaleSwimsuit": "attire_male_swimsuits.json",
            "AttireMaleTopwear": "attire_male_topwear.json",
            "AttireMaleTraditionalClothing": "attire_male_traditional_clothing.json",
            "AttireMaleUniformsAndCostumes": "attire_male_uniforms_and_costumes.json",
            "Fetishes": "fetishes.json",
            "SexToys": "sex_toys.json",
            "SexToysBondage": "sex_toys_bondage.json",
            "SexPositions": "sex_positions.json",
            "SexualActs": "sexual_acts.json",
            "SexualAttireBDSM": "sexual_attire_bdsm.json",
            "SexualAttireExposure": "sexual_attire_exposure.json",
            "SexualAttireLingerine": "sexual_attire_lingerine.json",
            "SexualAttireMiscellaneous": "sexual_attire_miscellaneous.json",
            "PostureArm": "posture_arm.json",
            "PostureBasic": "posture_basic.json",
            "PostureCarrying": "posture_carrying.json",
            "PostureHead": "posture_head.json",
            "PostureHips": "posture_hips.json",
            "PostureHug": "posture_hug.json",
            "PostureLeg": "posture_leg.json",
            "PostureMovement": "posture_movement.json",
            "PostureMultipleCharacter": "posture_multiple_character.json",
            "PostureOther": "posture_other.json",
            "PosturePoses": "posture_poses.json",
            "PostureTorso": "posture_torso.json",
            "NationalityByContinent": "nationality_by_continent.json",
            "NationalityByEthnic": "nationality_by_ethnic.json",
            "HairFemale": "hair_female.json",
            "HairMale": "hair_male.json",
            "SexyThings": "sexy_things.json",
            "Style": "styles.json",
            "InterestingIdeas": "interesting_ideas.json",
            "Material": "materials.json",
        }
        
        for trigger_word, filename in json_replacements.items():
            text = self.processor.replace_using_json(text, trigger_word, filename)
        
        return text

    def process_name_replacements(self, text: str) -> str:
        """Process name replacements based on gender context."""
        if not re.search(r"(\[Name(?:\:([\w\,]+?))?\])", text):
            return text
        
        is_male = bool(re.search(r"(?:^|\ |,)(male|boy|man)", text))
        is_female = bool(re.search(r"(?:^|\ |,)(female|girl|woman)", text))
        
        print(f"Name replacement - Is Male: {is_male}, Is Female: {is_female}")
        
        if is_male:
            filename = "names_male.json"
        elif is_female:
            filename = "names_female.json"
        else:
            # Default to female names if no gender context
            filename = "names_female.json"
        
        return self.processor.replace_using_json(text, "Name", filename)

    def process_conditional_statements(self, text: str) -> str:
        """Process conditional If statements."""
        if_pattern = r"(\[\[If:(Contains):([\w\ ]+?):(Add):([\w\ \[\]\:\,]+?)\]\])"
        
        while re.search(if_pattern, text):
            matches = re.findall(if_pattern, text)
            something_replaced = False
            
            for match in matches:
                complete_string, _, contains, _, value = match
                
                if contains in text:
                    text = text.replace(complete_string, value, 1)
                    something_replaced = True
                else:
                    text = text.replace(complete_string, "", 1)
            
            if not something_replaced:
                break
        
        return text

    def process_lora_replacements(self, text: str) -> str:
        """Process LoRA regex replacements."""
        lora_pattern = r"<RE\:(.+?)(?:\:(-?[0-9]+(?:\.[0-9]*)?)|(?:\:(-?[0-9]+(?:\.[0-9]*)?|))(?:\:(-?[0-9]+(?:\.[0-9]*)?|)))?>"""
        all_loras_regex = re.findall(lora_pattern, text)
        
        if not all_loras_regex:
            return text
        
        try:
            available_loras = folder_paths.get_filename_list("loras")
            print(f"Available LoRAs: {len(available_loras)} found")
            
            for lora_match in all_loras_regex:
                regex_pattern = lora_match[0]
                print(f"Processing LoRA regex: {regex_pattern}")
                
                try:
                    compiled_regex = re.compile(regex_pattern)
                    matching_loras = [lora for lora in available_loras if compiled_regex.match(lora)]
                    
                    if not matching_loras:
                        print(f"No LoRAs match regex: {regex_pattern}")
                        text = text.replace(f"<RE:{regex_pattern}>", "")
                        continue
                    
                    random_lora = matching_loras[numpy.random.randint(0, len(matching_loras))]
                    random_lora = re.sub(r"\.[a-zA-Z0-9]+$", "", random_lora)  # Remove extension
                    
                    print(f"Selected LoRA: {random_lora}")
                    text = text.replace(f"<RE:{regex_pattern}", f"<{random_lora}")
                    
                except re.error as e:
                    print(f"Invalid regex pattern '{regex_pattern}': {e}")
                    text = text.replace(f"<RE:{regex_pattern}>", "")
                    
        except Exception as e:
            print(f"Error processing LoRA replacements: {e}")
        
        return text

    def final_cleanup(self, text: str) -> str:
        """Perform final text cleanup."""
        # Remove empty lines and trim
        text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])
        
        # Remove multiple spaces
        text = re.sub(r"\s+", " ", text)
        
        # Remove multiple commas
        text = re.sub(r",+", ",", text)
        
        # Remove leading/trailing commas
        text = re.sub(r"^,|,$", "", text.strip())
        
        return text

    def process_random_replacements(self, text: str) -> str:
        """Process complex [Random] replacements with conditional logic."""
        random_pattern = r"(\[Random(?:\:[\w]+)*\])"
        
        if not re.search(random_pattern, text):
            return text
        
        random_matches = re.findall(random_pattern, text)
        
        for match in random_matches:
            complete_string = match
            new_text = ""
            
            # Check for person-related flags
            has_person = (numpy.random.randint(0, 3) == 0 or 
                         re.search(r"\[Random.*\:HasPerson.*\]", text) is not None) and \
                        re.search(r"\[Random.*\:HasNoPerson.*\]", text) is None
            
            if has_person:
                new_text += "[NationalityByEthnic] "
                
                # Determine gender
                is_female = (numpy.random.randint(0, 2) == 0 or 
                           re.search(r"\[Random.*\:IsFemale.*\]", text) is not None) and \
                          re.search(r"\[Random.*\:IsMale.*\]", text) is None
                
                if is_female:
                    new_text += "woman,[Name],[HairFemale],"
                    
                    # Add sexy elements if specified
                    if (numpy.random.randint(0, 2) == 0 or 
                        re.search(r"\[Random.*\:SexyOnly.*\]", text) is not None):
                        new_text += "[SexyThings],"
                        
                        # Add nudity elements
                        if numpy.random.randint(0, 2) == 0:
                            new_text += "[Nudity],"
                        
                        # Add clothing elements
                        if numpy.random.randint(0, 2) == 0:
                            if numpy.random.randint(0, 2) == 0:
                                new_text += "[AttireFemaleTopwear],"
                            if numpy.random.randint(0, 2) == 0:
                                new_text += "[AttireFemaleBottomwear],"
                        else:
                            new_text += "[AttireFemaleTopwear],"
                        
                        # Add accessories
                        if numpy.random.randint(0, 2) == 0:
                            new_text += "[AttireFemaleHeadwear],"
                        if numpy.random.randint(0, 2) == 0:
                            new_text += "[AttireFemaleFootwear],"
                    else:
                        # Regular clothing
                        if numpy.random.randint(0, 2) == 0:
                            if numpy.random.randint(0, 2) == 0:
                                new_text += "[AttireFemaleTopwear],"
                            if numpy.random.randint(0, 2) == 0:
                                new_text += "[AttireFemaleBottomwear],"
                        else:
                            new_text += "[AttireFemaleTopwear],"
                        
                        if numpy.random.randint(0, 2) == 0:
                            new_text += "[AttireFemaleHeadwear],"
                        if numpy.random.randint(0, 2) == 0:
                            new_text += "[AttireFemaleFootwear],"
                else:
                    # Male character
                    new_text += "man,[Name],[HairMale],"
                    
                    if (numpy.random.randint(0, 2) == 0 or 
                        re.search(r"\[Random.*\:SexyOnly.*\]", text) is not None):
                        new_text += "[SexyThings],"
                        
                        if numpy.random.randint(0, 2) == 0:
                            new_text += "[Nudity],"
                        
                        if numpy.random.randint(0, 2) == 0:
                            if numpy.random.randint(0, 2) == 0:
                                new_text += "[AttireMaleTopwear],"
                            if numpy.random.randint(0, 2) == 0:
                                new_text += "[AttireMaleBottomwear],"
                        else:
                            new_text += "[AttireMaleTopwear],"
                        
                        if numpy.random.randint(0, 2) == 0:
                            new_text += "[AttireMaleHeadwear],"
                        if numpy.random.randint(0, 2) == 0:
                            new_text += "[AttireMaleFootwear],"
                    else:
                        # Regular male clothing
                        if numpy.random.randint(0, 2) == 0:
                            if numpy.random.randint(0, 2) == 0:
                                new_text += "[AttireMaleTopwear],"
                            if numpy.random.randint(0, 2) == 0:
                                new_text += "[AttireMaleBottomwear],"
                        else:
                            new_text += "[AttireMaleTopwear],"
                        
                        if numpy.random.randint(0, 2) == 0:
                            new_text += "[AttireMaleHeadwear],"
                        if numpy.random.randint(0, 2) == 0:
                            new_text += "[AttireMaleFootwear],"
                
                # Add common person attributes
                if numpy.random.randint(0, 2) == 0:
                    new_text += "[FacialExpression],"
                if numpy.random.randint(0, 2) == 0:
                    new_text += "[HairColor] hair,"
                if numpy.random.randint(0, 2) == 0:
                    new_text += "[PostureBasic],"
                if numpy.random.randint(0, 2) == 0:
                    new_text += "[Color] eyes,"
                
                new_text += "(imperfections:0.3),(freckles:0.3),"
            
            # Add environmental elements
            if numpy.random.randint(0, 2) == 0:
                new_text += "[TimeOfDay],"
            
            # Add multiple locations
            location_quantity = numpy.random.randint(1, 4)
            for _ in range(location_quantity):
                if numpy.random.randint(0, 2) == 0:
                    new_text += "[Location],"
            
            # Add multiple scenarios
            scenario_quantity = numpy.random.randint(1, 4)
            for _ in range(scenario_quantity):
                if numpy.random.randint(0, 2) == 0:
                    new_text += "[Scenario],"
            
            # Add weather
            if numpy.random.randint(0, 2) == 0:
                new_text += "[Weather],"
            
            # Add style and interesting ideas
            new_text += "[Style],"
            if numpy.random.randint(0, 2) == 0:
                new_text += "[InterestingIdeas],"
            
            # Replace in text
            text = text.replace(complete_string, new_text, 1)
        
        return text

    def replace_text(self, text: str) -> Tuple[str]:
        """Main text replacement function."""
        print("=" * 50)
        print("Text Replacer - Starting Processing")
        print(f"Input text length: {len(text)}")
        
        try:
            # Initial cleanup
            text = self.clean_text(text)
            print(f"After initial cleanup: {text[:100]}...")
            
            # Process global filters
            text = self.processor.process_global_filters(text)
            print(f"Global whitelist: {self.processor.global_whitelist}")
            print(f"Global blacklist: {self.processor.global_blacklist}")
            
            # Process PickOne first (before other replacements)
            text = self.process_pick_one(text)
            text = self.clean_text(text)
            
            # Main replacement loop
            max_iterations = 10
            for iteration in range(max_iterations):
                initial_text = text
                
                # Process various replacement types
                text = self.process_basic_replacements(text)
                text = self.process_json_replacements(text)
                text = self.process_name_replacements(text)
                text = self.process_pick_multiple(text)
                text = self.process_random_replacements(text)
                text = self.process_conditional_statements(text)
                
                # If no changes were made, break the loop
                if text == initial_text:
                    break
                
                print(f"Iteration {iteration + 1} completed")
            
            # Clean up unmatched If statements
            while True:
                unmatched_if_pattern = r"(\[\[If:.*?\]\])"
                if not re.search(unmatched_if_pattern, text):
                    break
                
                matches = re.findall(unmatched_if_pattern, text)
                for match in matches:
                    text = text.replace(match, "", 1)
            
            # Process LoRA replacements
            text = self.process_lora_replacements(text)
            
            # Final cleanup
            text = self.final_cleanup(text)
            
            print(f"Final processed text: {text}")
            print("Text Replacer - Processing Complete")
            
            return (text,)
            
        except Exception as e:
            print(f"Error in text replacement: {e}")
            import traceback
            traceback.print_exc()
            return (text,)

    def IS_CHANGED(cls, **kwargs):
        return True

# Export the node classes
NODE_CLASS_MAPPINGS = {
    "TextSwitch": TextSwitch,
    "TextReplacer": TextReplacer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextSwitch": "Switch Text",
    "TextReplacer": "Replace Text",
}
