from .base_imports import *
from typing import Dict, List, Tuple, Optional, Any, Union

class LoraLoaderFromPrompt:
    def __init__(self):
        self.loaded_loras: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    FUNCTION = "load_loras"
    CATEGORY = "t4ggno/loaders"

    def load_loras(self, model: Any, clip: Any, prompt: str) -> Tuple[Any, Any, str]:
        print("=============================")
        print("== Load Loras from Prompt ==")

        available_loras = folder_paths.get_filename_list("loras")
        all_loras = self._extract_loras_from_prompt(prompt)
        prompt = self._remove_loras_from_prompt(prompt)
        
        processed_loras = self._process_lora_data(all_loras)
        resolved_loras = self._resolve_lora_names(processed_loras, available_loras)
        deduplicated_loras = self._deduplicate_loras(resolved_loras)
        
        return self._apply_loras(model, clip, deduplicated_loras, available_loras, prompt)

    def _extract_loras_from_prompt(self, prompt: str) -> List[Tuple[str, str, str]]:
        """Extract lora information from prompt using regex"""
        # Improved regex pattern to handle lora tags properly
        # Matches: <lora:name:model_strength:clip_strength>, <lora:name:model_strength>, <lora:name::clip_strength>, <lora:name>
        # Also matches: <name:model_strength:clip_strength>, <name:model_strength>, <name::clip_strength>, <name>
        lora_pattern = r"<(?:lora:)?([\w\-. \\]+?)(?::(-?[0-9]+(?:\.[0-9]*)?)?)?(?::(-?[0-9]+(?:\.[0-9]*)?)?)?>"
        matches = re.findall(lora_pattern, prompt)
        
        # Debug output
        for match in matches:
            print(f"DEBUG: Extracted lora match: {match}")
        
        return matches

    def _remove_loras_from_prompt(self, prompt: str) -> str:
        """Remove lora tags from prompt"""
        lora_pattern = r"<(?:lora:)?([\w\-. \\]+?)(?::(-?[0-9]+(?:\.[0-9]*)?)?)?(?::(-?[0-9]+(?:\.[0-9]*)?)?)?>"
        return re.sub(lora_pattern, "", prompt)

    def _process_lora_data(self, lora_matches: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
        """Convert lora matches to structured data with proper strength values"""
        processed_loras = []
        
        for match in lora_matches:
            name, model_strength, clip_strength = match
            
            # Debug output
            print(f"DEBUG: Processing lora - name: '{name}', model_strength: '{model_strength}', clip_strength: '{clip_strength}'")
            
            # Handle the different regex group combinations
            model_str = float(model_strength) if model_strength else 1.0
            clip_str = float(clip_strength) if clip_strength else model_str
            
            processed_loras.append({
                "name": name,
                "model_strength": model_str,
                "clip_strength": clip_str
            })
        
        return processed_loras

    def _resolve_lora_names(self, loras: List[Dict[str, Any]], available_loras: List[str]) -> List[Dict[str, Any]]:
        """Resolve lora names to match available files"""
        for lora in loras:
            current_name = lora["name"]
            
            if current_name in available_loras:
                continue
                
            matching_loras = [
                available_lora for available_lora in available_loras
                if current_name in available_lora
            ]
            
            if matching_loras:
                lora["name"] = numpy.random.choice(matching_loras)
                print(f"Resolved lora '{current_name}' to '{lora['name']}'")
            else:
                print(f"Lora not available: {current_name}")
        
        return loras

    def _deduplicate_loras(self, loras: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate loras by averaging their strengths"""
        lora_dict = {}
        
        for lora in loras:
            name = lora["name"]
            if name in lora_dict:
                existing = lora_dict[name]
                existing["model_strength"] += lora["model_strength"]
                existing["clip_strength"] += lora["clip_strength"]
                existing["count"] += 1
            else:
                lora_dict[name] = {
                    "name": name,
                    "model_strength": lora["model_strength"],
                    "clip_strength": lora["clip_strength"],
                    "count": 1
                }
        
        # Calculate averages
        deduplicated_loras = []
        for lora_data in lora_dict.values():
            count = lora_data["count"]
            deduplicated_loras.append({
                "name": lora_data["name"],
                "model_strength": lora_data["model_strength"] / count,
                "clip_strength": lora_data["clip_strength"] / count
            })
        
        return deduplicated_loras

    def _apply_loras(self, model: Any, clip: Any, loras: List[Dict[str, Any]], 
                    available_loras: List[str], prompt: str) -> Tuple[Any, Any, str]:
        """Apply loras to model and clip"""
        valid_loras = [lora for lora in loras if lora["name"] in available_loras]
        
        if not valid_loras:
            return (model, clip, prompt)
        
        for lora in valid_loras:
            try:
                model, clip, prompt = self._load_single_lora(model, clip, lora, prompt)
            except Exception as e:
                print(f"Error loading lora {lora['name']}: {e}")
                continue
        
        print(f"Text prompt after loading loras: {prompt}")
        return (model, clip, prompt)

    def _load_single_lora(self, model: Any, clip: Any, lora: Dict[str, Any], prompt: str) -> Tuple[Any, Any, str]:
        """Load a single lora file"""
        lora_path = folder_paths.get_full_path("loras", lora["name"])
        loaded_lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        
        model_strength = lora["model_strength"]
        clip_strength = lora["clip_strength"]
        
        model, clip = comfy.sd.load_lora_for_models(model, clip, loaded_lora, model_strength, clip_strength)
        print(f"Loaded lora: {lora_path} with model strength: {model_strength} and clip strength: {clip_strength}")
        
        return model, clip, self._process_trigger_words(lora_path, prompt)

    def _process_trigger_words(self, lora_path: str, prompt: str) -> str:
        """Extract and add trigger words from lora metadata if needed"""
        try:
            metadata = self._extract_lora_metadata(lora_path)
            if not metadata:
                return prompt
                
            trigger_words = self._get_trigger_words(metadata)
            if not trigger_words:
                return prompt
                
            if not any(word in prompt for word in trigger_words):
                random_trigger = numpy.random.choice(trigger_words)
                prompt += f" {random_trigger}"
                print(f"Added trigger word: '{random_trigger}' to prompt")
            
        except Exception as e:
            print(f"Error processing trigger words for {lora_path}: {e}")
        
        return prompt

    def _extract_lora_metadata(self, lora_path: str) -> Optional[Dict[str, Any]]:
        """Extract metadata from lora file"""
        try:
            with open(lora_path, 'r', encoding='ansi') as f:
                metadata_str = ""
                brace_count = 0
                
                for line in f:
                    for char in line:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            
                        if brace_count > 0:
                            metadata_str += char
                            
                        if brace_count == 0 and metadata_str:
                            return json.loads(metadata_str)
                            
        except Exception:
            pass
        
        return None

    def _get_trigger_words(self, metadata: Dict[str, Any]) -> List[str]:
        """Extract trigger words from metadata"""
        try:
            dataset_dirs = metadata.get("__metadata__", {}).get("ss_dataset_dirs")
            if not dataset_dirs:
                return []
                
            dataset_data = json.loads(dataset_dirs)
            trigger_keys = [key for key in dataset_data.keys() if re.match(r"^\d+_.+$", key)]
            
            return [key.split("_", 1)[1] for key in trigger_keys]
            
        except Exception:
            return []

class CheckpointLoaderByName:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "type": (["Detect (Manual)", "Detect (Random)"], {"default": "Detect (Manual)"}),
                "checkpoint_name": ("STRING", {"default": ""}),
                "fallback": (comfy_paths.get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "NAME_STRING")
    FUNCTION = "load_checkpoint"
    CATEGORY = "t4ggno/utils"
    OUTPUT_NODE = False

    def load_checkpoint(self, type_: str, checkpoint_name: str, fallback: str) -> Tuple[Any, Any, Any, str]:
        print("=============================")
        print("== Load Checkpoint by Name ==")
        print(f"Type: {type_}")

        available_checkpoints = comfy_paths.get_filename_list("checkpoints")

        if checkpoint_name:
            checkpoint_path = self._find_checkpoint(checkpoint_name, available_checkpoints)
            if checkpoint_path:
                return self._load_checkpoint_from_path(checkpoint_path, checkpoint_name)

        return self._load_fallback_checkpoint(type_, available_checkpoints, fallback)

    def _find_checkpoint(self, checkpoint_name: str, available_checkpoints: List[str]) -> Optional[str]:
        """Find checkpoint using multiple strategies"""
        # Strategy 1: Exact match
        checkpoint_path = comfy_paths.get_full_path("checkpoints", checkpoint_name)
        if checkpoint_path and os.path.isfile(checkpoint_path):
            print(f"Load checkpoint [Exact Match]: {checkpoint_name}")
            return checkpoint_path

        # Strategy 2: Match without extension
        name_without_ext = os.path.splitext(checkpoint_name)[0]
        matching_checkpoints = [cp for cp in available_checkpoints if name_without_ext in cp]
        if matching_checkpoints:
            checkpoint_path = comfy_paths.get_full_path("checkpoints", matching_checkpoints[0])
            print(f"Load checkpoint [Without Extension]: {name_without_ext}")
            return checkpoint_path

        # Strategy 3: Match without version and extension
        name_without_version = re.sub(r"-V\d+\.\d+", "", name_without_ext)
        matching_checkpoints = [cp for cp in available_checkpoints if name_without_version in cp]
        if matching_checkpoints:
            checkpoint_path = comfy_paths.get_full_path("checkpoints", matching_checkpoints[0])
            print(f"Load checkpoint [Without Version]: {name_without_version}")
            return checkpoint_path

        print("Missing or invalid checkpoint name")
        return None

    def _load_checkpoint_from_path(self, checkpoint_path: str, original_name: str) -> Tuple[Any, Any, Any, str]:
        """Load checkpoint from given path"""
        result = comfy.sd.load_checkpoint_guess_config(
            checkpoint_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=comfy_paths.get_folder_paths("embeddings")
        )
        
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        return (*result, checkpoint_name)

    def _load_fallback_checkpoint(self, type_: str, available_checkpoints: List[str], fallback: str) -> Tuple[Any, Any, Any, str]:
        """Load fallback checkpoint based on type"""
        if type_ == "Detect (Random)":
            selected_checkpoint = numpy.random.choice(available_checkpoints)
            print(f"Load checkpoint [Random]: {selected_checkpoint}")
        else:
            selected_checkpoint = fallback
            print(f"Load checkpoint [Fallback]: {selected_checkpoint}")

        checkpoint_path = comfy_paths.get_full_path("checkpoints", selected_checkpoint)
        return self._load_checkpoint_from_path(checkpoint_path, selected_checkpoint)

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        return float("nan")

class RandomCheckpointLoader:
    def __init__(self):
        self.checkpoint_usage: Dict[str, int] = defaultdict(int)

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
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

    def load_random_checkpoint(self, whitelist_regex: str, blacklist_regex: str, usage_limit: int) -> Tuple[Any, Any, Any, str]:
        print("=============================")
        print("== Loading random checkpoint")

        available_checkpoints = self._get_available_checkpoints()
        filtered_checkpoints = self._filter_checkpoints(available_checkpoints, whitelist_regex, blacklist_regex)
        eligible_checkpoints = self._get_eligible_checkpoints(filtered_checkpoints, usage_limit)

        return self._load_checkpoint(eligible_checkpoints)

    def _get_available_checkpoints(self) -> List[str]:
        """Get all available checkpoints and filter out None/empty values"""
        all_checkpoints = comfy_paths.get_filename_list("checkpoints")
        valid_checkpoints = [cp for cp in all_checkpoints if cp]
        
        print(f"Total checkpoints found: {len(all_checkpoints)}")
        print(f"Valid checkpoints: {len(valid_checkpoints)}")
        
        return valid_checkpoints

    def _filter_checkpoints(self, checkpoints: List[str], whitelist_regex: str, blacklist_regex: str) -> List[str]:
        """Filter checkpoints using whitelist and blacklist regex patterns"""
        # Process whitelist patterns
        if whitelist_regex.strip():
            whitelist_patterns = [
                re.compile(pattern.strip()) 
                for pattern in whitelist_regex.split('\n') 
                if pattern.strip()
            ]
            checkpoints = [
                cp for cp in checkpoints 
                if any(pattern.search(cp) for pattern in whitelist_patterns)
            ]
            print(f"Checkpoints after whitelist: {len(checkpoints)}")

        # Process blacklist patterns
        if blacklist_regex.strip():
            blacklist_patterns = [
                re.compile(pattern.strip()) 
                for pattern in blacklist_regex.split('\n') 
                if pattern.strip()
            ]
            checkpoints = [
                cp for cp in checkpoints 
                if not any(pattern.search(cp) for pattern in blacklist_patterns)
            ]
            print(f"Checkpoints after blacklist: {len(checkpoints)}")

        if not checkpoints:
            raise ValueError("No checkpoints available after applying whitelist and blacklist regex")

        return checkpoints

    def _get_eligible_checkpoints(self, checkpoints: List[str], usage_limit: int) -> List[str]:
        """Get checkpoints that haven't reached the usage limit"""
        eligible_checkpoints = [cp for cp in checkpoints if self.checkpoint_usage[cp] < usage_limit]
        print(f"Eligible checkpoints: {len(eligible_checkpoints)}")

        if not eligible_checkpoints:
            print("All checkpoints reached usage limit, resetting usage count")
            self.checkpoint_usage = defaultdict(int)
            eligible_checkpoints = checkpoints

        numpy.random.shuffle(eligible_checkpoints)
        return eligible_checkpoints

    def _load_checkpoint(self, eligible_checkpoints: List[str]) -> Tuple[Any, Any, Any, str]:
        """Attempt to load checkpoints until one succeeds"""
        for checkpoint_name in eligible_checkpoints:
            try:
                print(f"Attempting to load checkpoint: {checkpoint_name}")
                self.checkpoint_usage[checkpoint_name] += 1

                checkpoint_path = comfy_paths.get_full_path("checkpoints", checkpoint_name)
                print(f"Loading checkpoint from: {checkpoint_path}")

                result = comfy.sd.load_checkpoint_guess_config(
                    checkpoint_path,
                    output_vae=True,
                    output_clip=True,
                    embedding_directory=comfy_paths.get_folder_paths("embeddings")
                )

                if len(result) < 3:
                    raise ValueError(f"Unexpected number of return values: {len(result)}")

                model, clip, vae = result[:3]

                if model is None or clip is None or vae is None:
                    raise ValueError(f"Failed to load checkpoint components: {checkpoint_name}")

                print(f"Successfully loaded checkpoint: {checkpoint_name}")
                return (model, clip, vae, checkpoint_name)

            except Exception as e:
                print(f"Error loading checkpoint {checkpoint_name}: {e}")
                continue

        raise ValueError("Unable to load any checkpoints. All attempts failed.")

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        return float("nan")

NODE_CLASS_MAPPINGS = {
    "LoraLoaderFromPrompt": LoraLoaderFromPrompt,
    "CheckpointLoaderByName": CheckpointLoaderByName,
    "RandomCheckpointLoader": RandomCheckpointLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraLoaderFromPrompt": "Load Lora From Prompt",
    "CheckpointLoaderByName": "Checkpoint Loader By Name",
    "RandomCheckpointLoader": "Random Checkpoint Loader",
}
