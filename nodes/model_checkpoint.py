from .base_imports import *
from typing import Dict, List, Tuple, Optional, Any, Union

class LoraLoaderFromPrompt(comfy_io.ComfyNode):
    def __init__(self):
        self.loaded_loras: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="LoraLoaderFromPrompt",
            display_name="Load Lora From Prompt",
            category="t4ggno/loaders",
            inputs=[
                comfy_io.Model.Input("model"),
                comfy_io.Clip.Input("clip"),
                comfy_io.String.Input("prompt", multiline=True),
            ],
            outputs=[
                comfy_io.Model.Output(display_name="model"),
                comfy_io.Clip.Output(display_name="clip"),
                comfy_io.String.Output(display_name="prompt"),
            ]
        )

    @classmethod
    def execute(cls, model: Any, clip: Any, prompt: str, **kwargs) -> comfy_io.NodeOutput:
        print("=============================")
        print("== Load Loras from Prompt ==")

        available_loras = folder_paths.get_filename_list("loras")
        all_loras = cls._extract_loras_from_prompt(prompt)
        prompt = cls._remove_loras_from_prompt(prompt)

        processed_loras = cls._process_lora_data(all_loras)
        resolved_loras = cls._resolve_lora_names(processed_loras, available_loras)
        deduplicated_loras = cls._deduplicate_loras(resolved_loras)

        model, clip, prompt = cls._apply_loras(model, clip, deduplicated_loras, available_loras, prompt)
        return comfy_io.NodeOutput(model, clip, prompt)

    @classmethod
    def _extract_loras_from_prompt(cls, prompt: str) -> List[Tuple[str, str, str]]:
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

    @classmethod
    def _remove_loras_from_prompt(cls, prompt: str) -> str:
        """Remove lora tags from prompt"""
        lora_pattern = r"<(?:lora:)?([\w\-. \\]+?)(?::(-?[0-9]+(?:\.[0-9]*)?)?)?(?::(-?[0-9]+(?:\.[0-9]*)?)?)?>"
        return re.sub(lora_pattern, "", prompt)

    @classmethod
    def _process_lora_data(cls, lora_matches: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
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

    @classmethod
    def _resolve_lora_names(cls, loras: List[Dict[str, Any]], available_loras: List[str]) -> List[Dict[str, Any]]:
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

    @classmethod
    def _deduplicate_loras(cls, loras: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

    @classmethod
    def _apply_loras(cls, model: Any, clip: Any, loras: List[Dict[str, Any]],
                    available_loras: List[str], prompt: str) -> Tuple[Any, Any, str]:
        """Apply loras to model and clip"""
        valid_loras = [lora for lora in loras if lora["name"] in available_loras]

        if not valid_loras:
            return (model, clip, prompt)

        for lora in valid_loras:
            try:
                model, clip, prompt = cls._load_single_lora(model, clip, lora, prompt)
            except Exception as e:
                print(f"Error loading lora {lora['name']}: {e}")
                continue

        print(f"Text prompt after loading loras: {prompt}")
        return (model, clip, prompt)

    @classmethod
    def _load_single_lora(cls, model: Any, clip: Any, lora: Dict[str, Any], prompt: str) -> Tuple[Any, Any, str]:
        """Load a single lora file"""
        lora_path = folder_paths.get_full_path("loras", lora["name"])
        loaded_lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        model_strength = lora["model_strength"]
        clip_strength = lora["clip_strength"]

        model, clip = comfy.sd.load_lora_for_models(model, clip, loaded_lora, model_strength, clip_strength)
        print(f"Loaded lora: {lora_path} with model strength: {model_strength} and clip strength: {clip_strength}")

        return model, clip, cls._process_trigger_words(lora_path, prompt)

    @classmethod
    def _process_trigger_words(cls, lora_path: str, prompt: str) -> str:
        """Extract and add trigger words from lora metadata if needed"""
        try:
            metadata = cls._extract_lora_metadata(lora_path)
            if not metadata:
                return prompt

            trigger_words = cls._get_trigger_words(metadata)
            if not trigger_words:
                return prompt

            if not any(word in prompt for word in trigger_words):
                random_trigger = numpy.random.choice(trigger_words)
                prompt += f" {random_trigger}"
                print(f"Added trigger word: '{random_trigger}' to prompt")

        except Exception as e:
            print(f"Error processing trigger words for {lora_path}: {e}")

        return prompt

    @classmethod
    def _extract_lora_metadata(cls, lora_path: str) -> Optional[Dict[str, Any]]:
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

    @classmethod
    def _get_trigger_words(cls, metadata: Dict[str, Any]) -> List[str]:
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

class CheckpointLoaderByName(comfy_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="CheckpointLoaderByName",
            display_name="Checkpoint Loader By Name",
            category="t4ggno/utils",
            inputs=[
                comfy_io.Combo.Input("type", options=["Detect (Manual)", "Detect (Random)"], default="Detect (Manual)"),
                comfy_io.String.Input("checkpoint_name", default=""),
                comfy_io.Combo.Input("fallback", options=comfy_paths.get_filename_list("checkpoints"), lazy=True),
            ],
            outputs=[
                comfy_io.Model.Output(display_name="MODEL"),
                comfy_io.Clip.Output(display_name="CLIP"),
                comfy_io.Vae.Output(display_name="VAE"),
                comfy_io.String.Output(display_name="NAME_STRING"),
            ]
        )

    @classmethod
    def check_lazy_status(cls, type, checkpoint_name, fallback, **kwargs):
        needed = []
        # If checkpoint_name is provided and valid, we don't need fallback
        if checkpoint_name:
            available_checkpoints = comfy_paths.get_filename_list("checkpoints")
            if cls._find_checkpoint(checkpoint_name, available_checkpoints):
                return needed

        # Otherwise we need fallback
        if fallback is None:
            needed.append("fallback")
        return needed

    @classmethod
    def execute(cls, type: str, checkpoint_name: str, fallback: str, **kwargs) -> comfy_io.NodeOutput:
        print("=============================")
        print("== Load Checkpoint by Name ==")
        print(f"Type: {type}")

        available_checkpoints = comfy_paths.get_filename_list("checkpoints")

        if checkpoint_name:
            checkpoint_path = cls._find_checkpoint(checkpoint_name, available_checkpoints)
            if checkpoint_path:
                result = cls._load_checkpoint_from_path(checkpoint_path, checkpoint_name)
                return comfy_io.NodeOutput(*result)

        result = cls._load_fallback_checkpoint(type, available_checkpoints, fallback)
        return comfy_io.NodeOutput(*result)

    @classmethod
    def _find_checkpoint(cls, checkpoint_name: str, available_checkpoints: List[str]) -> Optional[str]:
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

    @classmethod
    def _load_checkpoint_from_path(cls, checkpoint_path: str, original_name: str) -> Tuple[Any, Any, Any, str]:
        """Load checkpoint from given path"""
        result = comfy.sd.load_checkpoint_guess_config(
            checkpoint_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=comfy_paths.get_folder_paths("embeddings")
        )

        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        return (*result, checkpoint_name)

    @classmethod
    def _load_fallback_checkpoint(cls, type_: str, available_checkpoints: List[str], fallback: str) -> Tuple[Any, Any, Any, str]:
        """Load fallback checkpoint based on type"""
        if type_ == "Detect (Random)":
            selected_checkpoint = numpy.random.choice(available_checkpoints)
            print(f"Load checkpoint [Random]: {selected_checkpoint}")
        else:
            selected_checkpoint = fallback
            print(f"Load checkpoint [Fallback]: {selected_checkpoint}")

        checkpoint_path = comfy_paths.get_full_path("checkpoints", selected_checkpoint)
        return cls._load_checkpoint_from_path(checkpoint_path, selected_checkpoint)

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

class RandomCheckpointLoader(comfy_io.ComfyNode):
    checkpoint_usage: Dict[str, int] = defaultdict(int)

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="RandomCheckpointLoader",
            display_name="Random Checkpoint Loader",
            category="t4ggno/loaders",
            description="""
            Load a random checkpoint from the available checkpoints. The checkpoint is selected randomly from the list of available checkpoints.
            The checkpoint is loaded only if it has not reached the usage limit. The usage limit is the maximum number of times a checkpoint can be loaded before it is reset.
            The checkpoint is selected based on the whitelist and blacklist regex patterns. The whitelist regex is used to filter the checkpoints that match the pattern.
            The blacklist regex is used to exclude the checkpoints that match the pattern.
            If no checkpoints are available after applying the whitelist and blacklist regex, an error is raised.
            If all checkpoints have reached the usage limit, the usage count is reset, and the process is repeated until an eligible checkpoint is found.
            """,
            inputs=[
                comfy_io.String.Input("whitelist_regex", default="", multiline=True),
                comfy_io.String.Input("blacklist_regex", default="", multiline=True),
                comfy_io.Int.Input("usage_limit", default=1, min=1, max=100),
            ],
            outputs=[
                comfy_io.Model.Output(display_name="model"),
                comfy_io.Clip.Output(display_name="clip"),
                comfy_io.Vae.Output(display_name="vae"),
                comfy_io.String.Output(display_name="checkpoint_name"),
            ]
        )

    @classmethod
    def execute(cls, whitelist_regex: str, blacklist_regex: str, usage_limit: int, **kwargs) -> comfy_io.NodeOutput:
        print("=============================")
        print("== Loading random checkpoint")

        available_checkpoints = cls._get_available_checkpoints()
        filtered_checkpoints = cls._filter_checkpoints(available_checkpoints, whitelist_regex, blacklist_regex)
        eligible_checkpoints = cls._get_eligible_checkpoints(filtered_checkpoints, usage_limit)

        result = cls._load_checkpoint(eligible_checkpoints)
        return comfy_io.NodeOutput(*result)

    @classmethod
    def _get_available_checkpoints(cls) -> List[str]:
        """Get all available checkpoints and filter out None/empty values"""
        all_checkpoints = comfy_paths.get_filename_list("checkpoints")
        valid_checkpoints = [cp for cp in all_checkpoints if cp]

        print(f"Total checkpoints found: {len(all_checkpoints)}")
        print(f"Valid checkpoints: {len(valid_checkpoints)}")

        return valid_checkpoints

    @classmethod
    def _filter_checkpoints(cls, checkpoints: List[str], whitelist_regex: str, blacklist_regex: str) -> List[str]:
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

    @classmethod
    def _get_eligible_checkpoints(cls, checkpoints: List[str], usage_limit: int) -> List[str]:
        """Get checkpoints that haven't reached the usage limit"""
        eligible_checkpoints = [cp for cp in checkpoints if cls.checkpoint_usage[cp] < usage_limit]
        print(f"Eligible checkpoints: {len(eligible_checkpoints)}")

        if not eligible_checkpoints:
            print("All checkpoints reached usage limit, resetting usage count")
            cls.checkpoint_usage = defaultdict(int)
            eligible_checkpoints = checkpoints

        numpy.random.shuffle(eligible_checkpoints)
        return eligible_checkpoints

    @classmethod
    def _load_checkpoint(cls, eligible_checkpoints: List[str]) -> Tuple[Any, Any, Any, str]:
        """Attempt to load checkpoints until one succeeds"""
        for checkpoint_name in eligible_checkpoints:
            try:
                print(f"Attempting to load checkpoint: {checkpoint_name}")
                cls.checkpoint_usage[checkpoint_name] += 1

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
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")
