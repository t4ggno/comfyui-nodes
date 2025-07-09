from .base_imports import *
from .ai_helpers import get_next_prompt
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

class PromptFromAIOpenAI:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "gpt_model": ([
                    "gpt-3.5-turbo", 
                    "gpt-3.5-turbo-16K", 
                    "gpt-4", 
                    "gpt-4-turbo", 
                    "gpt-4o", 
                    "Custom"
                ], {"default": "gpt-4o"}),
                "custom_model": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 2.0, "step": 0.1}),
                "frequency_penalty": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.1}),
                "presence_penalty": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.1}),
                "prompt_details": ("STRING", {"multiline": True}),
                "prefix": ("STRING", {"multiline": True}),
                "suffix": ("STRING", {"multiline": True}),
                "batch_quantity": ("INT", {"default": 1, "min": 1, "max": 10}),
                "images_per_batch": ("INT", {"default": 1, "min": 1, "max": 10}),
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

    def get_prompt(
        self, 
        api_key: str, 
        gpt_model: str, 
        custom_model: str, 
        temperature: float, 
        frequency_penalty: float, 
        presence_penalty: float, 
        prompt_details: str, 
        prefix: str, 
        suffix: str, 
        batch_quantity: int, 
        images_per_batch: int, 
        lora_whitelist_regex: str, 
        lora_blacklist_regex: str
    ) -> Tuple[str]:
        print("=============================")
        print("== Get prompt from OpenAI")

        next_prompt = get_next_prompt(self, prefix, suffix, images_per_batch)
        if next_prompt:
            return next_prompt

        available_loras = self._get_available_loras(lora_whitelist_regex, lora_blacklist_regex)
        keyword_list = self._load_keyword_list()
        
        prompt = self._generate_prompt_with_openai(
            api_key, gpt_model, custom_model, temperature, 
            frequency_penalty, presence_penalty, prompt_details, 
            batch_quantity, available_loras, keyword_list
        )
        
        self._save_prompt_and_keywords(prompt, keyword_list, api_key, gpt_model, custom_model)
        
        return self.get_prompt(
            api_key, gpt_model, custom_model, temperature, 
            frequency_penalty, presence_penalty, prompt_details, 
            prefix, suffix, batch_quantity, images_per_batch, 
            lora_whitelist_regex, lora_blacklist_regex
        )

    def _get_available_loras(self, whitelist_regex: str, blacklist_regex: str) -> str:
        lora_paths = folder_paths.get_folder_paths("loras")
        if not lora_paths:
            raise Exception("No lora paths found")
        
        available_loras = folder_paths.get_filename_list("loras")
        
        if whitelist_regex:
            whitelist_patterns = [
                re.compile(pattern.strip()) 
                for pattern in whitelist_regex.split('\n') 
                if pattern.strip()
            ]
            available_loras = [
                lora for lora in available_loras 
                if any(pattern.match(lora) for pattern in whitelist_patterns)
            ]
        
        if blacklist_regex:
            blacklist_patterns = [
                re.compile(pattern.strip()) 
                for pattern in blacklist_regex.split('\n') 
                if pattern.strip()
            ]
            available_loras = [
                lora for lora in available_loras 
                if not any(pattern.match(lora) for pattern in blacklist_patterns)
            ]
        
        return self._convert_loras_to_json(available_loras, lora_paths[0])

    def _convert_loras_to_json(self, loras: List[str], lora_path: str) -> str:
        converted_loras = {}
        
        for lora in loras:
            lora_parts = lora.split("\\")
            category = "General" if len(lora_parts) == 1 else lora_parts[0]
            
            if category not in converted_loras:
                converted_loras[category] = []
            
            lora_name = lora_parts[-1].replace(".safetensors", "")
            description = self._get_lora_description(lora_path, lora)
            
            lora_entry = {"name": lora_name}
            if description:
                lora_entry["description"] = description
            
            converted_loras[category].append(lora_entry)
        
        return json.dumps(converted_loras)

    def _get_lora_description(self, lora_path: str, lora: str) -> Optional[str]:
        description_file = os.path.join(lora_path, lora.replace(".safetensors", ".txt"))
        
        if os.path.isfile(description_file):
            try:
                with open(description_file, "r", encoding="utf-8") as f:
                    description = f.read().strip()
                    return description if description else None
            except Exception as e:
                print(f"Error reading description file {description_file}: {e}")
        
        return None

    def _load_keyword_list(self) -> str:
        try:
            with open("keywoard_list.txt", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print("keywoard_list.txt not found")
            return ""
        except Exception as e:
            print(f"Error loading keyword list: {e}")
            return ""

    def _generate_prompt_with_openai(
        self, 
        api_key: str, 
        gpt_model: str, 
        custom_model: str, 
        temperature: float,
        frequency_penalty: float, 
        presence_penalty: float, 
        prompt_details: str, 
        batch_quantity: int, 
        available_loras: str, 
        keyword_list: str
    ) -> str:
        client = OpenAI(api_key=api_key)
        
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(prompt_details, batch_quantity, available_loras, keyword_list)
        
        model = gpt_model if not custom_model else custom_model
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=4095,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        
        if not response.choices[0].message.content:
            print("No prompts found")
            return ""
        
        return response.choices[0].message.content

    def _build_system_prompt(self) -> str:
        return """You are a Stable Diffusion prompt generator. Generate detailed and creative prompts for unique and engaging scenes. Use keyword-rich sentences to describe the visuals. If the scene requires specific effects or characters, you may use loras in the format: `<lora:name:strength>`, ensuring only to use available loras. Emphasize important aspects of the scene by using brackets, for example: a white (cute cat) is playing with a (red ball:1.2). If generating multiple prompts, separate them with an empty line. Start directly with the prompt without any captions, titles, or numbering. Ensure clarity and creativity in each prompt."""

    def _build_user_prompt(self, details: str, batch_quantity: int, available_loras: str, keyword_list: str) -> str:
        prompt_parts = [
            f"Details for the prompt: {details}",
            "-" * 33,
            f"Quantity of prompts: {batch_quantity}",
            "-" * 33,
            f"Available loras: {available_loras}"
        ]
        
        if keyword_list:
            prompt_parts.extend([
                "-" * 33,
                f"Do NOT use the following keywords or similar keywords (IMPORTANT): {keyword_list}"
            ])
        
        return "\n".join(prompt_parts)

    def _save_prompt_and_keywords(self, prompt: str, keyword_list: str, api_key: str, gpt_model: str, custom_model: str):
        processed_prompt = self._process_prompt(prompt)
        updated_keywords = self._update_keyword_list(prompt, keyword_list, api_key, gpt_model, custom_model)
        
        self._write_files(processed_prompt, updated_keywords)

    def _process_prompt(self, prompt: str) -> str:
        prompt = re.sub(r"^\d+\.", "", prompt, flags=re.MULTILINE)
        prompt = re.sub(r"([^\n])\n([^\n])", r"\1 \2", prompt)
        prompt = re.sub(r"<(?:[\w\-. \\]+?)\:((?:[\w-]+?)(?:\-V\d+\.\d+))?(?:\.safetensors?)?>", r"<lora:\1>", prompt)
        prompt = re.sub(r"<loraname:", "<lora:", prompt)
        prompt = re.sub(r"^\s*[\-]+\s*$", "", prompt, flags=re.MULTILINE)
        
        return prompt

    def _update_keyword_list(self, prompt: str, current_keywords: str, api_key: str, gpt_model: str, custom_model: str) -> str:
        client = OpenAI(api_key=api_key)
        
        system_prompt = """You will receive an overview of prompts. Create a keyword list of prompts I can use to prevent further generations of the same or similar prompts later. Start directly with the keywords! Separate the keywords with a comma. If already a keyword list is provided you must add the new keywords to the existing list and avoid duplicates."""
        
        user_prompt = f"Prompts:\n\n{prompt}"
        if current_keywords:
            user_prompt += f"\n{'-' * 33}\nKeyword list: {current_keywords}"
        
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=4095,
            )
            
            if response.choices[0].message.content:
                print(f"Response keyword list: {response.choices[0].message.content}")
                return response.choices[0].message.content
        except Exception as e:
            print(f"Error updating keyword list: {e}")
        
        return current_keywords

    def _write_files(self, prompt: str, keywords: str):
        try:
            with open("prompt_from_ai.txt", "w", encoding="utf-8") as f:
                f.write(f"index:0\nimage:0\n\n{prompt}")
            
            with open("keywoard_list.txt", "w", encoding="utf-8") as f:
                f.write(keywords)
        except Exception as e:
            print(f"Error writing files: {e}")

    @classmethod
    def IS_CHANGED(self, **kwargs):
        return float("nan")

class PromptFromAIAnthropic:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "model": (["claude-3.5-sonnet", "claude-3-opus"], {"default": "claude-3.5-sonnet"}),
                "prompt_details": ("STRING", {"multiline": True}),
                "prefix": ("STRING", {"multiline": True}),
                "suffix": ("STRING", {"multiline": True}),
                "batch_quantity": ("INT", {"default": 1, "min": 1, "max": 10}),
                "images_per_batch": ("INT", {"default": 1, "min": 1, "max": 10}),
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

    def get_prompt(
        self, 
        api_key: str, 
        model: str, 
        prompt_details: str, 
        prefix: str, 
        suffix: str, 
        batch_quantity: int, 
        images_per_batch: int, 
        lora_whitelist_regex: str, 
        lora_blacklist_regex: str
    ) -> Tuple[str]:
        print("=============================")
        print("== Get prompt from Anthropic")

        next_prompt = get_next_prompt(self, prefix, suffix, images_per_batch)
        if next_prompt:
            return next_prompt

        available_loras = self._get_available_loras(lora_whitelist_regex, lora_blacklist_regex)
        keyword_list = self._load_keyword_list()
        
        prompt = self._generate_prompt_with_anthropic(
            api_key, model, prompt_details, batch_quantity, available_loras, keyword_list
        )
        
        if prompt:
            self._save_prompt_and_keywords(prompt, keyword_list, api_key, model)
        
        return self.get_prompt(
            api_key, model, prompt_details, prefix, suffix, 
            batch_quantity, images_per_batch, lora_whitelist_regex, lora_blacklist_regex
        )

    def _get_available_loras(self, whitelist_regex: str, blacklist_regex: str) -> str:
        lora_paths = folder_paths.get_folder_paths("loras")
        if not lora_paths:
            raise Exception("No lora paths found")
        
        available_loras = folder_paths.get_filename_list("loras")
        
        if whitelist_regex:
            whitelist_patterns = [
                re.compile(pattern.strip()) 
                for pattern in whitelist_regex.split('\n') 
                if pattern.strip()
            ]
            available_loras = [
                lora for lora in available_loras 
                if any(pattern.match(lora) for pattern in whitelist_patterns)
            ]
        
        if blacklist_regex:
            blacklist_patterns = [
                re.compile(pattern.strip()) 
                for pattern in blacklist_regex.split('\n') 
                if pattern.strip()
            ]
            available_loras = [
                lora for lora in available_loras 
                if not any(pattern.match(lora) for pattern in blacklist_patterns)
            ]
        
        return self._convert_loras_to_json(available_loras, lora_paths[0])

    def _convert_loras_to_json(self, loras: List[str], lora_path: str) -> str:
        converted_loras = {}
        
        for lora in loras:
            lora_parts = lora.split("\\")
            category = "General" if len(lora_parts) == 1 else lora_parts[0]
            
            if category not in converted_loras:
                converted_loras[category] = []
            
            lora_name = lora_parts[-1].replace(".safetensors", "")
            description = self._get_lora_description(lora_path, lora)
            
            lora_entry = {"name": lora_name}
            if description:
                lora_entry["description"] = description
            
            converted_loras[category].append(lora_entry)
        
        return json.dumps(converted_loras)

    def _get_lora_description(self, lora_path: str, lora: str) -> Optional[str]:
        description_file = os.path.join(lora_path, lora.replace(".safetensors", ".txt"))
        
        if os.path.isfile(description_file):
            try:
                with open(description_file, "r", encoding="utf-8") as f:
                    description = f.read().strip()
                    return description if description else None
            except Exception as e:
                print(f"Error reading description file {description_file}: {e}")
        
        return None

    def _load_keyword_list(self) -> str:
        try:
            with open("keywoard_list.txt", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print("keywoard_list.txt not found")
            return ""
        except Exception as e:
            print(f"Error loading keyword list: {e}")
            return ""

    def _generate_prompt_with_anthropic(
        self, 
        api_key: str, 
        model: str, 
        prompt_details: str, 
        batch_quantity: int, 
        available_loras: str, 
        keyword_list: str
    ) -> Optional[str]:
        client = Anthropic(api_key=api_key)
        
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(prompt_details, batch_quantity, available_loras, keyword_list)
        
        model_id = "claude-3-5-sonnet-20240620" if model == "claude-3.5-sonnet" else "claude-3-opus-20240229"
        
        print(f"Request prompt from AI:")
        print(f"API key: {api_key}")
        print(f"Model: {model}")
        print(f"System prompt: {system_prompt}")
        print(f"User prompt: {user_prompt}")
        
        try:
            response = client.messages.create(
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                model=model_id,
            )
            
            if response.content and response.content[0].text:
                print(f"Response prompt: {response.content[0].text}")
                return response.content[0].text
            else:
                print("No prompts found")
                return None
                
        except Exception as e:
            print(f"Error generating prompt: {e}")
            return None

    def _build_system_prompt(self) -> str:
        return """You are a Stable Diffusion prompt generator. Generate detailed and creative prompts for unique and engaging scenes. Use keyword-rich sentences to describe the visuals. If the scene requires specific effects or characters, you may use loras in the format: `<lora:name:strength>`, ensuring only to use available loras. Emphasize important aspects of the scene by using brackets, for example: a white (cute cat) is playing with a (red ball:1.2). If generating multiple prompts, separate them with an empty line. Start directly with the prompt without any captions, titles, or numbering. Ensure clarity and creativity in each prompt."""

    def _build_user_prompt(self, details: str, batch_quantity: int, available_loras: str, keyword_list: str) -> str:
        prompt_parts = [
            f"Details for the prompt: {details}",
            "-" * 33,
            f"Quantity of prompts: {batch_quantity}",
            "-" * 33,
            f"Available loras: {available_loras}"
        ]
        
        if keyword_list:
            prompt_parts.extend([
                "-" * 33,
                f"Do NOT use the following keywords or similar keywords (IMPORTANT): {keyword_list}"
            ])
        
        return "\n".join(prompt_parts)

    def _save_prompt_and_keywords(self, prompt: str, keyword_list: str, api_key: str, model: str):
        processed_prompt = self._process_prompt(prompt)
        updated_keywords = self._update_keyword_list(prompt, keyword_list, api_key, model)
        
        if updated_keywords:
            self._write_files(processed_prompt, updated_keywords)
        else:
            print("Failed to get keyword list")

    def _process_prompt(self, prompt: str) -> str:
        prompt = re.sub(r"^\d+\.", "", prompt, flags=re.MULTILINE)
        prompt = re.sub(r"([^\n])\n([^\n])", r"\1 \2", prompt)
        prompt = re.sub(r"<(?:[\w\-. \\]+?)\:((?:[\w-]+?)(?:\-V\d+\.\d+))?(?:\.safetensors?)?>", r"<lora:\1>", prompt)
        prompt = re.sub(r"<loraname:", "<lora:", prompt)
        prompt = re.sub(r"^\s*[\-]+\s*$", "", prompt, flags=re.MULTILINE)
        
        return prompt

    def _update_keyword_list(self, prompt: str, current_keywords: str, api_key: str, model: str) -> Optional[str]:
        client = Anthropic(api_key=api_key)
        
        system_prompt = """You will receive an overview of prompts. Create a keyword list of prompts I can use, to prevent further generations of the same or similar prompts later. Start directly with the keywords! Separate the keywords with a comma. If already a keyword list is provided you must add the new keywords to the existing list and avoid duplicates."""
        
        user_prompt = f"Prompts:\n\n{prompt}"
        if current_keywords:
            user_prompt += f"\n{'-' * 33}\nKeyword list: {current_keywords}"
        
        model_id = "claude-3-5-sonnet-20240620" if model == "claude-3.5-sonnet" else "claude-3-opus-20240229"
        
        try:
            response = client.messages.create(
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                model=model_id,
            )
            
            if response.content and response.content[0].text:
                print(f"Response keyword list: {response.content[0].text}")
                return response.content[0].text
        except Exception as e:
            print(f"Error updating keyword list: {e}")
        
        return current_keywords

    def _write_files(self, prompt: str, keywords: str):
        try:
            with open("prompt_from_ai.txt", "w", encoding="utf-8") as f:
                f.write(f"index:0\nimage:0\n\n{prompt}")
            
            with open("keywoard_list.txt", "w", encoding="utf-8") as f:
                f.write(keywords)
        except Exception as e:
            print(f"Error writing files: {e}")

    @classmethod
    def IS_CHANGED(self, **kwargs):
        return float("nan")

class PromptFromOllama:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "host": ("STRING", {"default": "http://localhost:11434/api/generate"}),
                "model": ("STRING", {"default": "mistral-nemo"}),
                "prompt_details": ("STRING", {"multiline": True}),
                "prefix": ("STRING", {"multiline": True}),
                "suffix": ("STRING", {"multiline": True}),
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

    def get_prompt(
        self, 
        host: str, 
        model: str, 
        prompt_details: str, 
        prefix: str, 
        suffix: str
    ) -> Tuple[str]:
        print("=============================")
        print("== Get prompt from Ollama")

        keyword_list = self._load_keyword_list()
        prompt = self._generate_prompt_with_ollama(host, model, prompt_details, keyword_list)
        
        if prompt:
            return (f"{prefix} {prompt} {suffix}".strip(),)
        else:
            return self.get_prompt(host, model, prompt_details, prefix, suffix)

    def _load_keyword_list(self) -> str:
        try:
            with open("keywoard_list.txt", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print("keywoard_list.txt not found")
            return ""
        except Exception as e:
            print(f"Error loading keyword list: {e}")
            return ""

    def _generate_prompt_with_ollama(
        self, 
        host: str, 
        model: str, 
        prompt_details: str, 
        keyword_list: str
    ) -> Optional[str]:
        system_prompt = "You are a Stable Diffusion / DALL-E / Midjourney prompt generator. The prompt should be detailed. Use Keyword sentences. The scene should be interesting and engaging. The prompt should be creative and unique. Start directly with the prompt and dont use a caption. Remember, its for an image and neither for a video nor for a book. The prompt should be 100 words long"
        
        user_prompt = f"Details for the prompt: {prompt_details}"
        
        if keyword_list:
            user_prompt += f"\n{'-' * 33}\nDONT use any of the following keywords or similar: {keyword_list}"

        print(f"Request prompt from AI:")
        print(f"Model: {model}")
        print(f"System prompt: {system_prompt}")
        print(f"User prompt: {user_prompt}")

        data = {
            "model": model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(host, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            
            response_data = response.json()
            prompt = response_data.get("response", "")
            
            if prompt:
                print(f"Response prompt: {prompt}")
                return prompt
            else:
                print("No prompts found")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error making request to Ollama: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    @classmethod
    def IS_CHANGED(self, **kwargs):
        return float("nan")

NODE_CLASS_MAPPINGS = {
    "PromptFromAIOpenAI": PromptFromAIOpenAI,
    "PromptFromAIAnthropic": PromptFromAIAnthropic,
    "PromptFromOllama": PromptFromOllama,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptFromAIOpenAI": "Prompt From AI (OpenAI)",
    "PromptFromAIAnthropic": "Prompt From AI (Anthropic)",
    "PromptFromOllama": "Prompt From Ollama",
}
