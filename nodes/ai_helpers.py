from .base_imports import *
from typing import Optional, Tuple, List

def get_next_prompt(instance, prefix: str, suffix: str, images_per_batch: int) -> Optional[Tuple[str]]:
    try:
        with open("prompt_from_ai.txt", "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print("prompt_from_ai.txt not found")
        return None
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return None

    parts = re.split(r"^\s*$", content, flags=re.MULTILINE)
    if len(parts) < 2:
        print("Invalid prompt file format")
        return None

    try:
        metadata_lines = parts[0].strip().split("\n")
        current_index = int(metadata_lines[0].split(":")[1])
        image_count = int(metadata_lines[1].split(":")[1])
    except (IndexError, ValueError) as e:
        print(f"Error parsing metadata: {e}")
        return None

    prompts = [p.strip() for p in parts[1:] if p.strip()]
    if not prompts:
        print("No prompts found")
        return None

    if image_count >= images_per_batch:
        print("Image count reached batch limit -> Reset count and increment index")
        image_count = 0
        current_index += 1

    if current_index >= len(prompts):
        print("Index exceeds available prompts")
        return None

    image_count += 1
    print(f"Index: {current_index}, Image count: {image_count}")

    selected_prompt = prompts[current_index]
    prompt_lines = [line.strip() for line in selected_prompt.split("\n") if line.strip()]
    
    if not prompt_lines:
        print("Selected prompt is empty")
        return None

    final_prompt = f"{prefix} {prompt_lines[0]} {suffix}".strip()
    
    try:
        with open("prompt_from_ai.txt", "w", encoding="utf-8") as f:
            f.write(f"index:{current_index}\nimage:{image_count}\n\n")
            f.write("\n".join(prompts))
    except Exception as e:
        print(f"Error writing prompt file: {e}")

    print(f"Prompt: {final_prompt}")
    return (final_prompt,)
