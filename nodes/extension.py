from comfy_api.latest import ComfyExtension, io

from .ai_llm import PromptFromAIOpenAI, PromptFromAIAnthropic, PromptFromOllama
from .image_processing import Base64ImageDecoder, ImageMetadataExtractor, AutoLoadImageForUpscaler, LoadImageWithMetadata, ImageSaver, ColorMatcher
from .layout_resolution import LayoutSwitch, PredefinedResolutions, ResolutionSwitch
from .model_checkpoint import LoraLoaderFromPrompt, CheckpointLoaderByName, RandomCheckpointLoader
from .prompt_generation import RandomJSONSelector, SmartPromptBuilder, PromptTemplateManager, PromptEnhancer, QuickPromptGenerator
from .text_processing import TextSwitch, TextReplacer
from .utility import CurrentDateTime, TimestampConverter, RandomSeed

class T4ggnoExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            PromptFromAIOpenAI, PromptFromAIAnthropic, PromptFromOllama,
            Base64ImageDecoder, ImageMetadataExtractor, AutoLoadImageForUpscaler, LoadImageWithMetadata, ImageSaver, ColorMatcher,
            LayoutSwitch, PredefinedResolutions, ResolutionSwitch,
            LoraLoaderFromPrompt, CheckpointLoaderByName, RandomCheckpointLoader,
            RandomJSONSelector, SmartPromptBuilder, PromptTemplateManager, PromptEnhancer, QuickPromptGenerator,
            TextSwitch, TextReplacer,
            CurrentDateTime, TimestampConverter, RandomSeed
        ]

async def comfy_entrypoint() -> ComfyExtension:
    return T4ggnoExtension()
