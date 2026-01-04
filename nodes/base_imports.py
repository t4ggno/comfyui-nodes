import io
import base64
from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo
import torch
import folder_paths
import numpy
import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import re
import urllib.parse
import urllib.request
import os
import json
import random
import folder_paths as comfy_paths
from datetime import datetime
import time
import subprocess
import sys
import importlib.util

def ensure_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    if importlib.util.find_spec(import_name) is None:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

ensure_package("openai")
ensure_package("anthropic")
ensure_package("pytz")
ensure_package("color-matcher", "color_matcher")

from openai import OpenAI
from anthropic import Anthropic
from collections import defaultdict
import hashlib
import node_helpers
import requests

dirPath = os.path.dirname(os.path.realpath(__file__))
ALLOWED_EXT = ('jpeg', 'jpg', 'png', 'tiff', 'gif', 'bmp', 'webp')
