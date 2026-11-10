import hashlib
import os
import threading
import time
from functools import lru_cache

from PIL import Image, ImageDraw

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "adapter")


class MockEngine:
    """Fake generator for development and tests. Needs no model and no GPU."""

    name = "mock"

    def generate(self, prompt: str, steps: int, seed: int) -> Image.Image:
        time.sleep(0.05 * steps)  # pretend that more steps take longer
        digest = hashlib.sha256(f"{prompt}|{seed}".encode()).digest()
        color = (digest[0], digest[1], digest[2])
        img = Image.new("RGB", (512, 512), color)
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), f"MOCK steps={steps} seed={seed}", fill="white")
        draw.text((20, 50), prompt[:60], fill="white")
        return img


class RealEngine:
    """Runs the LCM-LoRA student with the same settings as the notebook."""

    name = "real"

    def __init__(self) -> None:
        import torch
        from diffusers import LCMScheduler, StableDiffusionPipeline

        if torch.cuda.is_available():
            device, dtype = "cuda", torch.float16
        elif torch.backends.mps.is_available():
            device, dtype = "mps", torch.float32
        else:
            device, dtype = "cpu", torch.float32

        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=dtype, safety_checker=None
        )
        pipe.unet.load_lora_adapter(
            ADAPTER_DIR,
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="lcm",
            prefix=None,
        )
        pipe.unet.set_adapter("lcm")
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        self.pipe = pipe.to(device)
        self.torch = torch
        self.device = device
        self._lock = threading.Lock()  # one generation at a time

    def generate(self, prompt: str, steps: int, seed: int) -> Image.Image:
        with self._lock:
            gen = self.torch.Generator(device="cpu").manual_seed(seed)
            out = self.pipe(
                prompt,
                num_inference_steps=steps,
                guidance_scale=1.0,
                generator=gen,
            )
            return out.images[0]


@lru_cache(maxsize=1)
def get_engine():
    """ENGINE=real uses the model. Anything else uses the mock."""
    if os.getenv("ENGINE", "mock").lower() == "real":
        return RealEngine()
    return MockEngine()
