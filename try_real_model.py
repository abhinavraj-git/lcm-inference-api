import time
import torch
from diffusers import StableDiffusionPipeline, LCMScheduler

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

if torch.cuda.is_available():
    device, dtype = "cuda", torch.float16
elif torch.backends.mps.is_available():
    device, dtype = "mps", torch.float32
else:
    device, dtype = "cpu", torch.float32
print("device:", device)

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, torch_dtype=dtype, safety_checker=None
)
pipe.unet.load_lora_adapter(
    "adapter",
    weight_name="pytorch_lora_weights.safetensors",
    adapter_name="lcm",
    prefix=None,
)
pipe.unet.set_adapter("lcm")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

prompt = "A lighthouse on a rocky coast at sunset, oil painting style"
gen = torch.Generator(device="cpu").manual_seed(42)

t0 = time.perf_counter()
image = pipe(prompt, num_inference_steps=4, guidance_scale=1.0, generator=gen).images[0]
print(f"4-step time: {time.perf_counter() - t0:.1f}s")
image.save("smoke_test.png")
print("saved smoke_test.png")
