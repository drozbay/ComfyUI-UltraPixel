import os
import folder_paths
import torch
from . import ultrapixel as up
from huggingface_hub import snapshot_download


def get_models(base):
    models_path = os.path.join(folder_paths.models_dir, "ultrapixel")
    models = [base]
    if os.path.exists(models_path):
        for path in os.listdir(models_path):
            if path.lower().endswith(".safetensors"):
                if path not in models:
                    models.append(path)
    return models


def download_models(ultrapixel_directory, stablecascade_directory):
    models = [
        ["ultrapixel_t2i.safetensors", "2kpr/UltraPixel"],
        ["stage_a.safetensors", "stabilityai/stable-cascade"],
        ["previewer.safetensors", "stabilityai/stable-cascade"],
        ["effnet_encoder.safetensors", "stabilityai/stable-cascade"],
        ["stage_b_lite_bf16.safetensors", "stabilityai/stable-cascade"],
        ["stage_c_bf16.safetensors", "stabilityai/stable-cascade"],
        ["controlnet/canny.safetensors", "stabilityai/stable-cascade"],
    ]
    for model in models:
        if model[1].startswith("stabilityai"):
            if stablecascade_directory == "default":
                model_path = os.path.join(folder_paths.models_dir, "ultrapixel")
            elif not os.path.exists(stablecascade_directory):
                print(
                    f"{stablecascade_directory} does not exist, defaulting to {model_path}"
                )
                model_path = os.path.join(folder_paths.models_dir, "ultrapixel")
            else:
                model_path = stablecascade_directory
        else:
            if ultrapixel_directory == "default":
                model_path = os.path.join(folder_paths.models_dir, "ultrapixel")
            elif not os.path.exists(ultrapixel_directory):
                print(
                    f"{ultrapixel_directory} does not exist, defaulting to {model_path}"
                )
                model_path = os.path.join(folder_paths.models_dir, "ultrapixel")
            else:
                model_path = ultrapixel_directory
        if os.path.exists(os.path.join(model_path, model[0])):
            # print("Path already exists:", os.path.join(model_path, model[0]))
            continue
        print(f"Downloading from {model[1]} to:", os.path.join(model_path, model[0]))
        snapshot_download(
            repo_id=model[1],
            allow_patterns=[model[0]],
            local_dir=model_path,
        )


class UltraPixelLoad:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ultrapixel": (get_models("ultrapixel_t2i.safetensors"),),
                "stage_a": (get_models("stage_a.safetensors"),),
                "stage_b": (get_models("stage_b_lite_bf16.safetensors"),),
                "stage_c": (get_models("stage_c_bf16.safetensors"),),
                "effnet": (get_models("effnet_encoder.safetensors"),),
                "previewer": (get_models("previewer.safetensors"),),
                "controlnet": (get_models("controlnet/canny.safetensors"),),
                "ultrapixel_directory": (
                    "STRING",
                    {
                        "multiline": False,
                        "dynamicPrompts": True,
                        "default": "default",
                    },
                ),
                "stablecascade_directory": (
                    "STRING",
                    {
                        "multiline": False,
                        "dynamicPrompts": True,
                        "default": "default",
                    },
                ),
            },
        }

    RETURN_TYPES = ("ULTRAPIXELMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "ultrapixel"
    CATEGORY = "UltraPixel"

    def ultrapixel(
        self,
        ultrapixel,
        stage_a,
        stage_b,
        stage_c,
        effnet,
        previewer,
        controlnet,
        ultrapixel_directory,
        stablecascade_directory,
    ):
        download_models(ultrapixel_directory, stablecascade_directory)
        model = up.UltraPixel(
            ultrapixel,
            stage_a,
            stage_b,
            stage_c,
            effnet,
            previewer,
            controlnet,
            ultrapixel_directory,
            stablecascade_directory,
        )
        return (model,)


class UltraPixelProcess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("ULTRAPIXELMODEL",),
                "height": (
                    "INT",
                    {"default": 2048, "min": 512, "max": 5120, "step": 256},
                ),
                "width": (
                    "INT",
                    {"default": 2048, "min": 512, "max": 5120, "step": 256},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "dtype": (["bf16", "fp32"],),
                "stage_a_tiled": (["true", "false"],),
                "stage_b_steps": (
                    "INT",
                    {"default": 10, "min": 1, "max": 1000, "step": 1},
                ),
                "stage_b_cfg": (
                    "FLOAT",
                    {"default": 1.1, "min": 1.0, "max": 1000.0, "step": 0.1},
                ),
                "stage_c_steps": (
                    "INT",
                    {"default": 20, "min": 1, "max": 1000, "step": 1},
                ),
                "stage_c_cfg": (
                    "FLOAT",
                    {"default": 4.0, "min": 1.0, "max": 1000.0, "step": 0.1},
                ),
                "controlnet_weight": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.01, "max": 10.0, "step": 0.01},
                ),
                "prompt": (
                    "STRING",
                    {"multiline": True, "dynamicPrompts": True},
                ),
            },
            "optional": {
                "controlnet_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = ("image", "edge_preview")
    FUNCTION = "ultrapixel"
    CATEGORY = "UltraPixel"

    def ultrapixel(
        self,
        model,
        height,
        width,
        seed,
        dtype,
        stage_a_tiled,
        stage_b_steps,
        stage_b_cfg,
        stage_c_steps,
        stage_c_cfg,
        controlnet_weight,
        prompt,
        controlnet_image=None,
    ):
        model.set_config(
            height,
            width,
            seed,
            dtype,
            stage_a_tiled,
            stage_b_steps,
            stage_b_cfg,
            stage_c_steps,
            stage_c_cfg,
            controlnet_weight,
            prompt,
            controlnet_image,
        )
        image, edge_preview = model.process()
        return (image, edge_preview)


NODE_CLASS_MAPPINGS = {
    "UltraPixelLoad": UltraPixelLoad,
    "UltraPixelProcess": UltraPixelProcess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltraPixelLoad": "UltraPixel Load",
    "UltraPixelProcess": "UltraPixel Process",
}
