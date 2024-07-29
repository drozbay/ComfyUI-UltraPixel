import os
import yaml
import torch
import sys
import folder_paths

from .inference.utils import *
from .core.utils import load_or_fail
from .train import WurstCore_control_lrguide
from .gdf import (
    DDPMSampler,
)
from .train import WurstCore_t2i as WurstCoreC

from safetensors.torch import load_file as load_safetensors

class UltraPixel:
    def __init__(
        self,
        pretrained,
        stage_c,
        effnet,
        previewer,
        controlnet,
        ultrapixel_directory,
        stablecascade_directory,
    ):
        if ultrapixel_directory == "default":
            self.ultrapixel_path = os.path.join(folder_paths.models_dir, "ultrapixel")
        elif not os.path.exists(ultrapixel_directory):
            print(
                f"{ultrapixel_directory} does not exist, defaulting to {self.ultrapixel_path}"
            )
            self.ultrapixel_path = os.path.join(folder_paths.models_dir, "ultrapixel")
        else:
            self.ultrapixel_path = ultrapixel_directory
        if stablecascade_directory == "default":
            self.stablecascade_path = os.path.join(
                folder_paths.models_dir, "ultrapixel"
            )
        elif not os.path.exists(stablecascade_directory):
            print(
                f"{stablecascade_directory} does not exist, defaulting to {self.stablecascade_path}"
            )
            self.stablecascade_path = os.path.join(
                folder_paths.models_dir, "ultrapixel"
            )
        else:
            self.stablecascade_path = stablecascade_directory
        self.pretrained = os.path.join(self.ultrapixel_path, pretrained)
        self.stage_c = os.path.join(self.stablecascade_path, stage_c)
        self.effnet = os.path.join(self.stablecascade_path, effnet)
        self.previewer = os.path.join(self.stablecascade_path, previewer)
        self.controlnet = os.path.join(self.stablecascade_path, controlnet)

    def set_config(
        self,
        height_c,
        width_c,
        height_c_lr,
        width_c_lr,
        seed,
        dtype,
        stage_c_steps,
        stage_c_cfg,
        controlnet_weight,
        prompt,
        sampler,
        controlnet_image,
    ):
        self.height_c = height_c
        self.width_c = width_c
        self.height_c_lr = height_c_lr
        self.width_c_lr = width_c_lr
        self.seed = seed
        self.dtype = dtype
        self.stage_c_steps = stage_c_steps
        self.stage_c_cfg = stage_c_cfg
        self.controlnet_weight = controlnet_weight
        self.prompt = prompt
        self.sampler = sampler
        self.controlnet_image = controlnet_image




    def process(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        dtype = torch.bfloat16 if self.dtype == "bf16" else torch.float

        base_path = os.path.dirname(os.path.realpath(__file__))

        if self.controlnet_image == None:
            config_file = os.path.join(base_path, "configs/training/t2i.yaml")
        else:
            config_file = os.path.join(
                base_path, "configs/training/cfg_control_lr.yaml"
            )

        with open(config_file, "r", encoding="utf-8") as file:
            loaded_config = yaml.safe_load(file)
            loaded_config["effnet_checkpoint_path"] = self.effnet
            loaded_config["previewer_checkpoint_path"] = self.previewer
            loaded_config["generator_checkpoint_path"] = self.stage_c

        if self.controlnet_image == None:
            core = WurstCoreC(config_dict=loaded_config, device=device, training=False)
        else:
            core = WurstCore_control_lrguide(
                config_dict=loaded_config, device=device, training=False
            )

        extras = core.setup_extras_pre()
        models = core.setup_models(extras)
        models.generator.eval().requires_grad_(False)
        # print("STAGE C READY")
            
        sdd = load_safetensors(self.pretrained) # this is the equivalent code for loading the real safetensors versions of ultrapixel_t2i and lora_cat.
        collect_sd = {k: v for k, v in sdd.items()}
        collect_sd = {k[7:] if k.startswith('module.') else k: v for k, v in collect_sd.items()}

        if self.controlnet_image == None:
            models.train_norm.load_state_dict(collect_sd)
        else:
            models.train_norm.load_state_dict(collect_sd, strict=True)
            models.controlnet.load_state_dict(
                load_or_fail(self.controlnet), strict=True
            )

        models.generator.eval()   # stage C
        models.train_norm.eval()  # stage UP

        batch_size = 1
        edge_image = None

        if self.controlnet_image != None:
            self.controlnet_image = self.controlnet_image.squeeze(0)
            self.controlnet_image = self.controlnet_image.permute(2, 0, 1)
            images = (
                resize_image(
                    torchvision.transforms.functional.to_pil_image(
                        self.controlnet_image.clamp(0, 1)
                    ).convert("RGB")
                )
                .unsqueeze(0)
                .expand(batch_size, -1, -1, -1)
            )
            batch = {"images": images}
            cnet_multiplier = self.controlnet_weight  # 0.8 0.6 0.3  control strength

        """ 
        Calculate latent shapes. get_target_lr_size() preserves the aspect ratio. 1536x2560 becomes 24.79x41.31, which is not ideal. Rounds down with int() cast, then multiplies by 32, so this becomes 768x1312.
        stage_b_latent_shape_lr is unused.
        """
        
        stage_c_latent_shape    = (batch_size, 16, self.height_c,    self.width_c)
        stage_c_latent_shape_lr = (batch_size, 16, self.height_c_lr, self.width_c_lr)

        # Stage C Parameters
        extras.sampling_configs["cfg"] = self.stage_c_cfg
        extras.sampling_configs["shift"] = 1 if self.controlnet_image == None else 2
        extras.sampling_configs["timesteps"] = self.stage_c_steps
        extras.sampling_configs["t_start"] = 1.0
        #extras.sampling_configs["sampler"] = self.sampler
        extras.sampling_configs["sampler"] = DDPMSampler(extras.gdf)

        captions = [self.prompt] #POSITIVE PROMPT
        for cnt, caption in enumerate(captions):
            with torch.no_grad():
                models.generator.cpu()
                torch.cuda.empty_cache()
                models.text_model.cuda()
                if self.controlnet_image != None:
                    models.controlnet.cuda()

                if self.controlnet_image == None:
                    batch = {"captions": [caption] * batch_size}
                else:
                    batch["captions"] = [caption + " high quality"] * batch_size

                conditions = core.get_conditions(
                    batch,
                    models,
                    extras,
                    is_eval=True,
                    is_unconditional=False,
                    eval_image_embeds=False,
                )

                unconditions = core.get_conditions(
                    batch,
                    models,
                    extras,
                    is_eval=True,
                    is_unconditional=True,
                    eval_image_embeds=False,
                )

                if self.controlnet_image != None:
                    cnet, cnet_input = core.get_cnet(batch, models, extras)
                    cnet_uncond = cnet
                    conditions["cnet"] = [
                        c.clone() * cnet_multiplier if c is not None else c
                        for c in cnet
                    ]
                    unconditions["cnet"] = [
                        c.clone() * cnet_multiplier if c is not None else c
                        for c in cnet_uncond
                    ]
                    edge_images = show_images(cnet_input)
                    edge_image = edge_images[0]

                models.text_model.cpu()
                if self.controlnet_image != None:
                    models.controlnet.cpu()
                torch.cuda.empty_cache()
                models.generator.cuda()

                print("STAGE C GENERATION***************************")
                with torch.cuda.amp.autocast(dtype=dtype):
                    sampled_c = generation_c(
                        batch,
                        models,
                        extras,
                        core,
                        stage_c_latent_shape,
                        stage_c_latent_shape_lr,
                        device,
                        conditions,
                        unconditions,
                    )

                models.generator.cpu()
                torch.cuda.empty_cache()
                
                return ({'samples': sampled_c},)

                return edge_image
