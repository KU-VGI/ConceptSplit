# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    FusedAttnProcessor2_0,
    XFormersAttnProcessor,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_invisible_watermark_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput


if is_invisible_watermark_available():
    from .watermark import StableDiffusionXLWatermarker

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

import matplotlib.pyplot as plt
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import math
import gc
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from packaging import version

from util import attention_highlight
from utils.gaussian_smoothing import GaussianSmoothing
from utils.ptp_utils_LODA import AttentionStore, aggregate_attention
from utils import vis_utils
from scipy.ndimage import gaussian_filter


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionXLPipeline

        >>> pipe = StableDiffusionXLPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://arxiv.org/pdf/2305.08891.pdf).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.

    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class LODA_StableDiffusionXLPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    FromSingleFileMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
    IPAdapterMixin,
):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion XL.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion XL uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([` CLIPTextModelWithProjection`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
        add_watermarker (`bool`, *optional*):
            Whether to use the [invisible_watermark library](https://github.com/ShieldMnt/invisible-watermark/) to
            watermark output images. If not defined, it will default to True if the package is installed, otherwise no
            watermarker will be used.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "feature_extractor",
    ]
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.default_sample_size = (
            self.unet.config.sample_size
            if hasattr(self, "unet") and self.unet is not None and hasattr(self.unet.config, "sample_size")
            else 128
        )

        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder, lora_scale)

            if self.text_encoder_2 is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # textual inversion: process multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

                # We are only ALWAYS interested in the pooled output of the final text encoder
                if pooled_prompt_embeds is None and prompt_embeds[0].ndim == 2:
                    pooled_prompt_embeds = prompt_embeds[0]

                if clip_skip is None:
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                else:
                    # "2" because SDXL always indexes from the penultimate layer.
                    prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                if negative_pooled_prompt_embeds is None and negative_prompt_embeds[0].ndim == 2:
                    negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if self.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            if self.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds
    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        image_embeds = []
        if do_classifier_free_guidance:
            negative_image_embeds = []
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                image_embeds.append(single_image_embeds[None, :])
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    negative_image_embeds.append(single_negative_image_embeds)
                image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds = []
        for i, single_image_embeds in enumerate(image_embeds):
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds)

        return ip_adapter_image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                FusedAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def denoising_end(self):
        return self._denoising_end

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def _compute_harmonic_mean_bh(self, attention_store: AttentionStore,
                                    indices_to_alter: List[int],
                                    kernel_size: int = 7,
                                    sigma: int = 1,
                                    t: int = 10,
                                    is_x=False) -> torch.Tensor:
        # attention map 추출 및 보간
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=32,
            from_where=("up",),
            is_cross=True,
            select=0)  # [32,32,77]
        
        attention_maps = F.interpolate(
            attention_maps.permute(2, 0, 1).unsqueeze(0),
            size=(16, 16),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)
        
        padding = 1
        smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to(attention_maps.device)
        
        bhattacharyya_distances = []
        num_tokens = len(indices_to_alter)
        
        # indices_to_alter에 있는 모든 토큰 쌍에 대해 Bhattacharyya 거리 계산
        for i in range(num_tokens):
            for j in range(i + 1, num_tokens):
                token_i = indices_to_alter[i]
                token_j = indices_to_alter[j]
                
                # 두 토큰에 대한 attention map 추출
                tensor1 = attention_maps[:, :, token_i].to(torch.float32)
                tensor2 = attention_maps[:, :, token_j].to(torch.float32)
                
                # 패딩 및 Gaussian smoothing 적용
                tensor1 = F.pad(tensor1.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='reflect')
                tensor1 = smoothing(tensor1).squeeze(0).squeeze(0)
                
                tensor2 = F.pad(tensor2.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='reflect')
                tensor2 = smoothing(tensor2).squeeze(0).squeeze(0)
                
                epsilon = 1e-10  # 수치적 안정성 확보
                
                # 전체 분포 또는 X축 합산 분포로 정규화
                if is_x:
                    tensor1_prob = tensor1.sum(dim=0) / tensor1.sum()
                    tensor2_prob = tensor2.sum(dim=0) / tensor2.sum()
                else:
                    tensor1_prob = tensor1 / tensor1.sum()
                    tensor2_prob = tensor2 / tensor2.sum()
                
                # 안정성을 위한 작은 상수 추가
                tensor1_prob = tensor1_prob + epsilon
                tensor2_prob = tensor2_prob + epsilon
                
                # Bhattacharyya 계수: 두 분포의 제곱근 곱의 합
                b_coeff = torch.sum(torch.sqrt(tensor1_prob * tensor2_prob))
                # Bhattacharyya 거리: -log(계수)
                # b_distance = torch.log(b_coeff)
                b_distance = b_coeff
                
                bhattacharyya_distances.append(b_distance)
        
        # 모든 쌍에 대한 Bhattacharyya 거리를 텐서로 변환 후 조화평균 계산
        bhattacharyya_distances_tensor = torch.stack(bhattacharyya_distances)
        harmonic_mean_b = len(bhattacharyya_distances_tensor) / torch.sum(1.0 / bhattacharyya_distances_tensor)
        
        # 추가적으로 t와의 차이를 ReLU를 통해 반환 (필요에 따라 조정 가능)
        # harmonic_mean_b = torch.relu(t - harmonic_mean_b)
        
        return harmonic_mean_b.to(torch.float32)

    def _compute_harmonic_mean_hellinger(self, attention_store, indices_to_alter: List[int], kernel_size: int = 7, sigma: int = 1, is_x=False) -> torch.Tensor:
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=32,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0
        ).to(torch.float32)  # [32, 32, 77]
    
        # Resize to 16x16
        attention_maps = F.interpolate(
            attention_maps.permute(2, 0, 1).unsqueeze(0),
            size=(16, 16),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)
    
        padding = 1
        smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to(attention_maps.device)
        
        hellinger_values = []
    
        # 모든 토큰 간의 Hellinger Distance 계산
        num_tokens = len(indices_to_alter)
        for i in range(num_tokens):
            for j in range(i + 1, num_tokens):
                token_i = indices_to_alter[i]
                token_j = indices_to_alter[j]
                
                # 두 토큰에 대한 Attention Map 추출
                tensor1 = attention_maps[:, :, token_i]
                tensor2 = attention_maps[:, :, token_j]
    
                tensor1 = F.pad(tensor1.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='reflect')
                tensor1 = smoothing(tensor1).squeeze(0).squeeze(0)
    
                tensor2 = F.pad(tensor2.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='reflect')
                tensor2 = smoothing(tensor2).squeeze(0).squeeze(0)
    
                epsilon = 1e-10
                # Attention Map을 확률분포로 변환
                if is_x:
                    tensor1_prob = tensor1.sum(dim=0) / tensor1.sum()
                    tensor2_prob = tensor2.sum(dim=0) / tensor2.sum()
                else:
                    tensor1_prob = tensor1 / tensor1.sum()
                    tensor2_prob = tensor2 / tensor2.sum()
    
                # 안정성을 위한 작은 상수 추가
                tensor1_prob = tensor1_prob + epsilon
                tensor2_prob = tensor2_prob + epsilon
    
                # **Hellinger Distance 계산**
                hellinger_dist = torch.sqrt(0.5 * torch.sum((torch.sqrt(tensor1_prob) - torch.sqrt(tensor2_prob)) ** 2))
    
                hellinger_values.append(hellinger_dist)
    
        # Hellinger Distance 리스트를 텐서로 변환
        hellinger_tensor = torch.stack(hellinger_values)
    
        # 조화평균 Hellinger Distance 계산
        harmonic_mean_hellinger = len(hellinger_tensor) / torch.sum(1.0 / hellinger_tensor)
    
        # 학습을 위해 변환 (값이 작을수록 유사, 클수록 차이 → 손실로 사용 가능)
        loss = 1 - harmonic_mean_hellinger
    
        return loss
    def _compute_harmonic_mean_jsd(self, attention_store: AttentionStore,
                               indices_to_alter: List[int],
                               kernel_size: int = 7,
                               sigma: int = 1,
                               t: int = 10,
                               is_x=False) -> torch.Tensor:

        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=32,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0).to(torch.float32)  # [32, 32, 77]
        #Make it to 16x16 size. 
        attention_maps = F.interpolate(
            attention_maps.permute(2, 0, 1).unsqueeze(0),
            size=(16, 16),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)
        
        # padding = math.floor((kernel_size - 1) / 2)
        padding = 1
        smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to(attention_maps.device)
        
        jsd_values = []

        # indices_to_alter에 있는 모든 토큰 간의 JSD 계산
        num_tokens = len(indices_to_alter)
        for i in range(num_tokens):
            for j in range(i + 1, num_tokens):
                token_i = indices_to_alter[i]
                token_j = indices_to_alter[j] 
                # 두 토큰에 대한 Attention Map 추출
                tensor1 = attention_maps[:, :, token_i]
                tensor2 = attention_maps[:, :, token_j]
        
                tensor1 = F.pad(tensor1.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='reflect')
                tensor1 = smoothing(tensor1).squeeze(0).squeeze(0)
                
                tensor2 = F.pad(tensor2.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='reflect')
                tensor2 = smoothing(tensor2).squeeze(0).squeeze(0)
                
                epsilon = 1e-10
                # Attention Map을 확률분포로 변환
                if is_x:
                    tensor1_prob = tensor1.sum(dim=0) / tensor1.sum()
                    tensor2_prob = tensor2.sum(dim=0) / tensor2.sum()
                else:
                    tensor1_prob = tensor1 / tensor1.sum()
                    tensor2_prob = tensor2 / tensor2.sum()
        
                # 안정성을 위한 작은 상수 추가
                tensor1_prob = tensor1_prob + epsilon
                tensor2_prob = tensor2_prob + epsilon
        
                m = 0.5 * (tensor1_prob + tensor2_prob) # 두 분포의 평균 분포 M 계산
        
                # JSD 계산: 0.5 * (KL(P||M) + KL(Q||M))
                kl1 = torch.sum(tensor1_prob * torch.log(tensor1_prob / m))
                kl2 = torch.sum(tensor2_prob * torch.log(tensor2_prob / m))
                
                jsd = 0.5 * (kl1 + kl2)
                jsd = jsd / torch.log(torch.tensor(2.0)) #정규화
        
                jsd_values.append(jsd)
        
        # JSD 리스트를 텐서로 변환
        jsd_tensor = torch.stack(jsd_values)
        
        # 조화평균 JSD 계산
        harmonic_mean_jsd = len(jsd_tensor) / torch.sum(1.0 / jsd_tensor)
        
        # print(harmonic_mean_jsd)
        # harmonic_mean_jsd = torch.relu(t - harmonic_mean_jsd)
        harmonic_mean_jsd = 1-harmonic_mean_jsd
        
        # return harmonic_mean_jsd.to(torch.float16)
        return harmonic_mean_jsd

    def _compute_harmonic_mean_kl(self, attention_store: AttentionStore,
                                 indices_to_alter: List[int],
                                  kernel_size: int = 7,
                                  sigma: int = 1,
                                  t: int = 10,
                                 is_x = False) -> torch.Tensor:

        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=32,
            from_where=("down",),
            is_cross=True,
            select=0) # [32,32,77]
        # attention_maps = F.interpolate(
        #     attention_maps.permute(2, 0, 1).unsqueeze(0),
        #     size=(16, 16),
        #     mode='bilinear',
        #     align_corners=False
        # ).squeeze(0).permute(1, 2, 0)
        padding = 1
        smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to(attention_maps.device)
        
        kl_divergences = []
        
        # indices_to_alter에 있는 모든 토큰 간의 KL Divergence 계산
        num_tokens = len(indices_to_alter)
        for i in range(num_tokens):
            for j in range(i + 1, num_tokens):
                token_i = indices_to_alter[i]
                token_j = indices_to_alter[j]
                
                # 두 토큰에 대한 Attention Map 추출
                tensor1 = attention_maps[:, :, token_i].to(torch.float32) 
                tensor2 = attention_maps[:, :, token_j].to(torch.float32) 
    
                tensor1 = F.pad(tensor1.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='reflect')
                tensor1 = smoothing(tensor1).squeeze(0).squeeze(0)
                
                tensor2 = F.pad(tensor2.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='reflect')
                tensor2 = smoothing(tensor2).squeeze(0).squeeze(0)
                epsilon = 1e-10
                # KL Divergence 계산을 위해 두 Attention Map을 확률 분포로 변환
    
                # X축 또는 전체 분포로 합하여 확률 분포로 변환
                if is_x:
                    tensor1_prob = tensor1.sum(dim=0) / tensor1.sum()
                    tensor2_prob = tensor2.sum(dim=0) / tensor2.sum()
                else:
                    tensor1_prob = tensor1 / tensor1.sum()
                    tensor2_prob = tensor2 / tensor2.sum()
                
                # 작은 상수 추가하여 안정성 확보                    
                tensor1_prob = tensor1_prob + epsilon
                tensor2_prob = tensor2_prob + epsilon
    
                
                # KL Divergence 계산
                # print(f"var between {token_i} {token_j} is {variance_regulation}")
                kl_div = torch.sum(tensor1_prob * torch.log(tensor1_prob / tensor2_prob))
                
                # KL Divergence 값 저장
                kl_divergences.append(kl_div)
                
        # KL Divergence 리스트를 텐서로 변환
        kl_divergences_tensor = torch.stack(kl_divergences)

        
        # 조화평균 KL Divergence 계산
        harmonic_mean_kl = len(kl_divergences_tensor) / torch.sum(1.0 / kl_divergences_tensor)
        
        # print(harmonic_mean_kl)
        t = t
        harmonic_mean_kl = torch.relu((t-harmonic_mean_kl)) 
                
        return harmonic_mean_kl.to(torch.float32)

    def _compute_bhattacharyya_coefficient_loss(self, 
                                               attention_store, 
                                               indices_to_alter: List[int], 
                                               kernel_size: int = 7, 
                                               sigma: int = 1, 
                                               t: float = 10, 
                                               is_x: bool = False) -> torch.Tensor:
    
        # Attention Map Aggregation (주어진 코드 참고)
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=32,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0)  # Shape: [32, 32, 77]
    
        padding = 1
        smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to(attention_maps.device)
    
        bc_values = []
    
        num_tokens = len(indices_to_alter)
        for i in range(num_tokens):
            for j in range(i + 1, num_tokens):
                token_i = indices_to_alter[i]
                token_j = indices_to_alter[j]
    
                # 두 토큰의 Attention Map 추출
                tensor1 = attention_maps[:, :, token_i].to(torch.float32)  # Shape: [32, 32]
                tensor2 = attention_maps[:, :, token_j].to(torch.float32)  # Shape: [32, 32]
    
                # Gaussian Smoothing 적용
                tensor1 = F.pad(tensor1.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='reflect')
                tensor1 = smoothing(tensor1).squeeze(0).squeeze(0)
    
                tensor2 = F.pad(tensor2.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='reflect')
                tensor2 = smoothing(tensor2).squeeze(0).squeeze(0)
    
                epsilon = 1e-10
    
                # 확률 분포로 변환
                if is_x:
                    tensor1_prob = tensor1.sum(dim=0) / tensor1.sum()
                    tensor2_prob = tensor2.sum(dim=0) / tensor2.sum()
                else:
                    tensor1_prob = tensor1 / tensor1.sum()
                    tensor2_prob = tensor2 / tensor2.sum()
    
                # 작은 상수 추가해 안정성 확보
                tensor1_prob = tensor1_prob + epsilon
                tensor2_prob = tensor2_prob + epsilon
    
                # Bhattacharyya Coefficient 계산
                bc = torch.sum(torch.sqrt(tensor1_prob * tensor2_prob))
    
                # BC 값 저장
                bc_values.append(bc)
    
        # 리스트를 텐서로 변환
        bc_values_tensor = torch.stack(bc_values)
    
        # BC의 조화평균 (Harmonic Mean)
        harmonic_mean_bc = len(bc_values_tensor) / torch.sum(1.0 / (bc_values_tensor + 1e-10))
        print(harmonic_mean_bc)
        return harmonic_mean_bc.to(torch.float32)
        
    def apply_attention_masks(self, attention_store, tokens, sigma=1.5, p = 3, m = -1e8, threshold_percentile=90):
        """
        특정 토큰 인덱스에 대해 attention map을 처리하고, 생성된 mask를 모든 attn_processors에 추가.
    
        Parameters:
        - stable: Stable Diffusion 모델 객체
        - attention_store: attention map이 저장된 객체
        - tokens: 관심 있는 토큰 인덱스 리스트
        - sigma: Gaussian 필터의 표준 편차 (기본값: 1.5)
        - threshold_percentile: 주요 영역을 결정하기 위한 백분위수 (기본값: 90)
        """
        
        # Aggregate attention maps
        attention_map = aggregate_attention(attention_store=attention_store,
                                            res=32,
                                            from_where=("up", "down", "mid"),
                                            is_cross=True,
                                            select=0)
        # 각 토큰 인덱스에 대해 mask 생성
        masks = []
        for token_idx in tokens:
            # Convert attention maps for the token to numpy array
            attention_map_token = attention_map[:, :, token_idx].detach().cpu().to(torch.float32).numpy()
            
            # Apply Gaussian filter
            attention_map_token_g = gaussian_filter(attention_map_token, sigma=sigma)
            
            # Determine primary distribution areas by keeping the top values based on threshold_percentile
            threshold = np.percentile(attention_map_token_g, threshold_percentile)
            
            # Create mask for primary distribution areas
            mask_primary = attention_map_token_g >= threshold
            mask_primary_tensor = torch.tensor(mask_primary, dtype=torch.bool, device=attention_map.device)
            
            # Append mask to list
            masks.append(mask_primary_tensor)
        
        # 각 processor에 생성한 mask 추가
        for name in self.unet.attn_processors:
            processor = self.unet.attn_processors[name]
            processor.attention_indices = tokens
            for mask in masks:
                processor.masks.append(mask)
                processor.p = p
                processor.m = m
                
    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        # print("grad:", grad_cond.view(-1)[:5])
        latents = latents - step_size * grad_cond
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        attention_store : AttentionStore = None,
        token_idx = None,
        is_LODA: bool = False,
        LODA_steps: int = 10,
        inner_steps: int = 10,
        is_AFG: bool = False,
        AFG_steps: Optional[int] = None,
        LODA_idx = None,
        sigma_m = 1.5,
        sigma = 1,
        kernel_size = 7,
        p = 3,
        m = -1e8,
        threshold = 3,
        percentile = 90,
        scale_factor: int = 20,
        scale_range: Tuple[float, float] = (1., 0.5),
        visualize = False,
        store_global = False,
        **kwargs,
    ):

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds_nocfg = prompt_embeds.clone()
            add_text_embeds_nocfg = add_text_embeds.clone()
            add_time_ids_nocfg = add_time_ids.clone()
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        # print("prompt_embeds",prompt_embeds.shape)
        add_text_embeds = add_text_embeds.to(device)
        # print("add_text_embeds",add_text_embeds.shape)
        add_time_ids_nocfg = add_time_ids_nocfg.to(device).repeat(batch_size * num_images_per_prompt, 1)
        # print("atinocfg",add_time_ids_nocfg.shape)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
        # print("ati",add_time_ids.shape)
        

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 8.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)
        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                with torch.enable_grad():
                    if is_LODA:
                        if i == 0:
                            print("Inference with LODA")
                        if i < LODA_steps:
                            Divergence_loss = 100
                            latents = latents.clone().detach().requires_grad_(True)
                            
                            for step in range(inner_steps):
                                    
                                noise_pred_text = self.unet(
                                    latents,
                                    t,
                                    encoder_hidden_states=prompt_embeds_nocfg,
                                    timestep_cond=timestep_cond,
                                    cross_attention_kwargs=self.cross_attention_kwargs,
                                    added_cond_kwargs={"text_embeds": add_text_embeds_nocfg, "time_ids": add_time_ids_nocfg},
                                    return_dict=False,
                                )[0]
    
                                self.unet.zero_grad()
                                
                                Divergence_loss = self._compute_harmonic_mean_kl(attention_store=attention_store,
                                                                                 indices_to_alter=token_idx,
                                                                                 kernel_size = kernel_size,
                                                                                 sigma = sigma,
                                                                                 t = threshold)
                                # Divergence_loss = self._compute_harmonic_mean_bh(attention_store=attention_store,
                                #                                                   indices_to_alter=token_idx,
                                #                                                   kernel_size = kernel_size,
                                #                                                   sigma = sigma)

                                print(f"Divergence_loss is {Divergence_loss}")
                                latents = self._update_latent(latents=latents,
                                                              loss=Divergence_loss,
                                                              step_size=scale_factor * np.sqrt(scale_range[i]))
                                if visualize:
                                    vis_utils.show_cross_attention(attention_store=attention_store,
                                                           prompt=prompt,
                                                           tokenizer=self.tokenizer,
                                                               res=32,
                                                               from_where=("up", "down", "mid"),
                                                               indices_to_alter=token_idx,
                                                               orig_image=None)
                                

                    if is_AFG:
                        if AFG_steps == None:
                            AFG_steps = LODA_steps
                            
                        if i == AFG_steps:
                            print("Applying AFG")
                            self.apply_attention_masks(attention_store, tokens=token_idx, sigma = sigma_m, p = p, m =m, threshold_percentile = percentile)





                
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)


                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                
                if store_global:
                    attention_store.store_global()

                if visualize:
                    vis_utils.show_cross_attention(attention_store=attention_store,
                                           prompt=prompt,
                                           tokenizer=self.tokenizer,
                                               res=32,
                                               from_where=("up", "down", "mid"),
                                               indices_to_alter=token_idx,
                                               orig_image=None)

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()
        
        del latents.grad
        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            elif latents.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    self.vae = self.vae.to(latents.dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)