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
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, is_accelerate_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from scipy.ndimage import gaussian_filter


from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline


from util import attention_highlight
from utils.gaussian_smoothing import GaussianSmoothing
from utils.ptp_utils_LODA import AttentionStore, aggregate_attention
from utils import vis_utils

from IPython.display import display
logger = logging.get_logger(__name__)
def print_gpu_utilization():
    allocated = torch.cuda.memory_allocated() / 1024**3  # 기가바이트 단위로 변환
    reserved = torch.cuda.memory_reserved() / 1024**3    # 기가바이트 단위로 변환
    print(f'Allocated Memory: {allocated:.2f} GB')
    print(f'Reserved Memory:  {reserved:.2f} GB')
    
class AttendAndExcitePipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]



    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:

            # Process input prompt text #residual token embeddings
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

                    
            # Edit text embedding conditions with residual token embeddings.
            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds])

        return text_inputs, prompt_embeds

    def _compute_max_attention_per_index(self,
                                         attention_maps: torch.Tensor,
                                         indices_to_alter: List[int],
                                         smooth_attentions: bool = False,
                                         sigma: float = 0.5,
                                         kernel_size: int = 3,
                                         normalize_eot: bool = False) -> List[torch.Tensor]:
        """ Computes the maximum attention value for each of the tokens we wish to alter. """
        last_idx = -1
        if normalize_eot:
            prompt = self.prompt
            if isinstance(self.prompt, list):
                prompt = self.prompt[0]
            last_idx = len(self.tokenizer(prompt)['input_ids']) - 1
        attention_for_text = attention_maps[:, :, 1:last_idx]
        
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token
        indices_to_alter = [index - 1 for index in indices_to_alter]

        # Extract the maximum values
        max_indices_list = []
        for i in indices_to_alter:
            image = attention_for_text[:, :, i]
            if smooth_attentions:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cpu()
                input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                image = smoothing(input).squeeze(0).squeeze(0)
            max_indices_list.append(image.max())
        return max_indices_list
        
    def _compute_gaussian_distribution_per_index(self,
                                                 attention_store,
                                                 indices_to_alter: List[int]) -> List[Dict[str, torch.Tensor]]:
        """
        각 토큰에 대해 Attention Map의 공간적 좌표 기반으로 Gaussian 분포의 평균(중심 좌표)과 분산을 계산합니다.
        
        attention_maps: Attention Map (height, width, tokens) -> 3D 텐서
        indices_to_alter: Attention 값을 측정할 토큰의 인덱스 리스트
        """
        attention_maps = aggregate_attention(attention_store=attention_store,
                                             res=16,
                                             from_where=("up", "down", "mid"),
                                             is_cross=True,
                                             select=0)
        
        statistics = []
        
        # 좌표계 설정 (height와 width에 대한 좌표값 생성)
        height = attention_maps.size(0)
        width = attention_maps.size(1)
        
        x_coords = torch.arange(width).float()  # width에 대한 좌표값 (x축)
        y_coords = torch.arange(height).float()  # height에 대한 좌표값 (y축)
        
        # 좌표의 meshgrid 생성
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')  # (height, width)
        x_grid = x_grid.to(attention_maps.device)  # device에 맞추어 이동
        y_grid = y_grid.to(attention_maps.device)  # device에 맞추어 이동
        
        # 각 토큰에 대한 공간적 분포 계산
        for i in indices_to_alter:
            # Attention map에서 각 토큰에 대한 attention 값 (height, width)
            attention = attention_maps[:, :, i]  # (height, width)
            
            # Attention 값의 총합을 구함 (나중에 가중 평균 계산에 사용)
            sum_attention = attention.sum()  # 스칼라 값
            
            if sum_attention == 0:
                mean_x = torch.tensor(0.0, device=attention_maps.device)
                mean_y = torch.tensor(0.0, device=attention_maps.device)
                variance_x = torch.tensor(0.0, device=attention_maps.device)
                variance_y = torch.tensor(0.0, device=attention_maps.device)
            else:
                # X 좌표에 대한 가중 평균 (mean_x) 계산
                mean_x = (attention * x_grid).sum() / sum_attention  # Attention-weighted 평균 x좌표
                # Y 좌표에 대한 가중 평균 (mean_y) 계산
                mean_y = (attention * y_grid).sum() / sum_attention  # Attention-weighted 평균 y좌표
                
                # X 좌표에 대한 분산 (variance_x) 계산
                variance_x = (attention * (x_grid - mean_x) ** 2).sum() / sum_attention
                # Y 좌표에 대한 분산 (variance_y) 계산
                variance_y = (attention * (y_grid - mean_y) ** 2).sum() / sum_attention
            
            # 결과 저장
            statistics.append({
                'mean_x': mean_x,
                'mean_y': mean_y,
                'variance_x': variance_x,
                'variance_y': variance_y
            })
        
        return statistics
        
    # def update_weight(self, attention_store, i)
        
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
                                            res=16,
                                            from_where=("up", "down", "mid"),
                                            is_cross=True,
                                            select=0)
        # 각 토큰 인덱스에 대해 mask 생성
        masks = []
        for token_idx in tokens:
            # Convert attention maps for the token to numpy array
            attention_map_token = attention_map[:, :, token_idx].detach().cpu().numpy()
            
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

  

    def _compute_harmonic_mean_kl_raw(self, attention_store: AttentionStore,
                                     indices_to_alter: List[int],
                                      kernel_size: int = 7,
                                      sigma: int = 1,
                                     is_x = False) -> torch.Tensor:

        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=16,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0) # shape [16,16,77]
        attention_maps_32 = aggregate_attention(
            attention_store=attention_store,
            res=32,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0) # [32,32,77]
        attention_maps_8 = aggregate_attention(
            attention_store=attention_store,
            res=8,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0) #[8,8,77]

        attention_maps_32_resized = F.interpolate(
            attention_maps_32.permute(2, 0, 1).unsqueeze(0),
            size=(16, 16),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)

        attention_maps_8_resized = F.interpolate(
            attention_maps_32.permute(2, 0, 1).unsqueeze(0),
            size=(16, 16),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)

        attention_maps = (
            attention_maps +
            attention_maps_32_resized +
            attention_maps_8_resized
        ) / 3
        
        # padding = math.floor((kernel_size - 1) / 2)
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
                tensor1 = attention_maps[:, :, token_i]
                tensor2 = attention_maps[:, :, token_j]

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
        
        t = 3
        harmonic_mean_kl = torch.relu((t-harmonic_mean_kl)) 
        
        return harmonic_mean_kl

    @staticmethod
    def _compute_loss(max_attention_per_index: List[torch.Tensor], return_losses: bool = False) -> torch.Tensor:
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        losses = [max(0, 1. - curr_max) for curr_max in max_attention_per_index]
        loss = max(losses)
        if return_losses:
            return loss, losses
        else:
            return loss

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        print(grad_cond.view(-1)[:5])
        latents = latents - step_size * grad_cond
        return latents


    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            attention_store: AttentionStore,
            indices_to_alter: List[int],
            target_attention_store: AttentionStore = None,
            attention_res: int = 16,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            max_iter_to_alter: Optional[int] = 25,
            run_standard_sd: bool = False,
            thresholds: Optional[dict] = {0: 0.05, 10: 0.5, 20: 0.8},
            scale_factor: int = 20,
            scale_range: Tuple[float, float] = (1., 0.5),
            smooth_attentions: bool = True,
            sigma: float = 0.5,
            kernel_size: int = 3,
            sd_2_1: bool = False,
            p = 3,
            m = -1e8,
            sigma_m = 1.5,
            percentile = 90
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        Examples:
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
            :type attention_store: object
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        self.prompt = prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_inputs, prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

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
        
        # 6. Prepare extra step kargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))

        if max_iter_to_alter is None:
            max_iter_to_alter = len(self.scheduler.timesteps) + 1

        #param to train
        # to_k_params = [param for name, param in self.unet.named_parameters() if "to_k" in name]
        # optimizer = torch.optim.Adam(to_k_params, lr=5e-5)
        self.unet.requires_grad_(False)
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        # shape for add latent
        new_shape = (latents.shape[0], latents.shape[1], latents.shape[2] // 4, latents.shape[3] // 4)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                with torch.enable_grad():

                    if not run_standard_sd:
                        if i < max_iter_to_alter:
                            latents = latents.clone().detach().requires_grad_(True)

                            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                                self.unet = self.unet.to("cuda:1")
                                latents = latents.to("cuda:1")
                                prompt_embeds = prompt_embeds.to("cuda:1")
                            else:
                                None
                            
                            noise_pred_text = self.unet(latents,
                                                        t,
                                                        encoder_hidden_states=prompt_embeds[0].unsqueeze(0),
                                                        cross_attention_kwargs=cross_attention_kwargs).sample
                            
                            self.unet.zero_grad()
                            kl_loss = self._compute_harmonic_mean_kl_raw(attention_store=attention_store,
                                                                               indices_to_alter=indices_to_alter)

                            latents = self._update_latent(latents=latents, loss=3*kl_loss,
                                                                  step_size=scale_factor * np.sqrt(scale_range[i]))
                            
                            latents = latents.to("cuda:0")
                            self.unet = self.unet.to("cuda:0")
                            prompt_embeds = prompt_embeds.to("cuda:0")

                        
                            # kl_loss.backward()

                            
                            # optimizer.step()
                            print(f"kl_loss is {kl_loss}")

                        
                        if i == max_iter_to_alter:
                            self.apply_attention_masks(attention_store, tokens=indices_to_alter, sigma = sigma_m, p = p, m =m, threshold_percentile = percentile)



                            
                            


                            
                            
                            


                
                            
                # expand the latents if we are doing classifier free guidance
                
                
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents 
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample


                attention_store.store_global()

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)