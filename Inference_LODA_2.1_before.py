import argparse
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from utils.ptp_utils_LODA import AttentionStore, register_attention_control, aggregate_attention
from utils import vis_utils
from pipeline_sd2_1_loda import LODA_StableDiffusion2_1Pipeline
import warnings
import torch
import torch.nn as nn
import json
import re


warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Inference for LODA Stable Diffusion 2.1")
    parser.add_argument("--seeds", type=str, default="0", help="For Selective Tuning")
    parser.add_argument("--seed_range", type=int, default=None, help="range of seed for matric")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for image generation")
    parser.add_argument("--model_name",
                        type=str,
                        default="./models/stable-diffusion-2-1",
                        help="Model_NAme")
    parser.add_argument("--output_path", type=str, default="./outputs/sd2.1", help="output_path")
    parser.add_argument("--model_dir", type=str, default="./model_ckpt/sd_2.1/attn2.to_v", help="models_path")
    parser.add_argument("--rank", type=str, default="64", help="Rank of adapters")
    parser.add_argument("--height", type=int, default=768, help="Image height")
    parser.add_argument("--width", type=int, default=768, help="Image width")
    parser.add_argument("--is_LODA", action="store_true", help="Enable LODA")
    parser.add_argument("--LODA_steps", type=int, default=10, help="Number of LODA steps")
    parser.add_argument("--inner_steps", type=int, default=1, help="Number of inner steps")
    parser.add_argument("--is_AFG", action="store_true", help="Enable AFG")
    parser.add_argument("--AFG_steps", type=int, default=10, help="Number of AFG steps")
    parser.add_argument("--sigma", type=float, default=1.5, help="Sigma value for processing")
    parser.add_argument("--kernel_size", type=int, default=7, help="Kernel size")
    parser.add_argument("--p", type=int, default=0, help="Additional parameter p")
    parser.add_argument("--m", type=int, default=0, help="Additional parameter m")
    parser.add_argument("--threshold", type=int, default=1, help="Threshold value")
    parser.add_argument("--percentile", type=int, default=90, help="Percentile value")
    parser.add_argument("--scale_factor", type=int, default=20, help="Scale factor")
    parser.add_argument("--scale_range",
                        type=str,
                        default="1.0,0.5",
                        help="Scale range as comma-separated values")
    parser.add_argument("--config", type=str, default='{}', help="Configuration dictionary in JSON format")
    

    
    return parser.parse_args()

    
# Custom LoRA 레이어 정의
class CustomLoRALayer(nn.Module):
    def __init__(self, base_layer, lora_As, lora_Bs, *args, **kwargs):
        super(CustomLoRALayer, self).__init__()
        self.concept_num = len(lora_As)
        self.base_layer = base_layer
        
        # lora_As와 lora_Bs를 ModuleDict에 nn.Linear로 저장
        self.lora_As = nn.ModuleDict(lora_As)
        self.lora_Bs = nn.ModuleDict(lora_Bs)

        self.idx = []
    def reset(self):
        self.lora_AS = nn.ModuleDict({})
        self.lora_BS = nn.ModuleDict({})
        self.idx = []
        
    def forward(self, x):
        # 원래 base_layer를 통한 계산
        base_output = self.base_layer(x)
        lora_outputs = []
        # LoRA 계산
        for idx, key in zip(self.idx, self.lora_As):
            lora_A_output = self.lora_As[key](x[:,idx,:])  # nn.Linear로 forward 적용
            lora_B_output = self.lora_Bs[key](lora_A_output)  # nn.Linear로 forward 적용
            lora_outputs.append(lora_B_output)
        for idx,lora_output in zip(self.idx,lora_outputs):
            base_output[:,idx,:] += lora_output
        
        return base_output

def replace_lora_layers(stable, LoRAs, cas = ["k"]):
    # attn2.to_v의 LoRA 레이어를 CustomLoRALayer로 교체
    for name, module in stable.unet.named_modules():
        for ca in cas:
            if name.endswith(f"attn2.to_{ca}"):
                parent_name, child_name = name.rsplit('.', 1)
                parent_module = dict(stable.unet.named_modules())[parent_name]
    
                # 새로운 CustomLoRALayer 생성
                lora_As = {}
                lora_Bs = {}
    
                for LoRA in LoRAs:
                    # 여기서 LoRA는 하나의 concept에 대한 LoRA
                    for key in LoRA.keys():
                        if name in key:
                            # down weight는 lora_As에, up weight는 lora_Bs에 저장
                            if "lora_A" in key:
                                down_weight = LoRA[key]
                                # nn.Linear로 변환 (입력 차원과 출력 차원을 자동 추론)
                                in_features = down_weight.shape[1]
                                out_features = down_weight.shape[0]
                                lora_As[f'{len(lora_As)}'] = nn.Linear(in_features, out_features, bias=False)
                                lora_As[f'{len(lora_As)-1}'].weight = nn.Parameter(down_weight)
                            elif "lora_B" in key:
                                up_weight = LoRA[key]
                                in_features = up_weight.shape[1]
                                out_features = up_weight.shape[0]
                                lora_Bs[f'{len(lora_Bs)}'] = nn.Linear(in_features, out_features, bias=False)
                                lora_Bs[f'{len(lora_Bs)-1}'].weight = nn.Parameter(up_weight)
    
                print(lora_As)
                print(lora_Bs)
    
                new_module = CustomLoRALayer(
                    base_layer=module,
                    lora_As=lora_As,
                    lora_Bs=lora_Bs,
                )
    
                # setattr을 사용하여 실제 모듈 교체
                setattr(parent_module, child_name, new_module)
                print(f"Changed {name} from {module.__class__.__name__} to {new_module.__class__.__name__}")
    
    return stable

def update_custom_layer_indices(stable, new_indices):
    for name, module in stable.unet.named_modules():
        # Check if the module is an instance of CustomLoRALayer
        if isinstance(module, CustomLoRALayer):
            module.idx = new_indices  # Update the idx attribute
            print(f"Updated idx for {name} to {new_indices}")

def get_token_indices(prompt, config, tokenizer):
    """
    prompt 내에서 config의 각 key(예: dog_1, cat_1 등)가 의미하는
    기본 단어(dog, cat)가 등장하는 첫 위치(토큰 인덱스)를
    config 순서대로 찾아서 리스트로 반환한다.

    동일 단어가 여러 번 config에 등장할 경우(예: dog_1, dog_2),
    왼쪽부터 사용하지 않은 토큰 구간을 순차적으로 매칭한다.
    """
    # 1) 프롬프트 토큰화
    tokens = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    tokenized_text = tokenizer.convert_ids_to_tokens(tokens.input_ids[0])

    # 이미 사용한 토큰 인덱스를 기록(중복 매칭 방지)
    used_token_positions = set()

    result_indices = []
    print(f"The Given Prompt is {prompt}")
    # 2) config 순서대로 순회
    for key in config.keys():
        # 예: "dog_1" -> "dog", "dog_2" -> "dog", "cat_1" -> "cat"
        base_word = re.sub(r'_\d+', '', key)
        print(key)
        # base_word를 다시 토큰화해서, 그 토큰 시퀀스를 매칭해야 함
        word_tokens = tokenizer.tokenize(base_word)

        matched_index = None

        # 3) 왼쪽부터 가능한 매칭 구간 찾기
        for i in range(len(tokenized_text) - len(word_tokens) + 1):
            # 해당 구간이 이미 다른 매칭에 사용됐는지 체크
            if any((i + offset) in used_token_positions for offset in range(len(word_tokens))):
                continue

            # word_tokens 시퀀스와 정확히 일치하는지 비교
            if tokenized_text[i : i + len(word_tokens)] == word_tokens:
                matched_index = i
                # 이 구간(토큰들)을 사용 처리
                for offset in range(len(word_tokens)):
                    used_token_positions.add(i + offset)
                break

        # 4) 매칭되었다면 결과에 추가
        if matched_index is not None:
            print(f"{key} => {base_word} for idx {matched_index} : {word_tokens}")
            result_indices.append(matched_index)

    return result_indices

def get_object_indices(prompt, config, tokenizer):
    """
    prompt 내에서 config의 각 key(예: dog_1, cat_1 등)가 의미하는
    기본 단어(dog, cat)가 등장하는 첫 위치(토큰 인덱스)를
    config 순서대로 찾아서 리스트로 반환한다.

    동일 단어가 여러 번 config에 등장할 경우(예: dog_1, dog_2),
    왼쪽부터 사용하지 않은 토큰 구간을 순차적으로 매칭한다.
    """
    # 1) 프롬프트 토큰화
    tokens = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    tokenized_text = tokenizer.convert_ids_to_tokens(tokens.input_ids[0])

    # 이미 사용한 토큰 인덱스를 기록(중복 매칭 방지)
    used_token_positions = set()

    result_indices = []
    print(f"The Given Prompt is {prompt}")
    # 2) config 순서대로 순회
    Objects = ["dog","cat","bear"] #Object에 대한 index만 가져옴 => 이걸로 latent optimize 하기 때문에,
    for key in config.keys():
        # 예: "dog_1" -> "dog", "dog_2" -> "dog", "cat_1" -> "cat"
        base_word = re.sub(r'_\d+', '', key)
        print(key)
        # base_word를 다시 토큰화해서, 그 토큰 시퀀스를 매칭해야 함
        word_tokens = tokenizer.tokenize(base_word)

        matched_index = None

        # 3) 왼쪽부터 가능한 매칭 구간 찾기
        for i in range(len(tokenized_text) - len(word_tokens) + 1):
            # 해당 구간이 이미 다른 매칭에 사용됐는지 체크
            if any((i + offset) in used_token_positions for offset in range(len(word_tokens))):
                continue

            # word_tokens 시퀀스와 정확히 일치하는지 비교
            if tokenized_text[i : i + len(word_tokens)] == word_tokens:
                matched_index = i
                # 이 구간(토큰들)을 사용 처리
                for offset in range(len(word_tokens)):
                    used_token_positions.add(i + offset)
                break

        # 4) 매칭되었다면 결과에 추가
        if matched_index is not None:
            if base_word in Objects:
                print(f"{key} => {base_word} for idx {matched_index} : {word_tokens}")
                result_indices.append(matched_index)

    return result_indices

def get_token_key(prompt, config, tokenizer):
    """
    config의 각 key(예: "dog_1", "cat_1")에 대해
    - 해당 key가 의미하는 base_word (dog, cat 등)을 prompt 내에서 찾고
    - 못 찾으면 None
    - 찾으면 해당 시작 인덱스를 반환
    
    반환: { "dog_1": 인덱스 or None, "cat_1": 인덱스 or None, ... }
    """
    tokens = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    tokenized_text = tokenizer.convert_ids_to_tokens(tokens.input_ids[0])
    used_token_positions = set()

    results = {}
    for key in config.keys():
        base_word = re.sub(r'_\d+', '', key)
        word_tokens = tokenizer.tokenize(base_word)

        matched_index = None
        for i in range(len(tokenized_text) - len(word_tokens) + 1):
            # 이미 사용한 토큰과 겹치면 스킵
            if any((i + offset) in used_token_positions for offset in range(len(word_tokens))):
                continue

            # tokenized_text 구간과 word_tokens가 완전히 일치하는지 검사
            if tokenized_text[i : i + len(word_tokens)] == word_tokens:
                matched_index = i
                # 사용 처리
                for offset in range(len(word_tokens)):
                    used_token_positions.add(i + offset)
                break

        results[key] = matched_index

    return results

from safetensors.torch import load_file
def setting_lora(config, base_dir="./model_ckpt/sd_2.1/attn2.to_v", rank=64, concept_indices=None):
    """
    concept_indices: { "dog_1": some_index or None, ... }
    """
    loras = []
    for key, ckpt in config.items():
        # prompt에 해당 개념이 없으면(None이면), LoRA 로딩을 스킵
        if concept_indices is not None and concept_indices.get(key) is None:
            print(f"[WARNING] '{key}' 개념이 prompt에 없어 LoRA 로딩을 스킵합니다.")
            continue

        OUTPUT_DIR = f"{base_dir}_{key}"
        load_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{ckpt}")
        
        if not os.path.exists(load_dir):
            print(f"[ERROR] LoRA checkpoint path does not exist: {load_dir}")
            continue  # 파일이 없으면 로딩하지 않음

        print(f"Loading LoRA from: {load_dir}")
        lora_dict = load_file(os.path.join(load_dir, "pytorch_lora_weights.safetensors"))
        loras.append(lora_dict)
    return loras

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_image(image, args, seed, out_path):

    image_path = os.path.join(out_path, f"{seed}.png")
    args_path = os.path.join(out_path, f"{seed}_args.txt")
    image.save(image_path)
    print(f"Image saved at: {image_path}")

def save_ca(images, args, seed, out_path):
    out_path = os.path.join(out_path, "ca/")
    for idx,image in enumerate(images):
        image_path = os.path.join(out_path, f"{seed}_{idx}.png")
        
        Image.fromarray(image).save(image_path)
    print(f"CA saved at: {image_path}")
    
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ###Loading Models
    print(f"Loading model from {args.model_name}...")
    pipe = LODA_StableDiffusion2_1Pipeline.from_pretrained(args.model_name)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    tokenizer = pipe.tokenizer

    try:
        config = json.loads(args.config)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in --config argument.")

    token_indices = get_token_indices(args.prompt, config, tokenizer) #token to use ToVA
    token_key = get_token_key(args.prompt, config, tokenizer) #for update
    object_indices = get_object_indices(args.prompt, config, tokenizer) #
    print(token_indices)
    print(token_key)
    print("Object to indices is :", object_indices)
    ###Setting the ToVA
    
    loras = setting_lora(config, base_dir=args.model_dir, concept_indices=token_key)
    pipe = replace_lora_layers(pipe,loras,cas = ["k"]).to(device)

    ###Make a idx and update ToVa
    update_custom_layer_indices(pipe, token_indices)
    
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.seed_range is not None:
        seeds = range(args.seed_range)
    scale_range = tuple(map(float, args.scale_range.split(",")))
    
    
    # from datetime import datetime
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # out_path = os.path.join(args.output_path, args.prompt, os.path.basename(args.model_dir), timestamp)
    out_path = os.path.join(args.output_path, args.prompt)
    os.makedirs(out_path, exist_ok=True) #Folder for config and images
    # os.makedirs(os.path.join(out_path, "ca"), exist_ok=True) #folder for Cross Attention
    args_path = os.path.join(out_path, "config.json")
    print(f"Args saved at: {args_path}")
    
    # JSON 파일 저장
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    
    # 시드 설정
    set_seed(42)
    for seed in seeds:
        g = torch.Generator('cuda').manual_seed(seed)
        controller = AttentionStore()
        register_attention_control(pipe, controller)
        
        image = pipe(
            prompt=args.prompt,
            generator=g,
            attention_store=controller,
            token_idx = object_indices, #this is for loda idx
            height=args.height,
            width=args.width,
            is_LODA=args.is_LODA,
            LODA_steps=args.LODA_steps,
            inner_steps=args.inner_steps,
            is_AFG=args.is_AFG,
            AFG_steps=args.AFG_steps,
            sigma=args.sigma,
            kernel_size=args.kernel_size,
            p=args.p,
            m=args.m,
            threshold=args.threshold,
            percentile=args.percentile,
            scale_factor=args.scale_factor,
            scale_range=scale_range,
            num_inference_steps = 2
        ).images[0]
        
        # ca = vis_utils.show_cross_attention(
        #     attention_store=controller,
        #     prompt=args.prompt,
        #     tokenizer=tokenizer,
        #     res=24,
        #     from_where=("up", "down", "mid"),
        #     indices_to_alter = token_indices,
        # )

        save_image(image,args,seed,out_path)
        # save_ca(ca,args,seed,out_path)




if __name__ == "__main__":
    main()
