#!/usr/bin/env python
# coding: utf-8
"""
0512_ICCV_rebuttal_DINO_IA_eval.py
---------------------------------
• 폴더 단위로 TA / IA 계산
• --summary_file 옵션을 주면 한 방법론 전체 결과를 하나의 txt에 누적 저장
"""

import os
import sys
import argparse
import glob
import torch
import clip
from PIL import Image
import numpy as np

# ─────────────────────────────────────────────────────────────
# DINO 관련
# ─────────────────────────────────────────────────────────────
import timm
from torchvision import transforms as T




# ─────────────────────────────────────────────────────────────
# 1. 새 유틸 함수 –  level1 폴더명에 따라 barn 제거
# ─────────────────────────────────────────────────────────────
def filter_concepts_for_B_folders(concepts, level1_name):
    """
    O1B / O2B / O3B 와 같이 '...B' 로 끝나는 상위 폴더에서는
    background 용도로만 등장한 'barn'을 IA 계산에서 제외한다.
    """
    if level1_name.upper().endswith("B"):
        return [c for c in concepts if c != "barn"]
    return concepts





def load_dino(device):
    model = timm.create_model(
        "vit_base_patch16_224_dino",
        pretrained=True,
        num_classes=0,
    ).to(device).eval()

    preprocess = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return model, preprocess


def compute_dino_embedding_image(dino_model, dino_preprocess, device, image_path):
    img = Image.open(image_path).convert("RGB")
    inp = dino_preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = dino_model(inp)  # (1, 768)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0]


# ─────────────────────────────────────────────────────────────
# CLIP 관련
# ─────────────────────────────────────────────────────────────
def compute_clip_embedding_image(model, preprocess, device, image_path):
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features[0]


def compute_clip_embedding_text(model, device, text):
    text_input = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features[0]


def compute_similarity(a, b):
    return float((a * b).sum())


# # ─────────────────────────────────────────────────────────────
# # 레퍼런스 매핑 / 유틸
# # ─────────────────────────────────────────────────────────────
# REFERENCES_MAP = {
#     "bear": "/data/wysgene19/personalization_dataset/ICCV_Final_dataset/O_bear",
#     "cat": "/data/wysgene19/personalization_dataset/ICCV_Final_dataset/O_cat",
#     "dog": "/data/wysgene19/personalization_dataset/ICCV_Final_dataset/O_dog",
#     "barn": "/data/wysgene19/personalization_dataset/ICCV_Final_dataset/B_barn",
# }


# 예시: 개체별 레퍼런스 이미지 폴더 매핑
REFERENCES_MAP = {
    "bear": "/data/wysgene19/ICCV2025/Final_dataset/O_bear",
    "cat":  "/data/wysgene19/ICCV2025/Final_dataset/O_cat",
    "dog":  "/data/wysgene19/ICCV2025/Final_dataset/O_dog",
    "barn": "/data/wysgene19/ICCV2025/Final_dataset/B_barn",
}



def get_image_files(folder):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    image_files = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith(exts):
                image_files.append(os.path.join(root, fname))
    return image_files


def gather_reference_embeddings_dino(dino_model, dino_preprocess, device, concept):
    if concept not in REFERENCES_MAP:
        return None
    ref_folder = REFERENCES_MAP[concept]
    image_paths = get_image_files(ref_folder)
    if not image_paths:
        return None
    embeds = [
        compute_dino_embedding_image(dino_model, dino_preprocess, device, p)
        for p in image_paths
    ]
    embeds = torch.stack(embeds, dim=0)
    mean_embed = embeds.mean(dim=0)
    mean_embed = mean_embed / mean_embed.norm()
    return mean_embed


def parse_concepts_from_folder_name(folder_name):
    folder_name_lower = folder_name.lower()
    return [c for c in REFERENCES_MAP if c in folder_name_lower]


# ─────────────────────────────────────────────────────────────
# 파일 쓰기 헬퍼
# ─────────────────────────────────────────────────────────────
def write_results(path, text, append=False):
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        f.write(text)


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="예: /path/to/O1 등 분석 대상 폴더 경로",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default=None,
        help="한 방법론 전체 결과를 모을 txt 파일 경로",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
    dino_model, dino_preprocess = load_dino(device)

    # 1단계 하위 폴더
    level1_folders = [e.path for e in os.scandir(args.folder) if e.is_dir()]

    all_subfolder_TA, all_subfolder_IA = [], []
    results_per_level1 = {}

    for level1_folder in level1_folders:
        level1_name = os.path.basename(level1_folder)
        results_per_level1[level1_name] = {
            "subfolders": [],
            "TA_vals": [],
            "IA_vals": [],
        }

        level2_folders = [e.path for e in os.scandir(level1_folder) if e.is_dir()]

        for level2_folder in level2_folders:
            level2_name = os.path.basename(level2_folder)
            print("Analyzing:", level2_name)

            # TA
            text_embed = compute_clip_embedding_text(clip_model, device, level2_name)
            image_paths = get_image_files(level2_folder)
            if not image_paths:
                continue
            sims_ta = [
                compute_similarity(
                    text_embed,
                    compute_clip_embedding_image(
                        clip_model, clip_preprocess, device, p
                    ),
                )
                for p in image_paths
            ]
            mean_ta = float(np.mean(sims_ta))

            # IA
            concepts = parse_concepts_from_folder_name(level2_name)
            # ※ ‘...B’ 상위 폴더라면 barn 제거
            concepts = filter_concepts_for_B_folders(concepts, level1_name)
            if not concepts:
                mean_ia = 0.0
            else:
                ref_embeds = [
                    gather_reference_embeddings_dino(
                        dino_model, dino_preprocess, device, c
                    )
                    for c in concepts
                    if gather_reference_embeddings_dino(
                        dino_model, dino_preprocess, device, c
                    )
                    is not None
                ]
                if not ref_embeds:
                    mean_ia = 0.0
                else:
                    ref_mean = torch.stack(ref_embeds, dim=0).mean(dim=0)
                    sims_ia = [
                        compute_similarity(
                            ref_mean,
                            compute_dino_embedding_image(
                                dino_model, dino_preprocess, device, p
                            ),
                        )
                        for p in image_paths
                    ]
                    mean_ia = float(np.mean(sims_ia))

            # 저장
            rec = results_per_level1[level1_name]
            rec["subfolders"].append(level2_name)
            rec["TA_vals"].append(mean_ta)
            rec["IA_vals"].append(mean_ia)
            all_subfolder_TA.append(mean_ta)
            all_subfolder_IA.append(mean_ia)

    # 결과 문자열 구성
    result_lines = [f"=== IA/TA 계산 결과 (root: {args.folder}) ===\n"]
    for lvl1, rec in results_per_level1.items():
        if rec["TA_vals"]:
            lvl1_ta = float(np.mean(rec["TA_vals"]))
            lvl1_ia = float(np.mean(rec["IA_vals"]))
        else:
            lvl1_ta = lvl1_ia = 0.0
        result_lines.append(f"[{lvl1}] avg TA={lvl1_ta:.4f}, IA={lvl1_ia:.4f}")
        for sub, ta, ia in zip(
            rec["subfolders"], rec["TA_vals"], rec["IA_vals"]
        ):
            result_lines.append(f"   - {sub}: TA={ta:.4f}, IA={ia:.4f}")
        result_lines.append("")
    overall_ta = float(np.mean(all_subfolder_TA)) if all_subfolder_TA else 0.0
    overall_ia = float(np.mean(all_subfolder_IA)) if all_subfolder_IA else 0.0
    result_lines.append(f"Overall TA={overall_ta:.4f}, IA={overall_ia:.4f}\n")

    result_txt = "\n".join(result_lines)

    # 저장 방식 분기
    if args.summary_file:
        first_time = not os.path.exists(args.summary_file)
        if first_time:
            headline = "########## IA / TA SUMMARY ##########\n"
            write_results(args.summary_file, headline, append=False)
        write_results(args.summary_file, result_txt + "\n", append=True)
        print(f"[완료] 결과가 {args.summary_file} 에 추가되었습니다.")
    else:
        local_path = os.path.join(args.folder, "IA_DINO_TA_result.txt")
        write_results(local_path, result_txt, append=False)
        print(f"[완료] 결과가 {local_path} 에 저장되었습니다.")


if __name__ == "__main__":
    main()