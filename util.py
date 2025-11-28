import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import torch
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F


def attention_highlight(attention_maps, idx, visualize=True, device='cuda'):
    # 8번째 토큰의 attention map 추출
    token_attention_map = attention_maps[:, :, idx].detach().cpu().numpy()
    
    # 최대 attention 값을 갖는 위치 찾기
    max_attention_value = np.max(token_attention_map)
    max_position = np.unravel_index(np.argmax(token_attention_map), token_attention_map.shape)
    
    # 가우시안 분포 생성
    sigma = 2  # 가우시안 분포의 표준편차 설정 (분포의 넓이를 조정)
    gaussian_distribution = np.zeros_like(token_attention_map)
    gaussian_distribution[max_position] = 1  # 최대값 위치에 1 설정
    gaussian_distribution = gaussian_filter(gaussian_distribution, sigma=sigma)
    
    # 가우시안 분포를 attention map에 곱하여 적용
    filtered_attention_map = token_attention_map * gaussian_distribution
    
    if visualize:
        plt.imshow(filtered_attention_map, cmap='viridis', interpolation='nearest')
        plt.colorbar()  # 색상 바 추가
        plt.title(f'Before Shift: Gaussian Filtered Attention Map for Token {idx}')
        plt.show()
    
    # 텐서를 GPU로 보내기 (device에 맞게 이동)
    filtered_attention_map = torch.tensor(filtered_attention_map).to(device)
    
    # attention map의 평균값 계산
    mean_attention = filtered_attention_map.mean()
    
    # 평균보다 큰 값들만 남기고 나머지는 0으로 마스크 처리
    filtered_attention_map = torch.where(filtered_attention_map > mean_attention, filtered_attention_map, torch.tensor(0.0).to(device))
    
    if visualize:
        plt.imshow(filtered_attention_map.detach().cpu().numpy(), cmap='viridis', interpolation='nearest')
        plt.colorbar()  # 색상 바 추가
        plt.title(f'Filtered Attention Map for Token {idx}')
        plt.show()
    
    print(filtered_attention_map.shape)
    
    # Scale token_attention_maps from 16x16 to match 64x64 latents
    scaled_attention_maps = F.interpolate(filtered_attention_map.unsqueeze(0).unsqueeze(0), size=(64, 64), mode='nearest')
    
    # 배치 차원과 채널 차원 제거
    scaled_attention_maps = scaled_attention_maps.squeeze(0).squeeze(0)
    
    if visualize:
        plt.imshow(scaled_attention_maps.detach().cpu().numpy(), cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(f'Scaled Attention Map for Token {idx}')
        plt.show()

    return scaled_attention_maps 


def display_image_grid(folder_path, num_cols=4):
    """
    지정된 폴더 내의 이미지 파일들을 그리드 형태로 표시합니다.
    
    Parameters:
    folder_path (str): 이미지 파일들이 있는 폴더의 경로
    num_cols (int): 그리드의 열 수
    """
    # 폴더 내의 파일 목록 가져오기
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 이미지 그리드 설정
    num_images = len(image_files)
    num_rows = (num_images + num_cols - 1) // num_cols  # 필요한 행의 수 계산

    # 이미지 플롯 생성
    plt.figure(figsize=(2 * num_cols, 2 * num_rows))  # 전체 그리드의 크기 조정

    for i, image_file in enumerate(image_files):
        img = Image.open(os.path.join(folder_path, image_file))
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(img)
        plt.axis('off')  # 축 레이블 끄기
        plt.title(image_file)  # 이미지 제목으로 파일 이름 사용

    plt.tight_layout()  # 레이아웃 조정
    plt.show()


def plot_images(images, titles=None, rows=1, cols=1, figsize=(15, 10)):
    """
    플롯을 위해 이미지 리스트를 받아서 지정된 행과 열로 나누어 플롯하는 함수
    
    Parameters:
    - images (list): 이미지 데이터의 리스트 (PIL 이미지 객체 또는 NumPy 배열)
    - titles (list): 각 이미지에 대한 제목의 리스트 (옵션)
    - rows (int): 플롯할 행의 수
    - cols (int): 플롯할 열의 수
    - figsize (tuple): 전체 플롯의 크기 (기본값은 (15, 10))
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    else:
        axes = axes.flatten()  # 2D 배열을 1D 배열로 평탄화
    
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        ax.axis('off')  # 축을 끕니다.
        if titles is not None and i < len(titles):
            ax.set_title(titles[i])
    
    # 빈 subplot을 지웁니다.
    for ax in axes[len(images):]:
        fig.delaxes(ax)
    
    plt.tight_layout()
    plt.show()