import cv2
import numpy as np
from PIL import Image

# attention_maps = aggregate_attention(controller,
#                                      res=24,
#                                      from_where=("up", "down", "mid"),
#                                      is_cross = True,
#                                      select = 0).detach().cpu()

# attn1 = attention_maps[:,:,5]
# attn2 = attention_maps[:,:,10]

#이런식으로 attention 가져와서 사용
def combine_two_attention_maps(attn_map1, attn_map2, base_image=None):
    """
    두 개의 어텐션 맵을 받아서 하나의 RGB 이미지로 합성합니다.
    - attn_map1: 첫 번째 토큰에 해당하는 어텐션 맵 (2D numpy array)
    - attn_map2: 두 번째 토큰에 해당하는 어텐션 맵 (2D numpy array)
    - base_image: (선택) 원본 이미지 (PIL.Image 또는 numpy array)
    """

    # 0~1 범위로 정규화
    def normalize(attn):
        return (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    norm1 = normalize(attn_map1)
    norm2 = normalize(attn_map2)

    # 어텐션 맵의 크기 확인 (예: H x W)
    H, W = norm1.shape

    # ✅ 기본 배경을 옅은 회색 (128, 128, 128)로 설정
    composite = np.full((H, W, 3), fill_value=255, dtype=np.uint8)

    # 첫 번째 어텐션 맵은 빨간색 채널 (R 채널, index 2)
    composite[..., 1] = np.uint8((1-norm1) * 255)  # 128~255 범위로 설정
    
    # 두 번째 어텐션 맵은 초록색 채널 (G 채널, index 1)
    composite[..., 2] = np.uint8((1-norm2) * 255)  # 128~255 범위로 설정
    composite[..., 0] = np.uint8((1-(norm1+norm2)/2) * 255)  # 128~255 범위로 설정

    # (선택 사항) 원본 이미지와 혼합: 원본이 있다면 어텐션 맵을 반투명하게 오버레이
    if base_image is not None:
        if isinstance(base_image, Image.Image):
            base_image = np.array(base_image.resize((W, H)))

        # base_image가 3채널 컬러 이미지라고 가정
        composite = cv2.addWeighted(base_image, 0.5, composite, 0.5, 0)
    
    composed_rgb = cv2.cvtColor(composed, cv2.COLOR_BGR2RGB)
    composed_rgb = Image.fromarray(composed_rgb)
    large_image = composed_rgb.resize((512, 512), Image.NEAREST)  # 또는 Image.BILINEAR, Image.LANCZOS    
    return large_image