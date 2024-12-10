import torch
import legacy
import dnnlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# 모델 로드 함수
def load_stylegan2_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with dnnlib.util.open_url(model_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # StyleGAN2 모델 로딩
    return G, device


# 이미지 전처리 함수
def preprocess_image(image_path, target_size=(1024, 1024)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size, Image.LANCZOS)
    image = np.array(image).transpose(2, 0, 1)[None]  # HWC -> CHW
    image = torch.from_numpy(image).to(torch.float32) / 255.0
    return image


# 몽타주 합성 함수
def generate_montage(original_image, G, device):
    z = torch.randn([1, G.z_dim], device=device)  # 랜덤 잠재 벡터
    c = None  # StyleGAN2에서는 조건 벡터 c가 필요하지만, 여기서는 None으로 설정
    w = G.mapping(z, c)  # 잠재 공간에서의 매핑 (c는 None)
    generated_image = G.synthesis(w)

    # 원본 이미지와 생성된 이미지 섞기 (간단한 예시로 혼합)
    blended_image = (generated_image + original_image) / 2  # 이미지 합성 (간단한 예시)

    return blended_image


# 이미지를 표시하는 함수
def display_image(image_tensor):
    plt.imshow(image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0))
    plt.axis('off')
    plt.show()


# 메인 함수
def main():
    # 모델 경로 설정 (StyleGAN2 모델 경로)
    model_path = 'ffhq.pkl'  # StyleGAN2 모델 경로 (사용할 모델 파일을 지정)

    # 모델 로딩
    G, device = load_stylegan2_model(model_path)

    # 원본 이미지 경로 설정 (사용자가 제공한 이미지 파일)
    image_path = 'image.jpg'  # 이미지 파일 경로를 수정하세요

    # 원본 이미지 전처리
    original_image = preprocess_image(image_path)
    original_image = original_image.unsqueeze(0).to(device)

    # 몽타주 합성
    montage = generate_montage(original_image, G, device)

    # 결과 이미지 표시
    display_image(montage)


# 코드 실행
if __name__ == "__main__":
    main()
