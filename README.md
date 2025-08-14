# Segment-and-Tracking

**Segment Anything Model (SAM)**과 **Associating Objects with Transformers (AOT)**를 기반으로 한 CLI 기반 비디오 객체 분할 및 추적 도구입니다. 자동 다중 객체 추적과 대화형 선택을 통한 정밀한 단일 객체 추적을 모두 제공합니다.

**This repository was created for the purpose of removing objects from videos.**

**The key feature here is the video mask generation.**

## 기능

### 🎯 단일 객체 추적
- **대화형 선택**: 포인트 클릭 또는 바운딩 박스를 사용하여 객체 선택
- **화이트 마스크 출력**: 선택된 객체만 표시하는 깨끗한 화이트 마스크 비디오 생성
- **바운딩 박스 마스킹**: 정확한 분할 대신 사각형 바운딩 박스 마스크 생성 옵션
- **높은 정밀도**: 전체 비디오에서 특정 객체 추적
- **유연한 출력**: 마스크 전용 및 오버레이 시각화 옵션

### 🔄 다중 객체 추적  
- **자동 감지**: SAM 기반 자동 객체 감지
- **포괄적 추적**: 감지된 모든 객체를 동시에 추적
- **구성 가능한 매개변수**: 감지 민감도 및 추적 매개변수 조정

## 설치

### 사전 요구사항
- CUDA 지원 GPU (권장)
- Anaconda/Miniconda
- Python 3.9

### 환경 설정
```bash
# 저장소 복제
git clone https://github.com/BigJoon/Segment-and_Tracking.git
cd Segment-and_Tracking

# conda 환경 생성
conda create -n sam-track python=3.9 -y
conda activate sam-track

# PyTorch 설치 (필요에 따라 CUDA 버전 조정)
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio cudatoolkit=11.3 -c pytorch -y

# 의존성 패키지 설치
pip install opencv-python pillow numpy imageio gdown

# SAM 설치
pip install -e ./sam

# 모델 체크포인트 다운로드
mkdir -p ckpt
cd ckpt
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
gdown '1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ' --output R50_DeAOTL_PRE_YTB_DAV.pth
cd ..

# GroundingDINO 설정 (텍스트 감지 기능용)
git clone https://github.com/IDEA-Research/GroundingDINO.git
cp -r GroundingDINO/groundingdino .
```

## 빠른 시작

### 단일 객체 추적

포인트를 지정하여 특정 객체 추적:
```bash
python single_object_tracker.py video.mp4 --point 640 360 --mask-only -o output
```

바운딩 박스로 특정 객체 추적:
```bash
python single_object_tracker.py video.mp4 --bbox 500 200 700 400 -o output
```

바운딩 박스 마스크로 객체 추적 (객체 제거에 최적화):
```bash
python single_object_tracker.py video.mp4 --point 640 360 --bbox-mask --bbox-padding 10 --mask-only -o output
```

### 다중 객체 추적

모든 객체 자동 추적:
```bash
python cli_track.py video.mp4 -o output --sam-gap 10 --max-objects 50
```

## 사용 예제

### 단일 객체 추적 옵션
```bash
# 포인트 선택 (클릭 좌표)
python single_object_tracker.py input.mp4 --point X Y [옵션]

# 바운딩 박스 선택 (사각형 좌표)  
python single_object_tracker.py input.mp4 --bbox X1 Y1 X2 Y2 [옵션]

옵션:
  -o OUTPUT         출력 디렉토리 (기본값: ./single_track_output)
  --mask-only       화이트 마스크 비디오만 출력 (오버레이 없음)
  --bbox-mask       분할 마스크 대신 바운딩 박스 마스크 생성
  --bbox-padding N  바운딩 박스 주위에 추가 패딩 픽셀 (기본값: 0)
  --device DEVICE   사용할 장치: cuda/cpu (기본값: cuda)
```

### 다중 객체 추적 옵션
```bash
python cli_track.py input.mp4 [옵션]

옵션:
  -o OUTPUT           출력 디렉토리 (기본값: ./output)
  --sam-gap N         SAM 실행 간격 (기본값: 5)
  --max-objects N     추적할 최대 객체 수 (기본값: 255)
  --min-area N        최소 마스크 영역 (기본값: 200)
  --device DEVICE     사용할 장치 (기본값: cuda)
```

## 출력

### 단일 객체 추적
- `mask_video.mp4` - **화이트 마스크 비디오** (주요 출력)
- `overlay_video.mp4` - 추적 오버레이가 있는 원본 비디오
- `masks/` - PNG 파일로 된 개별 프레임 마스크

### 다중 객체 추적
- `output_video.mp4` - 모든 추적된 객체가 있는 비디오
- `output_masks.gif` - 애니메이션 마스크 시퀀스
- `masks/` - 객체 ID가 있는 개별 프레임 마스크

## 샘플 결과

이 도구는 다양한 비디오 유형에서 테스트되었습니다:
- **세포 현미경 비디오**: 정밀한 세포 추적 및 분열 감지
- **객체 운동 비디오**: 가려짐을 통한 강건한 추적
- **다중 객체 장면**: 여러 타겟의 동시 추적

## 기술적 세부사항

### 핵심 구성요소
- **SAM 통합**: 자동 마스크 생성 및 대화형 분할
- **AOT 추적**: Transformer 기반 프레임 간 객체 연관
- **메모리 관리**: 긴 비디오 시퀀스의 효율적 처리
- **CLI 인터페이스**: 사용자 친화적인 명령줄 도구

### 모델 사양
- **SAM 모델**: ViT-B (358MB) - Vision Transformer 백본
- **AOT 모델**: R50-DeAOTL (237MB) - DeAOT 레이어가 있는 ResNet-50
- **입력 해상도**: 다양한 비디오 해상도 지원
- **성능**: 최신 GPU에서 ~2-3 FPS 처리 속도

## 문제 해결

### 일반적인 문제
- **CUDA 메모리 부족**: 비디오 해상도 또는 배치 크기 감소
- **객체를 찾을 수 없음**: 선택 좌표 또는 영역 임계값 조정
- **추적 드리프트**: SAM 간격 및 IoU 임계값 미세 조정

### 성능 팁
- 정확도와 속도의 균형을 위해 `--sam-gap` 사용
- 작은 객체를 필터링하기 위해 `--min-area` 조정
- 더 빠른 처리를 위해 `--mask-only` 사용

## 기여

기여를 환영합니다! 이슈, 기능 요청 또는 풀 리퀘스트를 자유롭게 제출해 주세요.

## 라이선스

이 프로젝트는 여러 오픈소스 프로젝트를 기반으로 합니다:
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) - Meta AI
- [AOT (Associating Objects with Transformers)](https://github.com/yoxu515/aot-benchmark)
- [Segment-and-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything) - 원본 구현

## 감사의 말

Segment-and-Track-Anything의 원저자들과 컴퓨터 비전 및 객체 추적 분야의 획기적인 연구를 수행한 SAM 및 AOT 팀에게 특별한 감사를 표합니다.

---

## 인용

연구에서 이 작업을 사용하는 경우 원본 논문을 인용해 주세요:

```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

@article{yang2023aot,
  title={Associating Objects with Transformers for Video Object Segmentation},
  author={Yang, Zongxin and Wei, Yunchao and Yang, Yi},
  journal={Advances in Neural Information Processing Systems},
  year={2021}
}
```