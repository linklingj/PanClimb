# Hold Type SSL Pipeline

현재 `홀드 마스킹(holdseg)` 단계가 있으므로, 분류는 아래 4단계로 분리하면 안정적이다.

## 1) Hold crop 추출 (마스크 기반)
- 입력: 원본 이미지 + `holdseg_predictions.json` (COCO 스타일 polygon/bbox).
- 처리:
  - polygon으로 binary mask 재구성
  - `bbox + padding` 영역 crop
  - 배경 억제를 위해 mask 외 영역은 0으로 클리핑
- 출력:
  - `crops/*.png`
  - `metadata.csv` (crop 경로, 원본 이미지, ann_id, bbox)

## 2) SSL 사전학습 (Unlabeled)
- 목적: 라벨 없는 홀드 이미지에서 형태/질감 표현을 학습.
- 방식: SimCLR (InfoNCE).
- 입력: `metadata.csv` (label 없음 가능).
- 핵심:
  - 한 crop에서 강한 augment 2-view 생성
  - encoder(`resnet18`) + projection head 학습
  - checkpoint: `ssl_encoder.pth`

## 3) 소량 라벨 파인튜닝 (Labeled)
- 입력: labeled `metadata.csv` (label 컬럼 필수, 클래스명은 코드의 `HOLD_TYPES` 사용).
- 클래스(9):
  - `Jug`, `Sloper`, `Crimp`, `Pinch`, `Pocket`, `FootHold`, `Sidepull`, `Undercling`, `Volume`
- 방식:
  - SSL encoder 가중치 로드
  - classifier head 붙여 cross-entropy 학습
  - checkpoint: `holdtype_classifier.pth`

## 4) 추론 (Segmentation + Classification)
- 입력: 원본 이미지 + `holdseg_predictions.json` + `holdtype_classifier.pth`
- 처리:
  - holdseg annotation마다 동일한 crop/mask 로직 적용
  - hold type softmax 분류
- 출력:
  - 기존 holdseg json에 `hold_type`, `hold_type_id`, `hold_type_score` 추가한 결과 json

## 데이터 운영 권장
- split 권장:
  - Unlabeled: 다양한 홀/조명/촬영각
  - Labeled: class imbalance 최소화(특히 `Pocket`, `Undercling`, `FootHold`)
- 라벨 품질:
  - 애매 샘플은 `review`로 분리 후 제외
  - 초기 학습은 noisy label 허용, 이후 hard-negative 재라벨
- 평가지표:
  - Macro F1 (클래스 불균형 대응)
  - Confusion Matrix (예: Sloper vs Pinch 혼동)

## 실제 실행 순서
1. crop 생성
2. unlabeled SSL pretrain
3. labeled finetune
4. holdseg 결과에 type 추론

아래 스크립트로 실행:
- `src/holdtype_ssl.py`
- `src/holdtype_infer.py`
