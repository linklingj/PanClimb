# PanClimb
Indoor climbing Route Finding Project

### Summary
- Work in progress (25.01.20 ~ )
- PyTorch

### Plan
1. Hold segmentation
2. Hold type classification
3. Route visualize
4. Route solution
5. Visualize solution using game engine rigging (unity)

### Hold Type
- Jug
- Sloper
- Crimp
- Pinch
- Pocket
- FootHold
- Sidepull
- Undercling
- Volume

## Hold Type SSL Pipeline
세부 설계: `docs/holdtype_ssl_pipeline.md`

### 1) Hold crop 생성 (마스크 기반)
```bash
python src/holdtype_ssl.py build-crops \
  --image data/images/sm/0.jpeg \
  --seg-json outputs/holdseg_predictions.json \
  --output-dir data/holdtype/crops \
  --metadata-out data/holdtype/crops_metadata.csv
```

### 2) SSL 사전학습 (Unlabeled 가능)
```bash
python src/holdtype_ssl.py pretrain \
  --metadata data/holdtype/crops_metadata.csv \
  --save-path models/checkpoints/holdtype_ssl_encoder.pth
```

### 3) 분류기 파인튜닝 (Labeled 필요)
```bash
python src/holdtype_ssl.py finetune \
  --metadata data/holdtype/crops_metadata.csv \
  --save-path models/checkpoints/holdtype_classifier.pth
```

### 4) holdseg 결과에 타입 추론 붙이기
```bash
python src/holdtype_infer.py \
  --model models/checkpoints/holdtype_classifier.pth \
  --image data/images/sm/0.jpeg \
  --holdseg-json outputs/holdseg_predictions.json \
  --output outputs/holdtype_predictions.json \
  --override-volume-from-seg
```
