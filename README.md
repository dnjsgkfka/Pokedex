# Pokédex

**Pokédex**는 Transfer Learning을 활용하여 포켓몬 이미지를 입력받아 해당 포켓몬의 이름을 분류하는 프로그램입니다.

ImageNet으로 사전학습된 CNN Backbone(ResNet-50, EfficientNet-B0)을 fine-tuning하여 150종의 포켓몬을 구분합니다.

---

## 데모
<p>
  <img src="https://github.com/user-attachments/assets/4d90fe4a-0e3f-43a3-b51b-3eb1f87d420a" width="48%" alt="demo1">
  <img src="https://github.com/user-attachments/assets/9473d023-5f2d-4e93-bfa2-15fb7bb1721e" width="48%" alt="demo2">
</p>
<p>
  <img src="https://github.com/user-attachments/assets/1e5b831f-ae5c-49bd-aaf0-1ab0d3a83ce2" width="48%" alt="demo3">
  <img src="https://github.com/user-attachments/assets/5c632c7f-0cf5-41d4-a933-820021b9e576" width="48%" alt="demo5">
</p>
<p>
  <img src="https://github.com/user-attachments/assets/1ed47678-4811-4f30-b5ca-1f5a494f6cfb" width="48%" alt="demo4">
  <img src="https://github.com/user-attachments/assets/02730a18-6289-4db9-af22-d0806ec7ad19" width="48%" alt="demo6">
</p>

---

## 주요 기능

* **포켓몬 분류**: 이미지를 업로드하면 AI 모델이 포켓몬의 이름을 Top-5 확률과 함께 표시합니다.

* **모델 선택**: 4가지 실험 모델 중 원하는 모델을 선택하여 분류 결과를 비교할 수 있습니다.

---

## 실험 설계

| # | Backbone | Pretrained | Fine-Tune 범위 | 실험 목적 |
| :--- | :--- | :---: | :--- | :--- |
| **Exp 1** | ResNet-50 | O | Head Only (Backbone 동결) | Fine-tune 범위 비교 |
| **Exp 2** | ResNet-50 | O | Full (전체 레이어) | Fine-tune 범위 비교 |
| **Exp 3** | EfficientNet-B0 | O | Head Only | Backbone 아키텍처 비교 |
| **Exp 4** | ResNet-50 | X | Full (처음부터 학습) | Pretrain weight 유무 비교 |

---

## 실험 결과

| 실험 | Test Accuracy | Test Precision | Test Recall | Test F1 |
| :--- | :---: | :---: | :---: | :---: |
| Exp 1 — ResNet-50 Pretrained Head-Only | 0.8113 | 0.8024 | 0.8091 | 0.7941 |
| Exp 2 — ResNet-50 Pretrained Full-Tune | **0.9726** | **0.9738** | **0.9751** | **0.9735** |
| Exp 3 — EfficientNet-B0 Pretrained Head-Only | 0.7820 | 0.7712 | 0.7805 | 0.7623 |
| Exp 4 — ResNet-50 Scratch Full-Train | 0.6197 | 0.6081 | 0.6272 | 0.5954 |

### Learning Curves

<p>
  <img src="experiments\exp1_resnet50_pretrained_headonly\exp1_resnet50_pretrained_headonly_curve.png" width="48%" alt="exp1 curve">
  <img src="experiments\exp2_resnet50_pretrained_fulltune\exp2_resnet50_pretrained_fulltune_curve.png" width="48%" alt="exp2 curve">
</p>
<p>
  <img src="experiments\exp3_efficientnet_pretrained_headonly\exp3_efficientnet_pretrained_headonly_curve.png" width="48%" alt="exp3 curve">
  <img src="experiments\exp4_resnet50_scratch_fulltrain\exp4_resnet50_scratch_fulltrain_curve.png" width="48%" alt="exp4 curve">
</p>

---

## 데이터셋

* [7,000 Labeled Pokemon (150 classes)](https://www.kaggle.com/datasets/lantian773030/pokemonclassification) — Kaggle
* 분할 비율: Train 70% / Validation 15% / Test 15%
* Augmentation: RandomCrop, HorizontalFlip, ColorJitter, RandomRotation

---

## 설치 및 실행 방법

1. 필수 라이브러리를 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```

2. 데모 앱을 실행합니다.
   ```bash
   streamlit run app.py
   ```
