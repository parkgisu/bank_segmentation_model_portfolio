# Bank Customer Segmentation Model

## Overview
본 프로젝트는 은행 고객 데이터를 활용하여 고객의 회원 등급(Segment)을 예측하는 머신러닝 모델을 구축하는 것을 목표로 합니다.  

다양한 고객 정보(신용, 매출, 잔액, 채널 등)를 통합하여 피처를 구성하고,  
부스팅 기반 모델을 활용하여 다중 클래스 분류 문제를 해결하였습니다.

---

## My Contribution
본 프로젝트에서 저는 **데이터 전처리 이후 모델링 단계**를 담당하였습니다.

### 주요 작업
- PCA 기반 누적 설명분산 분석을 통해 적절한 피처 수 결정  
- XGBoost, LightGBM, CatBoost 모델 학습 및 비교  
- RandomizedSearchCV를 활용한 하이퍼파라미터 튜닝  
- Macro F1 score를 중심으로 한 성능 평가 설계  
- 심각한 클래스 불균형 문제를 해결하기 위한 class weight 적용  
- 희소 클래스 (Segment A)에 대한 별도 성능 분석 수행  

---

## Data Processing (Summary)
- 여러 데이터 소스(회원, 신용, 매출 등)를 ID 기준으로 병합  
- 수치형/범주형 피처 분리 및 재정의  
- ANOVA / Chi-square 기반 피처 중요도 평가  
- Top-N 피처 선택 (50 / 245 / 전체 비교)  

---

## Modeling Approach

### Feature Selection
- PCA 기반 cumulative explained variance 분석  
  - 80% → 114 features  
  - 90% → 185 features  
  - 95% → 245 features  

### Models
- XGBoost  
- LightGBM  
- CatBoost  

### Training Strategy
- Train / Validation / Test = 60 / 20 / 20  
- Stratified K-Fold Cross Validation (5-fold)  
- Evaluation Metric: Macro F1  

---

## Handling Class Imbalance
데이터의 클래스 분포가 매우 불균형하여,  
특히 소수 클래스(A)에 대한 성능이 낮은 문제가 발생했습니다.

이를 해결하기 위해:
- class weight 계산 및 적용  
- weight upper bound (max=50) 설정  
- A class에 대한 별도 evaluation 수행  

---

## Results

### Overall Performance
- Accuracy: ~0.89  
- Macro F1: ~0.45  

### Key Observation
- Accuracy는 높지만, 소수 클래스 성능은 매우 낮음  
- Macro F1이 더 적절한 평가 지표  
- Class imbalance가 모델 성능에 큰 영향을 미침  

---

## Project Structure

```text
.
├─ my_modeling.ipynb        # 모델링 및 실험 코드 (PCA 부터 본인 담당)
├─ data_analysis.ipynb      # 데이터 분석 및 전처리 (팀원 담당)
└─ README.md
