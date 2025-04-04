# Thompson Sampling in Reinforcement Learning

## 개요

Thompson Sampling은 **탐험(Exploration)**과 **이용(Exploitation)** 사이의 균형을 매우 자연스럽게 해결해주는 **확률 기반 강화 학습 기법**입니다. 특히 **Multi-Armed Bandit 문제**에서 널리 사용됩니다.

---

## Multi-Armed Bandit 문제 복습

> 여러 개의 슬롯머신(행동) 중에서 **보상을 최대화**하기 위해 어떤 머신을 선택할지 결정하는 문제

- 머신마다 평균 보상이 다르지만 알 수 없음
- 반복해서 머신을 선택하며 가장 보상이 높은 머신을 찾아야 함

---

## Thompson Sampling의 핵심 아이디어

> "확률적 베스트"를 선택하자!

- 각 머신의 **보상 확률 분포**를 베이지안 방식으로 계속 업데이트
- 매 라운드마다 **분포에서 샘플을 뽑아**, 그 중 **가장 높은 값을 가진 머신을 선택**

즉, 확률적으로 보상이 높을 **가능성이 큰** 머신을 선택하게 됩니다.

---

## 보통은 어떤 분포를 쓸까?

- **보상이 0 또는 1 (이진)**인 경우 → **베타 분포(Beta distribution)** 사용
  - 이 경우, 베타 분포는 **성공/실패 횟수**를 기반으로 업데이트됨
  - 초기에는 \( \text{Beta}(1, 1) \)로 시작 (균등 분포)

---

## 수식 개념

베타 분포:  
$$
\text{Beta}(a, b) \propto x^{a-1} (1 - x)^{b-1}
$$

- \(a\): 성공 횟수 + 1
- \(b\): 실패 횟수 + 1

---

## 알고리즘 흐름

1. 모든 머신의 초기 분포를 \( \text{Beta}(1, 1) \)로 설정
2. 매 라운드마다 각 머신에서 **확률 샘플**을 추출
3. 가장 높은 샘플 값을 가진 머신 선택
4. 보상을 관측하고 성공/실패로 기록
5. 해당 머신의 베타 분포 파라미터 업데이트
6. 반복

---

## ✅ 장점

- 수학적으로 우아한 방식으로 **탐험과 이용을 자동 조절**
- 구현이 매우 간단함
- 실제 문제에서도 성능이 매우 우수

---

## ❗ 단점

- 보상 분포에 대한 **사전 지식 필요** (예: 베르누이 분포, 정규분포 등)
- 이진 보상 문제에는 적합하지만, 연속적인 보상에서는 다른 분포 필요

---

## Python 코드 예제 (Thompson Sampling with Bernoulli Bandits)

```python
import numpy as np
import matplotlib.pyplot as plt

# 슬롯 머신 수
num_arms = 5

# 각 머신의 실제 성공 확률 (시뮬레이션용, 사용자는 모른다고 가정)
true_probs = np.random.rand(num_arms)
print("각 머신의 실제 성공 확률:", true_probs)

# 베타 분포 파라미터 (초기값: Beta(1,1) → 균등 분포)
successes = np.ones(num_arms)
failures = np.ones(num_arms)

n_rounds = 1000
total_rewards = []

for t in range(n_rounds):
    sampled_probs = np.random.beta(successes, failures)
    chosen_arm = np.argmax(sampled_probs)

    # 실제 보상 시뮬레이션 (1 또는 0)
    reward = np.random.rand() < true_probs[chosen_arm]  # 성공 확률에 따라 1 또는 0

    # 베타 분포 업데이트
    if reward:
        successes[chosen_arm] += 1
    else:
        failures[chosen_arm] += 1

    total_rewards.append(reward)

# 누적 평균 보상 시각화
cumulative_avg = np.cumsum(total_rewards) / (np.arange(n_rounds) + 1)

plt.plot(cumulative_avg)
plt.xlabel("Round")
plt.ylabel("Average Reward")
plt.title("Thompson Sampling을 사용한 평균 보상 변화")
plt.grid()
plt.show()
```

---

## 📊 결과 해석

- 알고리즘이 처음에는 여러 머신을 탐험하며 정보를 수집
- 시간이 지날수록 **성공 확률이 높은 머신을 더 자주 선택**
- 평균 보상이 점점 상승하며 안정화됨

---

## 🧩 확장 아이디어

- 연속적인 보상에서는 **정규 분포 기반**의 Thompson Sampling으로 확장
- 상태가 있는 환경에서는 **Contextual Bandit** 또는 **강화 학습 (MDP)**로 확장 가능

---
