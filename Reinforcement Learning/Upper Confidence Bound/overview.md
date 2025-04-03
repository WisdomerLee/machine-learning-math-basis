# 🧠 Upper Confidence Bound (UCB) in Reinforcement Learning

## 개요

강화 학습(특히 **Multi-Armed Bandit 문제**)에서 중요한 과제 중 하나는 **탐험(Exploration)**과 **이용(Exploitation)** 사이의 균형입니다.

- **탐험(Exploration)**: 아직 잘 모르지만 더 좋은 보상이 있을 수 있는 행동을 시도해보는 것
- **이용(Exploitation)**: 지금까지의 경험으로 보상이 가장 좋았던 행동을 반복하는 것

**Upper Confidence Bound (UCB)**는 이 두 가지를 **수학적으로 조절**해주는 방법 중 하나입니다.

---

## Multi-Armed Bandit 문제란?

> 슬롯 머신이 여러 개 있는 상황을 상상해보세요. 각 머신은 보상을 주지만, 어떤 머신이 가장 좋은지는 모릅니다.

- 목표는 **최대한 많은 보상**을 얻는 것
- 매번 어떤 머신을 선택할지를 결정해야 함

---

## UCB의 핵심 아이디어

UCB는 **각 행동(머신)**에 대해 아래 두 요소를 합쳐 점수를 계산합니다:

1. 지금까지의 **평균 보상**
2. 아직 그 행동을 충분히 시도하지 않았다는 **불확실성 보정 값**

### 수식

$$
UCB_i = \bar{X}_i + c \cdot \sqrt{\frac{\ln n}{n_i}}
$$

- \( \bar{X}_i \): 행동 \(i\)의 **평균 보상**
- \( n \): 전체 시행 횟수 (모든 행동을 포함한 총 시도 수)
- \( n_i \): 행동 \(i\)의 선택 횟수
- \( c \): 탐험을 얼마나 중시할지를 조절하는 **하이퍼파라미터**

---

## 예시로 이해하기

| 행동 | 평균 보상 | 선택 횟수 \(n_i\) | UCB 점수 |
|------|------------|------------------|----------|
| A    | 0.5        | 10               | 높음     |
| B    | 0.7        | 50               | 중간     |
| C    | 0.2        | 2                | 매우 높음 |

- 평균 보상만 보면 B가 제일 좋아 보이지만,
- C는 시도 횟수가 너무 적어서 아직 판단하기 이르므로 UCB 점수는 **높게 나옴**
- 이렇게 **덜 시도한 행동에 보정을 주어** 한 번쯤은 시도해보게 만드는 것이 UCB의 목적

---

## 알고리즘 흐름

1. 각 행동을 **최소 한 번씩** 시도
2. 매 라운드마다 모든 행동의 UCB 점수를 계산
3. 가장 높은 UCB 점수를 가진 행동을 선택
4. 해당 행동의 결과(보상)를 기록하고 평균 보상을 업데이트
5. 반복

---

## UCB의 장점

- 이론적으로 **탐험과 이용의 균형**을 잘 맞춰줌
- 단순한 수식이지만 성능이 우수함
- **확률적이지 않음**: 항상 같은 조건이면 같은 행동을 선택함

---

## 한계

- UCB 수식이 정확한 보상 분포에 대해 가정(예: 보상이 정규분포)을 포함할 수 있음
- 초기 선택이 결과에 영향을 많이 줄 수 있음
- 복잡한 상태 공간(예: MDP)에는 직접 적용이 어려움 → Contextual Bandit이나 RL로 확장 필요

---

## 마무리

UCB는 **심플하지만 강력한 강화 학습 기법**으로, 특히 **탐험-이용 딜레마**를 수학적으로 해결하고자 할 때 유용합니다. Multi-Armed Bandit 문제에 관심이 있다면 꼭 이해하고 넘어가야 할 개념입니다!

---
## 코드 예시

```python
import numpy as np
import matplotlib.pyplot as plt

# 슬롯머신 수 (팔 수)
num_arms = 5

# 각 슬롯머신의 실제 평균 보상 (우리가 모르는 값, 시뮬레이션용)
true_rewards = np.random.rand(num_arms)
print("각 머신의 실제 보상:", true_rewards)

# 각 머신의 선택 횟수
counts = np.zeros(num_arms)

# 각 머신의 현재 평균 보상
estimated_rewards = np.zeros(num_arms)

# 하이퍼파라미터 c (탐험 강도 조절)
c = 2

# 전체 시도 횟수
n_rounds = 1000

# 총 보상 추적용
total_rewards = []

for t in range(1, n_rounds + 1):
    ucb_values = np.zeros(num_arms)
    
    for i in range(num_arms):
        if counts[i] == 0:
            # 아직 선택 안 된 머신은 무조건 선택되게끔 UCB를 무한대로 설정
            ucb_values[i] = float('inf')
        else:
            # UCB 계산 공식
            avg_reward = estimated_rewards[i]
            delta = c * np.sqrt(np.log(t) / counts[i])
            ucb_values[i] = avg_reward + delta

    # UCB 값이 가장 높은 머신 선택
    chosen_arm = np.argmax(ucb_values)

    # 실제 보상 시뮬레이션 (정규분포 기반, 평균은 true_rewards)
    reward = np.random.randn() * 0.1 + true_rewards[chosen_arm]

    # 업데이트
    counts[chosen_arm] += 1
    estimated_rewards[chosen_arm] += (reward - estimated_rewards[chosen_arm]) / counts[chosen_arm]

    # 기록
    total_rewards.append(reward)

# 누적 평균 보상 시각화
cumulative_average = np.cumsum(total_rewards) / (np.arange(n_rounds) + 1)

plt.plot(cumulative_average)
plt.xlabel("Round")
plt.ylabel("Average Reward")
plt.title("UCB를 사용한 평균 보상 변화")
plt.grid()
plt.show()

```
