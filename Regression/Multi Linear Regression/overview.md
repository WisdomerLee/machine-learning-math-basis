P 값을 알아보기
통계적인 뜻을 알고 있어야 할 것
0.05로 설정해두었다면, 95%의 확률을 가진 값들, 혹은 그 상태들의 모음은 95%의 확률로 볼 수 있는 가능성들...

다중 선형 회귀일 때
연관된 모든 가능성을 확인하고, 그 중에 관계성이 낮은 것들은 버리는 과정이 들어감..
연관성이 높은, 관련성이 높은 부분들만 남기고 나머지는 버리는 과정이 포함

다중 선형 회귀의 모델을 만드는 방법은 다섯가지가 있음
1. All-in
2. Backward Elimination
3. Forward Selection
4. Bidirectional Elimination
5. Score Comparison

2,3,4는 Stepwise Regression

대체로 Stepwise Regression이라고 하면 4번을 가리키는 경우가 많음

All-in은 모든 종속변수를 모두 넣는 방법
이 경우는 이미 해당 내용에 대해 모두 알고 있기 때문에 굳이 모델을 만들어야 할 필요가 없음
혹은 Backward Elimination을 시행하기 전의 준비 단계

Backward Elinination
모형 내의 유의미한 확률을 설정해야 함 - SL = 0.05로 하면 95%의 신뢰도를 갖는 확률을 설정
모델에서 가능한 종속변수들을 모두 넣기(All-in)
P-value값이 높은지 확인, 즉 SL보다 P값이 크면, 다음으로 진행, 그렇지 않으면 종료
연관성이 낮은 종속변수 제거
모델에 해당 변수 없이 맞추기 - 변수 하나를 없애는 순간 다른 변수들의 조건들이 모두 변동....

Forward Selection
모형 내의 유의미한 확률을 설정해야 함 - SL = 0.05로 하면 95%의 신뢰도를 갖는 확률을 설정
가능한 모든 단순 회귀 모형들을 넣을 것... P-값이 가장 낮은 독립 변수가 있는 모형을 하나 선정
해당 변수를 유지하고, 또 다른 단순 회귀 모형을 하나 더 추가할 것...
그리고 P-value값이 SL보다 낮으면, 다시 위의 과정을 반복하고, 그렇지 않으면 종료함

Bidirectional Eliminiation
SLENTER, SLSTAY 값을 설정
Forward Selection을 선택 - 새 variables는 P < SLENTER를 만족해야 함
Backward Elimination의 모든 과정을 진행 (기존 variables들은 P< SLSTAY를 만족해야 함)

All Possible Models
선형회귀를 잘 맞추기 위한 좋은 방법을 선택 (Akaike criterion같은 것)
모든 가능한 Regression models들을 만들기 (2^n -1)개의 조합 가능
그 중에 가장 최상의 criterion을 선택
