# ROC curve and associated cost

Relationship between ROC curve, optimal threshold and ratio of costs of different mistakes.

Cost can be reduced with a better model, but the optimal cost (red vertical line) is dependent on the ratios of costs. 
![roc_costvary_performance](https://github.com/user-attachments/assets/e5bfef2e-fc01-4cd0-a4cc-283ac8cbc9e0)

The optimal cost is dependent on the ratio of costs of mistakes. One extreme it is a nuclear power plant, the other extreme is a spam filter.
![roc_costvary_ratio](https://github.com/user-attachments/assets/26fdaba8-8131-4e3e-b1d4-8800a8ddb833)

## Assumptions for the sake of visualisation

```math
\begin{equation*}
\begin{aligned}
\text{Cost} &= \text{TPR} \times \text{C}_{\text{TP}} + \text{TNR} \times \text{C}_{\text{TN}} \\
&+ \text{FPR} \times \text{C}_{\text{FP}} + \text{FNR} \times \text{C}_{\text{FN}}
\end{aligned}
\end{equation*}
```
- $\text{C}_{\text{TP}} = 0$ (No cost for correctly detecting a fault)
- $\text{C}_{\text{TN}} = 0$ (No cost for correctly detecting a healthy sample)
- $\text{C}_{\text{FP}}$ (False alarm cost) and $\text{C}_{\text{FN}}$ (Missed detection cost) are varied.
- $\text{C}_{\text{X}}$: Accounts for 1) Cost per occurrence 2) Prevalence of the fault
- FNR = 1 - TPR
