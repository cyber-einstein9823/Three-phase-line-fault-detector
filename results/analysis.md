# Task 6: Analysis

## 6.1 LR (Set R) vs linear SVM
Numbers: LR-R test acc 72.64%, linear SVM 73.59%.
Answer: Both are linear boundaries and score 72.64% and 73.59%. The bottleneck is the boundary’s shape, not the training method. 
The majority-class baseline is 54.3%.

## 6.2 Signs of I^2 and V^2 weights (Set Q)
Numbers: I^2 weights = +18.89, +4.75, +13.96; V^2 weights = -6.03, -6.46, -6.14.
Answer:I² weights are positive (+18.9, +4.8, +14.0) because current rises in a fault. V² weights are negative (−6.0, −6.5, −6.1) because voltage sags.
Explain why squaring removes the sign problem.

## 6.3 Support vectors, linear vs RBF
Numbers: linear 5849 / 7200; RBF 266 / 7200.
Answer: linear uses 5849 of 7200 points (81%) as support vectors, and RBF uses 266 (3.7%). Explain what a support vector is, 
and what it means when most points are support vectors.

## 6.4 Accuracy and support vectors as C increases
Numbers: linear accuracy stays 0.7258 for every C; RBF gamma=0.1: 0.9500 -> 0.9954, SV 1347 -> 174.
Answer:inear accuracy stays flat (SV 5849 → 5835). For RBF at γ=0.1, accuracy goes 0.9500 → 0.9954 and SV goes 1347 → 174.
Explain using what C penalises (margin violations).

## 6.5 Random split: optimistic or pessimistic?
Numbers: random split 0.9958 vs time-ordered 0.9938 (different test sets).
Answer: neighbouring samples are near-duplicates, so a random split leaks. Your experiment gave 0.9958 vs 0.9938, a small drop but on different test sets. Add what a real relay faces that the simulation lacks (noise, other fault types, resistances, locations). 
State your view and justify it.

    ## 6.6 Which model to deploy?
Numbers (missed faults / false alarms): LR-R 657/0, LR-Q 45/6, linear SVM 634/0, RBF 6/4.
Answer: The linear/R models never raise a false alarm, but they miss 634-657 of the 1096 test faults (about 58-60%). LR Q misses 45. RBF misses 6 with 4 false alarms. Argue from missed faults, 
since a missed fault means the breaker doesn’t trip.
