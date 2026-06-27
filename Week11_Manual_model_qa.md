# Week 11 Manual Calculation: Example Questions and Model Answers

This file is a practice bank for the calculation section of Week 11. The questions focus on K-means, K-modes, distance calculation, centroid calculation, and reassignment.

Use this rule throughout:

```text
K-means: numerical data -> mean centroid -> shortest distance wins
K-modes: categorical data -> mode center -> lowest dissimilarity wins
```

## Question 1: Euclidean Distance Between Two Observations

Given:

| Observation | x1 | x2 | x3 | x4 | x5 |
|---|---:|---:|---:|---:|---:|
| O1 | 10 | 2 | -1 | 4 | 0 |
| O2 | 12 | 4 | -5 | 4 | 1 |

Calculate the Euclidean distance between O1 and O2.

### Model Answer

Formula:

```text
d(O1, O2) = sqrt((x1_1 - x1_2)^2 + (x2_1 - x2_2)^2 + ... + (x5_1 - x5_2)^2)
```

Substitute the values:

```text
d(O1, O2)
= sqrt((10 - 12)^2 + (2 - 4)^2 + (-1 - -5)^2 + (4 - 4)^2 + (0 - 1)^2)
= sqrt((-2)^2 + (-2)^2 + (4)^2 + 0^2 + (-1)^2)
= sqrt(4 + 4 + 16 + 0 + 1)
= sqrt(25)
= 5
```

Final answer:

```text
d(O1, O2) = 5
```

## Question 2: Euclidean Distance With Negative Values

Given:

| Observation | x1 | x2 | x3 | x4 | x5 |
|---|---:|---:|---:|---:|---:|
| O1 | 10 | 2 | -1 | 4 | 0 |
| O3 | 10 | 6 | -6 | 4 | 0 |

Calculate the Euclidean distance between O1 and O3.

### Model Answer

```text
d(O1, O3)
= sqrt((10 - 10)^2 + (2 - 6)^2 + (-1 - -6)^2 + (4 - 4)^2 + (0 - 0)^2)
= sqrt(0^2 + (-4)^2 + 5^2 + 0^2 + 0^2)
= sqrt(0 + 16 + 25 + 0 + 0)
= sqrt(41)
= 6.403
```

Final answer:

```text
d(O1, O3) = 6.403
```

## Question 3: Manhattan Distance

Given:

| Observation | x1 | x2 | x3 |
|---|---:|---:|---:|
| O1 | 10 | 2 | -1 |
| O2 | 12 | 4 | -5 |

Calculate the Manhattan distance between O1 and O2.

### Model Answer

Formula:

```text
d(O1, O2) = |x1_1 - x1_2| + |x2_1 - x2_2| + |x3_1 - x3_2|
```

Substitute:

```text
d(O1, O2)
= |10 - 12| + |2 - 4| + |-1 - -5|
= |-2| + |-2| + |4|
= 2 + 2 + 4
= 8
```

Final answer:

```text
Manhattan distance = 8
```

## Question 4: K-Means Centroid Calculation

Given cluster A contains:

| Observation | x1 | x2 | x3 | x4 | x5 |
|---|---:|---:|---:|---:|---:|
| O1 | 10 | 2 | -1 | 4 | 0 |
| O4 | 9 | 2 | -1 | 5 | 0 |
| O7 | 8 | 4 | -5 | 5 | 1 |

Calculate the centroid of cluster A.

### Model Answer

For K-means, the centroid is the mean of each variable.

```text
x1 = (10 + 9 + 8) / 3 = 27 / 3 = 9
x2 = (2 + 2 + 4) / 3 = 8 / 3 = 2.667
x3 = (-1 + -1 + -5) / 3 = -7 / 3 = -2.333
x4 = (4 + 5 + 5) / 3 = 14 / 3 = 4.667
x5 = (0 + 0 + 1) / 3 = 1 / 3 = 0.333
```

Final answer:

```text
Centroid A = (9, 2.667, -2.333, 4.667, 0.333)
```

## Question 5: Assign Observations to Nearest Centroid

Given the following observations:

| Observation | x1 | x2 |
|---|---:|---:|
| O1 | 2 | 2 |
| O2 | 3 | 4 |
| O3 | 8 | 8 |
| O4 | 9 | 6 |

Given centroids:

```text
C1 = (2, 3)
C2 = (8, 7)
```

Using Euclidean distance, assign each observation to the nearest centroid.

### Model Answer

Calculate distance to both centroids.

For O1:

```text
d(O1, C1) = sqrt((2 - 2)^2 + (2 - 3)^2) = sqrt(1) = 1
d(O1, C2) = sqrt((2 - 8)^2 + (2 - 7)^2) = sqrt(36 + 25) = 7.810
```

For O2:

```text
d(O2, C1) = sqrt((3 - 2)^2 + (4 - 3)^2) = sqrt(2) = 1.414
d(O2, C2) = sqrt((3 - 8)^2 + (4 - 7)^2) = sqrt(25 + 9) = 5.831
```

For O3:

```text
d(O3, C1) = sqrt((8 - 2)^2 + (8 - 3)^2) = sqrt(36 + 25) = 7.810
d(O3, C2) = sqrt((8 - 8)^2 + (8 - 7)^2) = sqrt(1) = 1
```

For O4:

```text
d(O4, C1) = sqrt((9 - 2)^2 + (6 - 3)^2) = sqrt(49 + 9) = 7.616
d(O4, C2) = sqrt((9 - 8)^2 + (6 - 7)^2) = sqrt(2) = 1.414
```

Assignment table:

| Observation | d to C1 | d to C2 | Assigned cluster |
|---|---:|---:|---|
| O1 | 1.000 | 7.810 | C1 |
| O2 | 1.414 | 5.831 | C1 |
| O3 | 7.810 | 1.000 | C2 |
| O4 | 7.616 | 1.414 | C2 |

Final answer:

```text
C1 = O1, O2
C2 = O3, O4
```

## Question 6: Recalculate K-Means Centroids After Assignment

Using the final clusters from Question 5:

```text
C1 = O1, O2
C2 = O3, O4
```

with observations:

| Observation | x1 | x2 |
|---|---:|---:|
| O1 | 2 | 2 |
| O2 | 3 | 4 |
| O3 | 8 | 8 |
| O4 | 9 | 6 |

Recalculate the new centroids.

### Model Answer

For C1:

```text
C1 = O1, O2

x1 = (2 + 3) / 2 = 2.5
x2 = (2 + 4) / 2 = 3

New C1 = (2.5, 3)
```

For C2:

```text
C2 = O3, O4

x1 = (8 + 9) / 2 = 8.5
x2 = (8 + 6) / 2 = 7

New C2 = (8.5, 7)
```

Final answer:

```text
New C1 = (2.5, 3)
New C2 = (8.5, 7)
```

## Question 7: One Full K-Means Iteration

Given:

| i | x1 | x2 | x3 | Initial group |
|---:|---:|---:|---:|---|
| 1 | 10 | 2 | -1 | A |
| 2 | 11 | 3 | -1 | B |
| 3 | 18 | 5 | -1 | A |
| 4 | 20 | 4 | 0 | B |
| 5 | 19 | 3 | 0 | A |
| 6 | 8 | 2 | -1 | B |

First calculate the initial centroids. Then assign each observation to the nearest centroid using Euclidean distance.

### Model Answer

Initial groups:

```text
A = O1, O3, O5
B = O2, O4, O6
```

Centroid A:

```text
x1 = (10 + 18 + 19) / 3 = 15.667
x2 = (2 + 5 + 3) / 3 = 3.333
x3 = (-1 + -1 + 0) / 3 = -0.667

CA = (15.667, 3.333, -0.667)
```

Centroid B:

```text
x1 = (11 + 20 + 8) / 3 = 13
x2 = (3 + 4 + 2) / 3 = 3
x3 = (-1 + 0 + -1) / 3 = -0.667

CB = (13, 3, -0.667)
```

Distance and assignment:

| i | d to CA | d to CB | New group |
|---:|---:|---:|---|
| 1 | 5.831 | 3.180 | B |
| 2 | 4.690 | 2.028 | B |
| 3 | 2.887 | 5.395 | A |
| 4 | 4.435 | 7.102 | A |
| 5 | 3.416 | 6.037 | A |
| 6 | 7.789 | 5.110 | B |

Final answer after one iteration:

```text
A = O3, O4, O5
B = O1, O2, O6
```

## Question 8: K-Means Stopping Rule

After one K-means reassignment, the clusters become:

```text
A = O3, O4, O5
B = O1, O2, O6
```

The recalculated centroids are:

```text
CA = (19, 4, -0.333)
CB = (9.667, 2.333, -1)
```

After calculating distances again, the new assignment is:

```text
A = O3, O4, O5
B = O1, O2, O6
```

Should the K-means algorithm stop? Explain.

### Model Answer

Yes, the algorithm should stop.

Reason:

```text
The cluster assignments did not change after the new distance calculation.
```

In K-means, we repeat these two steps:

```text
1. Assign observations to nearest centroid.
2. Recalculate centroids.
```

We stop when the assignments remain the same.

Final answer:

```text
Stop, because the clusters are stable.
```

## Question 9: K-Means Within-Cluster Sum of Squares

Given final clusters:

| Observation | x1 | x2 | Cluster |
|---|---:|---:|---|
| O1 | 1 | 1 | A |
| O2 | 2 | 1 | A |
| O3 | 8 | 7 | B |
| O4 | 9 | 8 | B |

The centroids are:

```text
CA = (1.5, 1)
CB = (8.5, 7.5)
```

Calculate the within-cluster sum of squares, also called inertia.

### Model Answer

For each observation, calculate squared distance to its own centroid. Do not take the square root for sum of squares.

For cluster A:

```text
O1 to CA = (1 - 1.5)^2 + (1 - 1)^2 = 0.25 + 0 = 0.25
O2 to CA = (2 - 1.5)^2 + (1 - 1)^2 = 0.25 + 0 = 0.25
```

For cluster B:

```text
O3 to CB = (8 - 8.5)^2 + (7 - 7.5)^2 = 0.25 + 0.25 = 0.50
O4 to CB = (9 - 8.5)^2 + (8 - 7.5)^2 = 0.25 + 0.25 = 0.50
```

Total:

```text
inertia = 0.25 + 0.25 + 0.50 + 0.50 = 1.50
```

Final answer:

```text
Within-cluster sum of squares = 1.5
```

## Question 10: K-Modes Dissimilarity Score

Given:

| Object | x1 | x2 | x3 |
|---|---|---|---|
| O1 | A | L | M |
| C1 | A | L | C |
| C2 | B | P | C |

Calculate the dissimilarity score between O1 and C1, and between O1 and C2. Then assign O1 to a cluster.

### Model Answer

For K-modes:

```text
same = 0
different = 1
```

Compare O1 with C1:

| Feature | Compare | Score |
|---|---|---:|
| x1 | A vs A | 0 |
| x2 | L vs L | 0 |
| x3 | M vs C | 1 |

```text
d(O1, C1) = 0 + 0 + 1 = 1
```

Compare O1 with C2:

| Feature | Compare | Score |
|---|---|---:|
| x1 | A vs B | 1 |
| x2 | L vs P | 1 |
| x3 | M vs C | 1 |

```text
d(O1, C2) = 1 + 1 + 1 = 3
```

Since 1 is smaller than 3:

```text
O1 is assigned to C1
```

Final answer:

```text
d(O1, C1) = 1
d(O1, C2) = 3
O1 -> C1
```

## Question 11: Full K-Modes Assignment Table

Given observations:

| i | x1 | x2 | x3 |
|---:|---|---|---|
| 1 | A | L | M |
| 2 | B | P | C |
| 3 | A | L | C |
| 4 | B | L | C |
| 5 | A | P | M |

Initial modes:

```text
C1 = (A, L, C)
C2 = (B, P, C)
```

Calculate the dissimilarity to C1 and C2 for all observations.

### Model Answer

Use:

```text
same = 0
different = 1
```

| i | Observation | d to C1 | d to C2 | Assigned cluster |
|---:|---|---:|---:|---|
| 1 | (A, L, M) | 1 | 3 | C1 |
| 2 | (B, P, C) | 2 | 0 | C2 |
| 3 | (A, L, C) | 0 | 2 | C1 |
| 4 | (B, L, C) | 1 | 1 | tie |
| 5 | (A, P, M) | 2 | 2 | tie |

For O4:

```text
O4 = (B, L, C)
C1 = (A, L, C) -> different, same, same -> 1 + 0 + 0 = 1
C2 = (B, P, C) -> same, different, same -> 0 + 1 + 0 = 1
```

For O5:

```text
O5 = (A, P, M)
C1 = (A, L, C) -> same, different, different -> 0 + 1 + 1 = 2
C2 = (B, P, C) -> different, same, different -> 1 + 0 + 1 = 2
```

Final answer:

```text
O1 -> C1
O2 -> C2
O3 -> C1
O4 -> tie
O5 -> tie
```

If the exam question gives a tie rule, apply that rule.

## Question 12: K-Modes New Mode Calculation

Suppose a K-modes cluster contains:

| Observation | x1 | x2 | x3 |
|---|---|---|---|
| O1 | A | L | M |
| O3 | A | L | C |
| O5 | A | P | M |

Calculate the new mode for this cluster.

### Model Answer

For K-modes, choose the most frequent category in each column.

For x1:

```text
A, A, A -> A appears 3 times
mode x1 = A
```

For x2:

```text
L, L, P -> L appears 2 times
mode x2 = L
```

For x3:

```text
M, C, M -> M appears 2 times
mode x3 = M
```

Final answer:

```text
New mode = (A, L, M)
```

## Question 13: K-Modes Tie in Mode Calculation

Suppose a K-modes cluster contains:

| Observation | x1 | x2 | x3 |
|---|---|---|---|
| O1 | A | L | M |
| O3 | A | L | C |
| O4 | B | L | C |
| O5 | A | P | M |

Calculate the new mode. Identify any tie.

### Model Answer

For x1:

```text
A, A, B, A
A appears 3 times, B appears 1 time
mode x1 = A
```

For x2:

```text
L, L, L, P
L appears 3 times, P appears 1 time
mode x2 = L
```

For x3:

```text
M, C, C, M
M appears 2 times, C appears 2 times
```

There is a tie for x3.

Final answer:

```text
x1 = A
x2 = L
x3 = tie between M and C
```

So the new mode is:

```text
(A, L, M) or (A, L, C), depending on the tie rule.
```

In an exam, write the tie clearly instead of hiding it.

## Question 14: Identify the Correct Clustering Method

For each dataset, state whether K-means or K-modes is more suitable.

| Dataset | Variables |
|---|---|
| A | age, income, spending score |
| B | color, brand, product type |
| C | height, weight, blood pressure |
| D | operating system, browser, country |

### Model Answer

K-means is used for numerical variables.

K-modes is used for categorical variables.

| Dataset | Suitable method | Reason |
|---|---|---|
| A | K-means | age, income, and spending score are numerical |
| B | K-modes | color, brand, and product type are categorical |
| C | K-means | height, weight, and blood pressure are numerical |
| D | K-modes | operating system, browser, and country are categorical |

Final answer:

```text
A -> K-means
B -> K-modes
C -> K-means
D -> K-modes
```

## Question 15: K-Prototypes Concept Question

A dataset contains these variables:

| Variable | Type |
|---|---|
| Age | Numerical |
| Time Spent | Numerical |
| Operating System | Categorical |
| ISP | Categorical |

Which clustering method is most suitable: K-means, K-modes, or K-prototypes? Explain.

### Model Answer

K-means is mainly for numerical data.

K-modes is mainly for categorical data.

This dataset has both:

```text
Numerical: Age, Time Spent
Categorical: Operating System, ISP
```

Therefore, the most suitable method is K-prototypes.

Final answer:

```text
Use K-prototypes because the dataset contains both numerical and categorical variables.
```

## Final Exam Checklist

Before writing your final answer, check:

- Did I use mean for K-means?
- Did I use mode for K-modes?
- Did I square the differences for Euclidean distance?
- Did I use absolute values for Manhattan distance?
- Did I assign to the smallest distance or lowest dissimilarity?
- Did I recalculate the centroid or mode after reassignment?
- Did I clearly mention any tie?
- Did I stop only when the cluster assignment no longer changed?
