# Week 11 Manual Calculation Guide: Clustering

This guide is based on `week11.ipynb` and the class spreadsheet `K clustering calculation (230526).xlsx`.

Important exam habit: always use the values shown in the exam question. The notebook CSV and class spreadsheet have small differences in some example values, so do not memorize one table blindly.

## 1. What Week 11 Is Testing

Week 11 focuses on clustering, especially:

- K-means: numerical data, centroid is the mean.
- K-modes: categorical data, centroid is the mode.
- Euclidean distance: common distance for K-means.
- Manhattan distance: possible alternative distance.
- Reassignment step: assign each observation to the nearest cluster.
- Update step: recalculate the cluster center after assignment.

The manual calculation usually asks you to do one or more of these:

- calculate distance between two observations;
- assign observations to clusters;
- calculate new centroids for K-means;
- calculate dissimilarity scores for K-modes;
- calculate new modes for K-modes;
- repeat until cluster assignments no longer change.

## 2. Formula Sheet

### Euclidean Distance

Use for numerical data.

For two observations:

| Observation | x1 | x2 | x3 |
|---|---:|---:|---:|
| A | a1 | a2 | a3 |
| B | b1 | b2 | b3 |

The Euclidean distance is:

```text
d(A, B) = sqrt((a1 - b1)^2 + (a2 - b2)^2 + (a3 - b3)^2)
```

For 5 variables:

```text
d(A, B) = sqrt((x1A - x1B)^2 + (x2A - x2B)^2 + (x3A - x3B)^2 + (x4A - x4B)^2 + (x5A - x5B)^2)
```

### Manhattan Distance

Use if the question specifically asks for Manhattan distance.

```text
d(A, B) = |x1A - x1B| + |x2A - x2B| + |x3A - x3B| + ...
```

### K-Means Centroid

For numerical variables, the centroid is the average of each variable inside the cluster.

Example for cluster A with 3 observations:

```text
centroid x1 = (x1 of obs 1 + x1 of obs 2 + x1 of obs 3) / 3
centroid x2 = (x2 of obs 1 + x2 of obs 2 + x2 of obs 3) / 3
centroid x3 = (x3 of obs 1 + x3 of obs 2 + x3 of obs 3) / 3
```

Do this separately for every feature.

### K-Modes Dissimilarity

Use for categorical data.

For each feature:

```text
same category     = 0
different category = 1
```

Then add the scores.

Example:

| Object | x1 | x2 | x3 |
|---|---|---|---|
| O1 | A | L | M |
| C1 | A | L | C |

```text
x1: A vs A = 0
x2: L vs L = 0
x3: M vs C = 1

dissimilarity = 0 + 0 + 1 = 1
```

### K-Modes Mode

For categorical variables, the cluster center is the most frequent category for each feature.

Example:

| Object | x1 | x2 | x3 |
|---|---|---|---|
| O1 | A | L | M |
| O3 | A | L | C |
| O5 | A | P | M |

Mode for each column:

```text
x1: A, A, A -> A
x2: L, L, P -> L
x3: M, C, M -> M

new mode = (A, L, M)
```

## 3. Manual K-Means Procedure

Use this checklist in the exam.

1. Write down the initial centroids or initial cluster labels.
2. If initial labels are given, calculate the centroid of each cluster first.
3. For every observation, calculate distance to each centroid.
4. Assign the observation to the cluster with the smallest distance.
5. Recalculate the centroid of every cluster using the new members.
6. Repeat distance calculation and reassignment.
7. Stop when the cluster assignments do not change.

The key phrase is:

```text
nearest centroid wins
```

## 4. Worked Example: Euclidean Distance

From the class spreadsheet, K-means Example 1 uses this data:

| i | x1 | x2 | x3 | x4 | x5 | Initial group |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 10 | 2 | -1 | 4 | 0 | A |
| 2 | 12 | 4 | -5 | 4 | 1 | B |
| 3 | 10 | 6 | -6 | 4 | 0 | C |
| 4 | 9 | 2 | -1 | 5 | 0 | A |
| 5 | 10 | 6 | -3 | 4 | 0 | B |
| 6 | 9 | 4 | -4 | 5 | 1 | C |
| 7 | 8 | 4 | -5 | 5 | 1 | A |
| 8 | 10 | 6 | -1 | 5 | 0 | B |

### Distance Between Observation 1 and Observation 2

Observation 1:

```text
(10, 2, -1, 4, 0)
```

Observation 2:

```text
(12, 4, -5, 4, 1)
```

Step-by-step:

| Feature | Calculation | Squared difference |
|---|---:|---:|
| x1 | (10 - 12)^2 | 4 |
| x2 | (2 - 4)^2 | 4 |
| x3 | (-1 - -5)^2 = 4^2 | 16 |
| x4 | (4 - 4)^2 | 0 |
| x5 | (0 - 1)^2 | 1 |

```text
sum = 4 + 4 + 16 + 0 + 1 = 25
distance = sqrt(25) = 5
```

So:

```text
d(O1, O2) = 5
```

### Distance Between Observation 1 and Observation 3

Observation 3:

```text
(10, 6, -6, 4, 0)
```

```text
d(O1, O3)
= sqrt((10 - 10)^2 + (2 - 6)^2 + (-1 - -6)^2 + (4 - 4)^2 + (0 - 0)^2)
= sqrt(0 + 16 + 25 + 0 + 0)
= sqrt(41)
= 6.403
```

### Distance Between Observation 1 and Observation 4

Observation 4:

```text
(9, 2, -1, 5, 0)
```

```text
d(O1, O4)
= sqrt((10 - 9)^2 + (2 - 2)^2 + (-1 - -1)^2 + (4 - 5)^2 + (0 - 0)^2)
= sqrt(1 + 0 + 0 + 1 + 0)
= sqrt(2)
= 1.414
```

## 5. Worked Example: K-Means Centroid Calculation

Using the initial groups from Example 1:

```text
A = O1, O4, O7
B = O2, O5, O8
C = O3, O6
```

### Centroid A

Members:

| i | x1 | x2 | x3 | x4 | x5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 2 | -1 | 4 | 0 |
| 4 | 9 | 2 | -1 | 5 | 0 |
| 7 | 8 | 4 | -5 | 5 | 1 |

Calculate average column by column:

```text
x1 = (10 + 9 + 8) / 3 = 9
x2 = (2 + 2 + 4) / 3 = 2.667
x3 = (-1 + -1 + -5) / 3 = -2.333
x4 = (4 + 5 + 5) / 3 = 4.667
x5 = (0 + 0 + 1) / 3 = 0.333
```

So:

```text
CA = (9, 2.667, -2.333, 4.667, 0.333)
```

### Centroid B

Members:

```text
B = O2, O5, O8
```

```text
x1 = (12 + 10 + 10) / 3 = 10.667
x2 = (4 + 6 + 6) / 3 = 5.333
x3 = (-5 + -3 + -1) / 3 = -3
x4 = (4 + 4 + 5) / 3 = 4.333
x5 = (1 + 0 + 0) / 3 = 0.333
```

So:

```text
CB = (10.667, 5.333, -3, 4.333, 0.333)
```

### Centroid C

Members:

```text
C = O3, O6
```

```text
x1 = (10 + 9) / 2 = 9.5
x2 = (6 + 4) / 2 = 5
x3 = (-6 + -4) / 2 = -5
x4 = (4 + 5) / 2 = 4.5
x5 = (0 + 1) / 2 = 0.5
```

So:

```text
CC = (9.5, 5, -5, 4.5, 0.5)
```

## 6. Worked Example: Full K-Means Reassignment

From the spreadsheet K-means Example 2:

| i | x1 | x2 | x3 | Initial group |
|---:|---:|---:|---:|---|
| 1 | 10 | 2 | -1 | A |
| 2 | 11 | 3 | -1 | B |
| 3 | 18 | 5 | -1 | A |
| 4 | 20 | 4 | 0 | B |
| 5 | 19 | 3 | 0 | A |
| 6 | 8 | 2 | -1 | B |

### Step 1: Calculate Initial Centroids

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

### Step 2: Calculate Distance to Both Centroids

Use:

```text
d(Oi, CA) = sqrt((x1i - x1CA)^2 + (x2i - x2CA)^2 + (x3i - x3CA)^2)
```

| i | d to CA | d to CB | New group |
|---:|---:|---:|---|
| 1 | 5.831 | 3.180 | B |
| 2 | 4.690 | 2.028 | B |
| 3 | 2.887 | 5.395 | A |
| 4 | 4.435 | 7.102 | A |
| 5 | 3.416 | 6.037 | A |
| 6 | 7.789 | 5.110 | B |

New groups:

```text
A = O3, O4, O5
B = O1, O2, O6
```

Because the groups changed, recalculate the centroids.

### Step 3: Recalculate Centroids

New centroid A:

```text
A = O3, O4, O5

x1 = (18 + 20 + 19) / 3 = 19
x2 = (5 + 4 + 3) / 3 = 4
x3 = (-1 + 0 + 0) / 3 = -0.333

CA = (19, 4, -0.333)
```

New centroid B:

```text
B = O1, O2, O6

x1 = (10 + 11 + 8) / 3 = 9.667
x2 = (2 + 3 + 2) / 3 = 2.333
x3 = (-1 + -1 + -1) / 3 = -1

CB = (9.667, 2.333, -1)
```

### Step 4: Reassign Again

| i | d to CA | d to CB | New group |
|---:|---:|---:|---|
| 1 | 9.244 | 0.471 | B |
| 2 | 8.090 | 1.491 | B |
| 3 | 1.563 | 8.750 | A |
| 4 | 1.054 | 10.515 | A |
| 5 | 1.054 | 9.410 | A |
| 6 | 11.200 | 1.700 | B |

The groups are still:

```text
A = O3, O4, O5
B = O1, O2, O6
```

No change happened, so the K-means algorithm stops.

Final answer:

```text
Cluster A: O3, O4, O5
Cluster B: O1, O2, O6
Final CA = (19, 4, -0.333)
Final CB = (9.667, 2.333, -1)
```

## 7. Manual K-Modes Procedure

Use K-modes when the variables are categorical, such as:

```text
A, B, L, P, M, C
```

Procedure:

1. Write down the initial modes.
2. For each observation, compare it to every mode.
3. Put 0 if the category is the same.
4. Put 1 if the category is different.
5. Add the dissimilarity score.
6. Assign the observation to the cluster with the smallest score.
7. Recalculate the mode for each cluster.
8. Repeat until assignments do not change.

The key phrase is:

```text
lowest dissimilarity wins
```

## 8. Worked Example: K-Modes Dissimilarity

From the spreadsheet:

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

### Observation 1 Compared With C1

```text
O1 = (A, L, M)
C1 = (A, L, C)
```

| Feature | Compare | Score |
|---|---|---:|
| x1 | A vs A | 0 |
| x2 | L vs L | 0 |
| x3 | M vs C | 1 |

```text
d(O1, C1) = 0 + 0 + 1 = 1
```

### Observation 1 Compared With C2

```text
O1 = (A, L, M)
C2 = (B, P, C)
```

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
O1 belongs to C1
```

### Full First Assignment Table

| i | Oi | d to C1 | d to C2 | Assigned cluster |
|---:|---|---:|---:|---|
| 1 | (A, L, M) | 1 | 3 | C1 |
| 2 | (B, P, C) | 2 | 0 | C2 |
| 3 | (A, L, C) | 0 | 2 | C1 |
| 4 | (B, L, C) | 1 | 1 | tie |
| 5 | (A, P, M) | 2 | 2 | tie |

If there is a tie, follow the rule given by the lecturer or question. If no rule is given, state your tie decision clearly, for example:

```text
Tie rule used: keep the existing cluster / choose the first cluster / use lecturer's given assignment.
```

Do not silently hide a tie in your answer.

### Recalculate K-Modes Centers

Suppose after assignment:

```text
C1 = O1, O3, O4, O5
C2 = O2
```

For C1:

| Object | x1 | x2 | x3 |
|---|---|---|---|
| O1 | A | L | M |
| O3 | A | L | C |
| O4 | B | L | C |
| O5 | A | P | M |

Mode column by column:

```text
x1: A appears 3 times, B appears 1 time -> A
x2: L appears 3 times, P appears 1 time -> L
x3: M appears 2 times, C appears 2 times -> tie
```

If the question gives a tie rule, use it. If not, state the tie clearly.

Possible C1 mode:

```text
C1 = (A, L, M) or (A, L, C), depending on tie rule for x3
```

For C2:

```text
C2 = O2 = (B, P, C)
```

So:

```text
C2 = (B, P, C)
```

## 9. K-Means vs K-Modes

| Item | K-means | K-modes |
|---|---|---|
| Data type | Numerical | Categorical |
| Cluster center | Mean / centroid | Mode |
| Distance | Euclidean or Manhattan | Matching dissimilarity |
| Feature comparison | Squared difference or absolute difference | Same = 0, different = 1 |
| Update step | Average each variable | Most frequent category |

## 10. Exam Answer Templates

### Template: Euclidean Distance

```text
d(O__, O__) = sqrt((__ - __)^2 + (__ - __)^2 + (__ - __)^2)
            = sqrt(__ + __ + __)
            = sqrt(__)
            = __
```

For 5 variables:

```text
d(O__, O__) = sqrt((x1 difference)^2 + (x2 difference)^2 + (x3 difference)^2 + (x4 difference)^2 + (x5 difference)^2)
```

### Template: K-Means Centroid

```text
Cluster A members = O__, O__, O__

CA_x1 = (__ + __ + __) / 3 = __
CA_x2 = (__ + __ + __) / 3 = __
CA_x3 = (__ + __ + __) / 3 = __

CA = (__, __, __)
```

### Template: K-Means Assignment Table

| Observation | d to CA | d to CB | Assigned cluster |
|---:|---:|---:|---|
| O1 | | | |
| O2 | | | |
| O3 | | | |
| O4 | | | |

Decision rule:

```text
Choose the cluster with the smallest distance.
```

### Template: K-Modes Dissimilarity

```text
O__ = (__, __, __)
C__ = (__, __, __)

feature 1: same/different = __
feature 2: same/different = __
feature 3: same/different = __

dissimilarity = __ + __ + __ = __
```

### Template: K-Modes Assignment Table

| Observation | d to C1 | d to C2 | Assigned cluster |
|---:|---:|---:|---|
| O1 | | | |
| O2 | | | |
| O3 | | | |
| O4 | | | |

Decision rule:

```text
Choose the cluster with the lowest dissimilarity score.
```

### Template: K-Modes New Mode

```text
Cluster C1 members = O__, O__, O__

x1 values = __, __, __ -> mode = __
x2 values = __, __, __ -> mode = __
x3 values = __, __, __ -> mode = __

New C1 = (__, __, __)
```

## 11. Common Mistakes To Avoid

- Do not average categorical values. Use mode for K-modes.
- Do not use mode for numerical K-means. Use mean.
- Do not forget the square root in Euclidean distance.
- Do not take the square root before adding all squared differences.
- Do not compare only one feature unless the question asks for it.
- Do not forget to recalculate centroids after reassignment.
- Do not stop K-means after one iteration unless the question says to do only one iteration.
- Do not ignore ties in K-modes. State the tie rule.
- Do not mix up "highest frequency" and "lowest dissimilarity": mode uses highest frequency, assignment uses lowest dissimilarity.

## 12. Quick Final-Exam Flow

When you see numerical clustering:

```text
K-means -> calculate mean centroids -> calculate distance -> assign nearest -> update mean -> repeat
```

When you see categorical clustering:

```text
K-modes -> compare same/different -> assign lowest dissimilarity -> update mode -> repeat
```

When you see mixed numerical and categorical data:

```text
K-prototypes -> numerical part behaves like K-means, categorical part behaves like K-modes
```

For Week 11 manual calculation, the most important skills are:

```text
Euclidean distance
K-means centroid update
K-modes dissimilarity score
K-modes mode update
```
