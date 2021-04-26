# Data Science Bootcamp Notes
# Probability #
* Definition : the likelihood of an event occurring. Example flipping a coin has two possible outcomes, heads or tails. In other words quantifying how likely each event is on its own.
* P(A) = Preferred outcome / all outcomes, where A is event and P is the probability. 
* Probability of two independent events occurring at the same time is equal to product of all the probabilities of individual events. 
    * P(A and B) = P(A) . P(B)
## Expected Values ##
* The average outcome we expect if we run an experiment many times.
    * For example if we toss a coin 20 times and record the 20 outcomes that entire process is a single experiment
* All right the probabilities we get after conducting experiments are called experimental probabilities.
    * P(A) = successful trails / all trails
    * E(A) is the outcome we expect to occur when we run an experiment.
        * E(A) = P(A) * n where n is number of trails.
* We can use expected values to make predictions about the future based on past data.
## Frequency ##
* Probability frequecy distribution is a collection of the probabilities for each possilbe outcome. Example 7 is the most common sum of two dice, usually represented in graph.

### Probability Frequency Distribution ###
* A collection of all the probabilities for the various outcomes is called a probability frequency distribution

| Sum  | Frequency | Probability |
|------|-----------|-------------|
| 2    | 1         | 1/36        |
| 3    | 2         | 1/18        |
| 4    | 3         | 1/12        |
| 5    | 4         | 1/9         |
| 6    | 5         | 5/36        |
| 7    | 6         | 1/6         |
| 8    | 5         | 5/36        |
| 9    | 4         | 1/9         |
| 10   | 3         | 1/12        |
| 11   | 2         | 1/18        |
| 12   | 1         | 1/36        |
## Complements ##
*  Complement of an event is everything the event is not as the name suggests the complement.
* Probability expresses the likelihood of an event occurring so any probability less than 1 is not guaranteed to occur. Therefore there must be some part of the sample space we have not yet accounted for.
    * Representation A`
    * P(A) + P(B) + P(C) = 1
    * A` = B + C
    * P(A`) = 1 - P(A)
# Probability - Combinatorics
* For starters combinatorics deals with combinations of objects from a specific finite set.
* We will explore the three integral parts of combine matrix permutations variations and combinations.
## Permutations ##
* Permutations permutations represents the number of different possible ways we can arrange a set of elements.
* Pn = n * n-1 * n-2 * ... * 1 = n!
### Factoials ###
* n! is the product of the natural numbers from 1 to n.
* n! = 1 * 2 * 3 * ... * n
* Negative numbers dont have a factorial.
* 0! = 1
* (n+k)! = n! * (n+1) * (n+2) * ... * (n+k)
* (n-k)! = n! / (n-k+1) * (n-k+2) * ... * (n-k+k)
* n > k then n! / k! = (k+1) * (k+2) * ... * n
## Variations ##
* Variations express the total number of ways we can pick and arrange some elements of a given set.
* Variations = n ** p, where n is the total number of elements, we have available and p is the number of positions we need to fill.
* The number of variations with repetition.When picking P many elements out of n elements is equal to N to the power of P.
### Variation without repetition ###
* Variation = n! / (n-p)!
## Combinations ##
* All the different permutations of a single combination are different variations.
* Combinations = variations / permutations , C = V / P
* C = n! / p! * (n-p)!
* Combinations are symmetrical i.e. 10C3 = 10C7
* The likelihood of two independent events occurring simultaneously equals the product of their individual probabilities.

## With repetition ##
* Variations = n ** p
* Combination = (n+p-1)! / p * (n-1)!
