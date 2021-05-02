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
* Combination = (n+p-1)! / p! * (n-1)!
## Probability Distribution ##
*  distribution shows the possible values a variable can take and how frequently they occur.
* probability distributions or simply probabilities measure the likelihood of an outcome depending on how often it is featured in the sample space.
* regardless of whether we have a finite or infinite number of possibilities. We define distributions using only two characteristics mean and variance.
* standard deviation is simply the positive square root of variance as you may suspect.
* Types of probability distributions
    * Discrete distributions  
        * Uniform Distributions : all outcomes are equally likely i.e. Equiprobable.
        * Bernoulli Distributions : Events with only two possible outcomes.
        * Binomial Distributions : The outcomes for each iteration are two but we have many iterations.
        * Poisson Distributions : test out how unusual an event frequency is for a given interval.
    * Continuous distributions : the probability distribution would be a curve.
        * Normal distributions 
        * Student's-T distribution : limited data - A small sample approximation of normal distribution. And accommodates extreme values significantly better.
        * Chi-Squared distributions : Asymmetric, only consists of non-negative values. used in hypothesis testng.
        * Exponential distributions 
        * Logistic distributions : useful in forecast analysis, and determining a cut-off point for a successful outcome.
### Uniform Discrete distributions ##
* X ~ U(a,b) or X ~ U(3,7) : X follows a discrete uniform distribution ranging from three to seven events which follow the Uniform Distribution are ones where all outcomes have equal probability.
### Bernoulli Discrete distributions ##
* X ~ Bern(p) : X follows Aber newly distribution with a probability of success equal to P.
* one trial and two possible outcomes follows such a distribution These may include a coin flip a
### Binomial Discrete distribution ###
* In essence binomial events are a sequence of identical Bernoulli events.
* X ~ B(n,p) or X ~ B(10,0.6) X follows a binomial distribution with 10 trials and a likelihood of point six to succeed
### Poisson Discrete distributions ###
* deals with the frequency with which an event occurs within a specific interval
### Continuous distributions ###
* sample space is infinite, we cannot come up with probability distribution table. But it can be represented in graph.
* Graph is called PDF (probability density function) f(y) i.e. the associated probability for every possible value "y"
* Cumulative distribution function (CDF) is denoted by F(y) : especially useful when we want to estimate the probability of some interval. the cumulative probability is simply the probability of the interval from negative infinity
### Normal distribution ###
* The graph of normal distribution is bell shaped and symmetrical from mean.
* Expect value for normal distribution is mean value.
* 68, 95, 99.7 law state that 68% of all outcomes fall within one standard deviation away from the mean and 95% fall within two standard deviations and 99.7 within three.
*  transformation is a way in which we can alter every element of a distribution to get a new distribution with similar characteristics for normal distributions.
* standardizing is a special kind of transformation in which we make the expected value equal to zero and the variance equal to one the distribution we get after standardizing any normal distribution is called a standard normal distribution.
* In addition to these sixty eight ninety five ninety nine point seven rule a table exists which summarizes the most commonly used values for the CTF of a standard normal distribution this table is known as the standard normal distribution table or the z score table
### Student's T distribution ###
* t(k) : Y ~ t(3) t distribution with three degrees of freedom
* Small sample size approximation of a normal distribution.
* The graph of T distribution is bell shaped but with fatter tails to accommodate the occurrance of values far away from the mean.
### Chi-Squared Distribution ###
* Y ~ X**2(3) : a chi square distribution with three degrees of freedom
* graph will be asymmetric
### Exponential distribution ###
* X ~ Exp(1/2) : variable X follows an exponential distribution with a scale of a half.
* variables which most closely follow an exponential distribution are ones with a probability that initially decreases before eventually plateauing.
* The PD f of such a function would start off very high and sharply decrease within the first few timeframes.
* We require a rate parameter denoted by the Greek letter lambda. This parameter determines how fast the CTF curve reaches the point of plateauing and how spread out
### Logistic distributions ###
* Y ~ Logistic(6, 3) :  Y follows a logistic distribution with location 6 and a scale of 3.
*  logistic distribution is defined by two key features. It's mean and its scale parameter the former dictates the center of the graph whilst the latter shows how spread out the graph is going to be.
## Probability in other fields ##
* Finanace : calculating the probability of options going up/down
* Statistics : predominantly focuses on samples and incomplete data.
    * confidence intervals is that they simply approximate some margins for the mean of the entire population based on a small sample.
    * hypothesis is an idea that can be tested. The three crucial requirements for conducting successful hypothesis testing are knowing the mean variance and type of the distribution.
    * any distribution we try on predicts a value for all points within our dataset. This is what the distribution anticipates the actual data point to be. So it is essentially a type of anticipated average value.
* Data Science : We usually try to analyze past data and use some insight we find to make reasonable predictions about the future furthermore in mathematical modeling we often tend to run artificial simulations. data science is an expansion of probability statistics and programming that implements computational technology to solve more advanced questions.
## Statistics ##
* Population : collection of all items denote by N
* Sample : subset of the population denoted by n 
## Descriptive statistics ##
* Types of data 
    * Categorical
    * Numerical
        * Continuous
        * Discrete
* Measurement level
    * Qualitative
        * Nominal : Categories that cannot be put in any order.
        * Ordinal : Categories that can be put in order.
    * Quantitative 
        * Interval : Does not have true zero like temperature
        * Ratio : have true zero 
* Visualization techniques
    * Frequency distribution tables : shows the category and its corresponding absolute frequency
    * Bar charts : each bar represents a category
    * Pie Charts : share of an item as part of total 
    * Pareto diagrams : bar chart with in descending order along with cumulative frequency curve.
        * 80-20 rule : 80% of the effect come from 20% of the causes.
    * Cross tables and scatter plots
        * total is calculated column and rowwise and represented using side-by-side chart.
        * scatter plots are used while representing two numerical data and represents lot of data. good to detect outliers.
* Measures of Central tendency : Mean, median and mode
    * mean is simple average. most widely used.
    * median is the middle number in ordered data set. (n+1)/2
    * mode is the value that occurs most. can be used for categorical and numerical data. if all the data occurs only once then we can say that there is no mode.
* Measure of asymmetry : skewness
    * skewness is measure of asymmetry that indicates whether the observation in a dataset are concentrated on one side. it tells us where data is situated.
        * mean > median then right skewed.
        * mean = median = mode then distribution is symmetrical
        * mean < median then left skewed
* Measure of variability : variance, standard deviation, coefficient of variation.
  * variance measures the dispersion of a set of data points around their mean.
  * coefficient = standard deviation / mean and is needed because, comparing the standard deviations of two different data sets is meaningless. But comparing coefficients of variation is.
* Measures of relationships between variables : covariance and correlation
  * covariance is a measure of the joint variability of two variables. 
    * positive covariance means two variables move together. 
    * covariance zero means they are independent. 
    * negative covariance means that they move in opposite direction.
  * Correlation adjusts covariance, so that the relationship between two variables becomes easy and intuitive to interpret. covaiance can take any number but correlation ranges between -1 and 1. 
    * correlation coefficient = covariance / std(x) * std(y)
    * correlation does not imply causality 
## Infrential statistics ##
* A distribution is a function that shows the possible values for a variable and how often they occur.
* Normal and student's T distribution
  * they approximate a wide variety of random variables. 
  * distributions of sample means with large enough sample sizes could be approximated to normal distribution.
  * all computable statics are elegant
  * decisions based on normal distribution insight have a good track record.
  * normal distribution is also called Gaussian distribution 
* Standardization 
  * making mean to 0 and standard variance to 1. denoted by z ~ N(0,1)
* central limit theorem
  * The central limit theorem states that if you have a population with mean μ and standard deviation σ and take sufficiently large random samples from the population with replacement , then the distribution of the sample means will be approximately normally distributed
* Standard error : the variability of the means of the different samples we extracted. and standard error decreases as the sample size increases.
* Estimates
  * Point estimates : point estimate is located exactly in the middle of the confidence interval.
  * confidence estimates
  * The word statistic is the broader term a point estimate is a statistic.