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

* Complement of an event is everything the event is not as the name suggests the complement.
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
* Pn = n *n-1* n-2 *...* 1 = n!

### Factoials ###

* n! is the product of the natural numbers from 1 to n.
* n! = 1 *2* 3 *...* n
* Negative numbers dont have a factorial.
* 0! = 1
* (n+k)! = n! *(n+1)* (n+2) *...* (n+k)
* (n-k)! = n! / (n-k+1) *(n-k+2)* ... * (n-k+k)
* n > k then n! / k! = (k+1) *(k+2)* ... * n

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

* distribution shows the possible values a variable can take and how frequently they occur.
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
* transformation is a way in which we can alter every element of a distribution to get a new distribution with similar characteristics for normal distributions.
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
* logistic distribution is defined by two key features. It's mean and its scale parameter the former dictates the center of the graph whilst the latter shows how spread out the graph is going to be.

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

### Confidence interval ###

* The point estimate is the midpoint of the interval.
* confidence interval is the range within which you expect the population parameter to be.
* common confidence levels are 90%, 95%, 99% which means alpha is 0.1, 0.05 and 0.01 respectively.
* The Student’s T distribution approximates the Normal distribution but has fatter tails. This means the probability of values being far away from the mean is bigger. For big enough samples, the Student’s T distribution coincides with the Normal distribution.
* when population variance is unknow we use t-statistics.
* Margin error
  * The true population mean falls in the interval defined by the sample mean plus minus the margin of error.
  * Getting a smaller margin of error means that the confidence interval would be narrower
  * more observations there are in the sample the higher the chances of getting a good idea about the true mean of the entire population.
  * A higher statistic increases the ME. A higher standard deviation increases the ME. A higher sample size decreases the ME. Therefore: a) and c) are ambiguous. as we don’t know which effect is stronger. With b), the ME will definitely increase. In d) both statements decrease the ME.

## Hypothesis testing ##

* Steps in data driven decision making.
  * formulate a hypothesis
  * find the right test
  * execute the test
  * make a decision
* Hypothesis is an idea that can be tested.
  * Null hypothesis denoted by Ho
    * It is the statement we are trying to reject, therefore the null is the present state of affairs while the alternative is our personal opinion.
  * Altenative hypothesis denoted by H1 or Ha
  * Significance level
    * denoted by alpha : the probability of rejecting the null hypothesis, if it is true.
    * If the test value falls into the rejection region, you will reject the null hypothesis.
* Type I and type II errors
  * Type 1 error is when you reject a true null hypothesis. false positive. cause sololey depends on tester.
  * type 2 error is when you accept a false null hypothesis. false negative. causes could be sample size.
  * goal of hypothesis testing is to reject a false null hypothesis. probability 1-beta a.k.a power of the test.
* P-Value : it is the smallest level of significance at which we can still reject the null hypothesis, given the observed sample statistic. you should reject the null hypothesis if p-value is less than significance level.

## Regression Analysis ##

* Linear Regression is a linear approximation of a causal relationship between two or more variables.
* simple linear regression : y = a+bx+e, y is dependent and x is independent. e is error. this is for population data but for y^ = b0 + b1 x1 is for sample data can be considered.
* correlation vs regression
  * correlation does not imply causation.
  * correlation describes relationship but regression explains one variable affects the other.
  * correlation shows movement of different variables together, but regression shows cause and effect.
  * correlation between x and y is equal to correlation between y and x. but regression is one way.
* The R-squared shows how much of the total variability of the dataset is explained by your regression model. This may be expressed as: how well your model fits your data. It is incorrect to say your regression line fits the data, as the line is the geometrical representation of the regression equation. It also incorrect to say the data fits the model or the regression line, as you are trying to explain the data with a model, not vice versa.
* least sum of squared error is always desired. SSE and SSR are inversly proportional.
* Multiple linear regression - has more number of input/independent variables and y^ = b0+b1x1+b2x2+...+bnxn+e
  * Adjusted R-Squared : it measures how much of the total variability is explained by our model. the adjusted R-squared is smaller than the R-squared. The statement is not true only in the extreme
  occasions of small sample sizes and a high number of independent variables.

  <img src="https://render.githubusercontent.com/render/math?math=R^2_{adj.} = 1 - (1-R^2)*\frac{n-1}{n-p-1}">

  * Multiple regressions are always better than simple ones, as with each additional variable you add, the explanatory power may only increase or stay the same.
  * We should cherry pick the data/columns when dealing with multiple linear regression as even a single unnecessary column of data will effect the models performance by a large margin. R-Squared is a good way of analyzing this but not the only tool to do so.
  * a new parameter if adding it increases Larger image R-squared but decreases adjusted Larger image R-squared then the variable can be omitted since it holds no predictive power.
  * F-Statistic : it is used for testing the overall significance of the model. if all the betas are 0 then none of the x matters hence our model has no merit. the lower the F-statistic the closer to a non-significant model.
  * OLS Assumptions : If a regression assumption is violated then performing regressional analysis will yield an incorrect result.
    * Linearity : using a linear model for non-linear dataset will not make any sense. hence need to run non-linear regression or exponenential transformation  or log transformation and convert the data to a linear dataset as part of pre-processing.
    * No endogeneity of regressors : Omitted variable bias is the relevant column of dataset forgotten to include. everything that model does not explain will go into the error. This kind of error is hard to fix.
    * Normality and homoscedasticity : Normality is that we assume that error is normally distributed. Zero mean, if the mean is not expected to be zero then the line is not the best fitting one. however having an intercept sloves the problem. Homoscedasticity is to have equal variance. this can be prevented by looking at omitted variables or outliers or by log transforming the data. if we take the log transformation then the model is called semi-log model y^ = b0+b1(log x1). if we take log for both x and y then the model is called log-log model.
    * No autocorrelation : Autocorrelation is not observed in cross-sectional data. no serial correlation, commonly seen in stock market data becuase underlying assest is same but takes up different values on each day. this can be detected in multiple ways, common way is to plot all the residuals and look for patterns, if not found then we are good. another way is Durbin-Watson test provided by statsmodels and values range between 0 and 4 representing 2 as no correlation <1 or >3 cause an alarm. There is no workaround hence better to use autoregressive or moving average or autoregressive moving average or autoregressive integrated moving average models.
    * No multicollinearity : two or more variables having high collinearity. e.g. b = 2+5b, if the collinearity is high between two variables then one can be almost represented by other hence there is no point in using both in our model. hence transforming two variables into one (e.g. average price) would potentially solve the problem.
  * Dummy variables : Categorical colums cannot be considered directly into model. hence we introduce a new variable where we immitate categorical values to numericals as part of data preprocessing.
  * Standardization is the process of transforming data into a standard scale. i.e. sample value - mean / standard deviation.
  
  <img src="https://render.githubusercontent.com/render/math?math=standardization = \frac{x-\mu}{\sigma}">

  * Overfitting : our training has focused on the particular training set so much it has "missed the point"
  * Underfitting : The model has not captured the underlying logic of the data.

## Sklearn ##

* Machine learning package
* Advantages
  * Incredible documentation
  * variety
    * Regression
    * classification
    * clustering
    * support vector machines
    * dimensionality reduction
* disadvantages : deep learning

## Logistic regression ##

* Logistic regression implies that the possible outcomes are not numerical but categorical.
  * used for binary prediction.

<img src="https://render.githubusercontent.com/render/math?math=Pr(Y_i=1|X_i) = {\frac{exp(\beta_0 + \beta_1X_i + \beta_2X_2 + \beta_3X_3 + \beta_4X_4 + \beta_5X_5)}{1 + exp (\beta_0 + \beta_1X_i + \beta_2X_2 + \beta_3X_3 + \beta_4X_4 + \beta_5X_5)}}
">

* Maximum likelihood estimation : its a function which estimates the likely it is that the model at hand describes the real underlying relationship of the variables. Computer is going through different values until it finds a model for which the liklihood is the highest and when it can no longer improve it, it will just stop the optimization. another meric is log-likelihood which can be consider the bigger value the better model it is. LL-Null is the log likelihood of a model which has no independent variables. LLR p-value measures if our model is statistically different from LL-Null, a.k.a useless model.

## Cluster analysis ##
* cluster analysis is a multivariate statistical technique that groups observations on the basis some of their features or variables they are described by.
* the goal of clustering is to maximize the similarity of observations within a cluster and maximize the dissimilarity between cluster.
* applications
  * Market segmentation
  * Image segmentation
## K-Means clustering ##
* steps
  * choose the number of cluster (K)
  * specify the cluster seeds
  * assign each point to a centroid
  * adjust the centroid and reiterate from step 3
* How to choose right amount of clusters ?
  * Within cluster sum of squares or WCSS : is a measure developed within the ANOVA framework. if we minimize WCSS then we have reached the perfect clustering solution.
    * Analysis of variance (ANOVA) is a tool used to partition the observed variance in a particular variable into components attributable to different sources of variation. Analysis of variance (ANOVA) uses the same conceptual framework as linear regression.
  * Elbow method 
* Advantages
  * simple to understand
  * Fast to cluster
  * Widely available in many modules
  * Easy to implement
  * Always yields a result
* Disadvantages
  * we need to pick K, the elbow method helps in this case.
  * sensitive to initialization, solution is k-means++
  * sensitive to outliers, hence need to remove the outliers in data pre-processing. 
  * produces spherical solutions
  * standardization, if we know that one column data is more important than other then standardization should not be used.
* Types of analysis
  * Exploratory : get acquainted with data, search for patterns, plan
  * Confirmatory : explain a phenomenon, confirm a hypothesis, validate previous research
  * Explanatory : explain a phenomenon, confirm a hypothesis, validate previous research
* Types of clustering
  * Flat : K-Means
  * Hierarchical : Taxonom of the animal kingdom
    * Agglomerative (bottom-up) : Dendrogram 
      * shows all the possible linkage between clusters
      * we understand the data much better
      * No need to preset the number of clusters
      * Many methods to perform hierarchical clustering (Ward)
      * Computation intensive
    * Divisive (top-down)

## Mathematics : Linear Algebra ##
* Matrix : It is a collection of numbers ordered in rows and columns. matrix can only contain numbers, symbols or equations.
* Vectors and scalars :
  * a matrix with one row and one column is called scalar. represented by a point.
  * a vector has either one column or one row. simplest linear algebraic object. represented by a line.
    * Column vector
    * row vector 
  * length of vector is the number of elements in it.
* for addition or subtraction of two matrices the shape of them should be same.
* for multiplication of vectors the length of vectors should be the same. we can only multiply an m*n matrix with an n*k matrix meaning number of columns in first matrix should be equal to number of rows in second matrix.
* A tensor is generalization of the above concepts like scalars, vector and matrix
* scalar is tensor of rank 0,  vector is tensor of rank 1 and matrix is tensor of rank 2.
## Deep learning ##
* basic logic behind training a model.
  * Data
  * Model
  * Objective function : it is the measure used to evaluate how well the model output match the desired correct values.
    * Loss function : lower the loss function the higher the level of accuracy. any function that holds the basic property of higher for worse results, lower for better results can be a loss function 
      * L2-Norm : used in regression model. Norm comes from the fact it is the vector norm, or euclidean distance of the outputs and the targets
      * Coss Entropy : used in classification.
    * Reward function : Higher the reward function the higher the level of accuracy.
  * Optimization alogirthm 
    * Gradient descent : The gradient is a mathematical term. It is the multivariate generalization of the derivative concept.
      * eta is the learning rate 
      * oscillation : a repetitive variation around a central value.
      <img src="https://render.githubusercontent.com/render/math?math=x_{i 1}=x_i-n f`(x_i)">
      * generally we want the learning rate to be high enough so we can reach the closest minimum in a rational amount of time and low enough so we don't oscillate around minimum.