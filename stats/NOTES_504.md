# Notes on Stat504

Extraits de <https://online.stat.psu.edu/stat504/>

## Notations

- $\hat\theta$ : sampled
- casser la symétrie "explicative" du tableau : _response variable_ VS _explanatory variables_ or _predictors_

## 3: Two-Way Tables: Independence and Association

### 3.1 - Notation & Structure -- Saturated model

<https://online.stat.psu.edu/stat504/lesson/3/3.1#paragraph--440>

If the sample units are randomly selected from a large population, we can assume that the cell counts $(n_{11}, \dots, n_{IJ})$ have a multinomial distribution with index $n_{++} = n$ and parameters

$$\pi = (\pi_{11}, \dots, \pi_{IJ})$$

This is the general multinomial model, and it is often called the **saturated model**, because it contains the _maximum number of unknown parameters_. There are unknown parameters (elements) in the vector $\pi$ but because the elements of _must sum to one_ since this is a probability distribution, then there are really $I \times J - 1$ unknown parameters that we need to estimate.

### 3.2 - Sampling Schemes

The main generating probability mechanisms are Poisson, Binomial, and Multinomial models, but for two-way tables, the margins play a big role. We will discuss the following sampling schemes:

- Unrestricted sampling (Poisson)
- Sampling with fixed total sample size (Multinomial)
- Sampling with fixed certain marginal totals (Product-Multinomial, Hypergeometric)

<https://online.stat.psu.edu/stat504/lesson/3/3.3#paragraph--642>

### The general rule for Degrees of Freedom

_DF are equal to the number of parameters specified (estimated) under the alternative model (hypothesis) minus the number of parameters estimated under the null model (hypothesis)._

#### Computing Degrees of Freedom

Recall, under the saturated model, $\pi$ contains $IJ -1$ free (unique) parameters. And under the independence model, $\pi$ is a function of $(I-1) + (J-1)$ parameters since each joint probability $\pi_{ij} $can be written as the product of the marginals $\pi_{i+}\pi_{+j}$, each of which has the sum-to-one constraint. The degrees of freedom are therefore

$$ \nu = (IJ - 1) - (I - 1) - (J - 1) = (I - 1)(J - 1)$$

## 6: Binary Logistic Regression

we study binomial logistic regression, a special case of a generalized linear model.
