# Continuous Regression

## Description

It is a well known fact that a linear regression finds the optimal $\boldsymbol{\beta}$ on 
the following equation:

$$ y_i=\beta_1x_{i1}+\beta_2x_{i2}+ \dots +\beta_px_{ip} $$ 

That given n by p predictor matrix $\boldsymbol{X}$ and n-dimensional target column vector 
$\boldsymbol{Y}$, minimize the mean squared difference between predicted and target values:

$$ S(\boldsymbol\beta)=\| \boldsymbol{Y-X\beta}\|^2  $$ 

This minimum is reached when:

$$ \boldsymbol\beta= (\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{Y}  $$ 

But what if we had an infinite amount of individuals? Turns out that ordinary linear regression
with infinite samples can be used for a variety of applications.



