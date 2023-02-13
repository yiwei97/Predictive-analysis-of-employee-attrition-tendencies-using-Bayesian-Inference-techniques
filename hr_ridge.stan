data {
  // Define variables in data
   // Number of observations (an integer)
   int<lower=0> N;
   // Number of coefficients excluding constant term
   int<lower=0> p;
   // Variables
   int<lower=0,upper=1> y[N];
   real x1[N];
   real x2[N];
   real x3[N];
   real x4[N];
}
parameters {
  // Define parameters to estimate
  real alpha; //constant term
  real beta[p]; //coefficient estimates
  real<lower=0> lambda; //penalty term
} 
transformed parameters  {
   // Mean
   real mu[N];
   for (i in 1:N) {
     mu[i] = alpha + beta[1]*x1[i] + beta[2]*x2[i] + beta[3]*x3[i] + beta[4]*x4[i]; 
   }
}
model {
   //Prior
   alpha ~ cauchy(0,10); //Following Gelman(2008)
   lambda ~ cauchy(0,2.5); //Half cauchy, as in van Erp et al (2019)
   for (i in 1:p){
      beta[i] ~ normal(0,1/(2*lambda)); //Prior for coefficients
   }
   //Likelihood
   y ~ bernoulli_logit(mu);
}
generated quantities {
  vector[N] y_prob; //Estimated probability of attrition
  vector[N] y_hat; //Binary outcome for predicted attrition
  vector[N] indiv_log_likelihood; //log-likelihood for individual observation
  real N_real; //Converting from integer to real
  real log_likelihood; //Sum of individual log-likelihood
  real AIC; //Akaike Information Criterion
  real BIC; //Bayesian Information Criterion
  {
    for (i in 1:N)  {
      //Predicting attrition probability for each observation
      y_prob[i] = exp(mu[i])/(1+exp(mu[i])); 
      //Assuming positive attrition for individuals with attrition probability > 0.5
      if (y_prob[i]>0.5) 
         y_hat[i]=1;
      else
         y_hat[i]=0;
      //Summing up individual log-likelihood
      indiv_log_likelihood[i] = log(y_prob[i]*y[i] + y_prob[i]*(1-y[i])); 
          }
  }
  //Calculating AIC and BIC
  N_real = N;
  log_likelihood = sum(indiv_log_likelihood);
  //Number of free parameters: p coefficients + 1 constant + 1 penalty term
  AIC = 2*(p+2) - 2*log_likelihood;
  BIC = -2*log_likelihood + log(N)*(p+2);
}