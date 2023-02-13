data {
  // Define variables in data
   // Number of observations (an integer)
   int<lower=0> N;
   //Number of age groups
   int<lower=0> K;
   // Number of predictor variables
   int<lower=0> p;
   //Mapping function - mapping observation to job role
   int<lower=1,upper=K> JobRole[N];
   // Variables
   int<lower=0,upper=1> y[N];
   matrix[N, p] x;
}
parameters {
  // Define parameters to estimate
  real mu_alpha; //grand mean of constant term
  vector[p] mu_beta; //grand mean of coefficient terms 
  real<lower=0> sigma_alpha; //between variance for constant term
  vector<lower=0>[p] sigma_beta; //between variance for coefficient terms
  vector[K] alpha; //subpopulation constant terms
  matrix[K, p] beta; //subpopulation coefficient terms
} 
transformed parameters  {
   // Mean
   real mu[N];
   for (i in 1:N) {
     mu[i] = alpha[JobRole[i]] + beta[JobRole[i],1]*x[i,1] + beta[JobRole[i],2]*x[i,2] 
     + beta[JobRole[i],3]*x[i,3] + beta[JobRole[i],4]*x[i,4]; 
   }
}
model {
    //Prior
   mu_alpha ~ normal(0,10000); //Following Gelman(2008)
   mu_beta ~ normal(0, 10000); //Big variance prior
   sigma_alpha ~ normal(0, 10000); //Big variance prior
   sigma_beta ~ normal(0, 10000); //Big variance prior
   alpha ~ normal(mu_alpha, sigma_alpha); //Subpopulation prior for constant term
   for (i in 1:p){
      beta[,i] ~ normal(mu_beta[i], sigma_beta[i]); //Subpopulation prior for coefficient terms
   }
   //Likelihood
   y ~ bernoulli_logit(mu);
}
generated quantities {
  //Defining quantities to estimate
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
  //Number of free parameters: K*(p+1) subpopulation coefficients + (p+1) grand means + (p+1) between variances = 2+ 2*p + K + K*p
  AIC = 2*(2+2*p+K+K*p) - 2*log_likelihood;
  BIC = -2*log_likelihood + log(N_real)*(2+2*p+K+K*p);
}