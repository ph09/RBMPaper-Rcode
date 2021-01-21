forward <- function(stack, data, mode, nit){
  
  numLayers = ncol(stack)
  numcases = nrow(data)
  orgData = data
  
  if (strcmp(mode, 'deterministic')){
    for (i in 1:numLayers){
      hidbiases = stack[[i]]$hidbiases
      hidbias = repmat(hidbiases,numcases,1)
      #obtain hidden probabilities
      poshidprobs = 1/(1 + exp(-data*stack[[i]]$vishid - hidbias))
      #obtain hidden states
      hidStates = round(poshidprobs)
      data = hidStates
    }
    posteriorProbs = poshidprobs
  }
  
  else{ #mode equals 'stochastic'
    probs = zeros(nrow(data), nit)
    for (i in 1:nit){
      data = orgData
      for (j in 1:numLayers){
        hidbiases = stack[[j]]$hidbiases
        hidbias = repmat(hidbiases,numcases,1) 
        poshidprobs = 1/(1 + exp(-data*stack[[j]]$vishid - hidbias))
        hidStates = as.integer(poshidprobs > rand(dim(poshidprobs)))
        data = hidStates
      }
      probs[, i] = poshidprobs
    }
    posteriorProbs = mean(probs,2)
  }
  return(posteriotProbs)
}
