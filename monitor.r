monitor <- function(monitorInput){
  
  output = list()
  
  vishid = monitorInput$vishid
  visbiases = monitorInput$visbiases
  hidbiases = monitorInput$hidbiases
  
  
  trainData = monitorInput$trainData
  tDdim = dim(trainData)
  validationData = monitorInput$validationData
  vDdim = dim(validationData)
  
  numTrainPts = nrow(trainData)
  numValPts = nrow(validationData)
  
  if(length(dim(trainData))>2){
    rtD = rand(dim(trainData)[1]*dim(trainData)[2]*dim(trainData)[3])
    dim(rtD) = c(dim(trainData)[1], dim(trainData)[2], dim(trainData)[3])
    trainData = as.integer(as.logical(trainData > rtD))
    dim(trainData) = tDdim
  }else{
    trainData = as.integer(as.logical(trainData > rand(dim(trainData))))
    dim(trainData) = tDdim
  }
  if(length(dim(validationData))>2){
    rvD = rand(dim(validationData)[1]*dim(validationData)[2]*dim(validationData)[3])
    dim(rvD) = c(dim(validationData)[1], dim(validationData)[2], dim(validationData)[3])
    validationData = as.integer(as.logical(validationData > rvD))
    dim(validationData) = vDdim
  }else{
    validationData = as.integer(as.logical(validationData > rand(dim(validationData))))
    dim(validationData) = vDdim
  }
  
  trainVisbias = repmat(visbiases,numTrainPts,1)
  validationVisbias = repmat(visbiases,numValPts,1)
  
  trainHidbias = repmat(1%*%hidbiases,numTrainPts,1) 
  validationHidbias = repmat(1%*%hidbiases,numValPts,1)
  
  ##### START OF POSITIVE PHASE #####
  trainPoshidprobs = 1/(1 + exp(-trainData%*%(1*vishid) - trainHidbias))  
  tPdim = dim(trainPoshidprobs)
  validationPoshidprobs = 1/(1 + exp(-validationData%*%(1*vishid) - validationHidbias))
  vPdim = dim(validationPoshidprobs)
  ##### END OF POSITIVE PHASE  #####
    
  ##### START NEGATIVE PHASE #####
  trainPoshidstates = as.integer(as.logical(trainPoshidprobs > rand(dim(trainPoshidprobs))))
  dim(trainPoshidstates) = tPdim
  validationPoshidstates = as.integer(as.logical(validationPoshidprobs > rand(dim(validationPoshidprobs))))
  dim(validationPoshidstates) = vPdim
  
  trainNegdata = 1/(1 + exp(-trainPoshidstates%*%t(vishid) - trainVisbias))
  tNdim = dim(trainNegdata)
  validationNegdata = 1/(1 + exp(-validationPoshidstates%*%t(vishid) - validationVisbias))
  vNdim = dim(validationNegdata)
  
  trainNegdata = as.integer(as.logical(trainNegdata > rand(dim(trainNegdata))))
  dim(trainNegdata) = tNdim
  validationNegdata = as.integer(as.logical(validationNegdata > rand(dim(validationNegdata))))
  dim(validationNegdata) = vNdim
  
  trainNeghidprobs = 1/(1 + exp(-trainNegdata%*%(1*vishid) - trainHidbias))
  
  ## reconstruction error
  output$recErrTraining = sum(sum((trainData-trainNegdata)^2))/numTrainPts
  output$recErrValidation = sum(sum((validationData-validationNegdata)^2))/ numValPts
  
  ## free energy
  trainX = trainData%*%(1*vishid) + trainHidbias
  validationX = validationData%*%(1*vishid) + validationHidbias
  
  output$freeEnergyTraining = -sum(trainData%*%t(visbiases) + sum(log(1+exp(trainX)),2))/numTrainPts
  output$freeEnergyValidation = -sum(validationData%*%t(visbiases) + sum(log(1+exp(validationX)),2)) / numValPts
  
  return(output)
}


