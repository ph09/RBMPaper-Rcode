rm(list=ls())
graphics.off() 

monitor <- function(monitorInput){
  
  output = list()
  
  vishid = monitorInput$vishid
  visbiases = monitorInput$visbiases
  hidbiases = monitorInput$hidbiases
  
  
  trainData = monitorInput$trainData
  validationData = monitorInput$validationData
  
  numTrainPts = nrow(trainData)
  numValPts = nrow(validationData)
  
  trainData = trainData > rand(dim(trainData)) 
  validationData = validationData > rand(dim(validationData))
  
  trainVisbias = repmat(visbiases,numTrainPts,1)
  validationVisbias = repmat(visbiases,numValPts,1)
  
  trainHidbias = repmat(1%*%hidbiases,numTrainPts,1) 
  validationHidbias = repmat(1%*%hidbiases,numValPts,1)
  
  ##### START OF POSITIVE PHASE #####
  trainPoshidprobs = 1/(1 + exp(-trainData%*%(1*vishid) - trainHidbias))  
  validationPoshidprobs = 1/(1 + exp(-validationData%*%(1*vishid) - validationHidbias))  
  ##### END OF POSITIVE PHASE  #####
  
  ##### START NEGATIVE PHASE #####
  trainPoshidstates = trainPoshidprobs > rand(dim(trainPoshidprobs))
  validationPoshidstates = validationPoshidprobs > rand(dim(validationPoshidprobs))
  
  trainNegdata = 1/(1 + exp(-trainPoshidstates%*%t(vishid) - trainVisbias))
  validationNegdata = 1/(1 + exp(-validationPoshidstates%*%t(vishid) - validationVisbias))
  
  trainNegdata = trainNegdata > rand(dim(trainNegdata))
  validationNegdata = validationNegdata > rand(dim(validationNegdata))
  
  trainNeghidprobs = 1/(1 + exp(-trainNegdata%*%(1*vishid) - trainHidbias))
  
  ## reconstruction error
  output$recErrTraining = sum(sum((trainData-trainNegdata)^2))/numTrainPts
  output$recErrValidation = sum(sum((validationData-validationNegdata)^2))/ numValPts
  
  ## free energy
  trainX = trainData%*%(1*vishid) + trainHidbias
  validationX = validationData%*%(1*vishid) + validationHidbias
  
  output$freeEnergyTraining = -sum(trainData%*%t(visbiases) + sum(log(1+exp(trainX)),2))/numTrainPts
  output$freeEnergyValidation = -sum(validationData%*%t(visbiases) + sum(log(1+exp(validationX)),2)) / numValPts
  
  output = output
}

obtainHiddenRep <- function(rbmInput, rbmOutput){
  source('forwardPass.r')
  
  vishid = rbmOutput$vishid
  hidbiases = rbmOutput$hidbiases
  
  rbmInput$data$batchdata = forwardPass(rbmInput$data$batchdata, vishid, hidbiases)
  rbmInput$data$trainData = forwardPass(rbmInput$data$trainData, vishid, hidbiases)
  rbmInput$data$validationData = forwardPass(rbmInput$data$validationData, vishid, hidbiases)
  rbmInput$data$trainDataTable = forwardPass(rbmInput$data$trainDataTable, vishid, hidbiases)
  rbmInput$data$validationDataTable = forwardPass(rbmInput$data$validationDataTable, vishid, hidbiases)
  rbmInput$data$allDataTable = forwardPass(rbmInput$data$allDataTable, vishid, hidbiases)
  rbmInput$data = rbmInput$data
  
}

computeHiddenRepresentation <- function(vishid,hidbiases, data){
  hidbias = repmat(hidbiases, nrow(data),1)
  poshidprobs = 1/(1 + exp(-data*(vishid) - hidbias))
  hidRep = round(poshidprobs)
}

forwardPass <- function(data, vishid, hidbiases){
  
  numhid = ncol(vishid)
  numcases = dim(data)[1]
  numdims = dim(data)[2]
  numbatches = dim(data)[3]
  if(length(dim(data)) == 2){
    numbatches = 1
  }
  out=array(0, c(numcases, numhid, numbatches))
  
  hidbias = repmat(hidbiases,numcases,1)
  if(numbatches>1){
    for (i in 1:numbatches){
      ##### START POSITIVE PHASE #####
      batch = data[, , i]
      poshidprobs = 1/(1 + exp(-batch%*%vishid - hidbias))
      out[, , i] = round(poshidprobs)
      out[, , i] = poshidprobs>rand(dim(poshidprobs));
    }
  }
  out = out
}

rbm <- function(rbmInput){
  source('monitor.r')
  monitorInput = list()
  library(pracma)
  rbmOutput = list()
  batchdata <- rbmInput$data$batchdata
  
  decayLrAfter = rbmInput$decayLrAfter 
  epsilonw_0 = rbmInput$epsilonw       #Learning rate for weights 
  epsilonvb_0 = rbmInput$epsilonvb     #Learning rate for biases of visible units 
  epsilonhb_0 = rbmInput$epsilonhb     #Learning rate for biases of hidden units 
  initialmomentum = rbmInput$initialmomentum
  finalmomentum = rbmInput$finalmomentum
  lambda  =  rbmInput$weightPenalty   
  CD=rbmInput$CD
  maxEpoch = rbmInput$maxEpoch
  decayMomentumAfter = rbmInput$decayMomentumAfter
  numhid = rbmInput$numhid
  iIncreaseCD = rbmInput$iIncreaseCD
  
  numcases=dim(batchdata)[1]
  numdims=dim(batchdata)[2]
  numbatches=dim(batchdata)[3]
  
  restart = rbmInput$restart
  
  if (restart ==1){
    restart=0
    epoch=1
    
    #Initializing symmetric weights and biases. 
    vishid     = 0.01*matrix( rnorm(numdims*numhid,mean=0,sd=1), numdims, numhid) 
    hidbiases  = zeros(1,numhid)
    visbiases  = zeros(1,numdims)
    
    poshidprobs = zeros(numcases,numhid)
    neghidprobs = zeros(numcases,numhid)
    posprods    = zeros(numdims,numhid)
    negprods    = zeros(numdims,numhid)
    vishidinc  = zeros(numdims,numhid)
    hidbiasinc = zeros(1,numhid)
    visbiasinc = zeros(1,numdims)
    zeroData <- rep(0, numcases*numhid*numbatches)
    batchposhidprobs=array(zeroData, c(numcases, numhid, numbatches))
  }
  
  if (rbmInput$iMonitor){
    freeEnergyTraining = zeros(maxEpoch,1)
    freeEnergyValidation = zeros(maxEpoch,1)
    recErrTraining = zeros(maxEpoch,1)
    recErrValidation = zeros(maxEpoch,1)
  }
  
  for (epoch in epoch:maxEpoch){
    
    message(sprintf('epoch %d\r',epoch))
    if(iIncreaseCD){
      CD = ceiling(epoch/10)
    }
    if (epoch > decayLrAfter){
      factor = 10^-ceiling((epoch - decayLrAfter)/10)
      epsilonw = epsilonw_0*factor
      epsilonvb = epsilonvb_0*factor
      epsilonhb = epsilonhb_0*factor
    } else {
      epsilonw = epsilonw_0
      epsilonvb = epsilonvb_0
      epsilonhb = epsilonhb_0
    }
    errsum = 0
    
    for (batch in 1:numbatches){
      
      message(sprintf('epoch %d batch %d\r',epoch,batch))
      
      visbias = repmat(visbiases,numcases,1)
      hidbias = repmat(hidbiases,numcases,1) 
      
      ##### START POSITIVE PHASE #####
      data = batchdata[, , batch]
      data = data > rand(numcases,numdims)
      poshidprobs = 1/(1 + exp(-data%*%vishid - hidbias))    
      posprods = t(data)%*%poshidprobs
      poshidact   = sum(poshidprobs)
      posvisact = sum(data)
      ##### END OF POSITIVE PHASE #####
      
      poshidprobs_temp = poshidprobs
      
      ##### START NEGATIVE PHASE #####
      
      for (cditer in 1:CD){
        poshidstates = poshidprobs_temp > rand(numcases,numhid)
        negdata = 1/(1 + exp(-poshidstates%*%t(vishid) - visbias))
        negdata = negdata > rand(numcases,numdims)
        poshidprobs_temp = 1/(1 + exp(-negdata%*%vishid - hidbias))
      }
      
      neghidprobs = poshidprobs_temp
      negprods  = t(negdata)%*%neghidprobs
      neghidact = sum(neghidprobs)
      negvisact = sum(negdata)
      ##### END OF NEGATIVE PHASE #####
      
      err= sum(sum( (data-negdata)^2 ))
      errsum = err + errsum
      
      if (epoch>decayMomentumAfter){
        momentum=finalmomentum
      }
      
      else{
        momentum=initialmomentum
      }
      
      if (strcmp(rbmInput$reg_type, 'l2')){  #l_2 regularization
        vishidinc = momentum*vishidinc + epsilonw*( (posprods-negprods)/numcases - lambda*vishid)
        vishid = vishid + vishidinc
      }
      
      else{  #l_1 regularization
        vishidinc = momentum*vishidinc + epsilonw*((posprods-negprods)/numcases)
        vishid = softThresholding(vishid + vishidinc, lambda%*%epsilonw);
      }
      
      ##### UPDATE WEIGHTS AND BIASES #####
      visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact)
      hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact)
      
      visbiases = visbiases + visbiasinc
      hidbiases = hidbiases + hidbiasinc
      ##### END OF UPDATES #####
    }
    
    message(sprintf('epoch %4i error %6.1f  \n', epoch, errsum))
    if (rbmInput$iMonitor){
      monitorInput$trainData = rbmInput$data$trainDataTable
      monitorInput$validationData = rbmInput$data$validationDataTable
      monitorInput$visbiases = visbiases
      monitorInput$hidbiases = hidbiases
      monitorInput$vishid = vishid
      
      monitorOutput = monitor(monitorInput)
      freeEnergyTraining[epoch] = monitorOutput$freeEnergyTraining
      freeEnergyValidation[epoch] = monitorOutput$freeEnergyValidation
      recErrTraining[epoch] = monitorOutput$recErrTraining
      recErrValidation[epoch] = monitorOutput$recErrValidation
    }
    
    #RBM output
    rbmOutput$vishid = vishid
    rbmOutput$visbiases = visbiases
    rbmOutput$hidbiases = hidbiases
    rbmOutput$batchposhidprobs = batchposhidprobs
    rbmOutput$poshidstates = poshidstates
    
    if (rbmInput$iMonitor){
      rbmOutput$freeEnergyTraining = freeEnergyTraining
      rbmOutput$freeEnergyValidation = freeEnergyValidation
      rbmOutput$recErrTraining = recErrTraining
      rbmOutput$recErrValidation = recErrValidation
      plot(1:maxEpoch, freeEnergyTraining)
      title (main='Free Energies (Training)', xlab='Epoch', ylab='Avg Free Energy on validation data')
      plot(1:maxEpoch, freeEnergyValidation)
      title (main='Free Energies (Validation)', xlab='Epoch', ylab='Avg Free Energy on validation data')
      plot(1:maxEpoch, recErrTraining)
      title ('Reconstruction Error (Training)', xlab='Epoch', ylab='Avg Reconstruction Energy on validation data')
      plot(1:maxEpoch, recErrValidation)
      title ('Reconstruction Error (Validation)', xlab='Epoch', ylab='Avg Reconstruction Energy on validation data')
    }
  }
  rbmOutput = rbmOutput
}

forward <- function(stack, data, mode, nit){
  
  numLayers = ncol(stack)
  numcases = nrow(data)
  orgData = data
  
  if (strcmp(mode, 'deterministic')){
    for (i in 1:numLayers){
      hidbiases = stack$i$hidbiases
      hidbias = repmat(hidbiases,numcases,1)
      #obtain hidden probabilities
      poshidprobs = 1/(1 + exp(-data*stack$i$vishid - hidbias))
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
        hidbiases = stack$j$hidbiases
        hidbias = repmat(hidbiases,numcases,1) 
        poshidprobs = 1/(1 + exp(-data*stack$j$vishid - hidbias))
        hidStates = poshidprobs > rand(dim(poshidprobs))
        data = hidStates
      }
      probs[, i] = poshidprobs
    }
    posteriorProbs = mean(probs,2)
  }
}

readData <- function(dataSetName){
  data = readMat('dataset1.mat')
  #f: d x n matrix of -1,1
  #y: labels -1 or 1
  
  numClassifiers = dim(data$f)[1]
  numObs = dim(data$f)[2]
  
  p = sample(numObs)
  
  batchSize = 100
  data$batchSize = batchSize
  numObs = floor(numObs/batchSize)*batchSize
  
  orgData=t(data$f)
  orgData[orgData==-1]=0
  orgData = orgData[1:numObs, 1:numClassifiers]
  labels = t(data$y[1:numObs])
  labels[labels==-1]=0
  data$labels = labels
  
  
  # arrange in minibatches for RBM training
  numBatches = numObs/batchSize
  numValBatches = floor(numBatches/10)
  
  numTrainBatches = numBatches - numValBatches
  allData = t(orgData)
  dim(allData) =  c(numClassifiers, batchSize, numBatches)
  allData = aperm(allData, c(2,1,3))
  trainDataTable = orgData[1:(numTrainBatches*batchSize), ,drop=FALSE]
  trainData = trainDataTable
  trainData = t(trainData)
  dim(trainData) = c(numClassifiers, batchSize, numTrainBatches)
  trainData = aperm(trainData, c(2, 1, 3))
  validationDataTable = orgData[((numTrainBatches*batchSize)+1):((numTrainBatches + numValBatches)*batchSize),]
  validationData = t(validationDataTable)
  dim(validationData) = c(numClassifiers, batchSize, numValBatches)
  validationData = aperm(validationData, c(2, 1, 3))
  
  data$trainData = trainData
  data$validationData = validationData
  data$numTrainBatches = numTrainBatches
  data$numValBatches = numValBatches
  data$trainDataTable = trainDataTable
  data$validationDataTable = validationDataTable
  data$trainingLabels = labels[1:(numTrainBatches*batchSize)]
  data$validationLabels = labels[((numTrainBatches*batchSize)+1):((numTrainBatches + numValBatches)*batchSize)]
  data$allDataTable = orgData
  data$labels = labels
  data$batchdata = data$trainData
  
  data = data
}

sigmoid <- function(x){
  s = 1/(1+exp(-x))
}

softThresholding <- function(v,t){
  res =  sign(v) * max(abs(v)-t,zeros(dim(v)))
}

##read data
library(R.matlab)
library(lattice)

datasetName = 'dataset1.mat'; # path to dataset file.
#the dataset needs to be a .mat file, with a binary matrix f of size d x n 
#and (optionally, a binary label vector y). 
data = readData(datasetName)

## setup
rbmInput = list()
output = list()
rbmOutput = list()
monitorInput = list()
monitorInput$test = 1
rbmInput$restart=1

# the following are configurable hyperparameters for RBM

rbmInput$reg_type = 'l2'
rbmInput$weightPenalty = 1e-2 #\ell_2 weight penalty
weightPenaltyOrg = rbmInput$weightPenalty
rbmInput$epsilonw      = 5e-2 #5e-2 % Learning rate for weights 
rbmInput$epsilonvb     = 5e-2 #5e-2 % Learning rate for biases of visible units 
rbmInput$epsilonhb    = 5e-2 #5e-2 % Learning rate for biases of hidden units 
rbmInput$CD=10   # number of contrastive divergence iterations
rbmInput$initialmomentum  = 0
rbmInput$finalmomentum   = 0.9
rbmInput$maxEpoch = 150
rbmInput$decayLrAfter = 120
rbmInput$decayMomentumAfter = 90 # when to switch from initial to final momentum
rbmInput$iIncreaseCD = 0
#monitor free energy and likelihood change (on validation set) with time
rbmInput$iMonitor = 1


## train
sizes = list()
rbmInput$data <- data
rbmInput$data$trainData = data$trainData
rbmInput$data$validationData = data$validationData
rbmInput$data$trainDataTable = data$trainDataTable
rbmInput$data$validationDataTable = data$validationDataTable
rbmInput$data$allDataTable = data$allDataTable
rbmInput$data$batchdata = data$batchdata
rbmInput$numhid = ncol(data$allDataTable)
rbmInput = rbmInput
stack = list()
layerCounter = 1
addLayers = 1
while (addLayers){
  # train RBM
  rbmInput$weightPenalty = weightPenaltyOrg
  rbmOutput <- rbm(rbmInput)
  # collect params
  stack$layerCounter = list()
  stack$layerCounter$vishid = rbmOutput$vishid
  stack$layerCounter$hidbiases = rbmOutput$hidbiases
  stack$layerCounter$visbiases = rbmOutput$visbiases
  
  # SVD to determine number of hidden nodes
  tmp  = svd (stack$layerCounter$vishid)
  U = tmp$u
  D = unlist(tmp$d)
  V = tmp$v
  
  numhid = min(which(cumsum(diag(D))/sum(diag(D))>0.95))
  numdims = 0
  fprintf ('need %1.0f hidden units\n', numhid)
  print('paused, press Enter key to continue')
  scan(quiet=TRUE) 
  
  # Re-train RBM
  sizes = list(sizes, numhid)
  rbmInput$numhid = numhid
  rbmInput$weightPenalty = 0   #rbmInput$weightPenalty]/10
  rbmOutput = rbm(rbmInput)
  # collect params
  stack$layerCounter$vishid = rbmOutput$vishid
  stack$layerCounter$hidbiases = rbmOutput$hidbiases
  stack$layerCounter$visbiases = rbmOutput$visbiases
  v=c('weight matrix of RBM ', as.character(layerCounter))
  v = paste(v, collapse='')
  levelplot(stack$layerCounter$vishid, col.regions=terrain.colors(100), 
            #scales=list(x=list(0:((ncol(stack$layerCounter$vishid)>5)+1):ncol(stack$layerCounter$vishid)), 
            #y=list(0:((nrow(stack$layerCounter$vishid)>5)+1):nrow(stack$layerCounter$vishid))), 
            main=v, xlab='hidden units', ylab='visible units')
  # setup for next RBM
  rbmInput$data = obtainHiddenRep(rbmInput, rbmOutput)
  
  # stopping criterion
  if (numhid ==1){
    addLayers = 0
  }
  layerCounter = layerCounter + 1
}

numLayers = ncol(stack)
message(sprintf('trained a deep net with %1.0f layers, of sizes:\n', numLayers))
message(sprintf(sizes))
##obtain posterior probabilities
# deterministic
mode = 'deterministic'
posteriorProbsDet = forward(stack, data.allDataTable, mode)

#stochastic
mode = 'stochastic'
nit = 100
posteriorProbsStoch = forward(stack, data.allDataTable, mode, nit)

## predict labels
labels = t(data$labels)

#deterministic mode:
predictedLabels = round(posteriorProbsDet)
#check if predictedLables need to be flipped
m = mean(predictedLabels == data$allDataTable[,1])
if (m<0.5){
  predictedLabels = 1-predictedLabels
}
acc = mean(labels==predictedLabels)
inds1 = labels==1
inds0 = labels==0
sensitivity = mean(predictedLabels(inds1))
specificity = 1-mean(predictedLabels(inds0))

balAcc_rbmDet = (sensitivity + specificity)/2
print('Deterministic mode:')
print(1,'sensitivity: %0.3f%%\n',100*sensitivity)
print(1,'specificity: %0.3f%%\n',100*specificity)
print(1,'accuracy: %0.3f%%\n',100*acc)
print(1,'balanced accuracy: %0.3f%%\n',100*balAcc_rbmDet)

# stochastic mode:
predictedLabels = round(posteriorProbsStoch)
# check if predictedLables need to be flipped
m = mean(predictedLabels == data$allDataTable[,1])
if (m<0.5){
  predictedLabels = 1-predictedLabels
}
acc = mean(labels==predictedLabels)
inds1 = labels==1
inds0 = labels==0
sensitivity = mean(predictedLabels(inds1))
specificity = 1-mean(predictedLabels(inds0))

balAcc_rbmStoch = (sensitivity + specificity)/2
print('Stochastic mode:')
print(1,'sensitivity: %0.3f%%\n',100*sensitivity)
print(1,'specificity: %0.3f%%\n',100*specificity)
print(1,'accuracy: %0.3f%%\n',100*acc)
print(1,'balanced accuracy: %0.3f%%\n',100*balAcc_rbmStoch)