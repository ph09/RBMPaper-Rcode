rm(list=ls())
source('rbm.r')
source('readData.r')
source('obtainHiddenRep.r')
source('forward.r')
source('forwardPass.r')
source('sigmoid.r')
source('softThresholding.r')
graphics.off() 
##read data
library(R.matlab)
library(lattice)

datasetName = 'dataset1.mat' #path to dataset file.
#the dataset needs to be a .mat file, with a binary matrix f of size d x n 
#and (optionally, a binary label vector y). 
data = readData(datasetName)

## setup
rbmInput <- list()
rbmOutput <- list()
monitorInput <- list()
rbmInput$restart=1;

#the following are configurable hyperparameters for RBM

rbmInput$reg_type = 'l2'
rbmInput$weightPenalty = 1e-2 #\ell_2 weight penalty
weightPenaltyOrg = rbmInput$weightPenalty
rbmInput$epsilonw      = 5e-2 #5e-2 # Learning rate for weights 
rbmInput$epsilonvb     = 5e-2 #5e-2 # Learning rate for biases of visible units 
rbmInput$epsilonhb     = 5e-2 #5e-2 # Learning rate for biases of hidden units 
rbmInput$CD=10 #number of contrastive divergence iterations
rbmInput$initialmomentum  = 0
rbmInput$finalmomentum    = 0.9
rbmInput$maxEpoch = 150
rbmInput$decayLrAfter = 120
rbmInput$decayMomentumAfter = 90 #when to switch from initial to final momentum
rbmInput$iIncreaseCD = 0
# monitor free energy and likelihood change (on validation set) with time
rbmInput$iMonitor = 1


## train
sizes = list()
rbmInput$data = data
rbmInput$numhid = ncol(data$allDataTable)
stack = vector('list', 1)
layerCounter = 1
addLayers = 1
while (addLayers){
  # train RBM
  rbmInput$weightPenalty = weightPenaltyOrg
  rbmOutput = rbm(rbmInput)
  # collect params
  stack[[layerCounter]]$vishid = rbmOutput$vishid
  stack[[layerCounter]]$hidbiases = rbmOutput$hidbiases
  stack[[layerCounter]]$visbiases = rbmOutput$visbiases
  
  # SVD to determine number of hidden nodes
  tmp = svd (stack[[layerCounter]]$vishid)
  U = tmp$u
  D = diag(tmp$d)
  V = tmp$v
  numhid = min(which((cumsum(D)/sum(D))>0.95))
  message(sprintf ('need %1.0f hidden units\n', numhid))
  print('paused, press Enter key to continue')
  scan(quiet=TRUE) 
  
  # Re-train RBM
  sizes = c(sizes, numhid)
  rbmInput$numhid = numhid
  rbmInput$weightPenalty = 0 #rbmInput.weightPenalty/10;
  rbmOutput = rbm(rbmInput)
  # collect params
  stack[[layerCounter]]$vishid = rbmOutput$vishid
  stack[[layerCounter]]$hidbiases = rbmOutput$hidbiases
  stack[[layerCounter]]$visbiases = rbmOutput$visbiases
  v=c('weight matrix of RBM ', as.character(layerCounter))
  v = paste(v, collapse='')
  #levelplot(stack[[layerCounter]]$vishid, col.regions=terrain.colors(100), 
            #scales=list(x=list(0:((ncol(stack[[layerCounter]]$vishid)>5)+1):ncol(stack[[layerCounter]]$vishid)), 
            #y=list(0:((nrow(stack[[layerCounter]]$vishid)>5)+1):nrow(stack[[layerCounter]]$vishid))), 
            #main=v, xlab='hidden units', ylab='visible units')
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
print(sizes)
## obtain posterior probabilities
#deterministic
mode = 'deterministic'
posteriorProbsDet = forward (stack, data$allDataTable, mode)


# stochastic
mode = 'stochastic'
nit = 100
posteriorProbsStoch = forward(stack, data$allDataTable, mode, nit)

## predict labels
labels = data$labels

# deterministic mode:
predictedLabels = round(posteriorProbsDet)
# check if predictedLables need to be flipped
m = mean(predictedLabels == data$allDataTable[, 1])
if (m<0.5){
  predictedLabels = 1 - predictedLabels
}
acc = mean(labels==predictedLabels)
inds1 = labels[labels==1]
inds0 = labels[labels==0]
sensitivity = mean(predictedLabels[inds1])
specificity = 1-mean(predictedLabels[inds0])

balAcc_rbmDet = (sensitivity + specificity)/2
print('Deterministic mode:')
message(sprintf(1,'sensitivity: %0.3f%%\n',100*sensitivity))
message(sprintf(1,'specificity: %0.3f%%\n',100*specificity))
message(sprintf(1,'accuracy: %0.3f%%\n',100*acc))
message(sprintf(1,'balanced accuracy: %0.3f%%\n',100*balAcc_rbmDet))

#stochastic mode:
predictedLabels = round(posteriorProbsStoch)
#check if predictedLables need to be flipped
m = mean(predictedLabels == data$allDataTable[, 1])
if (m<0.5){
  predictedLabels = 1-predictedLabels
}
acc = mean(labels==predictedLabels)
inds1 = labels[labels==1]
inds0 = labels[labels==0]
sensitivity = mean(predictedLabels[inds1])
specificity = 1-mean(predictedLabels[inds0])

balAcc_rbmStoch = (sensitivity + specificity)/2
print('Stochastic mode:')
message(sprintf(1,'sensitivity: %0.3f%%\n',100*sensitivity))
message(sprintf(1,'specificity: %0.3f%%\n',100*specificity))
message(sprintf(1,'accuracy: %0.3f%%\n',100*acc))
message(sprintf(1,'balanced accuracy: %0.3f%%\n',100*balAcc_rbmStoch))