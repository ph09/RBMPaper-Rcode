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
  
  orgData = t(data$f)
  orgData[orgData==-1] = 0
  orgData = orgData[1:numObs, 1:numClassifiers]
  labels = t(data$y[1:numObs])
  labels[labels==-1] = 0
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
  data$traininglabels = labels[1:(numTrainBatches*batchSize)]
  data$validationlabels = labels[((numTrainBatches*batchSize)+1):((numTrainBatches + numValBatches)*batchSize)]
  data$allDataTable = orgData
  data$labels = labels
  data$batchdata = data$trainData
  
  data = data
}

