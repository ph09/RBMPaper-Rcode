rbm <- function(rbmInput){
  source('monitor.r')
  library(pracma)
  batchdata = rbmInput$data$batchdata
  
  decayLrAfter = rbmInput$decayLrAfter 
  epsilonw_0 = rbmInput$epsilonw       #Learning rate for weights 
  epsilonvb_0 = rbmInput$epsilonvb     #Learning rate for biases of visible units 
  epsilonhb_0 = rbmInput$epsilonhb     #Learning rate for biases of hidden units 
  initialmomentum = rbmInput$initialmomentum
  finalmomentum = rbmInput$finalmomentum
  lambda  =  rbmInput$weightPenalty   
  CD = rbmInput$CD
  maxEpoch = rbmInput$maxEpoch
  decayMomentumAfter = rbmInput$decayMomentumAfter
  numhid = rbmInput$numhid
  iIncreaseCD = rbmInput$iIncreaseCD
  
  numcases = dim(batchdata)[1]
  numdims = dim(batchdata)[2]
  numbatches = dim(batchdata)[3]
  
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
    batchposhidprobs <<- array(zeroData, c(numcases, numhid, numbatches))
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
      data = as.integer(as.logical(data > rand(numcases,numdims)))
      dim(data) = c(numcases, numdims)
      poshidprobs = 1/(1 + exp(-data%*%vishid - hidbias))    
      posprods = t(data)%*%poshidprobs
      poshidact   = sum(poshidprobs)
      posvisact = sum(data)
      ##### END OF POSITIVE PHASE #####
      
      poshidprobs_temp = poshidprobs
      
      ##### START NEGATIVE PHASE #####
  
      for (cditer in 1:CD){
        poshidstates = as.integer(as.logical(poshidprobs_temp > rand(numcases,numhid)))
        dim(poshidstates) = c(numcases, numhid)
        negdata = 1/(1 + exp(-poshidstates%*%t(vishid) - visbias))
        negdata = as.integer(as.logical(negdata > rand(numcases,numdims)))
        dim(negdata) = c(numcases, numdims)
        poshidprobs_temp = 1/(1 + exp(-negdata%*%vishid - hidbias))
      }
      
      neghidprobs = poshidprobs_temp
      negprods  = t(negdata)%*%neghidprobs
      neghidact = sum(neghidprobs)
      negvisact = sum(negdata)
      ##### END OF NEGATIVE PHASE #####
      
      err = sum(sum( (data-negdata)^2 ))
      errsum = err + errsum
      
      if (epoch>decayMomentumAfter){
        momentum = finalmomentum
      }
      
      else{
        momentum = initialmomentum
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
  return(rbmOutput)
}


 

 

