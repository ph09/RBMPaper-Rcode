forwardPass <- function(data, vishid, hidbiases){
  
  numhid = ncol(vishid)
  numcases = dim(data)[1]
  numdims = dim(data)[2]
  numbatches = dim(data)[3]
  if(length(dim(data)) == 2){
    numbatches = 1
  }
  out = array(0, c(numcases, numhid, numbatches))
  
  hidbias = repmat(hidbiases,numcases,1)
  if(numbatches>1){
    for (i in 1:numbatches){
      ##### START POSITIVE PHASE #####
      batch = data[, , i]
      poshidprobs = 1/(1 + exp(-batch%*%vishid - hidbias))
      out[, , i] = round(poshidprobs)
      out[, , i] = as.integer(poshidprobs>rand(dim(poshidprobs)))
    }
  }
  return(out)
}