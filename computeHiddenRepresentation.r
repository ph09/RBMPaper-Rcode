computeHiddenRepresentation <- function(vishid,hidbiases, data){
  hidbias = repmat(hidbiases, nrow(data),1)
  poshidprobs = 1/(1 + exp(-data*(vishid) - hidbias))
  hidRep = round(poshidprobs)
  return(hidRep)
}