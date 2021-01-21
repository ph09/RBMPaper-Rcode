sigmoid <- function(x){
  s = 1/(1+exp(-x))
  return(s)
}