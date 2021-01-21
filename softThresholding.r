softThresholding <- function(v,t){
  res =  sign(v) * max(abs(v)-t,zeros(dim(v)))
  return(res)
}