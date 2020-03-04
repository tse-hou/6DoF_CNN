rm(list=ls())
setwd("E:/silver/dqn/camera_parameters")
library("rjson")
library(dplyr)

datasets <- c("TechnicolorMuseum","ClassroomVideo", "TechnicolorHijack")

  for (n in datasets){
  # Give the input file name to the function.
  result <- fromJSON(file = paste0(n,".json"))
  
  names(result)
  
  Cols <- result$cameras%>% unlist() %>% names() %>% unique
  
  cam.para <- matrix(NA, ncol = length(Cols), nrow = length(result$cameras), 
                  dimnames = list(1:length(result$cameras),Cols))
  
  for (i in seq_along(result$cameras)) {
    cam.para[i, names( result$cameras[[i]] %>% unlist())] <- result$cameras[[i]] %>% unlist()
  }
  cam.para %>% View()
  
  write.csv(cam.para,file = paste0(n,".csv"),row.names = FALSE)
}