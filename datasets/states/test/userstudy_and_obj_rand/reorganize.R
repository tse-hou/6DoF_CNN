library(dplyr)
setwd("E:/silver/dqn/draft_dataset/random/")

db.fns <- c('TechnicolorHijack', 'TechnicolorMuseum','ClassroomVideo')
# db.fns <- c('UserStudy_1','UserStudy_2','UserStudy_3')
for(db.fn in db.fns){
  
  raw.db <- read.csv(paste0('raw_',db.fn,".csv"))
  
  db <- raw.db
  
  # raw.db %>% View()
  
  n_passes = 3
  
  views.list <- raw.db$X.viewperpasses %>%
    as.character() %>%
    strsplit(split=",") %>%
    lapply(function(x){tmp = as.numeric(x); max(tmp)}) %>%
    unlist()
  
  db$max_views <- views.list
  
  # db %>% View()
  
  theo.time <- raw.db %>% filter(X.passes ==1) %>% select(Dataset,Frame,Synthesized.View,X.viewperpasses,Execution.time..sec.)
  names(theo.time) <- c( "Dataset", "Frame", "Synthesized.View",'max_views', 'theo_time')
  
  theo.time$max_views <- theo.time$max_views %>% as.character() %>% as.numeric() 
  
  db <- left_join(db,theo.time)
  
  db$CEL <- (db$WS.PSNR-20)/db$theo_time
  
  db.view <- db %>% select(Dataset, Frame,Synthesized.View,X.passes, X.viewperpasses,WS.PSNR, CEL,theo_time)
  
  #db.view %>% View()
  
  # get passes and views
  passes_gen <- function(x, passes=n_passes){
    x <- as.numeric(x);#print(x)
    temp <- rep(0, times = passes);#print(temp)
    for(i in 1:length(x)){
      temp[i]=x[i]
    };#print(temp)
    temp
  }
  
  pv <- db.view$X.viewperpasses %>% as.character() %>% strsplit(split = ',') %>% #head(20) %>%
    lapply(passes_gen) %>%
    unlist() %>% matrix(ncol=n_passes,byrow = T) %>% as.data.frame() 
  
  colnames(pv) <- c("p1","p2","p3")
  
  db.view <- cbind(db.view,pv) #%>% View()
  db.view$X.viewperpasses <- NULL
  
  # get the sorted sequece of source views 
  sort.seq <-raw.db$Sorted_selected_view %>% as.character() %>% strsplit(split = ',') %>%
    lapply(function(x) as.numeric(x)) %>% unlist() %>% matrix(ncol=7,byrow = T) %>% as.data.frame() 
  
  s.views <-raw.db$Source.view %>% as.character() %>% strsplit(split = ',') %>%
    lapply(function(x) as.numeric(x)) %>% unlist() %>% matrix(ncol=7,byrow = T) %>% as.data.frame()
  s.views <- s.views[!duplicated(s.views),]
  try(if(s.views %>% nrow() != 1) stop("Mltiple source views combination!"))

  sorted.views <- apply(sort.seq+1, 1, function(x) s.views[x])  %>% unlist() %>% matrix(ncol=7,byrow = T) %>% as.data.frame()
  
  db.view <- cbind(db.view,sorted.views) #%>% View()
  
  # write file
  write.csv(db.view ,file=paste0(db.fn,".csv"))
}


