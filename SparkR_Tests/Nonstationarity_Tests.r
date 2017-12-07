# Databricks notebook source
library("SparkR")
install.packages("TSPred", repos = "http://cran.us.r-project.org")
library("TSPred")
numWorkers <- 2
spark.lapply(1:numWorkers, function(x) {
  install.packages("TSPred", repos = "http://cran.us.r-project.org")
  library("SparkR")
  library("TSPred")
})


# COMMAND ----------

data(NN3.A,NN3.A.cont)

NN3.lst <- list()
for(col in names(NN3.A)){
  NN3.lst[[col]] <- list(NN3.A[col],NN3.A.cont[col])
}

execTime <- system.time(
  fPolyR <- lapply(NN3.lst, function(ts){TSPred::fittestPolyR(ts[[1]],ts[[2]], maxorder=5, se.fit=TRUE)$MSE})
)
print(execTime)

fPolyR

# COMMAND ----------

data(NN3.A,NN3.A.cont)

NN3.lst <- list()
for(col in names(NN3.A)){
  NN3.lst[[col]] <- list(NN3.A[col],NN3.A.cont[col])
}

execTime <- system.time(
  fPolyR <- spark.lapply(NN3.lst, function(ts){
    require("TSPred")
    TSPred::fittestPolyR(ts[[1]],ts[[2]], maxorder=5, se.fit=TRUE)
  })
)
print(execTime)

# COMMAND ----------

data(NN3.A,NN3.A.cont)

NN3 <- data.frame(t(rbind(NN3.A,NN3.A.cont)))
NN3 <- cbind(nameTS=rownames(NN3),NN3)

NN3.df <- createDataFrame(NN3)

numWorkerCores <- 2
numWorkers <- 2
rNN3.df <- repartition(NN3.df, numPartitions = numWorkerCores*numWorkers)

execTime <- system.time(
  fPolyR <- dapplyCollect(rNN3.df, function(x) {
    require("TSPred")
    timeseries <- x
    lenTS <- ncol(x)
    nTS<- nrow(x)
    if(nTS >=1 & lenTS >=1){
      results <- NULL
      for(i_ts in 1:nTS){
        nameTS <- timeseries[i_ts,1]
        tsi <- na.omit(t(timeseries[i_ts,2:(lenTS-18)]))
        tsi.cont <- na.omit(t(timeseries[i_ts,(lenTS-17):lenTS]))

        results <- rbind(results,data.frame(TS = nameTS, MSE = TSPred::fittestPolyR(tsi,tsi.cont, maxorder=5, se.fit=TRUE)$MSE))        
      }
      results
    }
  })
)

print(execTime)

fPolyR