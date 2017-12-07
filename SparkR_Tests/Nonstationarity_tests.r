# Databricks notebook source
# MAGIC %md
# MAGIC ##Optimizing Moving Average Smoothers for Univariate Time Series Prediction
# MAGIC ###An Experiment using SparkR

# COMMAND ----------

# MAGIC %md ####1 - Introduction

# COMMAND ----------

# MAGIC %md Data preprocessing is a key step for mining and learning from data, and one of its main tasks is the transformation of data. This task is very important in the context of time series analysis, since most of the time series methods and models assume the property of stationarity, i.e., statistical properties do not change over time, which in practice is the exception and not the rule in most real datasets. There are several transformation methods designed to handle nonstationarity in time series, amomg them the moving average smoother (MAS). 
# MAGIC 
# MAGIC MAS has been widely used especially in finance and econometrics, being useful for highlighting seasonality and long-term trends and patterns in a time series. MAS can detect the evolving behavior of a time series by minimizing random noise, and it can also be used for seasonal adjustment of a time series (Shumway andStoffer 2010; Ogasawaraet al.2010).
# MAGIC 
# MAGIC This work presents an experiment for applying an optimization function that derives "best" parameters of MAS for each time series in a dataset. Since the computation performed by this function is The focus of this work is to generate practical knowledge regarding their advantages and limitations to the problem of time series prediction. A subset of the reviewed transformation methods are compared by means of an experimental evaluation in \proglang{R} using the Package \pkg{TSPred}, benchmark datasets from time series prediction competitions and other real macroeconomic datasets from Brazil.

# COMMAND ----------

# MAGIC %md ####1 - Installing and Loading Packages

# COMMAND ----------

# MAGIC %md #####Installing and Loading Packages on the Driver

# COMMAND ----------

library("SparkR")
install.packages("TSPred", repos = "http://cran.us.r-project.org")
library("TSPred")

# COMMAND ----------

# MAGIC %md #####Installing and Loading Packages on Workers

# COMMAND ----------

numWorkers <- 2
spark.lapply(1:numWorkers, function(x) {
  #Pacotes necessarios para funcoes em fittest_models_functions
  install.packages(c("TSPred","KFAS","car","forecast","wavelets","EMD","vars"), repos = "http://cran.us.r-project.org")
  #Pacotes necessarios para funcoes em stats_properties_functions
  install.packages(c("urca","tseries","stats","lmtest","car","nortest","plyr"), repos = "http://cran.us.r-project.org")
  #Pacotes necessarios para funcoes em results_analysis_functions
  install.packages(c("TSPred","KFAS","MuMIn","openair","ggplot2","corrplot","devtools","Cairo","plyr"), repos = "http://cran.us.r-project.org")
  devtools::install_github("vsimko/corrplot")
})

# COMMAND ----------

# MAGIC %md ####2 - First tests

# COMMAND ----------

# MAGIC %md #####No parallelization

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

# MAGIC %md #####How not to do it: spark.lapply

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

# MAGIC %md #####How not to do it 2: gapplyCollect

# COMMAND ----------

data(NN3.A,NN3.A.cont)

NN3 <- data.frame(t(rbind(NN3.A,NN3.A.cont)))
NN3 <- cbind(nameTS=rownames(NN3),NN3)

NN3.df <- createDataFrame(NN3)

execTime <- system.time(
  fPolyR <- gapplyCollect(NN3.df,"nameTS", function(key,x) {
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
        
        y <- data.frame(key, TSPred::fittestPolyR(tsi,tsi.cont, maxorder=5, se.fit=TRUE)$MSE)
        colnames(y) <- c("TS", "MSE")
        
        results <- rbind(results,y)        
      }
      results
    }
  })
)

print(execTime)

fPolyR


# COMMAND ----------

# MAGIC %md #####Now we're talking: dapplyCollect

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

# COMMAND ----------

# MAGIC %md ####3 - Evaluating Nonstationary Time Series Methods

# COMMAND ----------

# MAGIC %md #####Loading experiment datasets and functions

# COMMAND ----------

#exp_url <- url("https://github.com/RebeccaSalles/TSPred/blob/master/dev/12.17/exp/Exp_Base.RData")
#download.file(exp_url,"Exp_Base.RData",method="curl")
#load(exp_url)

install.packages("repmis", repos = "http://cran.us.r-project.org")
library(repmis)

source_data("https://github.com/RebeccaSalles/TSPred/blob/master/dev/12.17/exp/Exp_Base.RData?raw=True")

# COMMAND ----------

#Analysis of fittness and prediction of many series and many transforms
sparkr.MASExp <- function(timeseries,timeseries.test, numWorkers, numWorkerCores,
                          rank.by=c("MSE","NMSE","MAPE","sMAPE","MaxError","AIC","AICc","BIC","logLik")){
    
  n.ahead <- nrow(timeseries.test)

  dataset <- data.frame(t(rbind(timeseries,timeseries.test)))
  dataset <- cbind(nameTS=rownames(dataset),n.ahead=n.ahead,dataset)
  
  dataset.df <- createDataFrame(dataset)
  
  rdataset.df <- repartition(dataset.df, numPartitions = numWorkerCores*numWorkers)
  
  partitionExp <- function(timeseries){
    lenTS <- ncol(timeseries)
    nTS<- nrow(timeseries)
    
    results <- NULL
    for(i_ts in 1:nTS){
      nameTS <- timeseries[i_ts,1]
      n.ahead <- timeseries[i_ts,2]
      tsi <- na.omit(t(timeseries[i_ts,3:(lenTS-n.ahead)]))
      tsi.cont <- na.omit(t(timeseries[i_ts,(lenTS-n.ahead+1):lenTS]))
      
      print(paste("Computing results for time series ",nameTS,sep=""))
      #results of transfom experiments (rank and ranked.results)      
      transfResults <- fittestMAS(tsi,tsi.cont, model="arima", max.d=2, max.D=1, stationary=FALSE, rank.by=rank.by)
      results <- rbind(results,data.frame(TS = nameTS, Order = transfResults$order, MSE = transfResults$MSE))        
    }
    results
  }
  
  execTime <- system.time(
    transfResults <- dapplyCollect(rdataset.df, function(x) { partitionExp(x) })
  )
  
  return(list(time=execTime,results=transfResults))
}

# COMMAND ----------

sparkr.MASExp(CATS,CATS.cont, numWorkers=2, numWorkerCores=2, rank.by="MSE")

# COMMAND ----------

#Analysis of fittness and prediction of many series and many transforms
MASExp <- function(timeseries,timeseries.test,
                          rank.by=c("MSE","NMSE","MAPE","sMAPE","MaxError","AIC","AICc","BIC","logLik")){

  ts.lst <- list()
  for(col in names(timeseries)){
    ts.lst[[col]] <- list(timeseries[col],timeseries.test[col])
  }
  
  execTime <- system.time(
    transfResults <- lapply(ts.lst, function(ts){fittestMAS(ts[[1]],ts[[2]], model="arima", max.d=2, max.D=1, stationary=FALSE, rank.by=rank.by)})
  )
  
  results <- NULL
  for(ts in names(transfResults)){
    results <- rbind(results,data.frame(TS = ts, Order = transfResults[[ts]]$order, MSE = transfResults[[ts]]$MSE))
  }
  
  return(list(time=execTime,results=results))
}

# COMMAND ----------

MASExp(CATS,CATS.cont, rank.by="MSE")

# COMMAND ----------

times <- list()
for(i in 1:10){
  times[[i]] <- MASExp(CATS,CATS.cont, rank.by="MSE")$time
}

# COMMAND ----------

sparkr.times <- list()
for(i in 1:10){
  times[[i]] <- sparkr.MASExp(CATS,CATS.cont, numWorkers=2, numWorkerCores=2, rank.by="MSE")$time
}
