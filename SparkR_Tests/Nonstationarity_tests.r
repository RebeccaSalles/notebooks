# Databricks notebook source
# MAGIC %md
# MAGIC ##Optimizing Moving Average Smoothers for Univariate Time Series Prediction
# MAGIC ###An Experiment using SparkR

# COMMAND ----------

# MAGIC %md ####Parallel and Distributed Computing - Masters Degree Course on ComputerScience - CEFET/RJ
# MAGIC __Professor:__ Eduardo Ogasawara; __Student:__ Rebecca Pontes Salles; __December, 08 of 2017__

# COMMAND ----------

# MAGIC %md ####1 - Introduction

# COMMAND ----------

# MAGIC %md Data preprocessing is a key step for mining and learning from data, and one of its main tasks is the transformation of data. This task is very important in the context of time series analysis, since most of the time series methods and models assume the property of stationarity, i.e., statistical properties do not change over time, which in practice is the exception and not the rule in most real datasets. There are several transformation methods designed to handle nonstationarity in time series, amomg them the moving average smoother (__MAS__). 
# MAGIC 
# MAGIC __MAS__ has been widely used especially in finance and econometrics, being useful for highlighting seasonality and long-term trends and patterns in a time series. MAS can detect the evolving behavior of a time series by minimizing random noise, and it can also be used for seasonal adjustment of a time series (Shumway and Stoffer 2010; Ogasawara et al.2010).
# MAGIC 
# MAGIC This work presents an experiment for applying an optimization function that derives "best" parameters of __MAS__ for each time series in a dataset. This function, named __fittestMAS__, was implemented and will soon be published in a new version (4.0.0) of the R-package __TSPred__. Since this function presents a somewhat high computational cost, especially when applied to large datasets, this work presents the distribution of its computation using a cluster (in __Databricks__) and __Spark__. The __SparkR__ is also used as a frontend for Spark. For the experiments, __fittestMAS__ is applied to benchmark datasets from time series prediction competitions. All datasets used are available in the __TSPred__ package.

# COMMAND ----------

# MAGIC %md ####2 - The fittestMAS function

# COMMAND ----------

# MAGIC %md The __fittestMAS__ function implements the __MAS__ method, and optimizes its main parameter (order of moving average) according to the best prediction results on a cross-validation dataset. In this case, the best prediction results are defined by the smaller produced value of MSE (mean squared error) prediction accuracy measure. The general idea implemented by the function is:
# MAGIC 1. if no prior knowledge of the moving average order is given, it generates order options to be evaluated;
# MAGIC 2. it runs __MAS__ of each order on the provided time series and performs predictions with each of these transformed series;
# MAGIC 3. generates MSE errors for the predictions using validation data;
# MAGIC 4. selects the moving average order which generated the smaller MSE error as optim;
# MAGIC 5. using the selected optim order, generates predictions and evaluates the prediction error of the original time series using a test dataset.

# COMMAND ----------

# MAGIC %md ####3 - Spark and SparkR

# COMMAND ----------

# MAGIC %md __Apache Spark__ is an engine for distributed data processing, having a data-sharing abstraction called “Resilient Distributed Datasets” (RDDs). RDDs are a collection of data distributed over several nodes of a cluster that can be referenced as a single (local) collection. It can capture a wide range of processing workloads that previously needed separate engines, including SQL, streaming, machine learning, and graph processing (Zaharia et al.). Spark's applications are easier to develop using a unified API and it can run diverse functions over the same data, often in memory. One of the main libraries of Spark is the __SparkSQL__, which provides a higher-level abstraction called __DataFrames__, which conceptually are RDDs full of records (tuples) with a known schema (Zaharia et al.).
# MAGIC 
# MAGIC The Spark architecture follows the following image:
# MAGIC 
# MAGIC ![Spark architecture][logo]
# MAGIC [logo]: http://blog.cloudera.com/wp-content/uploads/2015/09/pyspark.png "Spark architecture"
# MAGIC 
# MAGIC * The Driver node: creates a the SparkContext and RDDs and stages ip or sends off transformation and action functions.
# MAGIC * The Cluster Manager: allocates resources
# MAGIC 
# MAGIC Recently it was developed, __SparkR__ an R package that provides a frontend to Apache Spark and uses Spark’s distributed computation engine to enable large scale data analysis from the R shell (Venkataraman et al.). It addresses a limitation of __R__, which is usually limited by single threaded computation and that can only process data sets that fit in a single machine’s memory. It uses the same structures as __SparkSQL__, that is the __DataFrames__, and also has general SQL functions in its API. Starting with Spark 1.4.x, SparkR provides a distributed DataFrame implementation that supports operations like selection, filtering, aggregation etc. (similar to R data frames, dplyr) but on large datasets.
# MAGIC 
# MAGIC Since the version 2.0 of __SparkR__, it presents functions like dapply, gapply and spark.lapply that allow the use of user defined functions (UDF) to be applied to DataFrames.  To this experiment this functions will be the most appropriate.

# COMMAND ----------

# MAGIC %md ####4 - Spark in the Databricks Cluster

# COMMAND ----------

# MAGIC %md Databricks is a unified analytics platform that is optimized to the use of Spark, which is natively installed and configured. Furthermore, with the release of Spark 1.4.0, SparkR was inlined in Apache Spark. At that time Databricks released R Notebooks to be the first company that officially supports SparkR. To further facilitate usage of SparkR, Databricks R notebooks imported SparkR by default and provided a working sqlContext object.
# MAGIC 
# MAGIC Due to the beneficial environment provided by Databricks it was selected for this experiment. Databricks makes it ease to create, configure and use a cluster. In particular Databricks runs on Amazon Web Services (AWS) for cloud infrastructure. This fact provides the possibilities of increasing productivity and collaboration, build on a secure, trusted cloud and scale without limits. AWS clusters and infrastructure management and data science are simplified with a fully managed and secure platform powered by Databricks Runtime.
# MAGIC 
# MAGIC In order to begin our experiments using Databricks the following steps were addressed:
# MAGIC 1. an account in Databricks was created;
# MAGIC 2. an account in AWS was created;
# MAGIC 3. information of the AWS account was configured in Databricks.
# MAGIC 
# MAGIC Occurred problems:
# MAGIC * __Regarding step 1__: The free account option of Databricks (Databricks Community) 

# COMMAND ----------

# MAGIC %md ####5 - Experiments

# COMMAND ----------

# MAGIC %md #####5.1 - Installing and Loading Packages

# COMMAND ----------

# MAGIC %md ######Installing and Loading Packages on the Driver

# COMMAND ----------

library("SparkR")
install.packages("TSPred", repos = "http://cran.us.r-project.org")
library("TSPred")

# COMMAND ----------

# MAGIC %md ######Installing and Loading Packages on Workers

# COMMAND ----------

numWorkers <- 2
spark.lapply(1:numWorkers, function(x) {
  #Pacotes necessarios para funcoes em fittest_models_functions
  install.packages(c("TSPred","forecast"), repos = "http://cran.us.r-project.org")
  #Pacotes necessarios para funcoes em results_analysis_functions
  install.packages(c("ggplot2","Cairo"), repos = "http://cran.us.r-project.org")
})

# COMMAND ----------

# MAGIC %md #####5.2 - First Tests with SparkR Functions

# COMMAND ----------

# MAGIC %md ######No Parallelization Test

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

# MAGIC %md ######How Not to Do It: Parallelization with spark.lapply

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

# MAGIC %md ######How Not to Do It (The Sequel): Parallelization with gapplyCollect

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

# MAGIC %md ######Now we're talking: Parallelization with dapplyCollect

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

# MAGIC %md #####5.3 - Evaluating fittestMAS with and without Parallelization

# COMMAND ----------

# MAGIC %md ######Loading experiment datasets and functions

# COMMAND ----------

#exp_url <- url("https://github.com/RebeccaSalles/TSPred/blob/master/dev/12.17/exp/Exp_Base.RData")
#download.file(exp_url,"Exp_Base.RData",method="curl")
#load(exp_url)

install.packages("repmis", repos = "http://cran.us.r-project.org")
library(repmis)

source_data("https://github.com/RebeccaSalles/TSPred/blob/master/dev/12.17/exp/Exp_Base.RData?raw=True")

# COMMAND ----------

#Analysis of fittness and prediction of many series and many transforms
sparkr.MASExp <- function(timeseries,timeseries.test, numProc,
                          rank.by=c("MSE","NMSE","MAPE","sMAPE","MaxError","AIC","AICc","BIC","logLik")){
    
  n.ahead <- nrow(timeseries.test)

  dataset <- data.frame(t(rbind(timeseries,timeseries.test)))
  dataset <- cbind(nameTS=rownames(dataset),n.ahead=n.ahead,dataset)
  
  dataset.df <- createDataFrame(dataset)
  
  rdataset.df <- repartition(dataset.df, numPartitions = numProc)
  
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

times <- NULL
for(i in 1:10){
  # Clear all cached tables in the current session
  clearCache()
  t <- MASExp(CATS,CATS.cont, rank.by="MSE")$time
  times <- rbind(times,data.frame(user=t[1],system=t[2],elapsed=t[3]))
}

# COMMAND ----------

par(mfrow = c(1, 2))
boxplot(times[,c("elapsed")], width=10, height=4)
boxplot(sparkr.times[,c("elapsed")], width=10, height=4)
#boxplot(times[,c("system")], width=10, height=4)
#boxplot(times[,c("elapsed")], width=10, height=4)

# COMMAND ----------

numWorkers<-2
numWorkerCores<-2
sparkr.times <- NULL
for(i in 1:10){
  t <- sparkr.MASExp(CATS,CATS.cont, numProc=numWorkers*numWorkerCores, rank.by="MSE")$time
  sparkr.times <- rbind(sparkr.times,data.frame(user=t[1],system=t[2],elapsed=t[3]))
}

# COMMAND ----------

par(mfrow = c(1, 3))
boxplot(times[,c("user")], width=10, height=4)
boxplot(times[,c("system")], width=10, height=4)
boxplot(times[,c("elapsed")], width=10, height=4)

# COMMAND ----------

numWorkers<-2
numWorkerCores<-2
sparkr.times.proc <- NULL
for(numProc in 1:(numWorkers*numWorkerCores)){
  for(i in 1:10){
    t <- sparkr.MASExp(CATS,CATS.cont, numProc=numProc, rank.by="MSE")$time
    sparkr.times.proc <- rbind(sparkr.times.proc,data.frame(numProc=numProc,user=t[1],system=t[2],elapsed=t[3]))
  }
}