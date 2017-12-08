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

# MAGIC %md Data preprocessing is a key step for mining and learning from data, and one of its main tasks is the transformation of data. This task is very important in the context of time series analysis, since most of the time series methods and models assume the property of stationarity, i.e., statistical properties do not change over time, which in practice is the exception and not the rule in most real datasets. There are several transformation methods designed to handle nonstationarity in time series, among them the moving average smoother (__MAS__). 
# MAGIC 
# MAGIC __MAS__ has been widely used especially in finance and econometrics, being useful for highlighting seasonality and long-term trends and patterns in a time series. MAS can detect the evolving behavior of a time series by minimizing random noise, and it can also be used for seasonal adjustment of a time series (Shumway and Stoffer 2010; Ogasawara et al.2010).
# MAGIC 
# MAGIC This work presents an experiment for applying an optimization function that derives "best" parameters of __MAS__ for each time series in a dataset. This function, named __fittestMAS__, was implemented and will soon be published in a new version (4.0.0) of the R-package __TSPred__ (Salles and Ogasawara 2017). Since this function presents a somewhat high computational cost, especially when applied to large datasets, this work presents the distribution of its computation using a cluster (in __Databricks__) and __Spark__. The __SparkR__ is also used as a frontend for Spark. For the experiments, __fittestMAS__ is applied to benchmark datasets from time series prediction competitions. All datasets used are available in the __TSPred__ package.

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

# MAGIC %md ####3 - Spark

# COMMAND ----------

# MAGIC %md __Apache Spark__ is an engine for distributed data processing, having a data-sharing abstraction called “Resilient Distributed Datasets” (RDDs). RDDs are a collection of data distributed over several nodes of a cluster that can be referenced as a single (local) collection. It can capture a wide range of processing workloads that previously needed separate engines, including SQL, streaming, machine learning, and graph processing (Zaharia et al. 2016). Spark's applications are easier to develop using a unified API and it can run diverse functions over the same data, often in memory. One of the main libraries of Spark is the __SparkSQL__, which provides a higher-level abstraction called __DataFrames__, which conceptually are RDDs full of records (tuples) with a known schema (Zaharia et al. 2016).
# MAGIC 
# MAGIC The Spark architecture follows the following image:
# MAGIC 
# MAGIC ![Spark architecture](https://0x0fff.com/wp-content/uploads/2015/03/Spark-Architecture-Official.png "Spark architecture")
# MAGIC 
# MAGIC * The Driver node: creates a the SparkContext and RDDs and stages ip or sends off transformation and action functions.
# MAGIC * The Cluster Manager: allocates resources accross cluster and manages scheduling (this role is adopted by Databricks. See next section.).
# MAGIC * The Worker nodes: actually run the application tasks, return results to Driver, and provide in-memory storage for cached RDDs.

# COMMAND ----------

# MAGIC %md ####4 - SparkR

# COMMAND ----------

# MAGIC %md Recently it was developed, __SparkR__ an R package that provides a frontend to Apache Spark and uses Spark’s distributed computation engine to enable large scale data analysis from the R shell (Venkataraman et al.). It addresses a limitation of __R__, which is usually limited by single threaded computation and that can only process data sets that fit in a single machine’s memory. It uses the same structures as __SparkSQL__, that is the __DataFrames__, and also has general SQL functions in its API. Starting with Spark 1.4.x, SparkR provides a distributed DataFrame implementation that supports operations like selection, filtering, aggregation etc. (similar to R data frames, dplyr) but on large datasets.
# MAGIC 
# MAGIC Since the version 2.0 of __SparkR__, it presents functions like dapply, gapply and spark.lapply that allow the use of user defined functions (UDF) to be applied to DataFrames (Falaki 2017).  To this experiment this functions will be the most appropriate as the objective is precisely evaluate the previously implemented R function __fittestMAS__.
# MAGIC 
# MAGIC ######__spark.lapply:__ runs a function over a list of elements. Usage: spark.lapply().
# MAGIC 
# MAGIC General algorithm:
# MAGIC 
# MAGIC For each element of a list:
# MAGIC 1. Sends the function to an R worker.
# MAGIC 2. Executes the function.
# MAGIC 3. Returns the results of all worker as a list to R driver.
# MAGIC 
# MAGIC spark.apply() control flow (source (Falaki 2017)):
# MAGIC 
# MAGIC ![spark apply](https://image.slidesharecdn.com/parallelizingexistingrpackages-170210000051/95/parallelizing-existing-r-packages-with-sparkr-9-638.jpg?cb=1486684983 "spark.apply control")
# MAGIC 
# MAGIC ######__dapply:__ applies a function over each partition of a SparkDataframe. Usage: dapply() or dapplyCollet().
# MAGIC 
# MAGIC General algorithm:
# MAGIC 
# MAGIC For each partition of a SparkDataframe:
# MAGIC 1. Collects each partition as an R data.frame.
# MAGIC 2. Sends the R function to the R worker.
# MAGIC 3. Executes the function.
# MAGIC 
# MAGIC dapply(sparkDF,func,schema): combines results as DataFrame with provided schema.
# MAGIC 
# MAGIC dapplyCollect(sparkDF,func): combines results as R data.frame.
# MAGIC 
# MAGIC dapplyCollect() control flow (source (Falaki 2017)):
# MAGIC 
# MAGIC ![dapply](https://image.slidesharecdn.com/parallelizingexistingrpackages-170210000051/95/parallelizing-existing-r-packages-with-sparkr-12-638.jpg?cb=1486684983 "dapply control")
# MAGIC 
# MAGIC ######__gapply:__ applies a function to each group within a SparkDataframe. Usage: gapply() or gapplyCollet().
# MAGIC 
# MAGIC General algorithm:
# MAGIC 
# MAGIC Groups a SparkDataframe on one or more columns:
# MAGIC 1. Collects each group as an R data.frame.
# MAGIC 2. Sends the R function to the R worker.
# MAGIC 3. Executes the function.
# MAGIC 
# MAGIC gapply(sparkDF,cols,func,schema): combines results as DataFrame with provided schema.
# MAGIC 
# MAGIC gapplyCollect(sparkDF,cols,func): combines results as R data.frame.
# MAGIC 
# MAGIC gapplyCollect() control flow (source (Falaki 2017)):
# MAGIC 
# MAGIC ![gapply](https://image.slidesharecdn.com/parallelizingexistingrpackages-170210000051/95/parallelizing-existing-r-packages-with-sparkr-14-638.jpg?cb=1486684983 "gapply control")

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
# MAGIC __Occurred problems__:
# MAGIC * __Regarding step 1__: The free account option of Databricks (Databricks Community) did not provide access to any Worker nodes, which are required in the Spark architecture. In order to overcome this limitation, a Full-platform 14-Day Free Trial account was created. However, this Free Trial account excluded AWS charges which are summed according to demand. 
# MAGIC * __Regarding step 2__: The Trial account required information of an AWS account. For that reason a free AWS account was created. This account includes charges of on demand nodes that are allocated. The verification and confirmation of this account was problematic and a call to the 24h support service of Amazon was necessary to resolve the issues. The American attendent was very helpful.
# MAGIC * __Regarding step 3__: The information of the AWS account and its configuration for its use together with Databricks were rather counter intuitive, and several tutorials had to be followed in order to make everything work properly and in order to actually being able to create a cluster.

# COMMAND ----------

# MAGIC %md ####5 - Experiments

# COMMAND ----------

# MAGIC %md ######Cluster Settings

# COMMAND ----------

# MAGIC %md For the experiments, a __cluster__ with the following settings was created:
# MAGIC * __Databricks Runtime Version__: 3.3 (includes Apache Spark 2.2.0, Scala 2.11)
# MAGIC * __Driver Node__: Amazon m4.large (8.0 GB Memory, 2 Cores)
# MAGIC * __Worker Nodes__: Amazon m4.large (8.0 GB Memory, 2 Cores)
# MAGIC * __Number of Worker Nodes__: 2

# COMMAND ----------

# MAGIC %md ######Datasets

# COMMAND ----------

# MAGIC %md In this experiment we use:
# MAGIC * __The CATS Competition dataset__: presents an artificial time series with 5,000 observations, among which 100 are unknown. The unknown observations are grouped into five non-consecutive gaps of 20 successive values. The prediction of each gap may be considered a different problem, and each subset of the series followed by a gap may be considered a different time series to be modeled. In this context, we consider the CATS dataset as being composed of 5 time series of 980 observations.
# MAGIC * __The NN3 Competition dataset__: present 111 time series having from 50 to 126 monthly observations drawn from homogeneous population of real empirical business time series.

# COMMAND ----------

# MAGIC %md #####5.1 - Installing and Loading Packages

# COMMAND ----------

# MAGIC %md ######Installing and Loading Packages on the Driver

# COMMAND ----------

# MAGIC %md First step to the experiments is installing and loading the required packages into the Driver node. For this experiment, that includes the SparkR and the package TSPred, that contains our UDF functions. We also install the ggplot package for graphic generation.

# COMMAND ----------

library("SparkR")
install.packages("TSPred", repos = "http://cran.us.r-project.org")
install.packages(c("ggplot2"), repos = "http://cran.us.r-project.org")
library("TSPred")
library("ggplot2")

# COMMAND ----------

# MAGIC %md ######Installing and Loading Packages on Workers

# COMMAND ----------

# MAGIC %md The SparkR closure capture does not include packages. So we need to import packages on each Worker inside a function. Therefore, a second step to the experiments is installing and loading the required packages into the Worker nodes. For this experiment, that includes the package TSPred and forecast, that contains our UDF functions, and is necessary to these functions, respectively.
# MAGIC 
# MAGIC For installing the packages on each node we can use the spark.lapply function that sends and executes the function provided in each worker node.

# COMMAND ----------

#Number of workers
numWorkers <- 2

#Executes the function on each worker node
spark.lapply(1:numWorkers, function(x) {
  #Pacotes necessarios para experimento com fittestMAS
  install.packages(c("TSPred","forecast"), repos = "http://cran.us.r-project.org")  
})

# COMMAND ----------

# MAGIC %md #####5.2 - Initial Tests with SparkR Functions

# COMMAND ----------

# MAGIC %md For the initial tests using the SparkR functions, the __NN3__ dataset was used together with a simpler function already published in the TSPred package, namely the __fittestPolyR__. Analogously to the __fittestMAS__, __fittestPolyR__ optimizes a polynomial regression to a given time series based on the smallest MSE prediction error. This choices were made so as to perform first tests with the SparkR functions without spending too much time, but still aproaching a large dataset. 

# COMMAND ----------

# MAGIC %md ######Local Single-threaded Computation Test

# COMMAND ----------

# MAGIC %md As a first test, we use the Driver to perform a non-distributed computation and measure the user, system and elapsed execution times.

# COMMAND ----------

#load the dataset from TSPred
data(NN3.A,NN3.A.cont)

#make a list of NN3 training and testing sets
NN3.lst <- list()
for(col in names(NN3.A)){
  NN3.lst[[col]] <- list(NN3.A[col],NN3.A.cont[col])
}

#apply the fittestPolyR function to the dataset using the R base lapply function
execTime <- system.time(
  fPolyR <- lapply(NN3.lst, function(ts){TSPred::fittestPolyR(ts[[1]],ts[[2]], maxorder=5, se.fit=TRUE)$MSE})
)

print("Execution times of the non-distributed and non-parallelized computation:")
print(execTime)

# COMMAND ----------

# MAGIC %md ######How Not to Do It: Data Distribution with spark.lapply

# COMMAND ----------

# MAGIC %md Now testing the UDF friendly API of SparkR, we start by performing the same test using the spark.lapply function which distributes the computation across each core of each Worker nodes. As can be seen in the results, this version of the computation actually takes much longer than the non-distributed one. This fact is due to the non-apropriate use of the spark.lapply function, since we are packing data in the closure and using spark.lapply() to try and distribute the large dataset. This forces the function to use the network to send the data to each Worker node through the function calls. This use of the network for data transfering is in fact very expensive and causes communication overhead, culminating in the increase in time for the computation to finish.

# COMMAND ----------

#apply the fittestPolyR function to the dataset using the SparkR base spark.lapply function
execTime <- system.time(
  fPolyR <- spark.lapply(NN3.lst, function(ts){
    #loading the package in the Worker nodes
    require("TSPred")
    TSPred::fittestPolyR(ts[[1]],ts[[2]], maxorder=5, se.fit=TRUE)
  })
)

print("Execution times of the spark.lapply distributed computation:")
print(execTime)

# COMMAND ----------

# MAGIC %md ######How Not to Do It (The Sequel): Data Distribution with gapplyCollect

# COMMAND ----------

# MAGIC %md Now performing the same test using the gapply function which groups the data based on a given column and then distributes the computation across each core of each Worker nodes. As can be seen in the results, this version of the computation takes the longest. This fact is probably due to the additional tasks of grouping and combining the results of the computation of each group. Furthermore, since our only possible grouping column regards the name of each time series, the grouping task does not reduce or facilitate computations which are done in each 111 time series groups of 1 time series.

# COMMAND ----------

#creating single data.frame with training and testing sets
NN3 <- data.frame(t(rbind(NN3.A,NN3.A.cont)))
NN3 <- cbind(nameTS=rownames(NN3),NN3)

#creating the SparkDataFrame
NN3.df <- createDataFrame(NN3)

#using SparkR's gapply to group the data into different time series and distribute the computations across Worker nodes
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

print("Execution times of the gapplyCollect distributed computation:")
print(execTime)

# COMMAND ----------

# MAGIC %md ######Now we're talking!: Data Distribution with dapplyCollect

# COMMAND ----------

# MAGIC %md Now performing the same test using the dapply function which performs computation for each data partition distributed across each core of each Worker nodes. For that the data is previously partitioned using the __repartition__ function of SparkR, responsable for returning a new SparkDataFrame that has the same partitions as the number of processors in the cluster (that is, the number of Worker nodes times the number of cores in each of these nodes). As can be seen in the results, this version of the computation is the fastest and it actually takes only 55% of the time of the non-distributed version.
# MAGIC 
# MAGIC Summary of comparison with the non-distributed version:
# MAGIC * __Elapsed time:__ 45% faster
# MAGIC * __Speedup:__ 1.8
# MAGIC 
# MAGIC The partitioning of the data corresponding to the number of processors in the cluster seem to contribute to the overall computational costs. Moreover, since the data partitions are located in-memory in the Worker nodes, no time series need to be passed in network communications and the data distributed computation is optimized.
# MAGIC 
# MAGIC Henceforth, this is the method adopted for evaluating the __fittestMAS__ function.

# COMMAND ----------

#creating the SparkDataFrame
NN3.df <- createDataFrame(NN3)

numWorkerCores <- 2
numWorkers <- 2
#Partitioning the NN3.df and returning a new SparkDataFrame that has exactly numPartitions
#numPartitions is equal to numWorkerCores*numWorkers, i.e., the number of different processors in the Spark cluster
rNN3.df <- repartition(NN3.df, numPartitions = numWorkerCores*numWorkers)

#using SparkR's dapply to apply the computations to each partition of (111/numPartitions) time series across Worker nodes
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

print("Execution times of the dapplyCollect distributed computation:")
print(execTime)

# COMMAND ----------

# MAGIC %md #####5.3 - Evaluating fittestMAS with and without Data Distribution

# COMMAND ----------

# MAGIC %md ######Loading experiment datasets and functions

# COMMAND ----------

# MAGIC %md For using the more complex __fittestMAS__ function, first we need to load the RData file containing its implementation, since it is not yet published in the TSPred package. The file also contains the smaller CATS dataset which is henceforth used in this experiment, in order to reduce the overall time of this experiment.

# COMMAND ----------

install.packages("repmis", repos = "http://cran.us.r-project.org")
library(repmis)

#loading RData file
source_data("https://github.com/RebeccaSalles/TSPred/blob/master/dev/12.17/exp/Exp_Base.RData?raw=True")

# COMMAND ----------

# MAGIC %md ######Experiment functions definition

# COMMAND ----------

# MAGIC %md The following functions __sparkr.MASExp__ and __MASExp__ perform the experiments of evaluating the __fittestMAS__ function using the given time series dataset with and without SparkR's data distribution (using dapplyCollect()), respectively. 

# COMMAND ----------

#Performs the experiment using SparkR's data distributed computation (with dapplyCollect())
sparkr.MASExp <- function(timeseries,timeseries.test, numProc){
    
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
      transfResults <- fittestMAS(tsi,tsi.cont, model="arima", max.d=2, max.D=1, stationary=FALSE, rank.by="MSE")
      results <- rbind(results,data.frame(timeSeries = nameTS, optimMASOrder = transfResults$order, MSE = transfResults$MSE))        
    }
    results
  }
  
  execTime <- system.time(
    transfResults <- dapplyCollect(rdataset.df, function(x) { partitionExp(x) })
  )
  
  return(list(time=execTime,results=transfResults))
}

# COMMAND ----------

#Performs the experiment with local and single-threaded computations (using simple lapply of R base)
MASExp <- function(timeseries,timeseries.test){

  ts.lst <- list()
  for(col in names(timeseries)){
    ts.lst[[col]] <- list(timeseries[col],timeseries.test[col])
  }
  
  execTime <- system.time(
    transfResults <- lapply(ts.lst, function(ts){fittestMAS(ts[[1]],ts[[2]], model="arima", max.d=2, max.D=1, stationary=FALSE, rank.by="MSE")})
  )
  
  results <- NULL
  for(ts in names(transfResults)){
    results <- rbind(results,data.frame(timeSeries = ts, optimMASOrder = transfResults[[ts]]$order, MSE = transfResults[[ts]]$MSE))
  }
  
  return(list(time=execTime,results=results))
}

# COMMAND ----------

# MAGIC %md ######Results of Local and Single-Threaded Computations

# COMMAND ----------

MASExp(CATS,CATS.cont)

# COMMAND ----------

# MAGIC %md ######Results of Data-Distributed Computations using SparkR

# COMMAND ----------

# MAGIC %md Summary of comparison with the non-distributed (local and single-threaded) version:
# MAGIC * __Elapsed time:__ 58% faster
# MAGIC * __Speedup:__ 2.42

# COMMAND ----------

numWorkers<-2
numWorkerCores<-2

sparkr.MASExp(CATS,CATS.cont, numProc = numWorkers * numWorkerCores)

# COMMAND ----------

# MAGIC %md ######Results over 10 Runs

# COMMAND ----------

times <- NULL
for(i in 1:10){
  # Clear all cached tables in the current session
  clearCache()
  t <- MASExp(CATS,CATS.cont)$time
  times <- rbind(times,data.frame(user=t[1],system=t[2],elapsed=t[3]))
}

# COMMAND ----------

numWorkers<-2
numWorkerCores<-2
sparkr.times <- NULL
for(i in 1:10){
  t <- sparkr.MASExp(CATS,CATS.cont, numProc=numWorkers*numWorkerCores)$time
  sparkr.times <- rbind(sparkr.times,data.frame(user=t[1],system=t[2],elapsed=t[3]))
}

# COMMAND ----------

ExpTimes <- rbind(data.frame(comp="Local",times),data.frame(comp="SparkR",sparkr.times))

ggplot(data = ExpTimes, aes(x=factor(1), y=elapsed, fill=comp)) +
  geom_boxplot() +  facet_wrap(~comp,scales="free",ncol=2)+
  theme(axis.text.x=element_blank(),
        axis.title.x=element_blank(), 
        axis.title.y = element_text("Elapsed time"))+
  ylab("Elapsed time")+
  labs(fill="Computation")


# COMMAND ----------

# MAGIC %md ######Mean SparkR Results over 10 Runs and Increasing Number of Processors

# COMMAND ----------

numWorkers <- 2
numWorkerCores <- 2
sparkr.times.proc <- NULL
for(numProc in 1:(numWorkers*numWorkerCores)){
  for(i in 1:10){
    t <- sparkr.MASExp(CATS,CATS.cont, numProc=numProc)$time
    sparkr.times.proc <- rbind(sparkr.times.proc,data.frame(numProc=numProc,user=t[1],system=t[2],elapsed=t[3]))
  }
}

# COMMAND ----------

#Aggregating results and taking mean elapsed times
sparkr.times.proc.agg <- cbind(aggregate(sparkr.times.proc$elapsed, by=list(sparkr.times.proc$numProc), FUN=mean),
                               aggregate(sparkr.times.proc$system, by=list(sparkr.times.proc$numProc), FUN=mean)[2],
                               aggregate(sparkr.times.proc$user, by=list(sparkr.times.proc$numProc), FUN=mean)[2])
names(sparkr.times.proc.agg) <- c("numProc","elapsed","system","user")
sparkr.times.proc.agg <- head(sparkr.times.proc.agg,4)

# COMMAND ----------

ggplot2::ggplot(data=sparkr.times.proc.agg, aes(x=numProc,y=elapsed))+
geom_line() +
geom_point()
#plot(sparkr.times.proc.agg$numProc,sparkr.times.proc.agg$elapsed,type="l",xlab="Number of Processors",ylab="Mean Elapsed Time",col="blue",lwd="2")

# COMMAND ----------

ggplot2::ggplot(data=sparkr.times.proc.agg, aes(x=numProc,y=elapsed[1]/elapsed))+
geom_line() +
geom_point()
plot(sparkr.times.proc.agg$numProc,sparkr.times.proc.agg$elapsed[1]/sparkr.times.proc.agg$elapsed,type="l",xlab="Number of Processors",ylab="Speedup",col="blue",lwd="2")

# COMMAND ----------

# MAGIC %md #####Reference

# COMMAND ----------

# MAGIC %md Falaki, H. Databricks. Parallelizing Existing R Packages with SparkR. Software. Presented at Spark Summit East, 2017. Retrieved from https://pt.slideshare.net/databricks/parallelizing-existing-r-packages-with-sparkr/5
# MAGIC 
# MAGIC Ogasawara E, Martinez LC, Oliveira Dd, Zimbr˜ao G, Pappa GL, Mattoso M (2010). “Adaptive
# MAGIC Normalization: A novel data normalization approach for non-stationary time series.” In
# MAGIC The 2010 International Joint Conference on Neural Networks (IJCNN), pp. 1–8. doi:
# MAGIC 10.1109/IJCNN.2010.5596746.
# MAGIC 
# MAGIC Salles RP, Ogasawara E (2017). TSPred: Functions for Benchmarking Time Series Prediction
# MAGIC Prediction. R package version 3.0.2, URL https://CRAN.R-project.org/package=
# MAGIC TSPred
# MAGIC 
# MAGIC Shumway RH, Stoffer DS (2010). Time Series Analysis and Its Applications: With R Examples.
# MAGIC 3rd edition. Springer, New York. ISBN 978-1-4419-7864-6.
# MAGIC 
# MAGIC Venkataraman, S., Yang, Z., Liu, D., Liang, E., Falaki, H., Meng, X., ... & Zaharia, M. (2016, June). Sparkr: Scaling r programs with spark. In Proceedings of the 2016 International Conference on Management of Data (pp. 1099-1104). ACM.
# MAGIC 
# MAGIC Zaharia, M., Xin, R. S., Wendell, P., Das, T., Armbrust, M., Dave, A., ... & Ghodsi, A. (2016). Apache Spark: A unified engine for big data processing. Communications of the ACM, 59(11), 56-65.