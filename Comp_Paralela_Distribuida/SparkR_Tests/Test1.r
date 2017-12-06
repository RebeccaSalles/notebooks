# Databricks notebook source
library("SparkR")
install.packages("TSPred")
library("TSPred")

# COMMAND ----------

data(CATS,CATS.cont)
execTime <- system.time(fPolyR <- TSPred::fittestPolyR(CATS[,1],CATS.cont[,1], maxorder=5, se.fit=TRUE))
print(execTime)
#predicted values and estimated standard errors
pred <- fPolyR$pred
#model information
fPolyR$rank[1,]
#View(fPolyR$rank)

# COMMAND ----------

#plotting the time series data
plot(c(CATS[,1],CATS.cont[,1]),type='o',lwd=2,xlim=c(960,1000),ylim=c(0,200),xlab="Time",ylab="PR")
#plotting predicted values
lines(ts(pred$fit,start=981),lwd=2,col='blue')
#plotting estimated standard errors (confidence interval)
lines(ts(pred$fit+pred$se.fit,start=981),lwd=2,col='light blue')
lines(ts(pred$fit-pred$se.fit,start=981),lwd=2,col='light blue')

# COMMAND ----------

data(CATS,CATS.cont)

CATS.lst <- list()
for(col in names(CATS)){
  CATS.lst[[col]] <- list(CATS[col],CATS.cont[col])
}

execTime <- system.time(
  fPolyR <- lapply(CATS.lst, function(ts){TSPred::fittestPolyR(ts[[1]],ts[[2]], maxorder=5, se.fit=TRUE)})
)
print(execTime)

# COMMAND ----------

fPolyR[["V1"]]

# COMMAND ----------

data(CATS,CATS.cont)

CATS.lst <- list()
for(col in names(CATS)){
  CATS.lst[[col]] <- list(CATS[col],CATS.cont[col])
}

execTime <- system.time(
  fPolyR <- spark.lapply(CATS.lst, function(ts){TSPred::fittestPolyR(ts[[1]],ts[[2]], maxorder=5, se.fit=TRUE)})
)
print(execTime)

# COMMAND ----------

fPolyR[["V1"]]

# COMMAND ----------

data(NN3.A,NN3.A.cont)

NN3.lst <- list()
for(col in names(NN3.A)){
  NN3.lst[[col]] <- list(NN3.A[col],NN3.A.cont[col])
}

execTime <- system.time(
  fPolyR <- lapply(NN3.lst, function(ts){TSPred::fittestPolyR(ts[[1]],ts[[2]], maxorder=5, se.fit=TRUE)})
)
print(execTime)

# COMMAND ----------

data(NN3.A,NN3.A.cont)

fPolyR <- list()
execTime <- system.time(
  for(ts in names(NN3.A)){
    fPolyR[[ts]] <- TSPred::fittestPolyR(NN3.A[[ts]],NN3.A.cont[[ts]], maxorder=5, se.fit=TRUE)
  }
)
print(execTime)

# COMMAND ----------

data(NN3.A,NN3.A.cont)

NN3.lst <- list()
for(col in names(NN3.A)){
  NN3.lst[[col]] <- list(NN3.A[col],NN3.A.cont[col])
}

execTime <- system.time(
  fPolyR <- spark.lapply(NN3.lst, function(ts){TSPred::fittestPolyR(ts[[1]],ts[[2]], maxorder=5, se.fit=TRUE)})
)
print(execTime)