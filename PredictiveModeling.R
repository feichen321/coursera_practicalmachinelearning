#Building Predictive Models in R Using the caret Package

rm(list=ls())
library(caret)
install.packages("QSARdata")
library(QSARdata)
data(Mutagen)

set.seed(1)

inTrain <- createDataPartition(Mutagen_Outcome, p = 3/4, list=FALSE)

trainDescr <- Mutagen_Dragon[inTrain,]
testDescr <- Mutagen_Dragon[-inTrain,]
trainClass <- Mutagen_Outcome[inTrain]
testClass <- Mutagen_Outcome[-inTrain]

prop.table(table(Mutagen_Outcome))

prop.table(table(trainClass))

ncol(trainDescr)

nsv <- nearZeroVar(trainDescr, saveMetrics = TRUE)
nsv

var.sel <- rownames(nsv)[nsv$nzv == FALSE]
trainDescr <- trainDescr[, var.sel]
testDescr <- testDescr[, var.sel]
ncol(trainDescr)

descrCorr <- cor(trainDescr)

highCorr <- findCorrelation(descrCorr, 0.90)

testDescr <- testDescr[, -highCorr]

ncol(trainDescr)

xTrans <- preProcess(trainDescr)

trainDescr <- predict(xTrans, trainDescr)

testDescr  <- predict(xTrans,  testDescr)

bootControl <- trainControl(number = 200)

set.seed(2)

svmFit <- train(trainDescr, trainClass, method = "svmRadial", tuneLength =5, trControl = bootControl, scaled = FALSE)

svmFit$finalModel
library(gbm)
gbmGrid <- expand.grid(.interaction.depth = (1:5)*2, 
                       .n.trees = (1:10)*25, .shrinkage = .1,
                       .n.minobsinnode = 10)
set.seed(2)
gbmFit <- train(trainDescr, trainClass, method = "gbm", 
                trControl = bootControl, verbose = FALSE,
                bag.fraction = 0.5, tuneGrid = gbmGrid)
