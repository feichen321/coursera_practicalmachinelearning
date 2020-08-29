install.packages("kernlab")
library(kernlab)
data(spam)
set.seed(333)
smallSpam <- spam[sample(dim(spam)[1], size =10),]
spamLabel <- (smallSpam$type == "spam")*1 + 1
plot(smallSpam$capitalAve, col = spamLabel)


rule1 <- function(x){
  prediction <- rep(NA, length(x))
  prediction[x > 2.7] <- "spam"
  prediction[x < 2.4] <- "nonspam"
  prediction[(x >= 2.40 & x <= 2.45)] <- "spam"
  prediction[(x > 2.45 & x <= 2.70)] <- "nonspam"
  return(prediction)
}

table(rule1(smallSpam$capitalAve), smallSpam$type)

table(rule1(spam$capitalAve), spam$type)

sum(rule1(spam$capitalAve) == spam$type)

rule2 <- function(x){
  prediction <- rep(NA, length(x))
  prediction[x > 2.8] <- "spam"
  prediction[x <= 2.8] <- "nonspam"
  return(prediction)
}

table(rule2(smallSpam$capitalAve), smallSpam$type)
table(rule2(spam$capitalAve), spam$type)

sum(rule2(spam$capitalAve) == spam$type)


#Week2
install.packages("caret")
library(caret); library(kernlab); data(spam)

inTrain <- createDataPartition(y = spam$type, p = 0.75, list = FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
dim(training)
install.packages("e1071")
set.seed(32343)
modelFit <- train(type ~ . , data = training, method = "glm")
modelFit$finalModel

predictions <- predict(modelFit, newdata = testing)
predictions

confusionMatrix(predictions, testing$type)

#Data Slicing
inTrain <- createDataPartition(y = spam$type, p = 0.75, list = FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
dim(training)

set.seed(32323)
folds <- createFolds(y = spam$type, k = 10, list = TRUE, returnTrain = TRUE)
sapply(folds, length)
#resampling
folds <- createResample(y = spam$type, times = 10, list = TRUE)
sapply(folds, length)

#Time Slices
tme <- 1:1000
folds <- createTimeSlices(y = tme, initialWindow = 20, horizon = 10)
names(folds)
folds$train[[1]]
folds$test[[1]]

## Training options
library(caret)
args(trainControl)

install.packages("ISLR")
library(ISLR)
library(ggplot2)
library(caret)
data(Wage)
summary(Wage)
inTrain <- createDataPartition(y=Wage$wage, p=0.7, list=FALSE)
training <- Wage[inTrain, ]
testing <- Wage[-inTrain,]
dim(training)
dim(testing)
featurePlot(x = training[,c("age", "education", "jobclass")], y=training$wage, plot="pairs")
qplot(age, wage, data=training)
qq <- qplot(age, wage, colour = jobclass, data=training)
qq + geom_smooth(method = "lm", formula = y ~ x)
install.packages("Hmisc")
library(Hmisc)
cutWage <- cut2(training$wage, g =3)
table(cutWage)
p1 <- qplot(cutWage, age, data=training, fill=cutWage, geom=c("boxplot"))
p1
library(gridExtra)
p2 <- qplot(cutWage, age, data=training, fill=cutWage, geom=c("boxplot", "jitter"))
grid.arrange(p1, p2, ncol=2)

t1 <- table(cutWage, training$jobclass)
t1
prop.table(t1, 1)

qplot(wage, colour = education, data = training, geom="density")

library(caret)
library(kernlab)
data(spam)
inTrain <- createDataPartition(y=spam$type, p=0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
hist(training$capitalAve, main="", xlab = "ave.capital run legnth")
mean(training$capitalAve)
sd(training$capitalAve)
#Standardizing
trainCapAve <- training$capitalAve
trainCapAveS <- (trainCapAve - mean(trainCapAve))/sd(trainCapAve)
mean(trainCapAveS)
sd(trainCapAveS)
#Standardizing-test set
testCapAve <- testing$capitalAve
testCapAveS <- (testCapAve - mean(trainCapAve))/sd(trainCapAve)
mean(testCapAveS)
sd(testCapAveS)

preObj <- preProcess(training[,-58], method = c("center", "scale"))
trainCapAves <- predict(preObj, training[,-58])$capitalAve
mean(trainCapAves)

testCapAveS <- predict(preObj, testing[,-58])$capitalAve
mean(testCapAveS)
sd(testCapAveS)

set.seed(32343)
modFit <- train(type ~ ., data= training, preProcess = c("center", "scale"), method = "glm")
modFit

#Box-Cox transfer
preObj <- preProcess(training[,-58], method=c("BoxCox"))
trainCapAveS <- predict(preObj, training[,-58])$capitalAve
par(mfrow=c(1,2))
hist(trainCapAveS)
qqnorm(trainCapAveS)
hist(training$capitalAve)

#Imputing data
set.seed(13343)
#Make some values NA
training$capAve <- training$capitalAve
selectNA <- rbinom(dim(training)[1], size = 1, prob = 0.05) == 1
training$capAve[selectNA] <- NA

#Impute and standardize
install.packages("RANN")
library(RANN)
preObj <- preProcess(training[,-58], method = "knnImpute")
capAve <- predict(preObj, training[,-58])$capAve

#standardize true values
capAveTruth <- training$capitalAve
capAveTruth <- (capAveTruth - mean(capAveTruth))/sd(capAveTruth)

quantile(capAve - capAveTruth)
quantile((capAve - capAveTruth)[selectNA])
quantile((capAve - capAveTruth)[!selectNA])

#Covariate creation

#Level 1: From raw data to covariate, depends on type of application
## the more knowledge of the system you have the better the job you will do
### Science is key
### Erro on overcreation of features
### In some applications automated feature creation is possible/necessary



#Level 2: Transforming tidy covariates:
## more necessary for regression svms than classification trees
## should be done only on the training set
## the best approach is through exploratory analysis
## new covariates should be added to data frames
library(kernlab)
data(spam)
spam$capitalAveSq <- spam$capitalAve^2

library(ISLR)
library(caret)
data(Wage)
inTrain <- createDataPartition(y=Wage$wage, p=0.7, list=FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]
table(training$jobclass)
dummies <- dummyVars(wage ~ jobclass, data=training)
head(predict(dummies, newdata=training))

nsv <- nearZeroVar(training, saveMetrics = TRUE)
nsv

library(splines)
bsBasis <- bs(training$age, df=3)
bsBasis

lm1 <- lm(wage ~ bsBasis, data=training)
plot(training$age, training$wage, pch=19, cex = 0.5)
points(training$age, predict(lm1, newdata = training), col="red", pch=19, cex=0.5)

predict(bsBasis, age=testing$age)

#PCA: we might not need every predictor, reduced number of predictor, reduced noise
library(caret)
library(kernlab)
data(spam)
inTrain <- createDataPartition(y=spam$type, p =0.75, list=FALSE)

training <- spam[inTrain,]
testing <- spam[-inTrain,]

M <- abs(cor(training[,-58]))
diag(M) <- 0
which(M > 0.8, arr.ind = T)
names(spam)[c(34,32)]
plot(spam[,34], spam[,32])

# We could rotate the plot

X <- 0.71*training$num415 + 0.71*training$num857
Y <- 0.71*training$num415 - 0.71*training$num857
plot(X, Y)

## Find a new set of multivariable variables that are uncorrelated and explain as much variance as possible

###SVD X = UDV^T

###PCA

smallSpam <- spam[,c(34,32)]
prComp <- prcomp(smallSpam)
plot(prComp$x[,1], prComp$x[,2])
prComp$rotation

###PCA on SPAM data
typeColor <- ((spam$type == "spam")*1 + 1)
prComp <- prcomp(log10(spam[, -58]+1))
plot(prComp$x[,1], prComp$x[,2], col=typeColor, xlab = "PC1", ylab="PC2")

preProc <- preProcess(log10(spam[,-58]+1), method="pca", pcaComp =2)
spamPC <- predict(preProc, log10(spam[,-58]+1))
plot(spamPC[,1], spamPC[,2], col=typeColor)

preProc <- preProcess(log10(training[,-58]+1), method="pca", pcaComp=2)
trainPC <- predict(preProc, log10(training[,-58]+1))
trainPC$type <- training$type
modFit <- train(type ~ ., method = "glm", data=trainPC)

testPC <- predict(preProc, log10(testing[,-58]+1))
#testPC$type <- testing$type
confusionMatrix(testing$type, predict(modFit, testPC))

#Alternative way to do it
modFit <- train(type ~., method="glm", preProcess="pca", data=training)
confusionMatrix(testing$type, predict(modFit, testing))
#most useful for linear-type models
#Watch out for outliers (transform first)

library(caret)
data(faithful)
set.seed(333)
inTrain <- createDataPartition(y=faithful$waiting, p = 0.5, list = FALSE)
trainFaith <- faithful[inTrain,]
testFaith <- faithful[-inTrain,]
head(trainFaith)
plot(trainFaith$waiting, trainFaith$eruptions, pch = 19, col="blue", xlab = "waiting", ylab = "Duration")
lm1 <- lm(eruptions ~ waiting, data = trainFaith)
summary(lm1)
newdata <- data.frame(waiting = 80)
predict(lm1, newdata)

par(mfrow = c(1,2))
plot(trainFaith$waiting, trainFaith$eruptions, pch = 19, col = "blue", xlab = "waiting", ylab = "Duration")
lines(trainFaith$waiting, predict(lm1), lwd=3)

plot(testFaith$waiting, testFaith$eruptions, pch=19, col="blue", xlab = "Waiting", ylab="Duration")
lines(testFaith$waiting, predict(lm1, newdata=testFaith), lwd=3)

sqrt(sum((lm1$fitted - trainFaith$eruptions)^2))
sqrt(sum((lm1$fitted - trainFaith$eruptions)^2))

pred1 <- predict(lm1, newdata=testFaith, interval = "prediction")
ord <- order(testFaith$waiting)
plot(testFaith$waiting, testFaith$eruptions, pch=19, col="blue")
matlines(testFaith$waiting[ord], pred1[ord,], type = "l", col=c(1,2,2), lty=c(1,1,1), lwd=3)

modFit <- train(eruptions ~ waiting, data = trainFaith, method = "lm")
summary(modFit$finalModel)

# Predicting with regression multiple covariates
library(ISLR)
library(ggplot2)
library(caret)
data(Wage)
Wage <- subset(Wage, select = -c(logwage))
summary(Wage)
inTrain <- createDataPartition(y=Wage$wage, p=0.7, list=FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]
dim(training)
dim(testing)

featurePlot(x=training[,c("age", "education", "jobclass")], y=training$wage, plot="pairs")
qplot(age, wage, data=training)
qplot(age, wage, colour = jobclass, data=training)
qplot(age, wage, colour = education, data=training)

modFit <- train(wage ~ age + jobclass + education, method = "lm", data = training)
finMod <- modFit$finalModel
print(modFit)

plot(finMod, 1, pch = 19, cex = 0.5, col="#00000010")
qplot(finMod$fitted, finMod$residuals, colour = race, data=training)
plot(finMod$residuals, pch = 19)
pred <- predict(modFit, testing)
qplot(wage, pred, colour = year, data = testing)

modFitAll <- train(wage ~ ., method = "lm", data = training)
pred <- predict(modFitAll, testing)
qplot(wage, pred, data=testing)

# Predicting with trees
## split variables into groups
## evaluate "homogeneity" within each group
## split again if necessary

# Basic algorithm
## Start with all variable in one group
## find the variable that best separates the outcomes
## Divide the data into two groups on that split
## Within each split, find the best variable that separates the outcomes
## continue unit the groups are too small or sufficiently "pure"

# Measures of impurity
## Misclassification error
### 0 = perfect purity
### 0.5 = no purity

## Gini index
### 0 = perfect purity
### 0.5 = no purity

## Deviance/information gain
###0 = perfect purity
###1 = no purity

data(iris)
library(ggplot2)
names(iris)
table(iris$Species)
inTrain <- createDataPartition(y=iris$Species, p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
dim(training)
dim(testing)
qplot(Petal.Width, Sepal.Width, colour = Species, data=training)

library(caret)
modFit <- train(Species ~., method = "rpart", data=training)
print(modFit$finalModel)
plot(modFit$finalModel, uniform = TRUE, main = "Classification Tree")
text(modFit$finalModel, use.n = TRUE, all=TRUE, cex = .8)

#prettier plots

install.packages("rattle")
library(rattle)
par(mfrow = c(1,1))
fancyRpartPlot(modFit$finalModel)
predict(modFit, newdata=testing)

## Bagging: boostrap aggregating

###Basic idea: resample cases and recalculate prediction
### average or majority vote

###Similar bias, reduced variance, most useful for nonlinear function
library(devtools)
install_github("cran/ElemStatLearn")
library(ElemStatLearn)

data(ozone, package = "ElemStatLearn")
ozone <- ozone[order(ozone$ozone),]
head(ozone)

ll <- matrix(NA, nrow = 10, ncol = 155)
for (i in 1:10){
  ss <- sample(1:dim(ozone)[1], replace = T)
  ozone0 <- ozone[ss,]
  ozone0 <- ozone0[order(ozone$ozone),]
  loess0 <- loess(temperature ~ ozone, data = ozone0, span = 0.2)
  ll[i,] <- predict(loess0, newdata= data.frame(ozone=1:155))
}

plot(ozone$ozone, ozone$temperature, pch=19, cex=0.5)
for (i in 1:10){lines(1:155, ll[i, ], col="grey", lwd = 2)}
lines(1:155, apply(ll, 2, mean), col="red", lwd = 2)

## Similar bias, reduced variability
## 1. bagEarth, 2. treebag, 3. bagFDA
library(caret)
predictors <- data.frame(ozone = ozone$ozone)

temperature <- ozone$temperature
install.packages("party")
library(party)
treebag <- bag(predictors, temperature, B = 10, 
               bagControl = bagControl(fit=ctreeBag$fit,
                                       predict = ctreeBag$pred,
                                       aggregate = ctreeBag$aggregate))
plot(ozone$ozone, temperature, col='lightgrey', pch=19)
points(ozone$ozone, predict(treebag$fits[[1]]$fit, predictors), pch=19, col="red")
points(ozone$ozone, predict(treebag, predictors), pch=19, col="blue")

## Parts of bagging

# Random forest
## bootstrap samples
## at each split, bootstrap variables
## growth multiple trees and vote

## Pros:
## Accuracy

## Cons:
## Speed, interpretability, overfitting

data(iris)
library(ggplot2)
inTrain <- createDataPartition(y= iris$Species, p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
library(caret)
modFit <- train(Species ~., data = training, method = "rf", prox = TRUE)
modFit
library(randomForest)
getTree(modFit$finalModel, k=2)
irisP <- classCenter(training[, c(3,4)], training$Species, modFit$finalModel$prox)
irisP <- as.data.frame(irisP)
irisP$Species <- rownames(irisP)
p <- qplot(Petal.Width, Petal.Length, col=Species, data=training)
p + geom_point(aes(x=Petal.Width, y=Petal.Length, col = Species), size =5, shape=4, data=irisP)

pred <- predict(modFit, testing)
testing$predRight <- pred == testing$Species
table(pred, testing$Species)
qplot(Petal.Width, Petal.Length, colour = predRight, data=testing, main="newdata predictions")

##Boosting
### Take lots of weak predictors
### Weigth them and add them up
### Get a stronger predictor

### k classifiers, create a classifier that combines classification functions

### Adaboost

library(ISLR)
data(Wage)
Wage <- subset(Wage, select = -c(logwage))
inTrain <- createDataPartition(y=Wage$wage, p=0.7, list=FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]
modFit <- train(wage ~ ., method = "gbm", data=training, verbose = FALSE)
print(modFit)

qplot(predict(modFit, testing), wage, data=testing)

#Model Based Prediction

## data follow a probablistic model
## use baye's theorem to identify optimal classifier

# Linear discriminant analysis
# Quadratic discrimant analysis
# Model based prediction assumes more complicated versions for the covariance matrix
# Naive Bayes assumes independence between features for model building
data(iris)
library(ggplot2)
names(iris)
inTrain <- createDataPartition(y = iris$Species, p =0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
dim(training)
dim(testing)


modlda <- train(Species ~ ., data = training, method = "lda")
modnb <- train(Species ~ ., data = training, method = "nb")
plda <- predict(modlda, testing)
pnb <- predict(modnb, testing)
table(plda, pnb)

equalPredictions <- (plda == pnb)
qplot(Petal.Width, Sepal.Width, colour = equalPredictions, data = testing)

#Regularized regression
## Fit a regression model
## Penalize large coefficients

## Help with the bias/variance tradeoff, help with model selection
library(ElemStatLearn)
data(prostate)
str(prostate)

small <- prostate[1:5,]
lm(lpsa ~ ., data = small)

## Ridge regression

## Lasso 

# Combining predictors
## combine classifiers by averaging / voting
library(ISLR)
data(Wage)
library(ggplot2)
library(caret)
Wage <- subset(Wage, select = -c(logwage))

inBuild <- createDataPartition(y=Wage$wage, p = 0.7, list = FALSE)
validation <- Wage[-inBuild, ]
buildData <- Wage[inBuild,]
inTrain <- createDataPartition(y=buildData$wage, p = 0.7, list = FALSE)
training <- buildData[inTrain,]
testing <- buildData[-inTrain,]

mod1 <- train(wage ~., method = "glm", data = training)
mod2 <- train(wage ~., method = "rf", data = training, trControl = trainControl(method = "cv"), number = 3)

pred1 <- predict(mod1, testing)
pred2 <- predict(mod2, testing)
qplot(pred1, pred2, colour = wage, data = testing)

predDF <- data.frame(pred1, pred2, wage = testing$wage)
combModFit <- train(wage ~., method = "gam", data = predDF)
combPred <- predict(combModFit, predDF)

sqrt(sum(pred1 - testing$wage)^2)

sqrt(sum(pred2 - testing$wage)^2)

sqrt(sum(combPred - testing$wage)^2)

pred1V <- predict(mod1, validation)
pred2V <- predict(mod2, validation)
predVDF <- data.frame(pred1 = pred1V, pred2 = pred2V)
combPredV <- predict(combModFit, predVDF)
sqrt(sum(pred1V - validation$wage)^2)
sqrt(sum(pred2V - validation$wage)^2)
sqrt(sum(combPredV - validation$wage)^2)

# Forecasting: data are dependent over time, trends, seasonal pattern
# cycles

## Be aware of spurious correlations
## Be aware of extrapolation
install.packages("quantmod")
library(quantmod)
from.dat <- as.Date("01/01/08", format = "%m/%d/%y")
to.dat <- as.Date("12/31/13", format = "%m/%d/%y")
getSymbols("GOOG", src="yahoo", from = from.dat, to = to.dat)
head(GOOG)
mGoog <- to.monthly(GOOG)
googOpen <- Op(mGoog)
ts1 <- ts(googOpen, frequency = 12)
plot(ts1, xlab = "Years + 1", ylab = "GOOG")

plot(decompose(ts1), xlab = "Years + 1")

#Training and test sets
ts1Train <- window(ts1, start = 1, end = 5)
ts1Test <- window(ts1, start = 5, end = (7-0.01))
ts1Train
plot(ts1Train)
install.packages("forecast")
library(forecast)
lines(ma(ts1Train, order=3), col="red")

#Exponential smoothing
ets1 <- ets(ts1Train, model = "MMM")
fcast <- forecast(ets1)
plot(fcast)
lines(ts1Test, col="red")
accuracy(fcast, ts1Test)

#Unsupervised prediction
data(iris)
library(ggplot2)
inTrain <- createDataPartition(y=iris$Species, p =0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
dim(training)
dim(testing)
kMeans1 <- kmeans(subset(training, select = -c(Species)), centers =3)
training$clusters <- as.factor(kMeans1$cluster)
qplot(Petal.Width, Petal.Length, colour = clusters, data=training)
table(kMeans1$cluster, training$Species)

modFit <- train(clusters ~., data=subset(training, select = -c(Species)), method="rpart")
table(predict(modFit, training), training$Species)
testClusterPred <- predict(modFit, testing)
table(testClusterPred, testing$Species)
