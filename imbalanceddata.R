setwd('/Users/hannahchz/Desktop/Project2018_9')
library(randomForest)
library(mlr)
library(MASS)
library(caret) #this package has the createDataPartition function
library(forcats)
library(ROSE)

#read in aircon dataset 
data<-read.csv("new.csv")
data$Country<-fct_lump(data$Country, 12)
data$ExpDate<-format(as.Date(data$ExpDate, format="%d/%m/%Y"),"%Y") #0020 
data$ExpDate<-as.factor(data$ExpDate)
cols <- sapply(data, is.logical)
data[,cols] <- lapply(data[,cols], as.numeric)
imp <- impute(data, classes = list(factor = imputeMode(), logical= imputeMode(), integer = imputeMedian(), numeric= imputeMedian()), dummy.classes = c("factor", "logical","integer","numeric"), dummy.type = "numeric")
impdata <- imp$data
new_impdata <- impdata[, which(names(impdata) %in% c("N.Standard", "Configuration2.unitmount", "C.Total.Cool.Rated", "EERtestAvg", "COPtestAvg", "AnnualOutputEER", "AnnualOutputCOP", "Product Class", "outdoortype","sri2010_heat", "Star2010_Cool", "Star2010_Heat", "EER", "Rated.cooling.power.input.kW", "Rated.ACOP", "sri2010_cool", "sri2010_cool_morethan2"))]

#I randomly divide the data into training and test sets (stratified by class) and perform Random Forest modeling with 10 x 10 
#repeated cross-validation. Final model performance is then measured on the test set.
str(new_impdata)
summarizeColumns(new_impdata)
levels(new_impdata$sri2010_cool_morethan2)
new_impdata$sri2010_cool_morethan2 <- factor(as.character(new_impdata$sri2010_cool_morethan2))
prop.table(table(new_impdata$sri2010_cool_morethan2))

set.seed(42)
index <- createDataPartition(new_impdata$sri2010_cool_morethan2, p = 0.7, list = FALSE)
train_data <- new_impdata[index, ]
test_data  <- new_impdata[-index, ]

#Besides over- and under-sampling, there are hybrid methods that 
#combine under-sampling with the generation of additional data.
library(rpart)
treeimb <- rpart(sri2010_cool_morethan2 ~ ., data = train_data)
pred.treeimb <- predict(treeimb, newdata = test_data)
accuracy.meas(test_data$sri2010_cool_morethan2, pred.treeimb[,2])
roc.curve(test_data$sri2010_cool_morethan2, pred.treeimb[,2], plotit = F)

data_balanced_both <- ovun.sample(sri2010_cool_morethan2 ~ ., data = train_data, method = "both", p=0.5,                             N=1000, seed = 1)$data
table(data_balanced_both$sri2010_cool_morethan2)

data.rose <- ROSE(sri2010_cool_morethan2 ~ ., data = train_data, seed = 1)$data
table(data.rose$sri2010_cool_morethan2)
data.rose$sri2010_cool

regr.task = makeRegrTask(data = data_balanced_both, target = "sri2010_cool")
regr.task
n = getTaskSize(regr.task) 
train.set = seq(1, n, by = 2)
test.set = seq(2, n, by = 2)
lrn = makeLearner("regr.gbm", n.trees = 100) 
mod = mlr::train(lrn, regr.task, subset = train.set)
task.pred = predict(mod, task = regr.task, subset = test.set)
task.pred
performance(task.pred)






