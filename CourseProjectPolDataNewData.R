
library(dplyr)
library(corrplot)
library(ggplot2)
library(MASS)
require(reshape2)
library(randomForest)
library(caTools)
library(miscTools)
library(mclust)
library(factoextra)
library(pROC)
library(DMwR)
library(caret)
library(ROSE)
library(ROCR)
library(RColorBrewer)



setwd('C:\\Users\\moose_f8sa3n2\\Google Drive\\Analytic Methods\\CourseProject')
df = read.csv('Election_Predict_Data4.csv')
df <- na.omit(df)


df[[1]] <- as.numeric(df[[1]])
df[[2]] <- as.numeric(df[[2]]) 
df[[3]] <- as.factor(df[[3]]) 
df[[4]] <- as.factor(df[[4]])
df[[6]] <- as.factor(df[[6]]) 
df[[7]] <- as.factor(df[[7]])
df[[8]] <- as.factor(df[[8]]) # Target feature of dataset: 1 for win 2 for loss. Is factor
df[[9]] <- as.numeric(df[[9]]) # Industry contributions
df[[14]] <- as.numeric(df[[14]])


dataVizContrib <- function() {
    # A histogram of the contributions over time
    gg <- ggplot(df,aes(x=df$Industry_Spending_Totals)) + geom_histogram(fill="red", bins=300)
    gg + scale_x_continuous(labels = function(x) format(x, scientific = FALSE))
}    

dataVizParties <- function() { 
   # A plot of the parties represented in the data
    ggplot(df,aes(x=df$General_Party)) + geom_bar(color="black", fill="orange")
}
 
dataVizIndust <- function() {   
    # A plot of the industries represented in the data
    ggplot(df,aes(x=df$Broad_Sector.id)) + geom_bar(color="black", fill="orange")
}

dataVizTotalsByIndustry <- function(){
    df <- data.frame(Industry_ID=c(4,2,5,8,6),
        Contributions=c(8308936.11, 57170161.74, 76571208.60, 142573152.46, 325025361.55))
    gg <- ggplot(df, aes(x=Industry_ID, y=Contributions)) + geom_bar(color="black", fill="orange", stat = "identity")
    gg + geom_text(aes(label=Contributions))
    gg + scale_y_continuous(labels = function(x) format(x, scientific = FALSE))
}


# Removing outliers in target variable dataset
# it looked like some outliers were in the target variable 
# in the plots above

summary(df[[9]])
#   Min.   1st Qu.   Median   Mean     3rd Qu    Max. 
#    0     2100      13950   49970    45813   10845112 

# Interquartile ranges of Industry Contributions
# Pct25th = 2100
# Pct75th = 45813
# IQR = 43713
# lowOut = 111382.5

OutliersOut <- function() {
    Pct25th <- summary(df$Industry_Spending_Totals)[["1st Qu."]]
    Pct75th <- summary(df$Industry_Spending_Totals)[["3rd Qu."]]
    IQR <- Pct75th - Pct25th
    highOut <- Pct75th + (1.5 * IQR)
    df <- subset(df,df$Industry_Spending_Totals<highOut)
    
    summary(df$Industry_Spending_Totals)
    # Min.   1st Qu.  Median    Mean   3rd Qu.    Max.
    #  0     1613     10720    21855   33050    111348
    
    gg <- ggplot(df,aes(x=df$Industry_Spending_Totals)) + geom_histogram(fill="red", bins=300)
    gg + scale_x_continuous(labels = function(x) format(x, scientific = FALSE))
}

################### Agglomerative Clustering 

scaledCon <- scale(cbind(df$Candidate.id,df$Industry_Spending_Totals,df$Office_Sought))

# Similarity matrix
d <- dist(scale(scaledCon), method = "euclidean")

# Hierarchical clustering using Complete Linkage
hc1 <- hclust(d, method = "complete" )

dendrogram <- function() {
    # Plot the obtained dendrogram
    #plot(hc1, cex = 0.6, hang = -1)
    fviz_dend(hc1, show_labels=FALSE, cex = 0.6, hang = -1, rect=TRUE)
}

# Assigning clusters for the tree as a feature
df$Agglom_clust <- as.factor(cutree(hc1, k = 5))


################### K-means Clustering 


# Performing the K means clustering
electCluster <- kmeans(df$Industry_Spending_Totals, df$Office_Sought.id,centers=5)

# Engineering features using Kmeans clustering
df$KmeansCluster <- as.factor(electCluster$cluster)


# Rearranging last features so target feature is on the end of dataframe
dfT <- df[c("Candidate.id","Incumbency_Status","Industry_Spending_Totals",
            "Broad_Sector.id","General_Party.id","Status_of_Candidate.id",
            "General_Party.id","Election_Type.id",
              "Broad_Sector.id","Incumbency_Status","Industry_Spending_Totals",
              "KmeansCluster","Agglom_clust","Status_of_Candidate.id")]


# splitting up the dataframe into train and test sets
set.seed(101)
smp_size = floor(0.75*nrow(dfT))
train_ind = sample(seq_len(nrow(dfT)),size = smp_size)
train = dfT[train_ind,]
test = dfT[-train_ind,]

# Turning the train and test sets into dataframes
train <- data.frame(cbind(train))
test <- data.frame(cbind(test))


# Showing the number of possible values in the target variable
table(train$Status_of_Candidate.id)
#   1    2 
# 4916 4195 

# Capturing the column names of training dataset
cols <- c(colnames(train))

# Training the random forest model on training data
rf.mdl <- randomForest(Status_of_Candidate.id ~ .-Candidate.id,
                            data = train[,cols], ntree=500, maxnodes=100, importance=TRUE)

# Plotting the random forest error rates
plotForest <- function() {
    plot(rf.mdl)
    legend("top", colnames(rf.mdl$err.rate), fill=1:ncol(rf.mdl$err.rate))
}



# Performing cross-validation on the training data
crossValidation <- function() {
    trainCV <- train[2:length(train)]
    colsCV <- c(colnames(trainCV))
    election.CV <- train(Status_of_Candidate.id ~ ., 
                         data = trainCV[,colsCV], 
                         method = "cforest", 
                         tuneGrid = data.frame(.mtry = 2),
                         trControl = trainControl(method = "oob"))
    print(election.CV)
    # Conditional Inference Random Forest 
    # 
    # 8252 samples
    # 7 predictor
    # 2 classes: '1', '2' 
    # 
    # No pre-processing
    # Resampling results:
    #     
    #     Accuracy   Kappa    
    #    0.8562773 0.6658975
    # 
    # Tuning parameter 'mtry' was held constant at a value of 8
    
    rf.cv <- rfcv(train[2:length(train)], rf.mdl$predicted, cv.fold=10)
    
    # Showing the error CV
    rf.cv$error.cv
    #      8          4          2          1 
    # 0.02261003 0.01251235 0.04631764 0.17495335 
}

# Results from the predictions made on training data 
trainPredictionResults <- function() {
    print(rf.mdl)
    # Call:
    #     randomForest(formula = Status_of_Candidate.id ~ . - Candidate.id,data = train[, cols], ntree = 1000, maxnodes = 100, importance = TRUE,      keep.forest = TRUE) 
    # Type of random forest: classification
    # Number of trees: 500
    # No. of variables tried at each split: 2
    # 
    # OOB estimate of  error rate: 15.65%
    # Confusion matrix:
    #     1    2 class.error
    # 1 4267  649   0.1320179
    # 2  777 3418   0.1852205
}


# Calculating variable importance from the random forest
importanceResults <- function() {
    import <- importance(rf.mdl)
    varImpPlot(rf.mdl, type = 1)
    print(import)
    # # #                             1        2         MeanDecreaseAccuracy MeanDecreaseGini
    # General_Party.id          49.820876  21.806826             52.84943       116.231094
    # Election_Type.id           6.572511  10.407057             11.05444         9.895273
    # Broad_Sector.id           53.888860  43.764683             63.19599       183.317090
    # Incumbency_Status        116.427325 173.146657            176.60229      1221.564557
    # Industry_Spending_Totals  53.287116  95.404859             98.10282       670.864106
    # KmeansCluster             60.549758   7.921762             63.97468        88.563992
    # Agglom_clust              22.805233  22.932265             33.12526       186.059536
}


rocScorePlot <- function() {
    rf.roc <- roc(train$Status_of_Candidate.id, rf.mdl$votes[,2])
    plot(rf.roc)
    auc(rf.roc) #Area under the curve: 0.9291
}


testDataPredictions <- function() {

    # Making predictions on the test data:
    PredictTest <- predict(rf.mdl,newdata=test)
    # Creating a new column in dataset for the test predictions
    test$PredictTest <- predict(rf.mdl,newdata=test)
    print(table(PredictTest,test$Status_of_Candidate.id))
    # PredictTest      1     2
                # 1  1454  278
                # 2  191  1115
    
    accuracy.meas(response = test$Status_of_Candidate.id, predicted = PredictTest)
    # Call: 
    #     accuracy.meas(response = test$Status_of_Candidate.id, predicted = PredictTest)
    # 
    # Examples are labelled as positive when predicted is greater than 0.5 
    # 
    # precision: 0.490
    # recall: 1.000
    # F: 0.329
    # print(test)
}

# Plotting the random forest margins for classifications
plotMargin <- function() {
    plot(margin(rf.mdl, test$Status_of_Candidate.id))
}

# dataVizContrib()
# dataVizParties()
# dataVizIndust()
# dataVizTotalsByIndustry()
# OutliersOut()
# dendrogram()
# plotForest()
# plotMargin()
# crossValidation()
# trainPredictionResults()
# importanceResults()
# rocScorePlot()
# testDataPredictions()



















