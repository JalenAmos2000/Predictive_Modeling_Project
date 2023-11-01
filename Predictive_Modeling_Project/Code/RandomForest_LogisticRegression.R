# Clearing anything in the global environment

rm(list=ls())

# Loading my packages 

library('easypackages')
libraries('tidyr','dplyr','boot','ROSE','randomForest','caret','e1071','pROC','MLmetrics','ROCR','ggplot2','ggcorrplot')



healthcare <- read.csv("../Data/healthcare-dataset-stroke-data.csv",sep=",")
#====
#===== Data Description
# Stroke is a binary response variable (1 = the patient had a stroke and 0 = the patient did not have a stroke). I may use binary logistic regression using the glm function. 
# ID is a unique identifier. I may remove this variable because it may not be useful.
# BMI is a numeric variable. it is the body mass index of each patient.
# Avg_glucose_level: is the average glucose level in blood of each patient, and it is a numeric variable. 
# Hypertension and heart disease are binary explanatory variables. 0 means the patient does not have __. 1 means the patient does have __.
# Age is the age of the patient
# Ever_married and Residence type are a character variable with two levels
# Gender is the gender of the patient, and it consists of three levels. However, I will drop one of the levels
# Smoking status is a character variable that has four levels. These four levels describes the smoking status of each patient.
# Work_type has 5 levels, and it describes what kind of work the patient does in life. 
#====


#===== 

#Data Cleaning

# Making gender a factor and dropping one of the levels 

table(healthcare$gender) # There are three levels, but there is only one subject in the "other" category. I will merge it with male category. 

healthcare$gender <- as.factor(healthcare$gender)

str(healthcare) #Checking to see if gender is a factor

healthcare$gender[(healthcare$gender=="Male")|healthcare$gender=="Other"] <- "Male"

healthcare$gender <- droplevels(healthcare$gender)

##### Using tidyr to clean my healthcare data and saving it as a new data frame. All I have to do is drop the missing values and change the categorical variables into factors 

new_healthcare <- healthcare %>%
    drop_na()  %>% 
    select(!(id)) %>%
    mutate(Residence_type = as.factor(Residence_type)) %>%
    mutate(ever_married = as.factor(ever_married))%>%
    mutate(work_type = as.factor(work_type)) %>%
    mutate(smoking_status = as.factor(smoking_status))%>%
    mutate(heart_disease = as.factor(heart_disease))%>%
    mutate(hypertension = as.factor(hypertension))%>%
    mutate(stroke = as.factor(stroke))

str(new_healthcare)

#Checking for missing values

sapply(new_healthcare, function(x) sum(is.na(x)))

# Data Visualization

# Histogram of BMI and Avg_glucose_level

ggplot(data = new_healthcare, aes(x = bmi))+
    geom_histogram(color = 'black',fill = 'blue',bins= 30)

ggplot(data = new_healthcare, aes(x = avg_glucose_level)) +
    geom_histogram(color = 'black', fill = 'purple',bins = 30)

# Box plots of BMI and Avg_glucose_level for the two levels of stroke

ggplot(data = new_healthcare, aes(x=as.character(stroke), y=bmi)) +
    geom_boxplot(fill="steelblue") +
    labs(title="BMI distribution by Stroke Status", x="Stroke", y="Bmi")

ggplot(data = new_healthcare, aes(x=as.character(stroke), y=avg_glucose_level)) +
    geom_boxplot(fill="red") +
    labs(title="AVG_Glc_Lv distribution by Stroke Status", x="Stroke", y="average_glucose_level")

#Bar chart of Stroke Status
ggplot(data = new_healthcare , aes(x=stroke))+
    geom_bar(stat="count", width=0.7, fill="steelblue")+
    theme_minimal()

# Correlation matrix plot of the new dataset  

ggcorrplot(cor(new_healthcare[sapply(new_healthcare,is.numeric)],use = "pairwise.complete.obs"))


# Logistic Regression (70% Train/ 30% Test split)

set.seed(1) #Makes the results reproducible.

indx <- sample(1:4909,size=4909*.7,replace=F)
x1   <- new_healthcare[indx,1:10] #Training X
x2   <- new_healthcare[-indx,1:10] #Test X
y1   <- new_healthcare[indx,11] #Training Y
y2   <- new_healthcare[-indx,11] #Test Y
NH_1 <- as.data.frame(x1); NH_1$Y <- y1
NH_2 <- as.data.frame(x2); NH_2$Y <- y2


## Random Forest (70% train/ 30% test split)
set.seed(1)

index <- sample(1:4909, size = 4909, replace = F)

NH.train <- new_healthcare[index[1:3436],]
NH.test <- new_healthcare[index[3437:4909],]



## Oversampling method

#There are more patients who did not have a stroke(ie. Stroke = 0) vs patients who did have a stroke. This issue leads to poor prediction performance for the class with less subjects.

# Two methods to visualize the data 

barplot(prop.table(table(new_healthcare$stroke)),
        col = rainbow(2),
        ylim = c(0, 0.9),
        main = "Class Distribution")


ggplot(data = new_healthcare , aes(x=stroke))+
    geom_bar(stat="count", width=0.7, fill="steelblue")+
    theme_minimal()

# We use a resampling method called oversampling. Oversampling will duplicate samples from the minority class to balance the binary response variable.
# We only apply the resampling method to the train set. We leave the test set imbalance. 
# Although oversampling is a solution to handling class imbalance, oversampling can lead to over fitting.For that reason it is not the absolute way of dealing with class imbalance. 

## I only apply the oversampling method to the train set, and the test set will remain imbalance. 
set.seed(1)

over <- ovun.sample(Y ~ ., data = NH_1, method = "over", N = 6588)$data

table(over$Y) # Checking to see if it is balanced

# The bar chart shows the two levels are equal 
barplot(prop.table(table(over$Y)),
        col = rainbow(2),
        ylim = c(0, 0.9),
        main = "Class Distribution")

ggplot(data = over , aes(x=Y))+
    geom_bar(stat="count", width=0.7, fill="steelblue")+
    theme_minimal()

# logistic regression 
over.glm <- glm(Y ~., family = binomial(link = logit), data = over)

summary(over.glm)


# Confusion Matrix 

over.pred <- predict(over.glm,  NH_2, type = "response") 
over_class <- ifelse(over.pred > 0.5, 1,0)

confusionMatrix(factor(over_class, levels = c(0,1)),NH_2$Y, mode = "everything", positive ="1") #Accuracy is a lot better, however it can be misleading because of the class imbalance. 

# Calculating the misclassifications rate for the logistic regression 

# The misclassification rate is used to measure the model's performance

log_error <- table(actual = NH_2$Y, predicted = factor(over_class))

log_misclass_rate <- 1-sum(diag(log_error))/sum(log_error)

log_misclass_rate

# The misclassification rate is 0.2593347 for logistic regression 

#plotting ROC and AUC For Logistic Regression

roc_lg <- roc(NH_2$Y,as.ordered(over_class))

auclg <- round(auc(NH_2$Y,as.ordered(over_class)),4)

ggroc(roc_lg, colour = 'steelblue', size = 2) +
    ggtitle(paste0('ROC Curve ', '(AUC = ', auclg, ')'))

# The AUC increased when we did the resampling method for the logistic regression model. The misclassification rate can be misleading because of the class imbalance. As a result, the proper metric would be the AUC.
# The logistic regression does good job distinguishing between classes based on the AUC score. However, in the medical field, researchers usually want scores to be higher than 0.95.

## Random Forest(Oversampling) using the optimal rf

set.seed(1)

over_rf <- ovun.sample(stroke ~ ., data = NH.train , method = "over", N = 6588)$data

table(over_rf$stroke)

# The Random Forest 

set.seed(1)

rfover <- randomForest(stroke ~., data = over_rf, xtest = NH.test[,-11], ytest = NH.test$stroke, ntree= 200 ,mtry = 9, nodesize = 8, keep.forest = TRUE)

rfover


# calculating the AUC score for the random forest 

rf_predicted_class <- rfover$test$predicted


# Plotting the AUC of Random forest 

roc_rf <- roc(NH.test$stroke,as.ordered(rf_predicted_class))

aucrf <- round(auc(NH.test$stroke,as.ordered(rf_predicted_class)),4)

ggroc(roc_rf, colour = 'steelblue', size = 2) +
    ggtitle(paste0('ROC Curve ', '(AUC = ', aucrf, ')'))

# The Random Forest perform poorly when we look the AUC score. The AUC score has a poor discrimination. THis model does no better than random guessing. 

# calculating the misclassification rate for the random forest 

rf_error <- table(actual = NH.test$stroke, predicted = rf_predicted_class)

rf_misclass_rate<- 1-sum(diag(rf_error))/sum(rf_error)

rf_misclass_rate

# The misclassification rate does not take into account the class imbalance so we will be using the AUC score.

# Confusion Matrix of Random Forest 

confusionMatrix(rf_predicted_class, NH.test$stroke, mode = "everything", positive = '1')


# OOB Error Rate 

par(mar=c(5,5,4,2),cex.axis=1.5,cex.lab=1.7,cex.main=2)
plot(rfover, main="Error rates on OOB samples")
legend("topright",legend=c("OOB",levels(over_rf$stroke)),col=1:7,lty=1:7,cex=1.5)

# Variable Importance Plot of Random Forest 

varImpPlot(rfover, sort=TRUE) 

# Age, Average_glucose_level, and Bmi are the most important variables in random forest using oversampling method 

# Partial Dependence plots of Age and Average Glucose Level  using over sampling method 

par(mfrow=c(2,2),cex.axis=1.5,cex.lab=1.7,cex.main=2)
partialPlot(rfover, over_rf, 'age',which.class='0',main="PDP of age for 0") # When the age increases, the log-odds of a patient to not have stroke montonocially decreases. Essentially, as the age of the patient increases, the patient not having a stroke is decreases. 
partialPlot(rfover, over_rf, 'age',which.class='1',main="PDP of age for 1") # As the age of the patient increases, the logit of a patient to have a stroke montonocially increases. 
partialPlot(rfover, over_rf, 'avg_glucose_level',which.class='0',main="PDP of A1C for 0") # As the average glucose level increases, the log-odds of a patient to not have a stroke monotonocially decreases
partialPlot(rfover, over_rf, 'avg_glucose_level',which.class='1',main="PDP of A1C for 1") # The log odds of a patient to have a stroke increases as the average glucose level increases. 

# Density Plot of Age on Stroke Status

d1 <- density(over_rf$age[which(over_rf$stroke=='1')])
d2 <- density(over_rf$age[which(over_rf$stroke=='0')])
par(mfrow=c(1,2),cex.axis=1.5,cex.lab=1.7)
plot(d1,xlim=range(d1$x,d2$x), ylim=range(d1$y,d2$y), col="red",
     main="Density plot for Age on Stroke status",
     xlab="age",lwd=2)
lines(d2,col="grey",lwd=2,lty=2)
legend("topleft",legend=c("stroke = 1","stroke = 0"),lty=c(1,2),col=c("red","grey"),lwd=2,cex=1.2)

## Density plots on average glucose level/age on stroke 1 

d3 <- density(over_rf$avg_glucose_level[which(over_rf$stroke=='1')])
plot(d1,xlim=range(d1$x,d3$x), ylim=range(d1$y,d3$y), col="red",
     main="Density plot for age and A1C on Stroke = 1",
     xlab="",lwd=2)
lines(d3,col="blue",lwd=2,lty=2)
legend("topleft",legend=c("Age","Avg_Glc_lv"),lty=c(1,2),col=c("red","blue"),lwd=2,cex=1.2)

# Stroke = 1 tend to have high value on Age than Average Glucose Level 
