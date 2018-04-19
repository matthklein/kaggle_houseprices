#Kaggle Competition --- House Prices: Advanced Regression Techniques
#https://www.kaggle.com/c/house-prices-advanced-regression-techniques
#
#Matthaeus Kleindessner, April  2018
#m.kleindessner[at]gmail.com


#############################################################
#############################################################
# set working directory
#setwd("~/R/Kaggle/HousePrices")

#load packages
library(bnlearn)
library(caret)
library(reshape)
library(missForest)
library(glmnet)
 
#helper functions
Mode <- function(x) {
   ux <- unique(x)
   ux[which.max(tabulate(match(x,ux)))]
}

Rmse <- function(x,y) {   
  rm <- sqrt(mean((x-y)^2))
}
#############################################################
#############################################################



#############################################################
#### PREPROCESSING ##########################################
#############################################################

#read train and test data; the first variable is an ID number --- we only need 
#it for the test data to submit to Kaggle
train.data <- read.csv('train.csv')
test.data <- read.csv('test.csv')
test.ID <- test.data$Id
 
#number of train and test points
nr.train=nrow(train.data)
nr.test=nrow(test.data)

#take logarithm of prices (then we work with ordinary L2-loss) and
#combine train and test data
train.SalePrice <- log(train.data$SalePrice)
dat <- rbind(train.data[,-c(1,81)],test.data[,-1])
nr.data=nrow(dat)



### Summary of the data and a few plots #####################
str(dat)
summary(dat)
sum(complete.cases(dat))   #=0 => there is no complete case

hist(train.SalePrice)
plot(train.data$YrSold,train.SalePrice)
feature.MonthsSinceJan2006<-(dat$YrSold-2006)*12+(dat$MoSold-1)
plot(feature.MonthsSinceJan2006[1:nr.train],train.SalePrice)   # => no correlation; we will treat YrSold and MoSold as categorical (below)
plot(train.data$YearBuilt,train.SalePrice)   # => there is some correlation; we want to include functions of this feature as new features (below)
plot(train.data$OverallQual,train.SalePrice)   # => there is a clear correlation
plot(train.data$BedroomAbvGr,train.SalePrice)
cor(train.data$BedroomAbvGr,train.SalePrice)  # => there is little correlation
cor(train.data$BedroomAbvGr[(train.data$BedroomAbvGr>1)&(train.data$BedroomAbvGr<5)],train.SalePrice[(
   train.data$BedroomAbvGr>1)&(train.data$BedroomAbvGr<5)])  # => there is more correlation; we want to include binned versions of this feature (below)
#############################################################



#there are too many NA's; for features Alley, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2,
#FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond, PoolQC, Fence, MiscFeature: NA means None
#(i.e., no alley, no pool, ...)
for (ell in c("Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu",
              "GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature")){
    dat[,ell] <- as.factor(ifelse(is.na(dat[,ell]),'None',as.character(dat[,ell])))
}



### Change data types and order factors #####################
dat$MSSubClass <- as.factor(dat$MSSubClass)
dat$OverallQual <- factor(dat$OverallQual,ordered=TRUE)
dat$OverallCond <- factor(dat$OverallCond,ordered=TRUE)
dat$BsmtFullBath <- as.factor(dat$BsmtFullBath)
dat$BsmtHalfBath <- as.factor(dat$BsmtHalfBath)
dat$FullBath <- as.factor(dat$FullBath)
dat$HalfBath <- as.factor(dat$HalfBath)
dat$KitchenAbvGr <- as.factor(dat$KitchenAbvGr)
dat$Fireplaces <- as.factor(dat$Fireplaces)
dat$MoSold <- as.factor(dat$MoSold)
dat$YrSold <- as.factor(dat$YrSold)

ord <- c("Po","Fa","TA","Gd","Ex")
for (ell in c("ExterQual","ExterCond","HeatingQC","KitchenQual")){
  dat[,ell] <- factor(dat[,ell],levels=ord,ordered=TRUE)
}
dat$Functional <- factor(dat$Functional,levels=c("Sal","Sev","Maj2","Maj1","Mod","Min2","Min1","Typ"),ordered=TRUE)
#############################################################



str(dat)
summary(dat)    #There are a lot of NA's in LotFrontage and GarageYrBlt (when there is no garage)

#Should we remove LotFrontage?
sum(is.na(test.data$LotFrontage))   #=227 => fraction of NA's in train and test data is roughly the same
plot(train.data$LotFrontage,train.SalePrice)
cor(train.data$LotFrontage,train.SalePrice,use="complete.obs")  #=0.36 => there is some correlation that we do not want to lose 
plot(dat$LotFrontage,dat$LotArea)
cor(dat$LotFrontage,dat$LotArea,use="complete.obs")   #=0.49 => there is a clear correlation between LotFrontage and LotArea, but not enough to remove LotFrontage  
dat$LotFrontageAvailable <- as.factor(ifelse(is.na(dat$LotFrontage),0,1))  #we include a feature telling whether LotFrontage=NA
# => we want to keep LotFrontage

#Fix the NA's for GarageYrBlt
temp1 <- rowSums(dat[,c("GarageType","GarageFinish","GarageQual","GarageCond","GarageCars","GarageArea")]==matrix(rep(
   c(rep("None",4),0,0),nr.data),nrow=nr.data,byrow = TRUE))
which(((temp1!=0) & (temp1!=6)) | is.na(temp1))   # record 2577 and record 2127, which are test points with GarageType=Detchd
                               #for which we accidentally set GarageFinish,GarageQual,GarageCond to None
dat[c(2127,2577),c("GarageFinish","GarageQual","GarageCond")]=NA
dat[,"GarageYrBlt"] <- ifelse(temp1==6,1000,dat[,"GarageYrBlt"])   #If there is no garage, we set GarageYrBlt to 1000
dat[,"GarageYN"] <- as.factor(ifelse(temp1==6,0,1))   #we include a feature telling whether there is a garage



### Further sanity checks' ##################################
#S1
temps <- c("BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","TotalBsmtSF")
temp1 <- rowSums(dat[,temps]==matrix(rep(c(rep("None",5),0),nr.data),nrow=nr.data,byrow = TRUE))
which(((temp1!=0) & (temp1!=6)) | is.na(temp1))   #records 333, 949, 1488, 2041, 2121, 2186, 2218, 2219,
                                                  #2349, 2525, for which we accidentally set some variables to None
dat[which(((temp1!=0) & (temp1!=6)) | is.na(temp1)),temps][dat[which(((temp1!=0) & (temp1!=6)) | is.na(temp1)),
                                      temps]=="None"]=NA

#S2
which((dat$FireplaceQu=='None')&(dat$Fireplaces!=0))   #=empty

#S4
which((dat$PoolQC=='None')&(dat$PoolArea!=0))   #records 2421, 2504, 2600, which are test points, for which we 
                                                #accidentally set PoolQC to none
dat$PoolQC[which((dat$PoolQC=='None')&(dat$PoolArea!=0))]=NA

#S5
which((dat$MiscFeature=='None')&(dat$MiscVal!=0)) #record 2550, which is a test point with MiscVal=17000,
                                                  #but MiscFeature=NA
dat$MiscFeature[2550]="Othr"   #Othr seems to be a general category
#############################################################



### Adding some features that might be meaningful ###########
### there might be much more possibilities!!! ###############
#Log and ^2 for numerical attributes
numericalAttr <- unlist(lapply(dat, is.numeric))
temp1=sum(numericalAttr <- unlist(lapply(dat, is.numeric)))  #=22 => there are not that many numerical attributes; we will include the transformations for all of them
feature.log <- lapply(dat[,numericalAttr],function(x) log(x+1))
feature.sq <- lapply(dat[,numericalAttr],function(x) x^2)
for (ell in 1:temp1){
  names(feature.log)[ell]=paste(names(feature.log)[ell],"Log",sep="")
  names(feature.sq )[ell]=paste(names(feature.sq)[ell],"Squared",sep="")
}

#Binning of BedroomAbvGr
feature.BedroomNew1 <-  as.factor(ifelse(dat$BedroomAbvGr>5,0,1))
feature.BedroomNew2 <- as.factor(ifelse(dat$BedroomAbvGr<2,0,1))

#Indicator whether there was a renovation at some point
feature.WasRenovatedNew <- as.factor(ifelse(dat$YearBuilt==dat$YearRemodAdd,0,1))

#Coupling OverallCond with Foundation 
CondBinned <- combine_factor(dat$OverallCond,c(0,0,0,1,1,1,2,2,2))
feature.CondCOMBFoundationNew <- model.matrix(~Cond:Found,data=data.frame("Cond"=CondBinned,"Found"=dat$Foundation))

dat <-cbind(dat,feature.log,feature.sq,feature.BedroomNew1,feature.BedroomNew2,feature.WasRenovatedNew,feature.CondCOMBFoundationNew[,-1])
#############################################################



### Removing features with small variance ###################
nzv <- nearZeroVar(dat, saveMetrics= TRUE)
nzv[nzv$nzv,]
summary(dat[,nzv$nzv])

#Binning of some of these features
feature.AlleyNew <- combine_factor(dat$Alley,c(0,1,0))
feature.LandContourNew <- combine_factor(dat$LandContour,c(0,0,0,1))
feature.LandSlopeNew <- combine_factor(dat$LandSlope,c(0,1,1))
feature.BsmtCondNew <- combine_factor(dat$BsmtCond,c(0,1,2,0,3))
feature.BsmtFinType2New <- combine_factor(dat$BsmtFinType2,c(0,1,0,2,3,4,5))
feature.BsmtFinSF2New <- as.factor(ifelse(dat$BsmtFinSF2>mean(dat$BsmtFinSF2[dat$BsmtFinSF2>0],na.rm=TRUE),1,0))
feature.LowQualFinSFNew <- as.factor(ifelse(dat$LowQualFinSF>mean(dat$LowQualFinSF[dat$LowQualFinSF>0],na.rm=TRUE),1,0))
feature.KitchenAbvGrNew <- combine_factor(dat$KitchenAbvGr,c(0,0,1,1))
feature.FunctionalNew <- combine_factor(dat$Functional,c(0,0,0,0,0,0,0,1))
feature.MiscFeatureNew <- combine_factor(dat$MiscFeature,c(0,1,0,2,0))

#We want to keep OpenPorchSF, EnclosedPorch, X3SsnPorch, PoolArea, MiscVal
temprem=nzv$nzv
for (ell in c("OpenPorchSF","EnclosedPorch","X3SsnPorch","PoolArea","MiscVal")){
  temprem[which(names(dat)==ell)]=FALSE
}
dat <- dat[,-which(temprem)]

tempadd=c("Alley","LandContour","LandSlope","BsmtCond","BsmtFinType2","BsmtFinSF2","LowQualFinSF","KitchenAbvGr",
          "Functional","MiscFeature")
for (ell in tempadd){
  dat[,paste(ell,"New",sep="")]=get(paste("feature.",ell,"New",sep=""))
}
#############################################################



### Centering and Scaling for numerical attributes ##########
preProcValues <- preProcess(dat[1:nr.train,], method = c("center", "scale"))
dat <- predict(preProcValues,dat)
#############################################################
 
 
 
### Imputing missing values #################################
sum(complete.cases(dat))   #=2394
mean(rowSums(is.na(dat)))   #0.55
which(rowSums(is.na(dat))>7)   #=2121 and 2577 (test points) 
# => there aren"t too many NA's

imp.dat <- missForest(dat,variablewise = TRUE)
imp.dat$OOBerror
names(dat)[which(imp.dat$OOBerror>0.2)]   # => Imputation might have worked reasonably 
imp.dat <- imp.dat$ximp
#############################################################



### Transforming into a model matrix ########################
dat.trans <- model.matrix(~.,data=imp.dat)

#Normalizing
preProcValues <- preProcess(dat.trans,method=c("center","scale"))  
dat.trans <- predict(preProcValues,dat.trans)

#Normalizing and removing variables with small variance, high
#inter-correlation and linear dependence 
preProcValues <- preProcess(dat.trans,method="nzv")  
dat.trans.nzv <- predict(preProcValues,dat.trans)
preProcValues <- preProcess(dat.trans.nzv,method="corr",cutoff=0.95,exact=TRUE)  
dat.trans.nzv.corr.lin <- predict(preProcValues,dat.trans.nzv)
comboInfo <- findLinearCombos(dat.trans.nzv.corr.lin)
dat.trans.nzv.corr.lin <- dat.trans.nzv.corr.lin[,-comboInfo$remove]
#############################################################

#############################################################
#############################################################
#############################################################





#############################################################
### MODEL TRAINING AND SELECTION ############################
#############################################################


### Knn and Ridge Regression as baseline ####################
# they are simple to use via caret

fitControl <- trainControl(method = "repeatedcv",number = 10,repeats = 10) #10-fold CV repeated for 10 times

#Knn
KnnGrid <- expand.grid(k=seq(5,19,2))
set.seed(825)
KnnFit <- train(dat.trans[1:nr.train,],train.SalePrice,method = "knn",trControl = fitControl,verbose = FALSE,tuneGrid = KnnGrid)
KnnFit

#RidgeRegression
#doesn't work with dat.trans
RidgeGrid <- expand.grid(lambda = 5^(-8:0))
set.seed(825)
RidgeFit.nzv.corr.lin <- train(dat.trans.nzv.corr.lin[1:nr.train,],train.SalePrice,method = "ridge",trControl = fitControl,verbose = FALSE,
                      tuneGrid = RidgeGrid)
RidgeFit.nzv.corr.lin

#Comparing the two
densityplot(RidgeFit.nzv.corr.lin)
densityplot(KnnFit)
resamps <- resamples(list(Ridge = RidgeFit.nzv.corr.lin,Knn = KnnFit))
theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
bwplot(resamps, layout = c(3, 1))
# => Ridge Regression appears to be superior
#############################################################



### Elastic-Net from Glmnet #################################
#I don't use caret because it's parameter selection works differently 

number=10
repeats=10
a.values=seq(0.5,1,0.1)
cverror_mat=matrix(0,nrow=repeats,ncol=length(a.values))

for (ell in 1:repeats){
  foldid=sample(rep(seq(number),length=nr.train))
    for (a in 1:length(a.values)){
      set.seed(90*ell+27)
      GLMNETFit.nzv.corr.lin=cv.glmnet(x=dat.trans.nzv.corr.lin[1:nr.train,],y=train.SalePrice,
                                       family="gaussian",alpha=a.values[a],nfolds=number,foldid=foldid)
      cverror_mat[ell,a]=sqrt(min(GLMNETFit.nzv.corr.lin$cvm))
    }
}

best.a=a.values[which.min(colMeans(cverror_mat))]
GLMNETFit.nzv.corr.lin=cv.glmnet(x=dat.trans.nzv.corr.lin[1:nr.train,],y=train.SalePrice,
                                 family="gaussian",alpha=best.a,nfolds=number,foldid=foldid)
#############################################################

list("MeanErrorGLMNET"=min(colMeans(cverror_mat)),"FinalErrorGLMNET"=sqrt(min(GLMNETFit.nzv.corr.lin$cvm)),"MeanErrorRidge"=
       mean(RidgeFit.nzv.corr.lin$resample[["RMSE"]]))

# => GLMNET might be slightly superior (note the final error [for the final lambda] of GLMNET
#is only over one repetition and the mean error of GLMNET is for the best lambda in each repetition,
#while the mean error of Ridge is for the final lambda over 10 repetitions)
#############################################################
#############################################################
#############################################################





#############################################################
### PREDICTION ##############################################
#############################################################

#predRidge <- predict(RidgeFit.nzv.corr.lin, newdata = dat.trans.nzv[(nr.train+1):nr.data,])
predGLMNET <- predict(GLMNETFit.nzv.corr.lin,dat.trans.nzv.corr.lin[(nr.train+1):nr.data,],
                      s = GLMNETFit.nzv.corr.lin$lambda.min,type="link")
pred <- predGLMNET
write.csv(data.frame("Id"=test.ID,"SalePrice"=exp(as.vector(pred))),file="submission.csv",row.names = FALSE,quote=FALSE)

#############################################################
#############################################################          
#############################################################