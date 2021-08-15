#SETUP
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

#DATA OVERVIEW
# loading census data
urlFile <- "https://raw.githubusercontent.com/christina-shao/CYO-Capstone/main/nyc-rolling-sales.csv"
sales <-read_csv(url(urlFile))

# overview census data
head(sales)
nrow(sales) # 84548 entries

#DATA WRANGLING
colnames(sales) <-c("X1", "borough", "neighbourhood", "buildingClass", "presentTaxClass", "block", 
                    "lot", "easement", "presentBuildingClass", "address", "apartmentNumber", 
                    "zipCode", "resUnits", 'commercialUnits', 'totalUnits', 'landSqft', 'grossSqft', 
                    'yearBuilt', 'saleTaxClass', 'saleBuildingClass', 'salePrice', 'saleDate') # column names
# convert salePrice to numeric for convenience
sales$salePrice <- as.numeric(sales$salePrice)
# convert borough to factor
sales$borough <- as.factor(sales$borough)
# convert zip code to factor
sales$zipCode <- as.factor(sales$zipCode)
# convert landSqft to numeric
sales$landSqft <- as.numeric(sales$landSqft)
# convert grossSqft to numeric
sales$grossSqft <- as.numeric(sales$grossSqft)

# data cleaning
# remove observations with no salePrice data
sales <- sales[!is.na(sales$salePrice), ]
# remove observations with salePrice of 0. this is a transfer of ownership without transfer of money. e.g. giving your kids your house
sales <- sales %>% filter(salePrice != 0)
# remove outliers function
removeOutliers <- function (i, df){
  Q <- quantile(df[i], probs=c(.25, .75), na.rm = TRUE)
  # find difference between 75th and 25th quartiles
  iqr <- IQR(unlist(df[i]), na.rm = TRUE)
  # find upper and lower range of interquartile space
  up <-  Q[2]+1.5*iqr # Upper Range  
  low<- Q[1]-1.5*iqr # Lower Range
  # remove outliers, return cleaned data
  subset(df, df[i] > low & df[i] < up)
}

# distribution of sales prices. there are some very expensive houses that seem to throw off the data
hist(sales$salePrice)
boxplot(sales$salePrice)
# eliminate outliers of price
priceClean <- removeOutliers(21, sales)
# the distribution now looks more reasonable
hist(priceClean$salePrice)

#DATA ANALYSIS
# borough. numerical code of: Manhattan (1), Bronx (2), Brooklyn (3), Queens (4), and Staten Island (5)
# boxplot of sale prices in boroughs
sales %>% ggplot(aes(borough, salePrice)) + geom_boxplot() #boroughs 1 and 3 consistently sell above 500k. 2 seems to sell for the least
# create dataframe of avg salePrice by borough
boroughAvg <- sales %>% group_by(borough) %>%
  summarize(mean = mean(salePrice)) 
# avg salePrice in NYC
avg <- mean(sales$salePrice)

# neighbourhood
# create dataframe of avg salePrice by borough
neighbourhoodAvg <- sales %>% group_by(neighbourhood) %>%
  summarize(mean = mean(salePrice)) 
# find most expensive neighbourhood
neighbourhoodMaxMin<- neighbourhoodAvg[which.max(neighbourhoodAvg$mean), ]
# find cheapest neighbourhood
neighbourhoodMaxMin <- rbind(neighbourhoodMaxMin[which.min(neighbourhoodAvg$mean), ])
# boxplot of little italy and pelham bay
sales %>% filter(neighbourhood=="LITTLE ITALY" | neighbourhood=="PELHAM BAY") %>%
  ggplot(aes(neighbourhood, salePrice)) + geom_boxplot()

# zip code
# create dataframe of avg salePrice by zip code
zipAvg <- sales %>% group_by(zipCode) %>%
  summarize(mean = mean(salePrice)) 
# find most expensive zip
zipMaxMin <- zipAvg[which.max(zipAvg$mean), ]
# find cheapest zip
zipMaxMin <- rbind(zipMaxMin, zipAvg[which.min(zipAvg$mean), ])
# boxplot of 10282 and 10105
sales %>% filter(zipCode==10282 | zipCode==10105) %>%
  ggplot(aes(zipCode, salePrice)) + geom_boxplot() 

# landSqft
# remove outliers (same method as in data wrangling)
landClean <- removeOutliers(16, sales)
# smooth plot of average sale prices by landSqft w/o outliers
landPrice <- landClean%>%
  ggplot(aes(landSqft, salePrice)) +
  geom_smooth() # clear trend but a bit bumpy
# if we don't subset we get the opposite trend due to the extremities
sales %>%
  ggplot(aes(landSqft, salePrice)) +
  geom_smooth()

# grossSqft
# remove outliers (same method as in data wrangling)
grossClean <- removeOutliers(17, sales)
# smooth plot of average sale prices by landSqft w/o outliers
grossPrice <- grossClean%>%
  ggplot(aes(grossSqft, salePrice)) +
  geom_smooth() # clearer trend

# year built
#smooth plot by yearBuilt
yearPrice <- sales %>% filter(yearBuilt >=1800) %>%ggplot(aes(yearBuilt, salePrice)) +
  geom_smooth() +
  scale_x_continuous(limits = c(1800, 2017)) #somewhat identifiable trend


# PARTITIONING
# split into volidation and training sets
nrow(sales) # 53388 without 0s and NAs. enough occurances to not expect to have to do ten-fold cross validation

# validation set will be 20% of the data set according to Pareto's principle
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = sales$salePrice, times = 1, p = 0.2, list = FALSE)
train <- sales[-test_index,]
validation <- sales[test_index,]

# function to create a test set and training set from the larger train set
createSet <- function (df, n){
  samp <- sample_n(df, n)
  test_index <- createDataPartition(y = samp$salePrice, times = 1, p = 0.2, 
                                    list = FALSE)
  train_set <- samp[-test_index,]
  test_set <- samp[test_index,]
  return(list(train_set, test_set))
}


#MODEL TRAINING (pure regression)
# create test and training set
set <- createSet(train, nrow(train))
train_set <- as.data.frame(set[1])
test_set <- as.data.frame(set[2])

# NAIVE RMSE. 6161066 w/o regularization
mu <- mean(train_set$salePrice)

# BOROUGH BIAS. 6100210 w/o regularization
borough_avgs <- train_set %>% 
  group_by(borough) %>% 
  summarize(b_b = mean(salePrice - mu))
# NAs to 0
borough_avgs$b_b[is.na(borough_avgs$b_b)] <- 0
#add borough biases to train_set for future use
train_set <- train_set %>% left_join(borough_avgs, by='borough')


# NEIGHBOURHOOD BIAS. 6099938 w/o regularization
neighbourhood_avgs <- train_set %>% 
  group_by(neighbourhood) %>% 
  summarize(b_n = mean(salePrice - mu - b_b))
# NAs to 0
neighbourhood_avgs$b_n[is.na(neighbourhood_avgs$b_n)] <- 0
#add neighbourhood biases to train_set for future use
train_set <- train_set %>% left_join(neighbourhood_avgs, by='neighbourhood')


# ZIP CODE BIAS. 6082665 w/o regularization
zip_avgs <- train_set %>% 
  group_by(zipCode) %>% 
  summarize(b_z = mean(salePrice - mu - b_b - b_n))
# NAs to 0
zip_avgs$b_z[is.na(zip_avgs$b_z)] <- 0
#add zip code biases to train_set for future use
train_set <- train_set %>% left_join(zip_avgs, by='zipCode')


# LAND SQFT BIAS. 3762000 w/o regularization. seems to be a very important factor in pricing.
land_avgs <- train_set %>% 
  group_by(landSqft) %>% 
  summarize(b_l = mean(salePrice - mu - b_b - b_n - b_z))
# NAs to 0
land_avgs$b_l[is.na(land_avgs$b_l)] <- 0
#add zip code biases to train_set for future use
train_set <- train_set %>% left_join(land_avgs, by='landSqft')


# GROSS SQFT BIAS. 3543360 w/o regularization. slightly smaller effect than land sqft.
gross_avgs <- train_set %>% 
  group_by(grossSqft) %>% 
  summarize(b_g = mean(salePrice - mu - b_b - b_n - b_z - b_l))
# NAs to 0
gross_avgs$b_g[is.na(gross_avgs$b_g)] <- 0
#add zip code biases to train_set for future use
train_set <- train_set %>% left_join(gross_avgs, by='grossSqft')


# YEAR BUILT BIAS. 3516079 w/o regularization
year_avgs <- train_set %>% 
  group_by(yearBuilt) %>% 
  summarize(b_y = mean(salePrice - mu - b_b - b_n - b_z - b_l - b_g))
# NAs to 0
year_avgs$b_y[is.na(year_avgs$b_y)] <- 0
#add year biases to train_set for future use
train_set <- train_set %>% left_join(year_avgs, by='yearBuilt')

#creating predicted prices based on bias information
predicted_price <- test_set %>% 
  left_join(borough_avgs, by = "borough") %>%
  left_join(neighbourhood_avgs, by = "neighbourhood") %>%
  left_join(zip_avgs, by = "zipCode") %>%
  left_join(land_avgs, by = "landSqft") %>%
  left_join(gross_avgs, by = "grossSqft") %>%
  left_join(year_avgs, by = "yearBuilt") %>%
  mutate(pred = mu + b_b + b_n + b_z + b_l + b_g + b_y) %>%
  pull(pred)

RMSE(predicted_price, test_set$salePrice, na.rm = TRUE) #3516079


# MODEL TRAINING (regularized)
# create test and training sets
set <- createSet(train, nrow(train))
train_set <- as.data.frame(set[1])
test_set <- as.data.frame(set[2])

lambdas <- seq(0, 125, 1)

regularization <- function(l, train_set, test_set){
  
  # NAIVE RMSE. 6161066 w/o regularization
  mu <- mean(train_set$salePrice)
  
  # BOROUGH BIAS. 6100210 w/o regularization
  borough_avgs <- train_set %>% 
    group_by(borough) %>% 
    summarize(b_b = sum(salePrice - mu)/(n() + l))
  # NAs to 0
  borough_avgs$b_b[is.na(borough_avgs$b_b)] <- 0
  #add borough biases to train_set for future use
  train_set <- train_set %>% left_join(borough_avgs, by='borough')
  
  
  # NEIGHBOURHOOD BIAS. 6099938 w/o regularization
  neighbourhood_avgs <- train_set %>% 
    group_by(neighbourhood) %>% 
    summarize(b_n = sum(salePrice - mu - b_b)/(n() + l))
  # NAs to 0
  neighbourhood_avgs$b_n[is.na(neighbourhood_avgs$b_n)] <- 0
  #add neighbourhood biases to train_set for future use
  train_set <- train_set %>% left_join(neighbourhood_avgs, by='neighbourhood')
  
  
  # ZIP CODE BIAS. 6082665 w/o regularization
  zip_avgs <- train_set %>% 
    group_by(zipCode) %>% 
    summarize(b_z = sum(salePrice - mu - b_b - b_n)/(n() + l))
  # NAs to 0
  zip_avgs$b_z[is.na(zip_avgs$b_z)] <- 0
  #add zip code biases to train_set for future use
  train_set <- train_set %>% left_join(zip_avgs, by='zipCode')
  
  
  # LAND SQFT BIAS. 3762000 w/o regularization. seems to be a very important factor in pricing.
  land_avgs <- train_set %>% 
    group_by(landSqft) %>% 
    summarize(b_l = sum(salePrice - mu - b_b - b_n - b_z)/(n() + l))
  # NAs to 0
  land_avgs$b_l[is.na(land_avgs$b_l)] <- 0
  #add zip code biases to train_set for future use
  train_set <- train_set %>% left_join(land_avgs, by='landSqft')
  
  
  # GROSS SQFT BIAS. 3543360 w/o regularization. slightly smaller effect than land sqft.
  gross_avgs <- train_set %>% 
    group_by(grossSqft) %>% 
    summarize(b_g = sum(salePrice - mu - b_b - b_n - b_z - b_l)/(n() + l))
  # NAs to 0
  gross_avgs$b_g[is.na(gross_avgs$b_g)] <- 0
  #add zip code biases to train_set for future use
  train_set <- train_set %>% left_join(gross_avgs, by='grossSqft')
  
  
  # YEAR BUILT BIAS. 3516079 w/o regularization
  year_avgs <- train_set %>% 
    group_by(yearBuilt) %>% 
    summarize(b_y = sum(salePrice - mu - b_b - b_n - b_z - b_l - b_g)/(n() + l))
  # NAs to 0
  year_avgs$b_y[is.na(year_avgs$b_y)] <- 0
  #add zip code biases to train_set for future use
  train_set <- train_set %>% left_join(year_avgs, by='yearBuilt')
  
  #creating predicted prices based on bias information
  predicted_price <- test_set %>% 
    left_join(borough_avgs, by = "borough") %>%
    left_join(neighbourhood_avgs, by = "neighbourhood") %>%
    left_join(zip_avgs, by = "zipCode") %>%
    left_join(land_avgs, by = "landSqft") %>%
    left_join(gross_avgs, by = "grossSqft") %>%
    left_join(year_avgs, by = "yearBuilt") %>%
    mutate(pred = mu + b_b + b_n + b_z + b_l + b_g + b_y) %>%
    pull(pred)
  
  RMSE(predicted_price, test_set$salePrice, na.rm = TRUE)
}
# apply regularization to find lambda value with the lowest resulting RMSE. Please allow a few minutes for it to run.
rmses <- sapply(lambdas, regularization, train_set=train_set, test_set=test_set)
min(rmses) 
lambda <- lambdas[which.min(rmses)]
lambda
regularization(lambda, train, validation) #RMSE: 2709453