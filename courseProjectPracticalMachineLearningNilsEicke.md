## Practical Machine Learning
## Course Project by Nils Eicke
In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.
They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website 
here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 
For this project training data and test data were used. The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.

The goal of this project is to predict the manner in which the participants did the exercise. This is the "classe" variable in the 
training set. Almost any of the other variables are used to predict with. It is described how the model has been built, how 
cross validation has been used, what I think the expected out of sample error is, and why I made the choices I did. 
The prediction model has been also used to predict 20 different test cases. 

Firstly packages and data has been loaded into R 

```r
#load packages
install.packages("AppliedPredictiveModeling")
```

```
## Warning: package 'AppliedPredictiveModeling' is in use and will not be
## installed
```

```r
install.packages("caret")
```

```
## Warning: package 'caret' is in use and will not be installed
```

```r
install.packages("randomForest")
```

```
## Warning: package 'randomForest' is in use and will not be installed
```

```r
install.packages("e1071")
```

```
## Warning: package 'e1071' is in use and will not be installed
```

```r
library(AppliedPredictiveModeling)
library(caret)
library(e1071)
library(randomForest)

#download training data
download.file(url="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile="data/pml-training.csv")

#download test data
download.file(url="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile="data/pml-testing.csv")

#import data; set empty values to NA
training_set <- read.csv("data/pml-training.csv", na.strings=c("NA",""), header=TRUE)
testing_set <- read.csv("data/pml-testing.csv", na.strings=c("NA",""), header=TRUE)
```

Secondly data has been explored.

```r
dim(training_set)
```

```
## [1] 19622   160
```

```r
dim(testing_set)
```

```
## [1]  20 160
```

```r
countNAs <- function(x) {
    #create count NAs vector
	countNA <- names(x)
	countNA <- rbind(countNA,0)

	#determine number of NAs for each column
    for (i in 1:length(names(x))) {
        countNA[2,i] <- length(which(is.na(x[,i])))
    }
	print(countNA)
}

test_countNAs <- countNAs(testing_set)
```

```
##         [,1] [,2]        [,3]                   [,4]                  
## countNA "X"  "user_name" "raw_timestamp_part_1" "raw_timestamp_part_2"
##         "0"  "0"         "0"                    "0"                   
##         [,5]             [,6]         [,7]         [,8]       
## countNA "cvtd_timestamp" "new_window" "num_window" "roll_belt"
##         "0"              "0"          "0"          "0"        
##         [,9]         [,10]      [,11]              [,12]               
## countNA "pitch_belt" "yaw_belt" "total_accel_belt" "kurtosis_roll_belt"
##         "0"          "0"        "0"                "20"                
##         [,13]                 [,14]               [,15]               
## countNA "kurtosis_picth_belt" "kurtosis_yaw_belt" "skewness_roll_belt"
##         "20"                  "20"                "20"                
##         [,16]                  [,17]               [,18]          
## countNA "skewness_roll_belt.1" "skewness_yaw_belt" "max_roll_belt"
##         "20"                   "20"                "20"           
##         [,19]            [,20]          [,21]           [,22]           
## countNA "max_picth_belt" "max_yaw_belt" "min_roll_belt" "min_pitch_belt"
##         "20"             "20"           "20"            "20"            
##         [,23]          [,24]                 [,25]                 
## countNA "min_yaw_belt" "amplitude_roll_belt" "amplitude_pitch_belt"
##         "20"           "20"                  "20"                  
##         [,26]                [,27]                  [,28]          
## countNA "amplitude_yaw_belt" "var_total_accel_belt" "avg_roll_belt"
##         "20"                 "20"                   "20"           
##         [,29]              [,30]           [,31]           
## countNA "stddev_roll_belt" "var_roll_belt" "avg_pitch_belt"
##         "20"               "20"            "20"            
##         [,32]               [,33]            [,34]         
## countNA "stddev_pitch_belt" "var_pitch_belt" "avg_yaw_belt"
##         "20"                "20"             "20"          
##         [,35]             [,36]          [,37]          [,38]         
## countNA "stddev_yaw_belt" "var_yaw_belt" "gyros_belt_x" "gyros_belt_y"
##         "20"              "20"           "0"            "0"           
##         [,39]          [,40]          [,41]          [,42]         
## countNA "gyros_belt_z" "accel_belt_x" "accel_belt_y" "accel_belt_z"
##         "0"            "0"            "0"            "0"           
##         [,43]           [,44]           [,45]           [,46]     
## countNA "magnet_belt_x" "magnet_belt_y" "magnet_belt_z" "roll_arm"
##         "0"             "0"             "0"             "0"       
##         [,47]       [,48]     [,49]             [,50]          
## countNA "pitch_arm" "yaw_arm" "total_accel_arm" "var_accel_arm"
##         "0"         "0"       "0"               "20"           
##         [,51]          [,52]             [,53]          [,54]          
## countNA "avg_roll_arm" "stddev_roll_arm" "var_roll_arm" "avg_pitch_arm"
##         "20"           "20"              "20"           "20"           
##         [,55]              [,56]           [,57]         [,58]           
## countNA "stddev_pitch_arm" "var_pitch_arm" "avg_yaw_arm" "stddev_yaw_arm"
##         "20"               "20"            "20"          "20"            
##         [,59]         [,60]         [,61]         [,62]        
## countNA "var_yaw_arm" "gyros_arm_x" "gyros_arm_y" "gyros_arm_z"
##         "20"          "0"           "0"           "0"          
##         [,63]         [,64]         [,65]         [,66]         
## countNA "accel_arm_x" "accel_arm_y" "accel_arm_z" "magnet_arm_x"
##         "0"           "0"           "0"           "0"           
##         [,67]          [,68]          [,69]              
## countNA "magnet_arm_y" "magnet_arm_z" "kurtosis_roll_arm"
##         "0"            "0"            "20"               
##         [,70]                [,71]              [,72]              
## countNA "kurtosis_picth_arm" "kurtosis_yaw_arm" "skewness_roll_arm"
##         "20"                 "20"               "20"               
##         [,73]                [,74]              [,75]         
## countNA "skewness_pitch_arm" "skewness_yaw_arm" "max_roll_arm"
##         "20"                 "20"               "20"          
##         [,76]           [,77]         [,78]          [,79]          
## countNA "max_picth_arm" "max_yaw_arm" "min_roll_arm" "min_pitch_arm"
##         "20"            "20"          "20"           "20"           
##         [,80]         [,81]                [,82]                
## countNA "min_yaw_arm" "amplitude_roll_arm" "amplitude_pitch_arm"
##         "20"          "20"                 "20"                 
##         [,83]               [,84]           [,85]           
## countNA "amplitude_yaw_arm" "roll_dumbbell" "pitch_dumbbell"
##         "20"                "0"             "0"             
##         [,86]          [,87]                    [,88]                    
## countNA "yaw_dumbbell" "kurtosis_roll_dumbbell" "kurtosis_picth_dumbbell"
##         "0"            "20"                     "20"                     
##         [,89]                   [,90]                   
## countNA "kurtosis_yaw_dumbbell" "skewness_roll_dumbbell"
##         "20"                    "20"                    
##         [,91]                     [,92]                  
## countNA "skewness_pitch_dumbbell" "skewness_yaw_dumbbell"
##         "20"                      "20"                   
##         [,93]               [,94]                [,95]             
## countNA "max_roll_dumbbell" "max_picth_dumbbell" "max_yaw_dumbbell"
##         "20"                "20"                 "20"              
##         [,96]               [,97]                [,98]             
## countNA "min_roll_dumbbell" "min_pitch_dumbbell" "min_yaw_dumbbell"
##         "20"                "20"                 "20"              
##         [,99]                     [,100]                    
## countNA "amplitude_roll_dumbbell" "amplitude_pitch_dumbbell"
##         "20"                      "20"                      
##         [,101]                   [,102]                
## countNA "amplitude_yaw_dumbbell" "total_accel_dumbbell"
##         "20"                     "0"                   
##         [,103]               [,104]              [,105]                
## countNA "var_accel_dumbbell" "avg_roll_dumbbell" "stddev_roll_dumbbell"
##         "20"                 "20"                "20"                  
##         [,106]              [,107]               [,108]                 
## countNA "var_roll_dumbbell" "avg_pitch_dumbbell" "stddev_pitch_dumbbell"
##         "20"                "20"                 "20"                   
##         [,109]               [,110]             [,111]               
## countNA "var_pitch_dumbbell" "avg_yaw_dumbbell" "stddev_yaw_dumbbell"
##         "20"                 "20"               "20"                 
##         [,112]             [,113]             [,114]            
## countNA "var_yaw_dumbbell" "gyros_dumbbell_x" "gyros_dumbbell_y"
##         "20"               "0"                "0"               
##         [,115]             [,116]             [,117]            
## countNA "gyros_dumbbell_z" "accel_dumbbell_x" "accel_dumbbell_y"
##         "0"                "0"                "0"               
##         [,118]             [,119]              [,120]             
## countNA "accel_dumbbell_z" "magnet_dumbbell_x" "magnet_dumbbell_y"
##         "0"                "0"                 "0"                
##         [,121]              [,122]         [,123]          [,124]       
## countNA "magnet_dumbbell_z" "roll_forearm" "pitch_forearm" "yaw_forearm"
##         "0"                 "0"            "0"             "0"          
##         [,125]                  [,126]                  
## countNA "kurtosis_roll_forearm" "kurtosis_picth_forearm"
##         "20"                    "20"                    
##         [,127]                 [,128]                 
## countNA "kurtosis_yaw_forearm" "skewness_roll_forearm"
##         "20"                   "20"                   
##         [,129]                   [,130]                 [,131]            
## countNA "skewness_pitch_forearm" "skewness_yaw_forearm" "max_roll_forearm"
##         "20"                     "20"                   "20"              
##         [,132]              [,133]            [,134]            
## countNA "max_picth_forearm" "max_yaw_forearm" "min_roll_forearm"
##         "20"                "20"              "20"              
##         [,135]              [,136]            [,137]                  
## countNA "min_pitch_forearm" "min_yaw_forearm" "amplitude_roll_forearm"
##         "20"                "20"              "20"                    
##         [,138]                    [,139]                 
## countNA "amplitude_pitch_forearm" "amplitude_yaw_forearm"
##         "20"                      "20"                   
##         [,140]                [,141]              [,142]            
## countNA "total_accel_forearm" "var_accel_forearm" "avg_roll_forearm"
##         "0"                   "20"                "20"              
##         [,143]                [,144]             [,145]             
## countNA "stddev_roll_forearm" "var_roll_forearm" "avg_pitch_forearm"
##         "20"                  "20"               "20"               
##         [,146]                 [,147]              [,148]           
## countNA "stddev_pitch_forearm" "var_pitch_forearm" "avg_yaw_forearm"
##         "20"                   "20"                "20"             
##         [,149]               [,150]            [,151]           
## countNA "stddev_yaw_forearm" "var_yaw_forearm" "gyros_forearm_x"
##         "20"                 "20"              "0"              
##         [,152]            [,153]            [,154]           
## countNA "gyros_forearm_y" "gyros_forearm_z" "accel_forearm_x"
##         "0"               "0"               "0"              
##         [,155]            [,156]            [,157]            
## countNA "accel_forearm_y" "accel_forearm_z" "magnet_forearm_x"
##         "0"               "0"               "0"               
##         [,158]             [,159]             [,160]      
## countNA "magnet_forearm_y" "magnet_forearm_z" "problem_id"
##         "0"                "0"                "0"
```

```r
train_countNAs <- countNAs(training_set)
```

```
##         [,1] [,2]        [,3]                   [,4]                  
## countNA "X"  "user_name" "raw_timestamp_part_1" "raw_timestamp_part_2"
##         "0"  "0"         "0"                    "0"                   
##         [,5]             [,6]         [,7]         [,8]       
## countNA "cvtd_timestamp" "new_window" "num_window" "roll_belt"
##         "0"              "0"          "0"          "0"        
##         [,9]         [,10]      [,11]              [,12]               
## countNA "pitch_belt" "yaw_belt" "total_accel_belt" "kurtosis_roll_belt"
##         "0"          "0"        "0"                "19216"             
##         [,13]                 [,14]               [,15]               
## countNA "kurtosis_picth_belt" "kurtosis_yaw_belt" "skewness_roll_belt"
##         "19216"               "19216"             "19216"             
##         [,16]                  [,17]               [,18]          
## countNA "skewness_roll_belt.1" "skewness_yaw_belt" "max_roll_belt"
##         "19216"                "19216"             "19216"        
##         [,19]            [,20]          [,21]           [,22]           
## countNA "max_picth_belt" "max_yaw_belt" "min_roll_belt" "min_pitch_belt"
##         "19216"          "19216"        "19216"         "19216"         
##         [,23]          [,24]                 [,25]                 
## countNA "min_yaw_belt" "amplitude_roll_belt" "amplitude_pitch_belt"
##         "19216"        "19216"               "19216"               
##         [,26]                [,27]                  [,28]          
## countNA "amplitude_yaw_belt" "var_total_accel_belt" "avg_roll_belt"
##         "19216"              "19216"                "19216"        
##         [,29]              [,30]           [,31]           
## countNA "stddev_roll_belt" "var_roll_belt" "avg_pitch_belt"
##         "19216"            "19216"         "19216"         
##         [,32]               [,33]            [,34]         
## countNA "stddev_pitch_belt" "var_pitch_belt" "avg_yaw_belt"
##         "19216"             "19216"          "19216"       
##         [,35]             [,36]          [,37]          [,38]         
## countNA "stddev_yaw_belt" "var_yaw_belt" "gyros_belt_x" "gyros_belt_y"
##         "19216"           "19216"        "0"            "0"           
##         [,39]          [,40]          [,41]          [,42]         
## countNA "gyros_belt_z" "accel_belt_x" "accel_belt_y" "accel_belt_z"
##         "0"            "0"            "0"            "0"           
##         [,43]           [,44]           [,45]           [,46]     
## countNA "magnet_belt_x" "magnet_belt_y" "magnet_belt_z" "roll_arm"
##         "0"             "0"             "0"             "0"       
##         [,47]       [,48]     [,49]             [,50]          
## countNA "pitch_arm" "yaw_arm" "total_accel_arm" "var_accel_arm"
##         "0"         "0"       "0"               "19216"        
##         [,51]          [,52]             [,53]          [,54]          
## countNA "avg_roll_arm" "stddev_roll_arm" "var_roll_arm" "avg_pitch_arm"
##         "19216"        "19216"           "19216"        "19216"        
##         [,55]              [,56]           [,57]         [,58]           
## countNA "stddev_pitch_arm" "var_pitch_arm" "avg_yaw_arm" "stddev_yaw_arm"
##         "19216"            "19216"         "19216"       "19216"         
##         [,59]         [,60]         [,61]         [,62]        
## countNA "var_yaw_arm" "gyros_arm_x" "gyros_arm_y" "gyros_arm_z"
##         "19216"       "0"           "0"           "0"          
##         [,63]         [,64]         [,65]         [,66]         
## countNA "accel_arm_x" "accel_arm_y" "accel_arm_z" "magnet_arm_x"
##         "0"           "0"           "0"           "0"           
##         [,67]          [,68]          [,69]              
## countNA "magnet_arm_y" "magnet_arm_z" "kurtosis_roll_arm"
##         "0"            "0"            "19216"            
##         [,70]                [,71]              [,72]              
## countNA "kurtosis_picth_arm" "kurtosis_yaw_arm" "skewness_roll_arm"
##         "19216"              "19216"            "19216"            
##         [,73]                [,74]              [,75]         
## countNA "skewness_pitch_arm" "skewness_yaw_arm" "max_roll_arm"
##         "19216"              "19216"            "19216"       
##         [,76]           [,77]         [,78]          [,79]          
## countNA "max_picth_arm" "max_yaw_arm" "min_roll_arm" "min_pitch_arm"
##         "19216"         "19216"       "19216"        "19216"        
##         [,80]         [,81]                [,82]                
## countNA "min_yaw_arm" "amplitude_roll_arm" "amplitude_pitch_arm"
##         "19216"       "19216"              "19216"              
##         [,83]               [,84]           [,85]           
## countNA "amplitude_yaw_arm" "roll_dumbbell" "pitch_dumbbell"
##         "19216"             "0"             "0"             
##         [,86]          [,87]                    [,88]                    
## countNA "yaw_dumbbell" "kurtosis_roll_dumbbell" "kurtosis_picth_dumbbell"
##         "0"            "19216"                  "19216"                  
##         [,89]                   [,90]                   
## countNA "kurtosis_yaw_dumbbell" "skewness_roll_dumbbell"
##         "19216"                 "19216"                 
##         [,91]                     [,92]                  
## countNA "skewness_pitch_dumbbell" "skewness_yaw_dumbbell"
##         "19216"                   "19216"                
##         [,93]               [,94]                [,95]             
## countNA "max_roll_dumbbell" "max_picth_dumbbell" "max_yaw_dumbbell"
##         "19216"             "19216"              "19216"           
##         [,96]               [,97]                [,98]             
## countNA "min_roll_dumbbell" "min_pitch_dumbbell" "min_yaw_dumbbell"
##         "19216"             "19216"              "19216"           
##         [,99]                     [,100]                    
## countNA "amplitude_roll_dumbbell" "amplitude_pitch_dumbbell"
##         "19216"                   "19216"                   
##         [,101]                   [,102]                
## countNA "amplitude_yaw_dumbbell" "total_accel_dumbbell"
##         "19216"                  "0"                   
##         [,103]               [,104]              [,105]                
## countNA "var_accel_dumbbell" "avg_roll_dumbbell" "stddev_roll_dumbbell"
##         "19216"              "19216"             "19216"               
##         [,106]              [,107]               [,108]                 
## countNA "var_roll_dumbbell" "avg_pitch_dumbbell" "stddev_pitch_dumbbell"
##         "19216"             "19216"              "19216"                
##         [,109]               [,110]             [,111]               
## countNA "var_pitch_dumbbell" "avg_yaw_dumbbell" "stddev_yaw_dumbbell"
##         "19216"              "19216"            "19216"              
##         [,112]             [,113]             [,114]            
## countNA "var_yaw_dumbbell" "gyros_dumbbell_x" "gyros_dumbbell_y"
##         "19216"            "0"                "0"               
##         [,115]             [,116]             [,117]            
## countNA "gyros_dumbbell_z" "accel_dumbbell_x" "accel_dumbbell_y"
##         "0"                "0"                "0"               
##         [,118]             [,119]              [,120]             
## countNA "accel_dumbbell_z" "magnet_dumbbell_x" "magnet_dumbbell_y"
##         "0"                "0"                 "0"                
##         [,121]              [,122]         [,123]          [,124]       
## countNA "magnet_dumbbell_z" "roll_forearm" "pitch_forearm" "yaw_forearm"
##         "0"                 "0"            "0"             "0"          
##         [,125]                  [,126]                  
## countNA "kurtosis_roll_forearm" "kurtosis_picth_forearm"
##         "19216"                 "19216"                 
##         [,127]                 [,128]                 
## countNA "kurtosis_yaw_forearm" "skewness_roll_forearm"
##         "19216"                "19216"                
##         [,129]                   [,130]                 [,131]            
## countNA "skewness_pitch_forearm" "skewness_yaw_forearm" "max_roll_forearm"
##         "19216"                  "19216"                "19216"           
##         [,132]              [,133]            [,134]            
## countNA "max_picth_forearm" "max_yaw_forearm" "min_roll_forearm"
##         "19216"             "19216"           "19216"           
##         [,135]              [,136]            [,137]                  
## countNA "min_pitch_forearm" "min_yaw_forearm" "amplitude_roll_forearm"
##         "19216"             "19216"           "19216"                 
##         [,138]                    [,139]                 
## countNA "amplitude_pitch_forearm" "amplitude_yaw_forearm"
##         "19216"                   "19216"                
##         [,140]                [,141]              [,142]            
## countNA "total_accel_forearm" "var_accel_forearm" "avg_roll_forearm"
##         "0"                   "19216"             "19216"           
##         [,143]                [,144]             [,145]             
## countNA "stddev_roll_forearm" "var_roll_forearm" "avg_pitch_forearm"
##         "19216"               "19216"            "19216"            
##         [,146]                 [,147]              [,148]           
## countNA "stddev_pitch_forearm" "var_pitch_forearm" "avg_yaw_forearm"
##         "19216"                "19216"             "19216"          
##         [,149]               [,150]            [,151]           
## countNA "stddev_yaw_forearm" "var_yaw_forearm" "gyros_forearm_x"
##         "19216"              "19216"           "0"              
##         [,152]            [,153]            [,154]           
## countNA "gyros_forearm_y" "gyros_forearm_z" "accel_forearm_x"
##         "0"               "0"               "0"              
##         [,155]            [,156]            [,157]            
## countNA "accel_forearm_y" "accel_forearm_z" "magnet_forearm_x"
##         "0"               "0"               "0"               
##         [,158]             [,159]             [,160]  
## countNA "magnet_forearm_y" "magnet_forearm_z" "classe"
##         "0"                "0"                "0"
```

```r
unique(test_countNAs[2,])
```

```
## [1] "0"  "20"
```

```r
unique(train_countNAs[2,])
```

```
## [1] "0"     "19216"
```

```r
#check if columnames of test and training data are equal (except classe and problem_id)  
test_colnames <- test_countNAs[1,test_countNAs[2,] == "0"]
train_colnames <- train_countNAs[1,train_countNAs[2,] == "0"]
test_colnames
```

```
##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
##  [7] "num_window"           "roll_belt"            "pitch_belt"          
## [10] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
## [13] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [16] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [19] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [22] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [25] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [28] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [31] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [34] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [37] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [40] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [43] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [46] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [49] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [52] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [55] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [58] "magnet_forearm_y"     "magnet_forearm_z"     "problem_id"
```

```r
train_colnames
```

```
##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
##  [7] "num_window"           "roll_belt"            "pitch_belt"          
## [10] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
## [13] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [16] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [19] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [22] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [25] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [28] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [31] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [34] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [37] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [40] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [43] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [46] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [49] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [52] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [55] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [58] "magnet_forearm_y"     "magnet_forearm_z"     "classe"
```

```r
all.equal(train_colnames[1:length(train_colnames)-1], test_colnames[1:length(test_colnames)-1])
```

```
## [1] TRUE
```

There are 160 variables in the training and testing data with 19622 training data values and 20 testing data values.
Either no value of a testing data variable is NA (not available) or all 20 values of a testing data variable are NA.
Either no value of a training data variable is NA or almost all values (19216 of 19622) of a training data variable are NA.

Thirdly data has been cleaned.

```r
#NA data are removed
dim(testing_set[,names(testing_set) %in% test_colnames])
```

```
## [1] 20 60
```

```r
dim(training_set[,names(training_set) %in% train_colnames])
```

```
## [1] 19622    60
```

```r
training_subset <- training_set[,names(training_set) %in% train_colnames]
testing_subset <- testing_set[,names(testing_set) %in% test_colnames]

#the first 7 columns are not necessary
training_subset <- training_subset[,8:length(colnames(training_subset))]
testing_subset <- testing_subset[,8:length(colnames(testing_subset))]
```

Lastly the prediction model has been built.

```r
#because the dataset is too big for using random forest, we use only 20% of the dataset
set.seed(1)
ids <- createDataPartition(y=training_subset$classe, p=0.2, list=FALSE)
training_small <- training_subset[ids,]

#devide the large training_set into a training and a test set
set.seed(1)
inTrain <- createDataPartition(y=training_small$classe, p=0.7, list=FALSE)
training_train_set <- training_small[inTrain,]
testing_train_set <- training_small[-inTrain,]

#train on training set with cross validation.
set.seed(1)
modFit <- train(classe ~ ., method="rf", trControl=trainControl(method = "cv", number = 4), data=training_train_set)
print(modFit, digits=3)
```

```
## Random Forest 
## 
## 2751 samples
##   52 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## Summary of sample sizes: 2063, 2064, 2063, 2063 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa
##    2    0.945     0.930
##   27    0.947     0.933
##   52    0.943     0.928
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
#predict testing
predictions <- predict(modFit, newdata=testing_train_set)
print(confusionMatrix(predictions, testing_train_set$classe), digits=4)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 330   8   0   0   0
##          B   2 218   6   1   2
##          C   2   1 199   3   2
##          D   0   0   0 189   5
##          E   0   1   0   0 207
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9719          
##                  95% CI : (0.9608, 0.9806)
##     No Information Rate : 0.284           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9645          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9880   0.9561   0.9707   0.9793   0.9583
## Specificity            0.9905   0.9884   0.9918   0.9949   0.9990
## Pos Pred Value         0.9763   0.9520   0.9614   0.9742   0.9952
## Neg Pred Value         0.9952   0.9894   0.9938   0.9959   0.9907
## Prevalence             0.2840   0.1939   0.1743   0.1641   0.1837
## Detection Rate         0.2806   0.1854   0.1692   0.1607   0.1760
## Detection Prevalence   0.2874   0.1947   0.1760   0.1650   0.1769
## Balanced Accuracy      0.9893   0.9723   0.9812   0.9871   0.9786
```

```r
#20 testing set
print(predict(modFit, newdata=testing_subset))
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

The accuracy of mtry = 27 has the largest value: 0.947
The out of Sample Error is the error rate on the testing data set: 1 - .9719 = 0.0281
The prediction (classe) of 20 testing set is: B A B A A E D B A A B C B A E E A B B B
