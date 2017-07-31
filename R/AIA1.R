# AIA1.R

library(tensorflow)
library(reticulate)
library(feather)

# Create Data
lf <- list.files("/home/thernandez/FeatherAIA2014/")
astropy <- import("astropy")

# Creates an index of date-times with all 8 channels and lists them as a matrix
# One row for time, one col for channel
all.channel.index.POSIX <- as.POSIXct(read.csv("/home/thernandez/AIA_index_allChannels.csv",
                                               stringsAsFactors = FALSE)[[1]],
                                      tz = "UTC")

all.channel.index <- outer(format(all.channel.index.POSIX, "%Y%m%d_%H%M"),
                           c("_0094", "_0131", "_0171", "_0193",
                             "_0211", "_0304", "_0335", "_1600"),
                           paste, sep = "")

y.mat <- read.csv("/home/thernandez/Flux_2010_2017_allY.csv", stringsAsFactors = F)
y.mat$Time <- as.POSIXct(y.mat$Time, tz = "UTC")
good.inds <- which(all.channel.index.POSIX %in% y.mat$Time)
y.mat <- y.mat[good.inds, ]
good.inds2 <- which(!is.na(y.mat[, "Flux"]))
y.mat <- y.mat[good.inds2, ]

# FIXX!!!!!!!!
# Double check things all line up.
# if(length(good.inds) != nrow(all.channel.index)){stop("Not same!!!")}

# Extracts matrices from fits file
fits2mat <- function(filename){
  temp <- astropy$io$fits$open(filename)
  temp$verify("fix")
  exp.time <- as.numeric(substring(strsplit(as.character(temp[[1]]$header),
                                            "EXPTIME")[[1]][2], 4, 12))
  temp.mat <- temp[[1]]$data
  temp.mat[temp.mat <= 0] <- 1
  log(t(temp.mat / exp.time))
}

###### Example usage:
temp.x <- fits2mat(filename = paste0("/home/thernandez/AIA2014/", lf[2]))
image(temp.x, zlim = c(-1, 5))
######

# Creates a 3D matrix of all 8 channels
indexTo3Dmat <- function(channel.index, channels.used = c(3), feather = TRUE){
  # array(., dim = c(1024, 1024, length(channels.used)))
  if(feather == TRUE){
    temp.mat <- as.matrix(read_feather(paste0("/home/thernandez/FeatherAIA2014/",
                                              channel.index)))[, channels.used]
  } else {
    temp.mat <- matrix(NA, nrow = 1024 ^ 2, ncol = length(channels.used))
    for(i in 1:length(channels.used)){
      temp.mat[, i] <- unlist(fits2mat(paste0("/home/thernandez/AIA2014/AIA",
                                              channel.index[channels.used],
                                              ".fits")))
    }
  }
  c(temp.mat)
}

###### Example usage:
# temp = indexTo3Dmat(channel.index = all.channel.index[1, ])
# temp2 = apply(temp, c(1, 2), mean)
# image(temp2, zlim = c(10, 100))
######
# Creates collection of 3D input matrices and corresponding Flux outputs
# Use for each to parallelize
mini.batch <- function(indices = 1:5,
                       target.y = c("Flux", "Flux_1h_1h12m",
                                    "Flux_1h_2h", "Flux_1h_1d1h_top1"),
                       normalization.x = TRUE, normalization.y = FALSE,
                       log.x = TRUE, log.y = TRUE,
                       channels.used = c(3), feather = TRUE){
  x = matrix(NA, nrow = length(indices),
             ncol = 1024 ^ 2 * length(channels.used))
  y = as.matrix(y.mat[indices, target.y])
  # cat("\n", dim(x))
  for(i in 1:length(indices)){
    x[i, ] <- c(indexTo3Dmat(channel.index = lf[indices[i]], #format(all.channel.index.POSIX, "%Y%m%d_%H%M"),
                             channels.used = channels.used))
  }
  if(log.x == TRUE){
    x <- log(x)
  }
  if(normalization.x == TRUE){
    x <- matrix(scale(c(x)), nrow = length(indices))
  }
  if(log.y == TRUE){
    y <- log(y)
  }
  if(normalization.y == TRUE){
    y <- matrix(scale(y), nrow = length(indices))
  }
  mb <- list(x = x, y = y)
}

###### Example usage:
temp = mini.batch(1:5, "Flux")
#############################################################
#############################################################
sess <- tf$InteractiveSession()

x <- tf$placeholder(tf$float32, shape(NULL, as.integer(1048576)))
y <- tf$placeholder(tf$float32, shape(NULL, 1L))

weight_variable <- function(shape) {
  initial <- tf$truncated_normal(shape, stddev=0.1)
  tf$Variable(initial)
}

bias_variable <- function(shape) {
  initial <- tf$constant(0.1, shape=shape)
  tf$Variable(initial)
}

conv2d <- function(x, W, strideX = 50, strideY = 50) {
  tf$nn$conv2d(x, W,
               strides=c(1L, as.integer(strideX), as.integer(strideY), 1L),
               padding='SAME')
}

max_pool_2x2 <- function(x, ksizeX = 50, ksizeY = 50,
                         strideX = 50, strideY = 50) {
  tf$nn$max_pool(
    x, 
    ksize=c(1L, as.integer(ksizeX), as.integer(ksizeY), 1L),
    strides=c(1L, as.integer(strideX), as.integer(strideY), 1L), 
    padding='SAME')
}

x_image <- tf$reshape(x, shape(-1L, 1024L, 1024L, 1L))

#############################################################
# Need to switch this all in-memory for speed
#############################################################
CV.mat <- as.data.frame(matrix(0, nrow = 100, ncol = 21))
colnames(CV.mat) <- c("LogRMSE", "RMSE",
                      "learning.rate", "fcl.num.units", "batch.size",
                      "conv.window.size1", "conv.depth1",
                      "conv.window.size2", "conv.depth2",
                      "strideXconv2D1", "strideYconv2D1",
                      "ksizeX1", "ksizeY1", "strideX1", "strideY1",
                      "strideXconv2D2", "strideYconv2D2", "ksizeX2", "ksizeY2",
                      "strideX2", "strideY2")
set.seed(1)
CV.mat[, "learning.rate"] <- 2 ^ -sample(8:15, nrow(CV.mat), replace = T)
CV.mat[, "fcl.num.units"] <- sample(32:256, nrow(CV.mat), replace = T)
CV.mat[, "batch.size"] <- sample(16:64, nrow(CV.mat), replace = T)
CV.mat[, "conv.window.size1"] <- sample(32:64, nrow(CV.mat), replace = T)
CV.mat[, "conv.depth1"] <- sample(8:64, nrow(CV.mat), replace = T)
CV.mat[, "conv.window.size2"] <- sample(16:128, nrow(CV.mat), replace = T)
CV.mat[, "conv.depth2"] <- sample(32:128, nrow(CV.mat), replace = T)
CV.mat[, "strideXconv2D1"] <- sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "strideYconv2D1"] <- sample(8:128, nrow(CV.mat), replace = T)
CV.mat[, "ksizeX1"] = sample(8:128, nrow(CV.mat), replace = T)
CV.mat[, "ksizeY1"] = sample(16:128, nrow(CV.mat), replace = T)
CV.mat[, "strideX1"] = sample(16:128, nrow(CV.mat), replace = T)
CV.mat[, "strideY1"] = sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "strideXconv2D2"] <- sample(8:128, nrow(CV.mat), replace = T)
CV.mat[, "strideYconv2D2"] <- sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "ksizeX2"] = sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "ksizeY2"] = sample(8:64, nrow(CV.mat), replace = T)
CV.mat[, "strideX2"] = sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "strideY2"] = sample(2:64, nrow(CV.mat), replace = T)

# test.batch <- mini.batch(indices = 2821:3525, target.y = "Flux")

for(i in 1:nrow(CV.mat)){
  
  cat(paste0(colnames(CV.mat), ":", CV.mat[i, ]), "\n")
  set.seed(1)
  conv.window.size1 <- as.integer(CV.mat$conv.window.size1[i])
  conv.depth1 <- as.integer(CV.mat$conv.depth1[i])
  conv.window.size2 <- as.integer(CV.mat$conv.window.size2[i])
  conv.depth2 <- as.integer(CV.mat$conv.depth2[i])
  strideXconv2D1 <- as.integer(CV.mat$strideXconv2D1[i])
  strideYconv2D1 <- as.integer(CV.mat$strideYconv2D1[i])
  ksizeX1 = as.integer(CV.mat$ksizeX1[i])
  ksizeY1 = as.integer(CV.mat$ksizeY1[i])
  strideX1 = as.integer(CV.mat$strideX1[i])
  strideY1 = as.integer(CV.mat$strideY1[i])
  strideXconv2D2 <- as.integer(CV.mat$strideXconv2D2[i])
  strideYconv2D2 <- as.integer(CV.mat$strideYconv2D2[i])
  ksizeX2 = as.integer(CV.mat$ksizeX2[i])
  ksizeY2 = as.integer(CV.mat$ksizeY2[i])
  strideX2 = as.integer(CV.mat$strideX2[i])
  strideY2 = as.integer(CV.mat$strideY2[i])
  fcl.num.units <- as.integer(CV.mat$fcl.num.units[i])
  batch.size <- as.integer(CV.mat$batch.size[i])

  W_conv1 <- weight_variable(shape(conv.window.size1, conv.window.size1, 1L,
                                   conv.depth1))
  b_conv1 <- bias_variable(shape(conv.depth1))
  
  h_conv1 <- tf$nn$relu(conv2d(x_image, W_conv1,
                               strideX = strideXconv2D1,
                               strideY = strideYconv2D1) + b_conv1)
  h_pool1 <- max_pool_2x2(h_conv1, ksizeX = ksizeX1, ksizeY = ksizeY1,
                          strideX = strideX1, strideY = strideY1)
  
  W_conv2 <- weight_variable(shape = shape(conv.window.size2, conv.window.size2,
                                           conv.depth1, conv.depth2))
  b_conv2 <- bias_variable(shape = shape(conv.depth2))
  
  h_conv2 <- tf$nn$relu(conv2d(h_pool1, W_conv2,
                               strideX = strideXconv2D2,
                               strideY = strideYconv2D2) + b_conv2)
  h_pool2 <- max_pool_2x2(h_conv2, ksizeX = ksizeX2, ksizeY = ksizeY2,
                          strideX = strideX2, strideY = strideY2)
  
  wfc1.shape <- as.integer(as.character(h_pool2$get_shape()[[1]]))
  wfc2.shape <- as.integer(as.character(h_pool2$get_shape()[[2]]))
  wfc3.shape <- as.integer(as.character(h_pool2$get_shape()[[3]]))
  
  W_fc1 <- weight_variable(shape(wfc1.shape * wfc2.shape * wfc3.shape, fcl.num.units))
  b_fc1 <- bias_variable(shape(fcl.num.units))
  
  h_pool2_flat <- tf$reshape(h_pool2, shape(-1L, wfc1.shape *
                                              wfc2.shape *
                                              wfc3.shape))
  h_fc1 <- tf$nn$relu(tf$matmul(h_pool2_flat, W_fc1) + b_fc1)
  
  keep_prob <- tf$placeholder(tf$float32)
  h_fc1_drop <- tf$nn$dropout(h_fc1, keep_prob)
  
  W_fc2 <- weight_variable(shape(fcl.num.units, 1L))
  b_fc2 <- bias_variable(shape(1L))
  
  y_conv <- tf$matmul(h_fc1_drop, W_fc2) + b_fc2
  
  loss <- tf$reduce_mean((y - y_conv) ^ 2)
  optimizer <- tf$train$AdamOptimizer(learning_rate = CV.mat[i, "learning.rate"]) # GradientDescentOptimizer(0.05) #
  train <- optimizer$minimize(loss)
  
  sess$run(tf$global_variables_initializer())
  
  train.mat <- matrix(0, nrow = 2820/batch.size, ncol = 2)
  colnames(train.mat) <- c("LogRMSE", "RMSE")
  test.err.calc <- TRUE

  for (j in 1:(2820/batch.size)) {
    batch <- mini.batch(indices = (j - 1) * batch.size + 1:batch.size,
                        target.y = "Flux")
    # cat(".")
    train$run(feed_dict = dict(x = batch[[1]], y = batch[[2]], keep_prob = 0.5))

    train_accuracy <- loss$eval(feed_dict = dict(x = batch[[1]],
                                                 y = batch[[2]],
                                                 keep_prob = 1.0))
    train_accuracy_exp <- y_conv$eval(feed_dict = dict(x = batch[[1]],
                                                       y = batch[[2]],
                                                       keep_prob = 1.0))
    
    train.mat[j, "LogRMSE"] <- round(sqrt(train_accuracy), 8)
    train.mat[j, "RMSE"] <- round(sqrt(mean((exp(batch[[2]]) -
                                              exp(train_accuracy_exp)) ^ 2)), 8)

    cat(paste0("Step ", j, " LogRMSE: ", train.mat[j, "LogRMSE"],
               " | RMSE: ", train.mat[j, "RMSE"], "\n"))
    if(round(train_accuracy, 8) == Inf | is.na(train_accuracy) | round(train_accuracy, 8) == 0){
      test.err.calc <- FALSE
      break()
    }
  }
  write.csv(train.mat, paste0("CNNtrainPars", i, ".csv"))
  #######################################################
  # Test against test set
  # Need batches because of memory
  #######################################################
  if(!exists("test.batch")){
    test.batch <- mini.batch(indices = 2821:3525, target.y = "Flux")
  }
  num.test.samples <- nrow(test.batch[[1]])
  y_hat <- c()
  expDiffs <- c()
  Diffs <- c()
  for(j in 1:ceiling(num.test.samples / 200)){
    cat("Testing batch", j, "of", ceiling(num.test.samples / 200), "\n")
    inds <- 1:200 + (j - 1) * 200
    if(inds[length(inds)] > nrow(test.batch[[1]])){
      inds <- inds[-which(inds > num.test.samples)]
    }
    testY <- y_conv$eval(feed_dict = dict(x = test.batch[[1]][inds, ],
                                          y = test.batch[[2]][inds, 1,
                                                              drop = F],
                                          keep_prob = 1.0))
    y_hat <- c(y_hat, testY)
    expDiffs <- c(expDiffs, (exp(testY) - exp(test.batch[[2]])[inds, ]))
    Diffs <- c(Diffs, testY - test.batch[[2]][inds, ])
  }
  # plot(test.batch[[2]], y_hat)
  
  CV.mat[i, "RMSE"] <- sqrt(mean(expDiffs ^ 2))
  CV.mat[i, "LogRMSE"] <- sqrt(mean(Diffs ^ 2))
  cat("RMSE: ", CV.mat[i, "RMSE"], "LogRMSE: ", CV.mat[i, "LogRMSE"], "\n" )
}

write.csv(CV.mat, paste0("CNNcvPars_", Sys.Date(), ".csv"), row.names = FALSE)

################# CV.mat Analysis #################################



CV.mat2 <- as.data.frame(CV.mat[order(CV.mat[, "TestMSE"]), -2])
# CV.mat2$ProxyMSE <- 0
# CV.mat2$ProxyMSE[which(CV.mat2$TestMSE > 1 & CV.mat2$TestMSE < 1.797693e+308)] <- 1
# CV.mat2$ProxyMSE[which(CV.mat2$TestMSE == 1.797693e+308)] <- 2
# CV.mat3 <- CV.mat2[, -1]
fit <- lm(TestMSE~., data = CV.mat2)
sort(fit$coefficients)

#    batch.size conv.window.size2           ksizeY1          strideY2 conv.window.size1       conv.depth1 
# -0.0613137391     -0.0307389430     -0.0270840946     -0.0172186164     -0.0131294311     -0.0130427407 
#      strideX1       conv.depth2           ksizeY2    strideXconv2D1    strideYconv2D1          strideX2 
# -0.0108363520     -0.0104540655     -0.0078710257     -0.0057511216     -0.0001315979      0.0016007235 
#      ksizeX2          strideY1     fcl.num.units           ksizeX1    strideYconv2D2    strideXconv2D2 
# 0.0033911206      0.0048058836      0.0055628938      0.0103174749      0.0127500333      0.0186606277 
#  (Intercept) 
# 4.7224334879 