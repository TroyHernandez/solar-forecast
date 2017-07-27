# AIA1.R

library(tensorflow)
library(reticulate)

# Create Data
lf <- list.files("AIA2014")
astropy <- import("astropy")

# Creates an index of date-times with all 8 channels and lists them as a matrix
# One row for time, one col for channel
all.channel.index.POSIX <- as.POSIXct(read.csv("/data/sw/AIA_index_allChannels.csv",
                                               stringsAsFactors = FALSE)[[1]],
                                      tz = "UTC")

all.channel.index <- outer(format(all.channel.index.POSIX, "%Y%m%d_%H%M"),
                           c("_0094", "_0131", "_0171", "_0193",
                             "_0211", "_0304", "_0335", "_1600"),
                           paste, sep = "")

y.mat <- read.csv("Flux_2010_2017_allY.csv", stringsAsFactors = F)
y.mat$Time <- as.POSIXct(y.mat$Time, tz = "UTC")
good.inds <- which(all.channel.index.POSIX %in% y.mat$Time)
y.mat <- y.mat[good.inds, ]
good.inds2 <- which(!is.na(y.mat[, "Flux"]))
y.mat <- y.mat[good.inds2, ]

# FIXX!!!!!!!!
# Double check things all line up.
if(length(good.inds) != nrow(all.channel.index)){stop("Not same!!!")}

# Extracts matrices from fits file
fits2mat <- function(filename){
  temp <- astropy$io$fits$open(filename)
  temp$verify("fix")
  exp.time <- as.numeric(substring(strsplit(as.character(temp[[1]]$header),
                                            "EXPTIME")[[1]][2], 4, 12))
  t(temp[[1]]$data / exp.time)
}

###### Example usage:
temp.x <- fits2mat(filename = paste0("AIA2014/", lf[1079]))
image(temp.x, zlim = c(-1, 2))
######

# Creates a 3D matrix of all 8 channels
indexTo3Dmat <- function(channel.index, channels.used = c(3)){
  # array(., dim = c(1024, 1024, length(channels.used)))
  temp.mat <- matrix(NA, nrow = 1024 ^ 2, ncol = length(channels.used))
  for(i in 1:length(channels.used)){
    temp.mat[, i] <- unlist(fits2mat(paste0("/home/thernandez/AIA2014/AIA",
                                       channel.index[channels.used],
                                       ".fits")))
  }
  c(temp.mat)
}

###### Example usage:
temp = indexTo3Dmat(channel.index = all.channel.index[1, ])
temp2 = apply(temp, c(1, 2), mean)
# image(temp2, zlim = c(10, 100))
######
# Creates collection of 3D input matrices and corresponding Flux outputs
# Use for each to parallelize
mini.batch <- function(indices = 1:5,
                       target.y = c("Flux", "Flux_1h_1h12m",
                                    "Flux_1h_2h", "Flux_1h_1d1h_top1"),
                       normalization = FALSE,
                       channels.used = c(3)){
  x = matrix(NA, nrow = length(indices),
             ncol = 1024 ^ 2 * length(channels.used))
  y = as.matrix(y.mat[indices, target.y])
  # cat("\n", dim(x))
  for(i in 1:length(indices)){
    x[i, ] <- c(indexTo3Dmat(channel.index = all.channel.index[indices[i], ],
                             channels.used = channels.used))
  }
  if(normalization == TRUE){
    x <- matrix(scale(x), nrow = length(indices))
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
#############################################################
CV.mat <- matrix(0, nrow = 25, ncol = 20)
colnames(CV.mat) <- c("TestMSE", "TrainMSE", "conv.window.size1", "conv.depth1",
                      "conv.window.size2", "conv.depth2",
                      "strideXconv2D1", "strideYconv2D1",
                      "ksizeX1", "ksizeY1", "strideX1", "strideY1",
                      "strideXconv2D2", "strideYconv2D2", "ksizeX2", "ksizeY2",
                      "strideX2", "strideY2",
                      "fcl.num.units", "batch.size")
set.seed(1)
CV.mat[, "conv.window.size1"] <- sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "conv.depth1"] <- sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "conv.window.size2"] <- sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "conv.depth2"] <- sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "strideXconv2D1"] <- sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "strideYconv2D1"] <- sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "ksizeX1"] = sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "ksizeY1"] = sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "strideX1"] = sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "strideY1"] = sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "strideXconv2D2"] <- sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "strideYconv2D2"] <- sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "ksizeX2"] = sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "ksizeY2"] = sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "strideX2"] = sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "strideY2"] = sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "fcl.num.units"] <- sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "batch.size"] <- sample(2:64, nrow(CV.mat), replace = T)

for(i in 3:nrow(CV.mat)){
  conv.window.size1 <- as.integer(CV.mat[i, 3])
  conv.depth1 <- as.integer(CV.mat[i, 4])
  conv.window.size2 <- as.integer(CV.mat[i, 5])
  conv.depth2 <- as.integer(CV.mat[i, 6])
  strideXconv2D1 <- as.integer(CV.mat[i, 7])
  strideYconv2D1 <- as.integer(CV.mat[i, 8])
  ksizeX1 = as.integer(CV.mat[i, 9])
  ksizeY1 = as.integer(CV.mat[i, 10])
  strideX1 = as.integer(CV.mat[i, 11])
  strideY1 = as.integer(CV.mat[i, 12])
  strideXconv2D2 <- as.integer(CV.mat[i, 13])
  strideYconv2D2 <- as.integer(CV.mat[i, 14])
  ksizeX2 = as.integer(CV.mat[i, 15])
  ksizeY2 = as.integer(CV.mat[i, 16])
  strideX2 = as.integer(CV.mat[i, 17])
  strideY2 = as.integer(CV.mat[i, 18])
  fcl.num.units <- as.integer(CV.mat[i, 19])
  batch.size <- as.integer(CV.mat[i, 20])

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
  optimizer <- tf$train$GradientDescentOptimizer(0.5)
  train <- optimizer$minimize(loss)
  
  sess$run(tf$global_variables_initializer())
  
  train.mat <- matrix(0, nrow = 10000/batch.size, ncol = 2)
  colnames(train.mat) <- c("MSE", "MeanFlux")
  test.err.calc <- TRUE

  for (j in 1:(10000/batch.size)) {
    batch <- mini.batch(indices = (j - 1) * batch.size + 1:batch.size,
                        target.y = "Flux", normalization = TRUE)
    # cat(".")
    train$run(feed_dict = dict(x = batch[[1]], y = batch[[2]], keep_prob = 0.5))

    train_accuracy <- loss$eval(feed_dict = dict(x = batch[[1]],
                                                 y = batch[[2]],
                                                 keep_prob = 1.0))
    train.mat[j, "MSE"] <- round(sqrt(train_accuracy), 8)
    train.mat[j, "MeanFlux"] <- round(mean(batch[[2]]), 8)

    cat(paste0("Step ", j, " MSE: ", round(train_accuracy, 8),
               " | Mean Flux: ", mean(batch[[2]]), "\n"))
    if(round(train_accuracy, 8) == Inf | is.na(train_accuracy)){
      test.err.calc <- FALSE
      break()
    }
  }
  write.csv(train.mat, paste0("CNNtrainPars", i, ".csv"))
  
  # if(test.err.calc == TRUE){
    batch <- mini.batch(indices = 10001:12811,
                        target.y = "Flux", normalization = FALSE)
    testY <- y_conv$eval(feed_dict = dict(x = batch[[1]],
                                          y = batch[[2]],
                                          keep_prob = 1.0))
    CV.mat$TestMSE[i] <- sqrt(mean((testY - batch[[2]]) ^ 2))
    CV.mat[i, "TrainMSE"] <- round(train_accuracy, 8)
  # }
}

write.csv(CV.mat, "CNNcvPars.csv", row.names = FALSE)