# Implementations of a set of univariate forecasting models
#
# Each function takes 2 parameters
# time_series - a ts object representing the time series that should be used with model training
# forecast_horizon - expected forecast horizon
#
# If a model fails to provide forecasts, it will return snaive forecasts

#utils::install.packages("reticulate")

#library("reticulate")
options(repos = c(
                    techtonique = "https://r-packages.techtonique.net",
                    CRAN = "https://cloud.r-project.org"
                ))

install.packages("ahead", source = TRUE, repos = "https://r-packages.techtonique.net")

library("ahead")
library("forecast")
library("MASS")

reticulate::py_install(c("nnetsauce", "scikit-learn", "numpy", "lightgbm"))

ns <- reticulate::import("nnetsauce")
sklearn_linear_model <- reticulate::import("sklearn.linear_model")
sklearn_ensemble <- reticulate::import("sklearn.ensemble")
sklearn_network <- reticulate::import("sklearn.neural_network")
sklearn_neighbors <- reticulate::import("sklearn.neighbors")
sklearn_tree <- reticulate::import("sklearn.tree")
sklearn_gaussian_process <- reticulate::import("sklearn.gaussian_process")
sklearn_kernel <- reticulate::import("sklearn.kernel_ridge")
np <- reticulate::import("numpy")
lgb <- reticulate::import("lightgbm")

get_nsmodel_forecasts <- function(time_series, forecast_horizon, model = NULL) {
  tryCatch({
    # Import required Python libraries 
    model_choice <- switch(
      model,
      "ElasticNetCV" = "ElasticNetCV()",
      "RidgeCV" = "RidgeCV()",
      "LassoCV" = "LassoCV()",
      "LassoLarsCV" = "LassoLarsCV()",
      "ElasticNet" = "ElasticNet()",
      "Ridge" = "Ridge()",
      "Lasso" = "Lasso()",
      "LinearRegression" = "LinearRegression()",
      "AdaBoostRegressor" = "AdaBoostRegressor()",
      "GradientBoostingRegressor" = "GradientBoostingRegressor()",
      "RandomRegressor" = "RandomForestRegressor()",
      "BaggingRegressor" = "BaggingRegressor()",
      "MLPRegressor" = "MLPRegressor()",
      "KernelRidgeRegressor" = "KernelRidgeRegressor()",
      "GaussianProcessRegressor" = "GaussianProcessRegressor()",
      "KNeighborsRegressor" = "KNeighborsRegressor()",
      "LGBMRegressor" = "LGBMRegressor(verbosity=-1L)",
      stop("Invalid model choice")
    )
    # Use the mapped model_choice instead of direct model name
    if (model %in% c("ElasticNetCV", "RidgeCV", "LassoCV", "LassoLarsCV", "ElasticNet", "Ridge", "Lasso", "LinearRegression")) {      
      regr <- eval(parse(text = paste0("sklearn_linear_model$", model_choice)))
    }

    if (model %in% c("AdaBoostRegressor", "GradientBoostingRegresssor", 
    "RandomRegressor", "BaggingRegressor")) {
      regr <- eval(parse(text = paste0("sklearn_ensemble$", model_choice)))
    }

    if (model == "MLPRegressor") {
      regr <- eval(parse(text = paste0("sklearn_network$", model_choice)))
    }

    if (model == "GaussianProcessRegressor") {
      regr <- eval(parse(text = paste0("sklearn_gaussian_process$", model_choice)))
    }

    if (model == "KNeighborsRegressor") {
      regr <- eval(parse(text = paste0("sklearn_neighbors$", model_choice)))
    }

    if (model == "KernelRidgeRegressor") {
      regr <- eval(parse(text = paste0("sklearn_kernelridge$", model_choice)))
    }

    if (model == "LGBMRegressor") {      
      regr <- eval(parse(text = paste0("lgb$", model_choice)))
    }    
    
    # Use max 15 lags, but no more than series length
    ts_length <- length(time_series)
    min_lags <- min(15L, ts_length - forecast_horizon)   
    
    if (ts_length < (min_lags + forecast_horizon)) {
      return(get_snaive_forecasts(time_series, forecast_horizon))
    } 
    # Initialize the MTS model with adaptive lags
    model <- ns$MTS(regr, 
    lags=as.integer(floor(min_lags)), 
    show_progress=FALSE)
    # Ensure values are numeric and convert to a NumPy array
    values <- np$array(as.numeric(time_series))
    values_reshaped <- np$reshape(values, c(-1L, 1L))
    # Fit and predict
    model$fit(values_reshaped)
    predictions <- model$predict(h = as.integer(forecast_horizon))
    # Return predictions as a time series
    ts(predictions, frequency = frequency(time_series))
  }, error = function(e) {
    warning(sprintf("nsmodel error with %s: %s", model, e$message))
    return(get_snaive_forecasts(time_series, forecast_horizon))
  })
}

get_lgb_forecasts <- function(time_series, forecast_horizon) {
  #regr <- lgb$LGBMRegressor()
}

# Calculate rlmtheta forecasts
get_rlmtheta_forecasts <- function(time_series, forecast_horizon){
  tryCatch(suppressWarnings(ahead::glmthetaf(time_series, h=forecast_horizon, fit_func=MASS::rlm, attention = TRUE, type_pi = "conformal-split", method = "adj")$mean),
  error = function(e) {
    warning(e)
    get_snaive_forecasts(time_series, forecast_horizon)
  })
}

# Calculate ets forecasts
get_ets_forecasts <- function(time_series, forecast_horizon){
  tryCatch(
    forecast(forecast:::ets(time_series), h = forecast_horizon)$mean
  ,error = function(e) {
    warning(e)
    get_snaive_forecasts(time_series, forecast_horizon)
  })
}


# Calculate simple exponential smoothing forecasts
get_ses_forecasts <- function(time_series, forecast_horizon){
  tryCatch(
    forecast(forecast:::ses(time_series, h = forecast_horizon))$mean
  , error = function(e) {   
    warning(e)
    get_snaive_forecasts(time_series, forecast_horizon)
  })
}


# Calculate theta forecasts
get_theta_forecasts <-function(time_series, forecast_horizon){
  tryCatch(
    forecast:::thetaf(y = time_series, h = forecast_horizon)$mean
  , error = function(e) {   
    warning(e)
    get_snaive_forecasts(time_series, forecast_horizon)
  })
}


# Calculate auto.arima forecasts
get_arima_forecasts <- function(time_series, forecast_horizon, model = NULL){
  if(is.null(model)){
    tryCatch({
      fit <- forecast:::auto.arima(time_series, lambda = 0)
    }, error = function(e) {
        tryCatch({
          fit <<- forecast:::auto.arima(time_series)
        }, error = function(e){
            fit <<- forecast:::auto.arima(time_series, seasonal = FALSE)
        })
    })
    
    tryCatch({
      f <- forecast:::forecast.Arima(fit, h = forecast_horizon)$mean
      list(f, fit)
    }, error = function(e) { 
        warning(e)
        f <- get_snaive_forecasts(time_series, forecast_horizon)
        list(f, fit)
    })
  }else{
    tryCatch(
      forecast(forecast:::Arima(time_series, model = model), h = forecast_horizon)$mean
    , error = function(e) {   
        warning(e)
        get_snaive_forecasts(time_series, forecast_horizon)
    })
  }
}


# Calculate tbats forecasts
get_tbats_forecasts <- function(time_series, forecast_horizon){
  tryCatch(
    forecast(forecast:::tbats(time_series), h = forecast_horizon)$mean
  , error = function(e) {   
    warning(e)
    get_snaive_forecasts(time_series, forecast_horizon)
  })
}


# Calculate dynamic harmonic regression arima forecasts
get_dhr_arima_forecasts <- function(time_series, forecast_horizon, model = NULL){
  if(is.null(model)){
    tryCatch({
      xreg <- forecast:::fourier(time_series, K = 1)
      model <- forecast:::auto.arima(time_series, xreg = xreg, seasonal = FALSE)
      xreg1 <- forecast:::fourier(time_series, K = 1, h = forecast_horizon)
      f <- forecast(model, xreg = xreg1)$mean
      list(f, model)
    }, error = function(e) {   
      warning(e)
      f <- get_snaive_forecasts(time_series, forecast_horizon)
      list(f, model)
    })
  }else{
    tryCatch({
      xreg <- forecast:::fourier(time_series, K = 1)
      xreg1 <- forecast:::fourier(time_series, K = 1, h = forecast_horizon)
      forecast(forecast:::Arima(time_series, model = model, xreg = xreg), xreg = xreg1)$mean
    }, error = function(e) {   
      warning(e)
      get_snaive_forecasts(time_series, forecast_horizon)
    })
  }
}


# Calculate snaive forecasts
get_snaive_forecasts <- function(time_series, forecast_horizon){
  forecast:::snaive(time_series, h = forecast_horizon)$mean
}
