setup_fixed_horizon <- function(dataset_name, method) {
  # Required directories
  dirs <- c(
    "results/fixed_horizon_forecasts",
    "results/fixed_horizon_errors",
    "results/fixed_horizon_execution_times"
  )
  
  # Create directories
  for (dir in dirs) {
    dir.create(file.path(BASE_DIR, dir), recursive = TRUE, showWarnings = FALSE)
  }
  
  # Clean up old results
  result_file <- file.path(BASE_DIR, "results", "fixed_horizon_forecasts", 
                          paste0(dataset_name, "_", method, ".txt"))
  if (file.exists(result_file)) {
    file.remove(result_file)
  }
}