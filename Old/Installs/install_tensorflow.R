# ==============================================================================
# Install TensorFlow for R
# ==============================================================================
#
# This script sets up TensorFlow properly for use with the keras3 package in R.
# Run this once to configure your environment.
#

message("Installing TensorFlow for R...")
message("")

# Install required packages
if (!requireNamespace("reticulate", quietly = TRUE)) {
  message("Installing reticulate package...")
  install.packages("reticulate")
}

if (!requireNamespace("keras3", quietly = TRUE)) {
  message("Installing keras3 package...")
  install.packages("keras3")
}

library(reticulate)
library(keras3)

# Step 1: Install a proper Python version using reticulate
message("Step 1: Installing Python 3.10...")
message("This may take several minutes...")
message("")

tryCatch({
  install_python(version = "3.10:latest")
  message("Python 3.10 installed successfully!")
}, error = function(e) {
  message("Note: Python installation may already exist or failed.")
  message("Error: ", e$message)
})

message("")

# Step 2: Install TensorFlow using virtual environment
message("Step 2: Installing TensorFlow via install_keras()...")
message("This may take several minutes...")
message("")

# Install TensorFlow - this creates a virtual environment with TensorFlow
# Using "virtualenv" method with Python 3.10
tryCatch({
  install_keras(
    method = "virtualenv",
    python_version = "3.10"
  )

  message("")
  message("Installation complete!")
  message("")
  message("To test if it works, run:")
  message('  library(keras3)')
  message('  tensorflow::tf$constant("Hello TensorFlow")')

}, error = function(e) {
  message("")
  message("Installation failed with error:")
  message(e$message)
  message("")
  message("Please try one of these alternatives:")
  message("1. Install Python manually from: https://www.python.org/downloads/")
  message("   Download Python 3.10.x and make sure to check 'Add Python to PATH'")
  message("")
  message("2. Or try installing via conda:")
  message("   install_keras(method = 'conda')")
})
