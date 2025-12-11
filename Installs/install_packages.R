# ==============================================================================
# PACKAGE INSTALLATION SCRIPT
# ==============================================================================
#
# This script installs all required packages for the meme stock LLM pipeline
#
# Run this ONCE before using the pipeline
#
# ==============================================================================

cat("\n")
cat(paste(rep("=", 70), collapse = ""))
cat("\n")
cat("INSTALLING REQUIRED PACKAGES\n")
cat(paste(rep("=", 70), collapse = ""))
cat("\n\n")

# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Function to install package if not already installed
install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat("Installing", pkg, "...\n")
    install.packages(pkg, dependencies = TRUE)
    cat("✓", pkg, "installed\n\n")
  } else {
    cat("✓", pkg, "already installed\n")
  }
}

# ==============================================================================
# CORE PACKAGES
# ==============================================================================

cat("\n--- Core Data Manipulation ---\n")
install_if_missing("tidyverse")    # Data manipulation & visualization
install_if_missing("lubridate")    # Date handling

cat("\n--- API & JSON ---\n")
install_if_missing("httr")         # HTTP requests
install_if_missing("jsonlite")     # JSON parsing
install_if_missing("glue")         # String interpolation

# ==============================================================================
# MACHINE LEARNING PACKAGES
# ==============================================================================

cat("\n--- Machine Learning Framework ---\n")
install_if_missing("tidymodels")   # ML framework

# Individual tidymodels components (in case tidymodels doesn't install all)
cat("\n--- Tidymodels Components ---\n")
install_if_missing("parsnip")      # Model specification
install_if_missing("recipes")      # Feature engineering
install_if_missing("workflows")    # Model workflows
install_if_missing("yardstick")    # Model metrics
install_if_missing("rsample")      # Data splitting
install_if_missing("tune")         # Hyperparameter tuning
install_if_missing("dials")        # Parameter grids

# Additional ML packages
cat("\n--- Additional ML Tools ---\n")
install_if_missing("glmnet")       # Regularized regression

# Optional: Variable importance (may not be available for older R versions)
cat("\n--- Optional Packages ---\n")
cat("Note: 'vip' package is optional and may not be available for all R versions\n")
tryCatch({
  install_if_missing("vip")        # Variable importance (optional)
}, error = function(e) {
  cat("⚠️  vip package not available (this is OK, it's optional)\n")
  cat("   The pipeline will work without it.\n\n")
})

# ==============================================================================
# NEURAL NETWORK PACKAGES (OPTIONAL)
# ==============================================================================

cat("\n--- Neural Networks (Optional) ---\n")
cat("Note: Keras installation may take several minutes\n")

install_neural <- readline(prompt = "Install keras for neural networks? (y/n): ")

if (tolower(install_neural) == "y") {
  install_if_missing("keras")

  cat("\nInstalling Keras backend (TensorFlow)...\n")
  cat("This may take 5-10 minutes. Please be patient.\n\n")

  tryCatch({
    keras::install_keras()
    cat("✓ Keras installed successfully\n\n")
  }, error = function(e) {
    cat("✗ Keras installation failed:", e$message, "\n")
    cat("You can skip neural networks and use only logistic regression.\n\n")
  })
} else {
  cat("Skipping keras. You can install it later with:\n")
  cat("  install.packages('keras')\n")
  cat("  keras::install_keras()\n\n")
}

# ==============================================================================
# VISUALIZATION PACKAGES
# ==============================================================================

cat("\n--- Visualization ---\n")
install_if_missing("patchwork")    # Combining plots

# ==============================================================================
# VERIFICATION
# ==============================================================================

cat("\n")
cat(paste(rep("=", 70), collapse = ""))
cat("\n")
cat("VERIFYING INSTALLATION\n")
cat(paste(rep("=", 70), collapse = ""))
cat("\n\n")

required_packages <- c(
  "tidyverse", "lubridate", "httr", "jsonlite", "glue",
  "tidymodels", "parsnip", "recipes", "workflows", "yardstick",
  "glmnet", "patchwork"
)

optional_packages <- c("vip", "keras")

all_installed <- TRUE

for (pkg in required_packages) {
  if (require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat("✓", pkg, "\n")
  } else {
    cat("✗", pkg, "MISSING\n")
    all_installed <- FALSE
  }
}

cat("\nOptional packages:\n")
for (pkg in optional_packages) {
  if (require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat("✓", pkg, "\n")
  } else {
    cat("⚠️ ", pkg, "(optional - not installed)\n")
  }
}

cat("\n")

if (all_installed) {
  cat(paste(rep("=", 70), collapse = ""))
  cat("\n")
  cat("SUCCESS! All required packages installed correctly.\n")
  cat(paste(rep("=", 70), collapse = ""))
  cat("\n\n")

  cat("Note: Optional packages (vip, keras) are not required.\n")
  cat("      The pipeline will work without them.\n\n")

  cat("You can now run:\n")
  cat("  source('simple_test.R')        # Test the pipeline\n")
  cat("  source('meme_stock_llm_pipeline.R')  # Load main functions\n\n")

} else {
  cat(paste(rep("=", 70), collapse = ""))
  cat("\n")
  cat("WARNING: Some packages failed to install.\n")
  cat(paste(rep("=", 70), collapse = ""))
  cat("\n\n")

  cat("Try installing missing packages manually:\n")
  cat("  install.packages(c('tidymodels', 'glmnet', 'vip'))\n\n")
}

# ==============================================================================
# SESSION INFO
# ==============================================================================

cat("\nR Session Info:\n")
cat("R version:", R.version.string, "\n")
cat("Platform:", R.version$platform, "\n\n")

cat("Package versions:\n")
if (require("tidyverse", quietly = TRUE)) {
  cat("  tidyverse:", as.character(packageVersion("tidyverse")), "\n")
}
if (require("tidymodels", quietly = TRUE)) {
  cat("  tidymodels:", as.character(packageVersion("tidymodels")), "\n")
}
if (require("httr", quietly = TRUE)) {
  cat("  httr:", as.character(packageVersion("httr")), "\n")
}

cat("\n")
cat(paste(rep("=", 70), collapse = ""))
cat("\n")
cat("Installation complete!\n")
cat(paste(rep("=", 70), collapse = ""))
cat("\n\n")
