# Install all R packages required for this project
# Run this script in R or RStudio to set up your R environment

cat("Installing R packages for Meme Stock Detection project...\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

packages <- c(
  "tidyverse",
  "lubridate",
  "purrr",
  "tidymodels",
  "slider",
  "torch"
)

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(paste0("\nInstalling ", pkg, "...\n"))
    install.packages(pkg, dependencies = TRUE)
    cat(paste0("✓ ", pkg, " installed successfully\n"))
  } else {
    cat(paste0("✓ ", pkg, " already installed\n"))
  }
}

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("All R packages installed successfully!\n")
cat(paste(rep("=", 60), collapse = ""), "\n")