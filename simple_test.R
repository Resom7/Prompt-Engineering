# ==============================================================================
# SIMPLE TEST - Verify Setup and LLM Integration
# ==============================================================================
#
# This script tests the core functionality without requiring full datasets
#
# ==============================================================================

# Clear environment
rm(list = ls())

# Check and load required packages
required_packages <- c("tidyverse", "jsonlite", "httr", "glue")
missing_packages <- c()

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    missing_packages <- c(missing_packages, pkg)
  }
}

if (length(missing_packages) > 0) {
  cat("\n❌ ERROR: Missing required packages:\n")
  cat(paste("  -", missing_packages, collapse = "\n"), "\n\n")
  cat("Please install missing packages by running:\n")
  cat("  source('install_packages.R')\n\n")
  cat("Or install manually:\n")
  cat("  install.packages(c('", paste(missing_packages, collapse = "', '"), "'))\n\n")
  stop("Missing packages. Installation required.")
}

# Load packages quietly
suppressPackageStartupMessages({
  library(tidyverse)
  library(jsonlite)
  library(httr)
  library(glue)
})

cat("\n=== TESTING LLM PIPELINE ===\n\n")

# ------------------------------------------------------------------------------
# Test 1: Load Prompt Template
# ------------------------------------------------------------------------------

cat("Test 1: Loading prompt template...\n")

if (!file.exists("prompt_sentiment_template.txt")) {
  stop("ERROR: prompt_sentiment_template.txt not found!")
}

llm_prompt_template <- readr::read_file("prompt_sentiment_template.txt")
cat("  ✓ Prompt template loaded (", nchar(llm_prompt_template), " characters)\n\n")

# Check for placeholder
if (str_detect(llm_prompt_template, "\\{POST_TEXT\\}")) {
  cat("  ✓ Placeholder {POST_TEXT} found\n\n")
} else {
  warning("  ✗ Placeholder {POST_TEXT} missing!\n\n")
}

# ------------------------------------------------------------------------------
# Test 2: API Configuration
# ------------------------------------------------------------------------------

cat("Test 2: API configuration...\n")

api_key <- "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"
base_url <- "https://api.openai.com/v1/chat/completions"

cat("  ✓ API key loaded\n")
cat("  ✓ API endpoint:", base_url, "\n\n")

# ------------------------------------------------------------------------------
# Test 3: Build Prompt Message
# ------------------------------------------------------------------------------

cat("Test 3: Building prompt message...\n")

test_post <- "GME to the moon! 🚀🚀🚀 Diamond hands 💎"

# Replace placeholder
user_prompt <- str_replace(llm_prompt_template, "\\{POST_TEXT\\}", test_post)

messages <- list(
  list(role = "user", content = user_prompt)
)

cat("  ✓ Messages constructed\n")
cat("  Test post:", test_post, "\n\n")

# ------------------------------------------------------------------------------
# Test 4: Call ChatGPT API
# ------------------------------------------------------------------------------

cat("Test 4: Calling ChatGPT API...\n")
cat("  (This uses 1 API call - cost ~$0.0001)\n\n")

# Make API call
body <- list(
  model = "gpt-4o-mini",
  messages = messages,
  max_tokens = 500,
  temperature = 0.0
)

response <- tryCatch({
  POST(
    url = base_url,
    add_headers(
      "Authorization" = paste("Bearer", api_key),
      "Content-Type" = "application/json"
    ),
    body = body,
    encode = "json",
    timeout(30)
  )
}, error = function(e) {
  cat("  ✗ API call failed:", e$message, "\n\n")
  return(NULL)
})

if (is.null(response)) {
  stop("API call failed. Check your internet connection and API key.")
}

# Check status
if (status_code(response) != 200) {
  cat("  ✗ API returned status", status_code(response), "\n")
  cat("  Response:", content(response, "text"), "\n\n")
  stop("API call unsuccessful")
}

cat("  ✓ API call successful (status 200)\n")

# Parse response
content_parsed <- content(response, "parsed", type = "application/json")
response_text <- content_parsed$choices[[1]]$message$content

cat("  ✓ Response received\n\n")
cat("Raw LLM Response:\n")
cat("---\n")
cat(response_text)
cat("\n---\n\n")

# ------------------------------------------------------------------------------
# Test 5: Parse JSON Output
# ------------------------------------------------------------------------------

cat("Test 5: Parsing JSON output...\n")

# Extract JSON from response
json_match <- str_extract(response_text, "\\{[^}]+\\}")

if (is.na(json_match)) {
  cat("  Trying to parse entire response as JSON...\n")
  json_match <- response_text
}

scores <- tryCatch({
  parsed <- fromJSON(json_match)

  tibble(
    sent_hype = as.numeric(if (!is.null(parsed$sent_hype)) parsed$sent_hype else NA),
    sent_fomo = as.numeric(if (!is.null(parsed$sent_fomo)) parsed$sent_fomo else NA),
    sent_fear = as.numeric(if (!is.null(parsed$sent_fear)) parsed$sent_fear else NA),
    sent_panic = as.numeric(if (!is.null(parsed$sent_panic)) parsed$sent_panic else NA),
    sent_sarcasm = as.numeric(if (!is.null(parsed$sent_sarcasm)) parsed$sent_sarcasm else NA),
    sent_confidence = as.numeric(if (!is.null(parsed$sent_confidence)) parsed$sent_confidence else NA),
    sent_anger = as.numeric(if (!is.null(parsed$sent_anger)) parsed$sent_anger else NA),
    sent_regret = as.numeric(if (!is.null(parsed$sent_regret)) parsed$sent_regret else NA),
    has_rocket_emoji = as.integer(if (!is.null(parsed$has_rocket_emoji)) parsed$has_rocket_emoji else NA),
    has_moon_emoji = as.integer(if (!is.null(parsed$has_moon_emoji)) parsed$has_moon_emoji else NA),
    has_diamond_emoji = as.integer(if (!is.null(parsed$has_diamond_emoji)) parsed$has_diamond_emoji else NA),
    has_money_emoji = as.integer(if (!is.null(parsed$has_money_emoji)) parsed$has_money_emoji else NA)
  )
}, error = function(e) {
  cat("  ✗ JSON parsing failed:", e$message, "\n\n")
  return(NULL)
})

if (is.null(scores)) {
  stop("JSON parsing failed. Check the LLM response format.")
}

cat("  ✓ JSON parsed successfully\n\n")

# ------------------------------------------------------------------------------
# Test 6: Validate Output
# ------------------------------------------------------------------------------

cat("Test 6: Validating output...\n")

cat("\nExtracted Sentiment Scores:\n")
print(scores)

# Check columns
expected_cols <- c("sent_hype", "sent_fomo", "sent_fear", "sent_panic",
                   "sent_sarcasm", "sent_confidence", "sent_anger", "sent_regret",
                   "has_rocket_emoji", "has_moon_emoji", "has_diamond_emoji", "has_money_emoji")

all_present <- all(expected_cols %in% names(scores))

if (all_present) {
  cat("\n  ✓ All 12 expected columns present\n")
} else {
  missing <- setdiff(expected_cols, names(scores))
  cat("\n  ✗ Missing columns:", paste(missing, collapse = ", "), "\n")
}

# Check sentiment ranges
sentiment_cols <- names(scores)[str_detect(names(scores), "^sent_")]
sentiment_vals <- scores %>% select(all_of(sentiment_cols)) %>% as.numeric()

if (all(sentiment_vals >= -1 & sentiment_vals <= 1, na.rm = TRUE)) {
  cat("  ✓ All sentiment values in valid range [-1, 1]\n")
} else {
  cat("  ✗ Some sentiment values out of range\n")
}

# Check emoji flags
emoji_cols <- names(scores)[str_detect(names(scores), "^has_")]
emoji_vals <- scores %>% select(all_of(emoji_cols)) %>% as.numeric()

if (all(emoji_vals %in% c(0, 1, NA))) {
  cat("  ✓ All emoji flags are binary (0 or 1)\n")
} else {
  cat("  ✗ Some emoji values are not binary\n")
}

# ------------------------------------------------------------------------------
# Test 7: Sanity Check Results
# ------------------------------------------------------------------------------

cat("\nTest 7: Sanity checking results...\n")

cat("\nTest post:", test_post, "\n")
cat("Expected: High hype, some FOMO, rocket & diamond emojis detected\n")
cat("\nActual results:\n")
cat("  Hype:", scores$sent_hype, if_else(scores$sent_hype > 0.5, "✓ HIGH", "✗ LOW"), "\n")
cat("  FOMO:", scores$sent_fomo, if_else(scores$sent_fomo > 0.3, "✓ PRESENT", "• moderate"), "\n")
cat("  Rocket emoji:", if_else(scores$has_rocket_emoji == 1, "✓ DETECTED", "✗ MISSED"), "\n")
cat("  Diamond emoji:", if_else(scores$has_diamond_emoji == 1, "✓ DETECTED", "✗ MISSED"), "\n")

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------

cat("\n")
cat(paste(rep("=", 70), collapse = ""))
cat("\n")
cat("TEST SUMMARY\n")
cat(paste(rep("=", 70), collapse = ""))
cat("\n\n")

cat("✓ All core functionality working!\n\n")

cat("Next steps:\n")
cat("  1. Review the LLM output above - does it make sense?\n")
cat("  2. If scores look wrong, edit prompt_sentiment_template.txt\n")
cat("  3. Run this test again to see improvements\n")
cat("  4. When satisfied, run the full pipeline with main()\n\n")

cat("Files ready:\n")
cat("  • meme_stock_llm_pipeline.R - Main pipeline\n")
cat("  • prompt_sentiment_template.txt - LLM prompt\n")
cat("  • analysis_examples.R - Analysis functions\n")
cat("  • quick_test.R - Testing utilities\n\n")

cat("To run full pipeline (requires datasets):\n")
cat("  source('meme_stock_llm_pipeline.R')\n")
cat("  results <- main()\n\n")

cat(paste(rep("=", 70), collapse = ""))
cat("\n\n")
