# ==============================================================================
# LLM-Powered Sentiment Analysis for Predicting Meme Stock Alerts
# ==============================================================================
#
# This script implements a Human-Centered Prompting pipeline for meme stock
# prediction using Reddit data and ChatGPT-5 Nano sentiment analysis.
#
# Human-Centered Prompting Framework:
# 1. Problem Understanding: Meme stocks exhibit elevated volatility and trading
#    volumes driven by social media hype rather than fundamentals.
# 2. Solution Understanding: Multi-dimensional LLM sentiment scores capture
#    nuanced emotional signals (hype, FOMO, fear) better than traditional methods.
# 3. Prompt Engineering: External prompt template allows iterative refinement
#    based on expert feedback and performance metrics.
# 4. Prompt Evaluation: Model performance on held-out test set measures prompt
#    effectiveness in capturing predictive sentiment signals.
# 5. Iteration & Refinement: Modify prompt_sentiment_template.txt and rerun
#    pipeline to improve performance.
#
# ==============================================================================

# ------------------------------------------------------------------------------
# 0. SETUP
# ------------------------------------------------------------------------------

# Load required packages
suppressPackageStartupMessages({
  library(tidyverse)    # Data manipulation and visualization
  library(lubridate)    # Date handling
  library(jsonlite)     # JSON parsing
  library(httr)         # HTTP requests for API calls
  library(glue)         # String interpolation
  library(tidymodels)   # ML framework

  # Optional packages - load if available
  if (requireNamespace("vip", quietly = TRUE)) {
    library(vip)        # Variable importance (optional)
  }
  if (requireNamespace("keras", quietly = TRUE)) {
    library(keras)      # Neural networks (optional)
  }
  if (requireNamespace("furrr", quietly = TRUE)) {
  library(furrr)     # Parallel map
} else {
  warning("Package 'furrr' not installed - install.packages('furrr') for parallel scoring.")
}
})

# Set seed for reproducibility
set.seed(123)

# Read OpenAI API key from environment
# IMPORTANT: In production, use Sys.getenv(). For this project, we'll set it directly.
# Normally you would set this in .Renviron file
api_key <- Sys.getenv("OPENAI_API_KEY")
if (api_key == "") {
  # For this project, set directly (NOT recommended for production)
  api_key <- "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"
  message("Using hardcoded API key. In production, set OPENAI_API_KEY environment variable.")
}

# Define API endpoint
base_url <- "https://api.openai.com/v1/chat/completions"

# Load LLM prompt template
# This template will be designed and refined through Human-Centered Prompting
# Students will iterate on this file during the project
llm_prompt_template_file <- "prompt_sentiment_template.txt"

if (!file.exists(llm_prompt_template_file)) {
  warning(glue("Prompt template file '{llm_prompt_template_file}' not found. ",
               "Creating a default template. Please refine this through the ",
               "Human-Centered Prompting process."))

  # Create default template (students will refine this)
  default_prompt <- 'You are an expert financial sentiment analyst specializing in social media analysis of meme stocks.

Analyze the following Reddit post and return ONLY a valid JSON object with sentiment scores.

Return exactly this JSON structure with numeric values in [-1.0000, 1.0000] and binary flags (0 or 1):

{
  "sent_hype": 0.0000,
  "sent_fomo": 0.0000,
  "sent_fear": 0.0000,
  "sent_panic": 0.0000,
  "sent_sarcasm": 0.0000,
  "sent_confidence": 0.0000,
  "sent_anger": 0.0000,
  "sent_regret": 0.0000,
  "has_rocket_emoji": 0,
  "has_moon_emoji": 0,
  "has_diamond_emoji": 0,
  "has_money_emoji": 0
}

POST TEXT:
{POST_TEXT}

Return ONLY the JSON object, no additional text.'

  write_file(default_prompt, llm_prompt_template_file)
}

llm_prompt_template <- read_file(llm_prompt_template_file)

message("✓ Setup complete. API key loaded, prompt template ready.")

# ------------------------------------------------------------------------------
# 1. LOAD & CLEAN REDDIT DATA
# ------------------------------------------------------------------------------

#' Load and clean Reddit data
#'
#' @param file_path Path to Reddit CSV file
#' @param min_text_length Minimum text length to keep (default: 10)
#' @param target_subreddits Optional vector of subreddits to filter to
#' @return Cleaned tibble with Reddit posts
load_and_clean_reddit <- function(file_path = "dataset_meme_reddit_historical_1.csv",
                                   min_text_length = 10,
                                   target_subreddits = NULL) {

  message("Loading Reddit data from: ", file_path)

  df <- read_csv(file_path, show_col_types = FALSE)

  message(glue("  Loaded {nrow(df)} posts"))

  # Convert created_utc to POSIXct if it's numeric (Unix timestamp)
  if (is.numeric(df$created_utc)) {
    df <- df %>%
      mutate(created_dt = as_datetime(created_utc, tz = "UTC"))
  } else if (is.character(df$created_dt)) {
    df <- df %>%
      mutate(created_dt = as_datetime(created_dt, tz = "UTC"))
  }

  # Add date column
  df <- df %>%
    mutate(date = as_date(created_dt))

  # Handle missing values in title and selftext
  df <- df %>%
    mutate(
      title = replace_na(title, ""),
      selftext = replace_na(selftext, ""),
      text = paste(title, selftext, sep = "\n\n")
    )

  # Filter short/empty posts
  df <- df %>%
    filter(nchar(text) >= min_text_length)

  message(glue("  After removing short posts: {nrow(df)} posts"))

  # Optional: filter to target subreddits
  if (!is.null(target_subreddits)) {
    df <- df %>%
      filter(subreddit %in% target_subreddits)
    message(glue("  After subreddit filter: {nrow(df)} posts"))
  }

  message("✓ Reddit data cleaned")

  return(df)
}

# ------------------------------------------------------------------------------
# 2. EXTRACT TICKERS FROM TEXT
# ------------------------------------------------------------------------------

#' Extract stock tickers from text
#'
#' Finds ticker symbols in text (e.g., $GME or GME as standalone word)
#' and matches against valid tradeable tickers
#'
#' @param text Character vector of text to search
#' @param tradeable_tickers Character vector of valid ticker symbols
#' @return Character vector of matched tickers
extract_tickers <- function(text, tradeable_tickers) {

  if (is.na(text) || text == "") return(character(0))

  # Pattern 1: $TICKER format
  dollar_tickers <- str_extract_all(text, "\\$[A-Z]{1,5}\\b")[[1]]
  dollar_tickers <- str_remove(dollar_tickers, "\\$")

  # Pattern 2: Standalone uppercase words (2-5 letters)
  # Split on word boundaries and extract uppercase sequences
  words <- str_split(text, "\\s+")[[1]]
  word_tickers <- str_extract(words, "^[A-Z]{2,5}$")
  word_tickers <- word_tickers[!is.na(word_tickers)]

  # Combine and deduplicate
  all_tickers <- unique(c(dollar_tickers, word_tickers))

  # Filter to tradeable tickers only
  valid_tickers <- intersect(all_tickers, tradeable_tickers)

  return(valid_tickers)
}

#' Extract tickers from all posts and unnest to post-ticker pairs
#'
#' @param df_posts Tibble of posts with 'text' column
#' @param tradeable_tickers Character vector of valid tickers
#' @return Tibble with one row per (post_id, ticker) pair
extract_and_unnest_tickers <- function(df_posts, tradeable_tickers) {

  message("Extracting tickers from posts...")

  df_posts <- df_posts %>%
    mutate(tickers = map(text, ~extract_tickers(.x, tradeable_tickers)))

  # Count posts with tickers
  n_with_tickers <- sum(map_int(df_posts$tickers, length) > 0)
  message(glue("  Posts mentioning valid tickers: {n_with_tickers}"))

  # Unnest to post-ticker level and rename column
  df_posts_ticker <- df_posts %>%
    filter(map_int(tickers, length) > 0) %>%
    unnest(tickers, names_repair = "check_unique") %>%
    rename(ticker = tickers)  # Rename tickers (plural) to ticker (singular)

  message(glue("  Total post-ticker pairs: {nrow(df_posts_ticker)}"))
  message("✓ Ticker extraction complete")

  return(df_posts_ticker)
}

# ------------------------------------------------------------------------------
# 3. CHATGPT API CALLS (SCORING POSTS)
# ------------------------------------------------------------------------------

#' Low-level ChatGPT API call
#'
#' @param messages List of message objects for Chat API
#' @param model Model identifier (default: "gpt-4o-mini" for GPT-5 Nano)
#' @param api_key OpenAI API key
#' @param base_url API endpoint URL
#' @param max_tokens Maximum tokens in response
#' @param temperature Sampling temperature
#' @return Character string of assistant response, or NA on error
call_chatgpt_raw <- function(messages,
                              model = "gpt-4o-mini",
                              api_key,
                              base_url,
                              max_tokens = 150,
                              temperature = 0.0) {

  # Construct request body
  body <- list(
    model = model,
    messages = messages,
    max_tokens = max_tokens,
    temperature = temperature
  )

  # Make API call with error handling
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
    warning("HTTP request failed: ", e$message)
    return(NULL)
  })

  if (is.null(response)) return(NA_character_)

  # Check status
  if (status_code(response) != 200) {
    warning("API returned status ", status_code(response))
    return(NA_character_)
  }

  # Parse response
  content <- tryCatch({
    content(response, "parsed", type = "application/json")
  }, error = function(e) {
    warning("Failed to parse response: ", e$message)
    return(NULL)
  })

  if (is.null(content)) return(NA_character_)

  # Extract assistant message
  if (!is.null(content$choices) && length(content$choices) > 0) {
    return(content$choices[[1]]$message$content)
  }

  return(NA_character_)
}

#' Build messages for LLM using prompt template
#'
#' @param text Post text to analyze
#' @param prompt_template Template string with {POST_TEXT} placeholder
#' @return List of message objects for Chat API
build_llm_messages <- function(text, prompt_template = llm_prompt_template) {

  # Replace placeholder with actual text
  user_prompt <- str_replace(prompt_template, "\\{POST_TEXT\\}", text)

  # Return messages format for Chat API
  messages <- list(
    list(role = "user", content = user_prompt)
  )

  return(messages)
}

#' Score a single post with LLM
#'
#' @param text Post text
#' @param api_key OpenAI API key
#' @param base_url API endpoint
#' @param prompt_template Prompt template string
#' @return One-row tibble with sentiment scores
score_post_with_llm <- function(text,
                                 api_key,
                                 base_url,
                                 prompt_template = llm_prompt_template) {

  # Truncate very long posts (GPT-4o-mini has token limits)
  if (nchar(text) > 3000) {
    text <- str_sub(text, 1, 3000)
  }

  # Build messages
  messages <- build_llm_messages(text, prompt_template)

  # Call API
  response_text <- call_chatgpt_raw(
    messages = messages,
    model = "gpt-4o-mini",  # GPT-5 Nano equivalent
    api_key = api_key,
    base_url = base_url
  )

  # Default scores (if parsing fails)
  default_scores <- tibble(
    sent_hype = NA_real_,
    sent_fomo = NA_real_,
    sent_fear = NA_real_,
    sent_panic = NA_real_,
    sent_sarcasm = NA_real_,
    sent_confidence = NA_real_,
    sent_anger = NA_real_,
    sent_regret = NA_real_,
    has_rocket_emoji = NA_integer_,
    has_moon_emoji = NA_integer_,
    has_diamond_emoji = NA_integer_,
    has_money_emoji = NA_integer_
  )

  if (is.na(response_text)) {
    return(default_scores)
  }

  # Parse JSON response
  scores <- tryCatch({
    # Extract JSON from response (in case there's extra text)
    # Use more flexible regex to handle nested braces
    json_match <- str_extract(response_text, "\\{[^{}]*(?:\\{[^{}]*\\}[^{}]*)*\\}")
    if (is.na(json_match)) {
      # Try to parse entire response
      parsed <- fromJSON(response_text, simplifyVector = FALSE)
    } else {
      parsed <- fromJSON(json_match, simplifyVector = FALSE)
    }

    # Convert to tibble with expected columns
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
    warning("JSON parsing failed: ", e$message)
    default_scores
  })

  return(scores)
}

#' Score posts in batch with progress reporting
#'
#' @param df Tibble of posts with 'text' column
#' @param text_col Name of text column (default: "text")
#' @param max_calls Maximum number of API calls (for debugging/cost control)
#' @param api_key OpenAI API key
#' @param base_url API endpoint
#' @return Tibble with original columns plus LLM scores

score_posts_batch <- function(df,
                               text_col = "text",
                               max_calls = Inf,
                               api_key,
                               base_url,
                               workers = 16) {   # <– how many parallel R processes

  message(glue("Scoring {min(nrow(df), max_calls)} posts with ChatGPT..."))

  # Limit to max_calls for cost control
  if (nrow(df) > max_calls) {
    message(glue("  Limiting to first {max_calls} posts"))
    df <- df %>% slice(1:max_calls)
  }

  n_total <- nrow(df)

  if (!requireNamespace("furrr", quietly = TRUE)) {
    warning("furrr not available – falling back to sequential scoring.")
    # Your original sequential loop:
    scores_list <- vector("list", n_total)
    for (i in seq_len(n_total)) {
      if (i %% 10 == 0) {
        message(glue("  Progress: {i}/{n_total} ({round(100*i/n_total)}%)"))
      }
      scores_list[[i]] <- score_post_with_llm(
        text = df[[text_col]][i],
        api_key = api_key,
        base_url = base_url,
        prompt_template = llm_prompt_template
      )
    }
  } else {
    # ---- PARALLEL VERSION ----
    message(glue("  Using parallel scoring with {workers} workers"))

    future::plan(future::multisession, workers = workers)

    # Parallel map over indices; each worker calls your existing scorer
    scores_list <- furrr::future_map(
      seq_len(n_total),
      function(i) {
        score_post_with_llm(
          text = df[[text_col]][i],
          api_key = api_key,
          base_url = base_url,
          prompt_template = llm_prompt_template
        )
      },
      .progress = TRUE
    )

    # After scoring, you may want to reset the plan in long sessions:
    future::plan(future::sequential)
  }

  # Bind scores to original data
  scores_df <- bind_rows(scores_list)
  df_scored <- bind_cols(df, scores_df)

  # Report scoring success rate
  n_failed <- sum(is.na(scores_df$sent_hype))
  success_rate <- round(100 * (n_total - n_failed) / n_total, 1)
  message(glue("  Successfully scored: {n_total - n_failed}/{n_total} ({success_rate}%)"))
  if (n_failed > 0) {
    message(glue("  Failed scores: {n_failed} (will be imputed with neutral values)"))
  }
  message("✓ LLM scoring complete")

  return(df_scored)
}

# ------------------------------------------------------------------------------
# 4. AGGREGATE LLM FEATURES TO (TICKER, DATE)
# ------------------------------------------------------------------------------

#' Aggregate post-level scores to daily ticker level
#'
#' @param df_posts_scored Tibble with post-level LLM scores
#' @return Tibble with one row per (ticker, date)
aggregate_reddit_to_daily <- function(df_posts_scored) {

  message("Aggregating Reddit features to (ticker, date) level...")

  df_daily <- df_posts_scored %>%
    group_by(ticker, date) %>%
    summarise(
      # Post volume
      num_posts = n(),

      # Sentiment: mean and max (replace NaN with 0 for days with no valid scores)
      mean_sent_hype = if_else(is.nan(mean(sent_hype, na.rm = TRUE)), 0, mean(sent_hype, na.rm = TRUE)),
      max_sent_hype = if_else(is.infinite(max(sent_hype, na.rm = TRUE)), 0, max(sent_hype, na.rm = TRUE)),
      mean_sent_fomo = if_else(is.nan(mean(sent_fomo, na.rm = TRUE)), 0, mean(sent_fomo, na.rm = TRUE)),
      max_sent_fomo = if_else(is.infinite(max(sent_fomo, na.rm = TRUE)), 0, max(sent_fomo, na.rm = TRUE)),
      mean_sent_fear = if_else(is.nan(mean(sent_fear, na.rm = TRUE)), 0, mean(sent_fear, na.rm = TRUE)),
      mean_sent_panic = if_else(is.nan(mean(sent_panic, na.rm = TRUE)), 0, mean(sent_panic, na.rm = TRUE)),
      mean_sent_sarcasm = if_else(is.nan(mean(sent_sarcasm, na.rm = TRUE)), 0, mean(sent_sarcasm, na.rm = TRUE)),
      mean_sent_confidence = if_else(is.nan(mean(sent_confidence, na.rm = TRUE)), 0, mean(sent_confidence, na.rm = TRUE)),
      mean_sent_anger = if_else(is.nan(mean(sent_anger, na.rm = TRUE)), 0, mean(sent_anger, na.rm = TRUE)),
      mean_sent_regret = if_else(is.nan(mean(sent_regret, na.rm = TRUE)), 0, mean(sent_regret, na.rm = TRUE)),

      # Share of posts with strong signals (replace NaN with 0)
      share_hype_posts = if_else(is.nan(mean(sent_hype > 0.5, na.rm = TRUE)), 0, mean(sent_hype > 0.5, na.rm = TRUE)),
      share_fomo_posts = if_else(is.nan(mean(sent_fomo > 0.5, na.rm = TRUE)), 0, mean(sent_fomo > 0.5, na.rm = TRUE)),

      # Emoji indicators (replace NaN with 0)
      share_rocket_emoji = if_else(is.nan(mean(has_rocket_emoji == 1, na.rm = TRUE)), 0, mean(has_rocket_emoji == 1, na.rm = TRUE)),
      share_moon_emoji = if_else(is.nan(mean(has_moon_emoji == 1, na.rm = TRUE)), 0, mean(has_moon_emoji == 1, na.rm = TRUE)),
      share_diamond_emoji = if_else(is.nan(mean(has_diamond_emoji == 1, na.rm = TRUE)), 0, mean(has_diamond_emoji == 1, na.rm = TRUE)),
      share_money_emoji = if_else(is.nan(mean(has_money_emoji == 1, na.rm = TRUE)), 0, mean(has_money_emoji == 1, na.rm = TRUE)),

      # Engagement metrics
      avg_score = mean(score, na.rm = TRUE),
      avg_num_comments = mean(num_comments, na.rm = TRUE),

      .groups = "drop"
    ) %>%
    # Replace any remaining Inf/-Inf/NaN with 0 (neutral)
    mutate(across(where(is.numeric), ~if_else(is.infinite(.x) | is.nan(.x), 0, .x)))

  message(glue("  Aggregated to {nrow(df_daily)} (ticker, date) rows"))
  message("✓ Aggregation complete")

  return(df_daily)
}

# ------------------------------------------------------------------------------
# 5. LOAD PRICE DATA & CREATE GOLD-STANDARD LABEL
# ------------------------------------------------------------------------------

#' Load and prepare price data with meme alert label
#'
#' @param file_path Path to price CSV
#' @param return_threshold Minimum next-day return for meme event (default: 0.10)
#' @param volume_multiplier Minimum volume spike multiple (default: 2.0)
#' @return Tibble with price data and meme_alert label
load_and_prepare_prices <- function(file_path = "dataset_yahoo_top100_most_volatile_us_stocks.csv",
                                     return_threshold = 0.10,
                                     volume_multiplier = 2.0) {

  message("Loading price data from: ", file_path)

  df <- read_csv(file_path, show_col_types = FALSE)

  message(glue("  Loaded {nrow(df)} daily observations"))

  # Ensure correct types and rename columns to match pipeline expectations
  # Dataset has: adj_close, return_log_daily, avg_20d_vol
  # Pipeline expects: close, ret, vol_ma20
  df <- df %>%
    mutate(
      ticker = str_to_upper(ticker),
      date = as_date(date),
      # Rename and convert columns to expected format
      close = as.numeric(adj_close),
      ret = as.numeric(return_log_daily),
      vol_ma20 = as.numeric(avg_20d_vol),
      volume = as.numeric(volume)
    ) %>%
    # Keep only columns needed for pipeline
    select(ticker, date, close, volume, ret, vol_ma20, everything()) %>%
    # Remove duplicate columns
    select(-any_of(c("adj_close", "return_log_daily", "avg_20d_vol"))) %>%
    arrange(ticker, date)

  message("✓ Price data loaded and columns standardized")

  return(df)
}

#' Create meme alert label based on next-day price action
#'
#' Label definition: meme_alert = 1 if:
#'   - Next-day return > return_threshold (e.g., 10%)
#'   - Next-day volume > volume_multiplier × vol_ma20 (e.g., 2x average)
#'
#' @param df_price Tibble with price data
#' @param return_threshold Return threshold for alert
#' @param volume_multiplier Volume spike multiplier
#' @return Tibble with meme_alert label
create_meme_alert_label <- function(df_price,
                                     return_threshold = 0.10,
                                     volume_multiplier = 2.0) {

  message("Creating meme alert label...")
  message(glue("  Return threshold: {return_threshold*100}%"))
  message(glue("  Volume multiplier: {volume_multiplier}x"))

  df_labeled <- df_price %>%
    group_by(ticker) %>%
    arrange(date) %>%
    mutate(
      # Lead values for next day
      ret_lead1 = lead(ret, 1),
      vol_lead1 = lead(volume, 1),

      # Create binary label
      meme_alert = if_else(
        !is.na(ret_lead1) & !is.na(vol_lead1) & !is.na(vol_ma20) &
          ret_lead1 > return_threshold &
          vol_lead1 > volume_multiplier * vol_ma20,
        1L,
        0L
      )
    ) %>%
    ungroup() %>%
    # Drop last day per ticker where lead values are NA
    filter(!is.na(ret_lead1))

  n_alerts <- sum(df_labeled$meme_alert == 1, na.rm = TRUE)
  pct_alerts <- round(100 * n_alerts / nrow(df_labeled), 2)

  message(glue("  Meme alerts: {n_alerts} ({pct_alerts}% of days)"))
  message("✓ Label creation complete")

  return(df_labeled)
}

# ------------------------------------------------------------------------------
# 6. MERGE REDDIT FEATURES WITH PRICE + LABEL
# ------------------------------------------------------------------------------

#' Merge Reddit features with labeled price data
#'
#' @param df_reddit_daily Daily Reddit features
#' @param df_price_labeled Price data with meme_alert label
#' @return Merged modeling dataset
merge_reddit_and_prices <- function(df_reddit_daily, df_price_labeled) {

  message("Merging Reddit features with price data...")

  # Inner join on ticker and date
  df_model <- inner_join(
    df_reddit_daily,
    df_price_labeled,
    by = c("ticker", "date")
  )

  message(glue("  Merged dataset: {nrow(df_model)} rows"))
  message(glue("  Unique tickers: {n_distinct(df_model$ticker)}"))
  message(glue("  Date range: {min(df_model$date)} to {max(df_model$date)}"))

  # Check for missing values
  na_counts <- df_model %>%
    summarise(across(everything(), ~sum(is.na(.x)))) %>%
    pivot_longer(everything(), names_to = "column", values_to = "na_count") %>%
    filter(na_count > 0) %>%
    arrange(desc(na_count))

  if (nrow(na_counts) > 0) {
    message("  Columns with missing values:")
    print(na_counts)
  }

  # Check for rows with missing LLM features and impute instead of removing
  llm_cols <- names(df_model)[str_detect(names(df_model), "^(sent_|has_|share_|mean_|max_)")]

  # Count NA rows for reporting
  df_model <- df_model %>%
    rowwise() %>%
    mutate(n_na_llm = sum(is.na(c_across(all_of(llm_cols))))) %>%
    ungroup()

  n_with_nas <- sum(df_model$n_na_llm > 0)
  message(glue("  Rows with missing LLM features: {n_with_nas}"))

  # Impute missing LLM features with 0 (neutral sentiment) instead of removing rows
  df_model <- df_model %>%
    mutate(across(all_of(llm_cols), ~if_else(is.na(.x), 0, .x))) %>%
    select(-n_na_llm)

  message(glue("  After imputing missing values: {nrow(df_model)} rows"))
  message("✓ Merge complete")

  return(df_model)
}

# ------------------------------------------------------------------------------
# 7. TIME-BASED TRAIN/VALIDATION/TEST SPLIT
# ------------------------------------------------------------------------------

#' Create time-based train/validation/test splits
#'
#' Splits by date to respect temporal ordering and avoid look-ahead bias
#'
#' @param df_model Modeling dataset
#' @param train_pct Percentage for training (default: 0.70)
#' @param val_pct Percentage for validation (default: 0.15)
#' @return List with train_df, val_df, test_df
make_time_splits <- function(df_model, train_pct = 0.70, val_pct = 0.15) {

  message("Creating time-based train/val/test splits...")

  # Get unique dates sorted
  dates_sorted <- sort(unique(df_model$date))
  n_dates <- length(dates_sorted)

  # Calculate split points
  train_end_idx <- floor(n_dates * train_pct)
  val_end_idx <- floor(n_dates * (train_pct + val_pct))

  train_end_date <- dates_sorted[train_end_idx]
  val_end_date <- dates_sorted[val_end_idx]

  # Split data
  train_df <- df_model %>% filter(date <= train_end_date)
  val_df <- df_model %>% filter(date > train_end_date & date <= val_end_date)
  test_df <- df_model %>% filter(date > val_end_date)

  message(glue("  Train: {nrow(train_df)} rows ({min(train_df$date)} to {max(train_df$date)})"))
  message(glue("  Val:   {nrow(val_df)} rows ({min(val_df$date)} to {max(val_df$date)})"))
  message(glue("  Test:  {nrow(test_df)} rows ({min(test_df$date)} to {max(test_df$date)})"))

  # Check class balance
  message("  Class balance (meme_alert):")
  message(glue("    Train: {sum(train_df$meme_alert==1)}/{nrow(train_df)} ({round(100*mean(train_df$meme_alert==1),1)}%)"))
  message(glue("    Val:   {sum(val_df$meme_alert==1)}/{nrow(val_df)} ({round(100*mean(val_df$meme_alert==1),1)}%)"))
  message(glue("    Test:  {sum(test_df$meme_alert==1)}/{nrow(test_df)} ({round(100*mean(test_df$meme_alert==1),1)}%)"))

  message("✓ Data splits created")

  return(list(
    train_df = train_df,
    val_df = val_df,
    test_df = test_df
  ))
}

# ------------------------------------------------------------------------------
# 8. BASELINE MODEL: LOGISTIC REGRESSION
# ------------------------------------------------------------------------------

#' Train logistic regression model using tidymodels
#'
#' @param train_df Training data
#' @param val_df Validation data
#' @return List with workflow and validation metrics
train_logistic_model <- function(train_df, val_df) {

  message("Training logistic regression model...")

  # Convert meme_alert to factor BEFORE creating recipe
  train_df <- train_df %>%
    mutate(meme_alert = factor(as.character(meme_alert), levels = c("0", "1")))

  # Define recipe
  logit_recipe <- recipe(meme_alert ~ ., data = train_df) %>%
    # Remove ID columns
    step_rm(ticker, date, ret_lead1, vol_lead1) %>%
    # Handle missing values
    step_impute_median(all_numeric_predictors()) %>%
    # Normalize predictors
    step_normalize(all_numeric_predictors()) %>%
    # Remove zero-variance predictors
    step_zv(all_predictors())

  # Define model (with regularization)
  logit_spec <- logistic_reg(penalty = 0.01, mixture = 1) %>%
    set_engine("glmnet") %>%
    set_mode("classification")

  # Create workflow
  logit_workflow <- workflow() %>%
    add_recipe(logit_recipe) %>%
    add_model(logit_spec)

  # Fit model
  message("  Fitting model...")
  logit_fit <- fit(logit_workflow, data = train_df)

  # Evaluate on validation set
  message("  Evaluating on validation set...")
  # Convert validation target to factor to match training data
  val_df <- val_df %>%
    mutate(meme_alert = factor(as.character(meme_alert), levels = c("0", "1")))

  val_pred <- predict(logit_fit, val_df, type = "prob") %>%
    bind_cols(predict(logit_fit, val_df, type = "class")) %>%
    bind_cols(val_df %>% select(meme_alert))

  # Calculate metrics
  val_metrics <- val_pred %>%
    metrics(truth = meme_alert, estimate = .pred_class, .pred_1)

  message("  Validation metrics:")
  print(val_metrics)

  message("✓ Logistic regression training complete")

  return(list(
    workflow = logit_fit,
    val_metrics = val_metrics,
    val_predictions = val_pred
  ))
}

#' Evaluate model on test set
#'
#' @param model_fit Fitted workflow
#' @param test_df Test data
#' @param model_name Name for display
#' @return List with metrics and predictions
evaluate_model <- function(model_fit, test_df, model_name = "Model") {

  message(glue("Evaluating {model_name} on test set..."))

  # Prepare test data - convert target to factor to match training
  test_df <- test_df %>%
    mutate(meme_alert = factor(as.character(meme_alert), levels = c("0", "1")))

  # Predictions
  test_pred <- predict(model_fit, test_df, type = "prob") %>%
    bind_cols(predict(model_fit, test_df, type = "class")) %>%
    bind_cols(test_df %>% select(meme_alert))

  # Metrics
  test_metrics <- test_pred %>%
    metrics(truth = meme_alert, estimate = .pred_class, .pred_1)

  # Additional metrics
  test_metrics_detailed <- bind_rows(
    test_metrics,
    test_pred %>% precision(truth = meme_alert, estimate = .pred_class),
    test_pred %>% recall(truth = meme_alert, estimate = .pred_class),
    test_pred %>% f_meas(truth = meme_alert, estimate = .pred_class)
  )

  message(glue("  {model_name} test metrics:"))
  print(test_metrics_detailed)

  # Confusion matrix
  conf_mat <- test_pred %>%
    conf_mat(truth = meme_alert, estimate = .pred_class)

  message("  Confusion matrix:")
  print(conf_mat)

  # ROC curve
  roc_curve_data <- test_pred %>%
    roc_curve(truth = meme_alert, .pred_1)

  roc_plot <- roc_curve_data %>%
    ggplot(aes(x = 1 - specificity, y = sensitivity)) +
    geom_line(color = "steelblue", linewidth = 1) +
    geom_abline(linetype = "dashed", color = "gray50") +
    labs(
      title = glue("{model_name} - ROC Curve"),
      x = "False Positive Rate (1 - Specificity)",
      y = "True Positive Rate (Sensitivity)"
    ) +
    theme_minimal()

  print(roc_plot)

  message(glue("✓ {model_name} evaluation complete"))

  return(list(
    metrics = test_metrics_detailed,
    predictions = test_pred,
    confusion_matrix = conf_mat,
    roc_curve = roc_curve_data,
    roc_plot = roc_plot
  ))
}

# ------------------------------------------------------------------------------
# 9. NEURAL NETWORK CLASSIFIER
# ------------------------------------------------------------------------------

#' Prepare data matrices for neural network
#'
#' @param train_df Training data
#' @param val_df Validation data
#' @param test_df Test data
#' @return List with prepared matrices and preprocessing info
prepare_nn_data <- function(train_df, val_df, test_df) {

  message("Preparing data for neural network...")

  # Remove ID columns and target
  feature_cols <- setdiff(
    names(train_df),
    c("ticker", "date", "meme_alert", "ret_lead1", "vol_lead1")
  )

  # Extract features
  X_train <- train_df %>% select(all_of(feature_cols))
  X_val <- val_df %>% select(all_of(feature_cols))
  X_test <- test_df %>% select(all_of(feature_cols))

  # Handle missing values
  X_train <- X_train %>%
    mutate(across(everything(), ~replace_na(.x, median(.x, na.rm = TRUE))))

  medians <- X_train %>% summarise(across(everything(), ~median(.x, na.rm = TRUE)))

  X_val <- X_val %>%
    mutate(across(everything(), ~replace_na(.x, medians[[cur_column()]])))
  X_test <- X_test %>%
    mutate(across(everything(), ~replace_na(.x, medians[[cur_column()]])))

  # Normalize
  means <- X_train %>% summarise(across(everything(), mean))
  sds <- X_train %>% summarise(across(everything(), sd))

  X_train_norm <- X_train %>%
    mutate(across(everything(), ~(.x - means[[cur_column()]]) / sds[[cur_column()]]))
  X_val_norm <- X_val %>%
    mutate(across(everything(), ~(.x - means[[cur_column()]]) / sds[[cur_column()]]))
  X_test_norm <- X_test %>%
    mutate(across(everything(), ~(.x - means[[cur_column()]]) / sds[[cur_column()]]))

  # Convert to matrices
  X_train_mat <- as.matrix(X_train_norm)
  X_val_mat <- as.matrix(X_val_norm)
  X_test_mat <- as.matrix(X_test_norm)

  # Extract labels
  y_train <- train_df$meme_alert
  y_val <- val_df$meme_alert
  y_test <- test_df$meme_alert

  message(glue("  Feature matrix shape: ({nrow(X_train_mat)}, {ncol(X_train_mat)})"))
  message("✓ Data preparation complete")

  return(list(
    X_train = X_train_mat,
    X_val = X_val_mat,
    X_test = X_test_mat,
    y_train = y_train,
    y_val = y_val,
    y_test = y_test,
    feature_cols = feature_cols,
    normalization = list(means = means, sds = sds, medians = medians)
  ))
}

#' Train neural network classifier
#'
#' @param nn_data Prepared data from prepare_nn_data()
#' @param epochs Number of training epochs
#' @param batch_size Batch size
#' @return Trained keras model
train_nn_model <- function(nn_data, epochs = 50, batch_size = 32) {

  message("Training neural network classifier...")

  # Define model architecture
  model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu", input_shape = ncol(nn_data$X_train)) %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 1, activation = "sigmoid")

  # Compile model
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss = "binary_crossentropy",
    metrics = c("accuracy")
  )

  message("  Model architecture:")
  print(model)

  # Early stopping
  early_stop <- callback_early_stopping(
    monitor = "val_loss",
    patience = 10,
    restore_best_weights = TRUE
  )

  # Train model
  message("  Training...")
  history <- model %>% fit(
    x = nn_data$X_train,
    y = nn_data$y_train,
    epochs = epochs,
    batch_size = batch_size,
    validation_data = list(nn_data$X_val, nn_data$y_val),
    callbacks = list(early_stop),
    verbose = 1
  )

  message("✓ Neural network training complete")

  return(list(
    model = model,
    history = history
  ))
}

#' Evaluate neural network on test set
#'
#' @param model Trained keras model
#' @param nn_data Prepared data
#' @return Evaluation results
evaluate_nn_model <- function(model, nn_data) {

  message("Evaluating neural network on test set...")

  # Predictions
  y_pred_prob <- model %>% predict(nn_data$X_test) %>% as.vector()
  y_pred_class <- if_else(y_pred_prob > 0.5, 1L, 0L)

  # Create predictions tibble
  test_pred <- tibble(
    .pred_1 = y_pred_prob,
    .pred_class = factor(y_pred_class, levels = c("0", "1")),
    meme_alert = factor(nn_data$y_test, levels = c("0", "1"))
  )

  # Metrics
  test_metrics <- bind_rows(
    test_pred %>% accuracy(truth = meme_alert, estimate = .pred_class),
    test_pred %>% roc_auc(truth = meme_alert, .pred_1),
    test_pred %>% precision(truth = meme_alert, estimate = .pred_class),
    test_pred %>% recall(truth = meme_alert, estimate = .pred_class),
    test_pred %>% f_meas(truth = meme_alert, estimate = .pred_class)
  )

  message("  Neural network test metrics:")
  print(test_metrics)

  # Confusion matrix
  conf_mat <- test_pred %>%
    conf_mat(truth = meme_alert, estimate = .pred_class)

  message("  Confusion matrix:")
  print(conf_mat)

  message("✓ Neural network evaluation complete")

  return(list(
    metrics = test_metrics,
    predictions = test_pred,
    confusion_matrix = conf_mat
  ))
}

# ------------------------------------------------------------------------------
# 10. SAVING OUTPUTS
# ------------------------------------------------------------------------------

#' Save all outputs (data, models, results)
#'
#' @param df_model Full modeling dataset
#' @param logit_results Logistic regression results
#' @param nn_model Neural network model
#' @param output_dir Directory for outputs
save_outputs <- function(df_model,
                          logit_results,
                          nn_model,
                          nn_data,
                          output_dir = ".") {

  message("Saving outputs...")

  # Save dataset
  write_csv(df_model, file.path(output_dir, "dataset_meme_with_llm.csv"))
  message("  Saved: dataset_meme_with_llm.csv")

  # Save logistic regression workflow
  saveRDS(logit_results$workflow, file.path(output_dir, "logistic_meme_model.rds"))
  message("  Saved: logistic_meme_model.rds")

  # Save neural network model (if available)
  if (!is.null(nn_model) && !is.null(nn_model$model)) {
    tryCatch({
      save_model_hdf5(nn_model$model, file.path(output_dir, "nn_meme_model.h5"))
      message("  Saved: nn_meme_model.h5")
    }, error = function(e) {
      message("  Skipped saving NN model (keras not available)")
    })
  }

  # Save normalization parameters for NN (if available)
  if (!is.null(nn_data) && !is.null(nn_data$normalization)) {
    saveRDS(nn_data$normalization, file.path(output_dir, "nn_normalization.rds"))
    message("  Saved: nn_normalization.rds")
  }

  message("✓ All outputs saved")
}

#' Predict meme alert for a new observation
#'
#' @param model Fitted workflow or keras model
#' @param new_row One-row tibble with features
#' @param model_type Either "logistic" or "nn"
#' @param nn_norm Normalization parameters (for NN only)
#' @return Probability of meme_alert = 1
predict_meme_alert_for_row <- function(model, new_row,
                                        model_type = "logistic",
                                        nn_norm = NULL) {

  if (model_type == "logistic") {
    pred <- predict(model, new_row, type = "prob")
    return(pred$.pred_1)
  } else if (model_type == "nn") {
    # Preprocess
    feature_cols <- setdiff(names(new_row), c("ticker", "date", "meme_alert", "ret_lead1", "vol_lead1"))
    X_new <- new_row %>% select(all_of(feature_cols))

    # Impute and normalize
    X_new <- X_new %>%
      mutate(across(everything(), ~replace_na(.x, nn_norm$medians[[cur_column()]]))) %>%
      mutate(across(everything(), ~(.x - nn_norm$means[[cur_column()]]) / nn_norm$sds[[cur_column()]]))

    X_new_mat <- as.matrix(X_new)

    pred <- model %>% predict(X_new_mat) %>% as.vector()
    return(pred)
  }
}

# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================

main <- function() {

  message("\n" %>% paste(rep("=", 80), collapse = ""))
  message("LLM-POWERED MEME STOCK PREDICTION PIPELINE")
  message(rep("=", 80) %>% paste(collapse = ""))
  message("")

  # --------------------------------------------------------------------------
  # STEP 1: Load and process Reddit data
  # --------------------------------------------------------------------------

  message("\n[STEP 1] Loading and cleaning Reddit data")
  message(rep("-", 80) %>% paste(collapse = ""))

  df_reddit <- load_and_clean_reddit(
    file_path = "dataset_meme_reddit_historical_1.csv",
    min_text_length = 10,
    target_subreddits = NULL  # NULL = use all subreddits
  )

  # --------------------------------------------------------------------------
  # STEP 2: Load price data to get valid tickers
  # --------------------------------------------------------------------------

  message("\n[STEP 2] Loading price data")
  message(rep("-", 80) %>% paste(collapse = ""))

  df_price_raw <- load_and_prepare_prices(
    file_path = "dataset_yahoo_top100_most_volatile_us_stocks.csv"
  )

  tradeable_tickers <- unique(df_price_raw$ticker)
  message(glue("  Found {length(tradeable_tickers)} tradeable tickers"))

  # --------------------------------------------------------------------------
  # STEP 3: Extract tickers from Reddit posts
  # --------------------------------------------------------------------------

  message("\n[STEP 3] Extracting tickers from posts")
  message(rep("-", 80) %>% paste(collapse = ""))

  df_posts_ticker <- extract_and_unnest_tickers(df_reddit, tradeable_tickers)

  # --------------------------------------------------------------------------
  # STEP 4: Score posts with ChatGPT (LLM sentiment analysis)
  # --------------------------------------------------------------------------

  message("\n[STEP 4] Scoring posts with ChatGPT")
  message(rep("-", 80) %>% paste(collapse = ""))
  message("Human-Centered Prompting: Using prompt_sentiment_template.txt")
  message("This template has been refined through iterative design with domain experts")

  # For demonstration, limit API calls (remove max_calls in production)
  df_posts_scored <- score_posts_batch(
  df = df_posts_ticker,
  text_col = "text",
#  batch_size = 100,
  api_key = api_key,
  base_url = base_url
)

  # --------------------------------------------------------------------------
  # STEP 5: Aggregate to (ticker, date) level
  # --------------------------------------------------------------------------

  message("\n[STEP 5] Aggregating to daily ticker level")
  message(rep("-", 80) %>% paste(collapse = ""))

  df_reddit_daily <- aggregate_reddit_to_daily(df_posts_scored)

  # --------------------------------------------------------------------------
  # STEP 6: Create meme alert label
  # --------------------------------------------------------------------------

  message("\n[STEP 6] Creating gold-standard meme alert label")
  message(rep("-", 80) %>% paste(collapse = ""))

  df_price_labeled <- create_meme_alert_label(
    df_price = df_price_raw,
    return_threshold = 0.10,
    volume_multiplier = 2.0
  )

  # --------------------------------------------------------------------------
  # STEP 7: Merge Reddit and price data
  # --------------------------------------------------------------------------

  message("\n[STEP 7] Merging Reddit features with price data")
  message(rep("-", 80) %>% paste(collapse = ""))

  df_model <- merge_reddit_and_prices(df_reddit_daily, df_price_labeled)

  # --------------------------------------------------------------------------
  # STEP 8: Create train/val/test splits
  # --------------------------------------------------------------------------

  message("\n[STEP 8] Creating time-based data splits")
  message(rep("-", 80) %>% paste(collapse = ""))

  splits <- make_time_splits(df_model, train_pct = 0.70, val_pct = 0.15)
  train_df <- splits$train_df
  val_df <- splits$val_df
  test_df <- splits$test_df

  # --------------------------------------------------------------------------
  # STEP 9: Train logistic regression model
  # --------------------------------------------------------------------------

  message("\n[STEP 9] Training logistic regression baseline")
  message(rep("-", 80) %>% paste(collapse = ""))

  logit_results <- train_logistic_model(train_df, val_df)

  # Evaluate on test set
  logit_test_results <- evaluate_model(
    model_fit = logit_results$workflow,
    test_df = test_df,
    model_name = "Logistic Regression"
  )

  # --------------------------------------------------------------------------
  # STEP 10: Train neural network model (optional)
  # --------------------------------------------------------------------------

  message("\n[STEP 10] Training neural network classifier")
  message(rep("-", 80) %>% paste(collapse = ""))

  # Check if keras/tensorflow is available
  nn_available <- requireNamespace("keras", quietly = TRUE)

  if (nn_available) {
    # Try to use keras - if TensorFlow not installed, skip gracefully
    nn_results <- tryCatch({
      nn_data <- prepare_nn_data(train_df, val_df, test_df)
      nn_model <- train_nn_model(nn_data, epochs = 50, batch_size = 32)
      nn_test <- evaluate_nn_model(nn_model$model, nn_data)
      list(data = nn_data, model = nn_model, test = nn_test)
    }, error = function(e) {
      message("  Neural network training skipped:")
      message("  ", e$message)
      message("  To enable neural networks, install TensorFlow:")
      message("    install.packages('keras')")
      message("    library(keras)")
      message("    install_keras()")
      NULL
    })

    if (!is.null(nn_results)) {
      nn_data <- nn_results$data
      nn_results <- nn_results$model
      nn_test_results <- nn_results$test
    } else {
      nn_data <- NULL
      nn_results <- NULL
      nn_test_results <- NULL
    }
  } else {
    message("  Keras package not available. Skipping neural network training.")
    message("  To enable neural networks, install keras:")
    message("    install.packages('keras')")
    nn_data <- NULL
    nn_results <- NULL
    nn_test_results <- NULL
  }

  # --------------------------------------------------------------------------
  # STEP 11: Save outputs
  # --------------------------------------------------------------------------

  message("\n[STEP 11] Saving outputs")
  message(rep("-", 80) %>% paste(collapse = ""))

  save_outputs(
    df_model = df_model,
    logit_results = logit_results,
    nn_model = nn_results,
    nn_data = nn_data,
    output_dir = "."
  )

  # --------------------------------------------------------------------------
  # FINAL SUMMARY
  # --------------------------------------------------------------------------

  message("\n" %>% paste(rep("=", 5), collapse = ""))
  message("PIPELINE COMPLETE - SUMMARY")
  message(rep("=", 5) %>% paste(collapse = ""))

  message("\nHuman-Centered Prompting Results:")
  message("  Prompt template: prompt_sentiment_template.txt")
  message("  LLM features extracted: 12 (8 sentiments + 4 emoji flags)")
  message("")
  message("Model Performance:")
  message("  Logistic Regression:")
  logit_acc <- logit_test_results$metrics %>%
    filter(.metric == "accuracy") %>%
    pull(.estimate)
  message(glue("    Accuracy: {round(logit_acc, 4)}"))

  if (!is.null(nn_test_results)) {
    message("  Neural Network:")
    nn_acc <- nn_test_results$metrics %>%
      filter(.metric == "accuracy") %>%
      pull(.estimate)
    message(glue("    Accuracy: {round(nn_acc, 4)}"))
  } else {
    message("  Neural Network: Skipped (TensorFlow not available)")
  }

  message("\nIteration & Refinement:")
  message("  To improve performance, modify prompt_sentiment_template.txt and rerun")
  message("  Compare metrics across iterations to evaluate prompt effectiveness")

  message("\n" %>% paste(rep("=", 5), collapse = ""))

  # Return results
  invisible(list(
    data = df_model,
    splits = splits,
    logit = logit_results,
    logit_test = logit_test_results,
    nn = nn_results,
    nn_test = nn_test_results
  ))
}

# ==============================================================================
# RUN PIPELINE
# ==============================================================================

if (interactive()) {
  message("\n\nTo run the full pipeline, execute: results <- main()")
  message("To run individual steps, call the functions directly.\n")
} else {
  # Run automatically if sourced non-interactively
  results <- main()
}
