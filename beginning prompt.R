# ============================================================================
# MEME STOCK PREDICTOR - Reddit Sentiment Analysis with OpenAI GPT-4
# ============================================================================
# This script analyzes Reddit posts to predict potential meme stock movements
# using OpenAI GPT-4 Turbo API for advanced sentiment and momentum analysis
# ============================================================================

library(httr)
library(jsonlite)
library(dplyr)
library(tidyr)
library(lubridate)
library(stringr)

# ============================================================================
# CONFIGURATION
# ============================================================================

# OpenAI API Configuration
API_KEY <- "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"
API_ENDPOINT <- "https://api.openai.com/v1/chat/completions"
MODEL_NAME <- "gpt-4-turbo-preview"  # Use "gpt-3.5-turbo" for lower cost

# ============================================================================
# OPTIMIZED PROMPT FOR MEME STOCK ANALYSIS
# ============================================================================
# This prompt is designed to extract actionable insights from Reddit posts
# about potential meme stock movements

create_meme_stock_prompt <- function(reddit_posts_batch) {

  # Format the Reddit posts for the prompt
  formatted_posts <- paste(
    sprintf(
      "POST %d:\nSubreddit: %s | Score: %d | Comments: %d | Date: %s\nTitle: %s\nText: %s\n---",
      seq_len(nrow(reddit_posts_batch)),
      reddit_posts_batch$subreddit,
      reddit_posts_batch$score,
      reddit_posts_batch$num_comments,
      reddit_posts_batch$created_dt,
      reddit_posts_batch$title,
      substr(reddit_posts_batch$selftext, 1, 500)  # Limit text to avoid token limits
    ),
    collapse = "\n\n"
  )

  # The main prompt - optimized for meme stock prediction
  prompt <- sprintf('You are a financial analyst specializing in retail investor sentiment analysis and meme stock prediction. Your task is to analyze Reddit posts from r/wallstreetbets and related subreddits to identify stocks with high bullish momentum potential in the next 3-7 days.

ANALYSIS FRAMEWORK:
Analyze the following Reddit posts and extract:

1. **Stock Mentions**: Identify all stock tickers, company names, or asset symbols mentioned
2. **Sentiment Intensity**: Rate bullish sentiment on a scale of 1-10 (10 = extremely bullish)
3. **Momentum Indicators**: Look for:
   - Frequency of mentions across posts
   - Use of action-oriented language (buy, YOLO, moon, rocket, etc.)
   - Community engagement (upvotes, comments)
   - Short squeeze potential mentions
   - Technical/fundamental catalysts mentioned
4. **Risk Signals**: Identify red flags (pump-and-dump language, new accounts, too-good-to-be-true claims)
5. **Temporal Urgency**: Does the post suggest immediate action or longer-term holds?

REDDIT POSTS TO ANALYZE:
%s

OUTPUT FORMAT (JSON):
Return ONLY a valid JSON object with this exact structure:
{
  "stocks_analyzed": [
    {
      "ticker": "STOCK_SYMBOL",
      "company_name": "Company Name if mentioned",
      "bullish_score": 8.5,
      "mention_count": 12,
      "sentiment_summary": "Brief summary of why this stock is gaining attention",
      "key_catalysts": ["catalyst1", "catalyst2"],
      "risk_level": "low/medium/high",
      "urgency": "immediate/short-term/long-term",
      "community_engagement_score": 7.2,
      "prediction_confidence": "high/medium/low",
      "predicted_direction": "bullish/neutral/bearish",
      "key_phrases": ["rocket emoji mentions", "short squeeze talk"]
    }
  ],
  "overall_market_sentiment": "bullish/neutral/bearish",
  "top_3_picks": ["TICKER1", "TICKER2", "TICKER3"],
  "analysis_timestamp": "%s"
}

CRITICAL INSTRUCTIONS:
- Only include stocks with bullish_score >= 6.0
- Prioritize stocks with both high sentiment AND high engagement
- Be conservative with prediction_confidence - only mark "high" if multiple strong signals align
- Exclude cryptocurrencies unless they have associated stock tickers (e.g., COIN for Coinbase)
- Return ONLY the JSON object, no additional text or explanations',
    formatted_posts,
    Sys.time()
  )

  return(prompt)
}

# ============================================================================
# API CALL FUNCTION
# ============================================================================

call_openai_api <- function(prompt, temperature = 0.3, max_tokens = 2000) {

  if (API_KEY == "") {
    stop("API_KEY not set. Please configure your OpenAI API key.")
  }

  cat(sprintf("Making API call to %s...\n", MODEL_NAME))

  response <- tryCatch({
    POST(
      url = API_ENDPOINT,
      add_headers(
        "Authorization" = paste("Bearer", API_KEY),
        "Content-Type" = "application/json"
      ),
      body = toJSON(list(
        model = MODEL_NAME,
        messages = list(
          list(
            role = "system",
            content = "You are a financial sentiment analysis expert specializing in retail trading communities and meme stock prediction."
          ),
          list(
            role = "user",
            content = prompt
          )
        ),
        temperature = temperature,
        max_tokens = max_tokens,
        response_format = list(type = "json_object")  # Force JSON response
      ), auto_unbox = TRUE),
      encode = "json"
    )
  }, error = function(e) {
    warning(paste("API call failed:", e$message))
    return(NULL)
  })

  if (is.null(response)) {
    return(NULL)
  }

  if (status_code(response) != 200) {
    error_content <- content(response, "text", encoding = "UTF-8")
    warning(paste("API returned status code:", status_code(response)))
    warning(paste("Error details:", error_content))
    return(NULL)
  }

  # Parse response
  tryCatch({
    response_content <- content(response, "text", encoding = "UTF-8")
    result <- fromJSON(response_content)

    # Extract the analysis from the response
    if (!is.null(result$choices) && length(result$choices) > 0) {
      analysis_json <- result$choices[[1]]$message$content
      analysis <- fromJSON(analysis_json)
      return(analysis)
    } else {
      warning("Unexpected API response structure")
      return(NULL)
    }
  }, error = function(e) {
    warning(paste("Failed to parse API response:", e$message))
    return(NULL)
  })
}

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

load_reddit_data <- function(file_path) {
  data <- read.csv(file_path, stringsAsFactors = FALSE)

  # Clean and prepare data
  data <- data %>%
    mutate(
      # Handle different date formats
      created_dt = tryCatch(
        as.POSIXct(created_dt, format = "%Y-%m-%d %H:%M:%S", tz = "UTC"),
        error = function(e) as.POSIXct(created_dt, tz = "UTC")
      ),
      # Handle missing or removed text
      selftext = ifelse(is.na(selftext) | selftext == "[removed]" | selftext == "[deleted]", "", selftext),
      title = ifelse(is.na(title), "", title),
      combined_text = paste(title, selftext, sep = " ")
    ) %>%
    filter(
      nchar(combined_text) > 20  # Filter out very short posts
    ) %>%
    arrange(desc(created_dt))

  cat(sprintf("Loaded and cleaned %d posts\n", nrow(data)))

  return(data)
}

# Process data in batches to avoid API token limits
analyze_reddit_data <- function(reddit_data, batch_size = 10, max_batches = NULL) {

  all_analyses <- list()
  total_batches <- ceiling(nrow(reddit_data) / batch_size)

  if (!is.null(max_batches)) {
    total_batches <- min(total_batches, max_batches)
  }

  cat(sprintf("Processing %d batches of %d posts each...\n", total_batches, batch_size))

  for (i in 1:total_batches) {
    start_idx <- (i - 1) * batch_size + 1
    end_idx <- min(i * batch_size, nrow(reddit_data))

    cat(sprintf("Analyzing batch %d/%d (posts %d-%d)...\n", i, total_batches, start_idx, end_idx))

    batch <- reddit_data[start_idx:end_idx, ]
    prompt <- create_meme_stock_prompt(batch)

    analysis <- call_openai_api(prompt)

    if (!is.null(analysis)) {
      all_analyses[[i]] <- analysis
      Sys.sleep(1)  # Rate limiting - adjust as needed
    } else {
      warning(sprintf("Batch %d failed to analyze", i))
    }
  }

  return(all_analyses)
}

# ============================================================================
# AGGREGATION AND SCORING
# ============================================================================

aggregate_predictions <- function(analyses) {

  if (length(analyses) == 0) {
    warning("No analyses to aggregate")
    return(data.frame())
  }

  # Extract all stock predictions
  all_stocks <- lapply(analyses, function(x) {
    if (!is.null(x) && !is.null(x$stocks_analyzed) && length(x$stocks_analyzed) > 0) {
      return(x$stocks_analyzed)
    }
    return(NULL)
  })

  # Remove NULL entries
  all_stocks <- all_stocks[!sapply(all_stocks, is.null)]

  if (length(all_stocks) == 0) {
    warning("No stocks found in analyses")
    return(data.frame())
  }

  all_stocks <- bind_rows(all_stocks)

  if (nrow(all_stocks) == 0) {
    warning("No stocks found after binding")
    return(data.frame())
  }

  # Aggregate by ticker
  aggregated <- all_stocks %>%
    group_by(ticker) %>%
    summarise(
      company_name = ifelse(length(na.omit(company_name)) > 0, first(na.omit(company_name)), "Unknown"),
      avg_bullish_score = mean(bullish_score, na.rm = TRUE),
      total_mentions = sum(mention_count, na.rm = TRUE),
      avg_engagement_score = mean(community_engagement_score, na.rm = TRUE),
      high_confidence_count = sum(prediction_confidence == "high", na.rm = TRUE),
      sentiment_summaries = paste(unique(sentiment_summary), collapse = " | "),
      all_catalysts = paste(unique(unlist(key_catalysts)), collapse = ", "),
      dominant_risk_level = ifelse(length(risk_level) > 0, names(which.max(table(risk_level))), "unknown"),
      bullish_mentions = sum(predicted_direction == "bullish", na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      # Calculate composite score
      composite_score = (avg_bullish_score * 0.4) +
                       (log1p(total_mentions) * 2 * 0.3) +
                       (avg_engagement_score * 0.2) +
                       (high_confidence_count * 2 * 0.1),
      # Normalize to 0-10 scale
      composite_score = pmin(composite_score, 10)
    ) %>%
    arrange(desc(composite_score)) %>%
    select(ticker, company_name, composite_score, avg_bullish_score, total_mentions,
           avg_engagement_score, high_confidence_count, bullish_mentions,
           dominant_risk_level, sentiment_summaries, all_catalysts)

  return(aggregated)
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main <- function() {

  cat("=== MEME STOCK PREDICTOR ===\n\n")

  # Load data
  cat("Loading Reddit data...\n")
  reddit_data <- load_reddit_data("dataset_meme_reddit_historical_1.csv")
  cat(sprintf("Loaded %d posts\n\n", nrow(reddit_data)))

  # Analyze data
  cat("Analyzing posts with OpenAI GPT-4 Turbo API...\n")
  analyses <- analyze_reddit_data(reddit_data, batch_size = 10, max_batches = 5)  # Adjust max_batches as needed

  # Aggregate results
  cat("\nAggregating predictions...\n")
  predictions <- aggregate_predictions(analyses)

  if (nrow(predictions) == 0) {
    cat("\nNo predictions generated. This could be due to:\n")
    cat("  - API errors (check warnings above)\n")
    cat("  - No stocks mentioned in the analyzed posts\n")
    cat("  - All stocks had bullish_score < 6.0\n")
    return(data.frame())
  }

  # Save results
  output_file <- paste0("meme_stock_predictions_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv")
  write.csv(predictions, output_file, row.names = FALSE)
  cat(sprintf("\nResults saved to: %s\n", output_file))

  # Display top predictions
  cat("\n=== TOP 10 BULLISH PREDICTIONS ===\n")
  top_predictions <- head(predictions, 10)
  if (ncol(predictions) >= 5) {
    print(top_predictions %>% select(ticker, composite_score, avg_bullish_score,
                                      total_mentions, dominant_risk_level))
  } else {
    print(top_predictions)
  }

  return(predictions)
}

# Run the analysis (comment out if sourcing this file)
# predictions <- main()
