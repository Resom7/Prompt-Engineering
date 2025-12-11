############################################################
# Meme Stocks: Modelling Using Pre-Scored LLM Sentiment
# ---------------------------------------------------------
#  1) Load Reddit + price data (LLM sentiment already added)
#  2) Extract tickers from Reddit text
#  3) Aggregate to (ticker, date) level
#  4) Create meme_alert label from prices
#  5) Merge Reddit + price features
#  6) Train:
#       - Logistic regression (tidymodels)
#       - Neural net (torch MLP)
#  7) Evaluate on time-based train/val/test split
#  8) Save dataset + models
############################################################

## 0. SETUP ------------------------------------------------

library(tidyverse)
library(lubridate)
library(purrr)
library(jsonlite)
library(rsample)
library(tidymodels)
library(slider)
library(torch)

set.seed(123)

# ---- File paths (EDIT THESE IF NEEDED) ----
reddit_path <- "~/Documents/PEE/LLM_dataset.csv"
price_path  <- "~/Documents/PEE/price_yahoo_dataset.csv"

############################################################
# 1. LOAD & PREPARE PRICE DATA
############################################################

load_and_prepare_prices <- function(price_path) {
  price_raw <- readr::read_csv(price_path, show_col_types = FALSE)
  names(price_raw) <- tolower(names(price_raw))
  
  # ticker
  if (!"ticker" %in% names(price_raw)) {
    if ("symbol" %in% names(price_raw)) {
      price_raw <- dplyr::rename(price_raw, ticker = symbol)
    } else {
      stop("Could not find a 'ticker' or 'symbol' column. Columns: ",
           paste(names(price_raw), collapse = ", "))
    }
  }
  
  # date
  if (!"date" %in% names(price_raw)) {
    if ("timestamp" %in% names(price_raw)) {
      price_raw <- dplyr::rename(price_raw, date = timestamp)
    } else {
      stop("Could not find a 'date' or 'timestamp' column. Columns: ",
           paste(names(price_raw), collapse = ", "))
    }
  }
  
  # close
  if (!"close" %in% names(price_raw)) {
    if ("adjclose" %in% names(price_raw)) {
      price_raw <- dplyr::rename(price_raw, close = adjclose)
    } else if ("adj_close" %in% names(price_raw)) {
      price_raw <- dplyr::rename(price_raw, close = adj_close)
    } else {
      stop("Could not find 'close' (or adjclose/adj_close) column. Columns: ",
           paste(names(price_raw), collapse = ", "))
    }
  }
  
  # volume
  if (!"volume" %in% names(price_raw)) {
    stop("Could not find 'volume' column. Columns: ",
         paste(names(price_raw), collapse = ", "))
  }
  
  price_df <- price_raw %>%
    mutate(
      ticker = toupper(as.character(ticker)),
      date   = as.Date(date)
    ) %>%
    arrange(ticker, date)
  
  # ret
  if (!"ret" %in% names(price_df)) {
    price_df <- price_df %>%
      group_by(ticker) %>%
      mutate(ret = close / dplyr::lag(close) - 1) %>%
      ungroup()
  }
  
  # vol_ma20
  if (!"vol_ma20" %in% names(price_df)) {
    price_df <- price_df %>%
      group_by(ticker) %>%
      mutate(
        vol_ma20 = slider::slide_dbl(
          volume,
          mean,
          .before   = 19,
          .complete = TRUE
        )
      ) %>%
      ungroup()
  }
  
  needed <- c("ticker", "date", "close", "volume", "ret", "vol_ma20")
  missing_cols <- setdiff(needed, names(price_df))
  if (length(missing_cols) > 0) {
    stop("Missing columns in price dataset even after processing: ",
         paste(missing_cols, collapse = ", "))
  }
  
  price_df
}

price_df <- load_and_prepare_prices(price_path)
tradeable_tickers <- unique(price_df$ticker)
message("Loaded price data for ", length(tradeable_tickers), " tickers.")

############################################################
# 2. LOAD & CLEAN REDDIT DATA (WITH LLM SENTIMENT)
############################################################

load_and_clean_reddit <- function(reddit_path) {
  df <- readr::read_csv(reddit_path, show_col_types = FALSE)
  
  if (!"created_dt" %in% names(df) && "created_utc" %in% names(df)) {
    df <- df %>%
      mutate(created_dt = as_datetime(created_utc, tz = "UTC"))
  }
  
  df <- df %>%
    mutate(
      created_dt = ymd_hms(created_dt, tz = "UTC"),
      date       = as.Date(created_dt),
      title      = coalesce(title, ""),
      selftext   = coalesce(selftext, ""),
      text_raw   = paste(title, selftext, sep = "\n\n"),
      text       = stringr::str_squish(text_raw)
    ) %>%
    filter(!is.na(text), nchar(text) >= 10)
  
  df
}

reddit_df <- load_and_clean_reddit(reddit_path)
message("Loaded Reddit posts: ", nrow(reddit_df))

############################################################
# 3. EXTRACT TICKERS FROM TEXT
############################################################

extract_tickers <- function(text, tradeable_tickers) {
  if (is.na(text) || !nzchar(text)) return(character(0))
  
  txt <- toupper(text)
  
  dollar_syms <- stringr::str_extract_all(txt, "\\$[A-Z]{1,5}") %>%
    unlist() %>%
    stringr::str_sub(2)
  
  bare_syms <- stringr::str_extract_all(txt, "\\b[A-Z]{2,5}\\b") %>%
    unlist()
  
  candidates <- unique(c(dollar_syms, bare_syms))
  intersect(candidates, tradeable_tickers)
}

reddit_df <- reddit_df %>%
  mutate(
    tickers = map(text, extract_tickers, tradeable_tickers = tradeable_tickers)
  )

reddit_posts_ticker <- reddit_df %>%
  filter(map_int(tickers, length) > 0) %>%
  unnest(tickers, names_repair = "check_unique") %>%
  rename(ticker = tickers) %>%
  mutate(ticker = toupper(ticker))

message("Number of post–ticker pairs: ", nrow(reddit_posts_ticker))

############################################################
# 4. AGGREGATE LLM FEATURES TO (TICKER, DATE)
############################################################

aggregate_reddit_to_daily <- function(df_posts) {
  df_posts %>%
    mutate(date = as.Date(date)) %>%
    group_by(ticker, date) %>%
    summarise(
      num_posts            = n(),
      mean_sent_hype       = mean(sent_hype, na.rm = TRUE),
      max_sent_hype        = ifelse(
        all(is.na(sent_hype)), NA_real_,
        max(sent_hype, na.rm = TRUE)
      ),
      mean_sent_fomo       = mean(sent_fomo, na.rm = TRUE),
      mean_sent_fear       = mean(sent_fear, na.rm = TRUE),
      mean_sent_panic      = mean(sent_panic, na.rm = TRUE),
      mean_sent_sarcasm    = mean(sent_sarcasm, na.rm = TRUE),
      mean_sent_confidence = mean(sent_confidence, na.rm = TRUE),
      mean_sent_anger      = mean(sent_anger, na.rm = TRUE),
      mean_sent_regret     = mean(sent_regret, na.rm = TRUE),
      share_rocket_emoji   = mean(has_rocket_emoji == 1, na.rm = TRUE),
      share_moon_emoji     = mean(has_moon_emoji   == 1, na.rm = TRUE),
      share_diamond_emoji  = mean(has_diamond_emoji == 1, na.rm = TRUE),
      share_money_emoji    = mean(has_money_emoji  == 1, na.rm = TRUE),
      avg_score            = mean(score, na.rm = TRUE),
      avg_num_comments     = mean(num_comments, na.rm = TRUE),
      .groups = "drop"
    )
}

reddit_daily <- aggregate_reddit_to_daily(reddit_posts_ticker)
message("Daily Reddit feature rows: ", nrow(reddit_daily))

############################################################
# 5. CREATE GOLD-STANDARD LABEL FROM PRICE DATA
############################################################

create_meme_alert_label <- function(price_df,
                                    return_threshold  = 0.03,
                                    volume_multiplier = 1.3,
                                    horizon_days      = 3) {
  price_df %>%
    arrange(ticker, date) %>%
    group_by(ticker) %>%
    mutate(
      # Use slider::slide_* over the next H days
      future_ret_max = slider::slide_dbl(
        ret,
        ~ max(.x, na.rm = TRUE),
        .after   = horizon_days,
        .before  = 0,
        .complete = FALSE
      ),
      future_vol_max = slider::slide_dbl(
        volume,
        ~ max(.x, na.rm = TRUE),
        .after   = horizon_days,
        .before  = 0,
        .complete = FALSE
      ),
      meme_alert = if_else(
        !is.na(future_ret_max) & !is.na(vol_ma20) &
          future_ret_max > return_threshold &
          future_vol_max > volume_multiplier * vol_ma20,
        1L, 0L
      )
    ) %>%
    ungroup()
}

price_labeled <- create_meme_alert_label(price_df)

############################################################
# 6. MERGE REDDIT FEATURES WITH PRICE + LABEL
############################################################

df_model <- inner_join(
  reddit_daily,
  price_labeled,
  by = c("ticker", "date")
)

message("Final modelling dataset rows: ", nrow(df_model))

############################################################
# 7. TIME-BASED TRAIN/VAL/TEST SPLIT
############################################################

make_time_splits <- function(df_model, train_prop = 0.7, val_prop = 0.15) {
  df_model  <- df_model %>% arrange(date)
  all_dates <- sort(unique(df_model$date))
  n_dates   <- length(all_dates)
  
  train_end_idx <- floor(train_prop * n_dates)
  val_end_idx   <- floor((train_prop + val_prop) * n_dates)
  
  train_dates <- all_dates[1:train_end_idx]
  val_dates   <- all_dates[(train_end_idx + 1):val_end_idx]
  test_dates  <- all_dates[(val_end_idx + 1):n_dates]
  
  list(
    train = df_model %>% filter(date %in% train_dates),
    val   = df_model %>% filter(date %in% val_dates),
    test  = df_model %>% filter(date %in% test_dates)
  )
}

df_model_clean <- df_model %>% drop_na()

splits   <- make_time_splits(df_model_clean)
train_df <- splits$train
val_df   <- splits$val
test_df  <- splits$test

message("Train rows: ", nrow(train_df),
        " | Val rows: ", nrow(val_df),
        " | Test rows: ", nrow(test_df))

############################################################
# 8. BASELINE MODEL: LOGISTIC REGRESSION (tidymodels)
############################################################

train_logistic_model <- function(train_df) {
  train_df <- train_df %>%
    mutate(meme_alert = factor(meme_alert, levels = c(0, 1)))
  
  id_cols <- c("ticker", "date")
  outcome <- "meme_alert"
  
  recipe_spec <- recipes::recipe(
    as.formula(paste(outcome, "~ .")),
    data = train_df %>% select(-all_of(id_cols))
  ) %>%
    update_role(meme_alert, new_role = "outcome") %>%
    step_zv(all_predictors()) %>%
    step_normalize(all_numeric_predictors())
  
  logit_spec <- logistic_reg(mode = "classification") %>%
    set_engine("glm")
  
  workflow() %>%
    add_model(logit_spec) %>%
    add_recipe(recipe_spec) %>%
    fit(train_df)
}

logit_fit <- train_logistic_model(train_df)

############################################################
# 9. NEURAL NETWORK CLASSIFIER (torch MLP)
############################################################

train_nn_model <- function(train_df, val_df, epochs = 30, batch_size = 64, lr = 1e-3) {
  id_cols <- c("ticker", "date")
  outcome <- "meme_alert"
  
  # make sure meme_alert is numeric 0/1
  train_df <- train_df %>% mutate(meme_alert = as.numeric(meme_alert))
  val_df   <- val_df   %>% mutate(meme_alert = as.numeric(meme_alert))
  
  # recipe: standardize predictors, but *not* the outcome
  recipe_spec <- recipes::recipe(
    as.formula(paste(outcome, "~ .")),
    data = train_df %>% select(-all_of(id_cols))
  ) %>%
    update_role(meme_alert, new_role = "outcome") %>%
    step_zv(all_predictors()) %>%
    step_normalize(all_numeric_predictors())
  
  prep_rec <- prep(recipe_spec)
  
  # helper: bake + convert to torch tensors
  df_to_tensors <- function(df) {
    baked <- bake(prep_rec, new_data = df)
    
    x_mat <- baked %>%
      select(-meme_alert) %>%
      as.matrix()
    
    y_vec <- baked$meme_alert
    
    # force to 0/1
    y_vec <- ifelse(is.na(y_vec), 0, y_vec)
    y_vec <- ifelse(y_vec > 0, 1, 0)
    
    # replace any non-finite values in predictors (NaN, Inf) by 0
    x_mat[!is.finite(x_mat)] <- 0
    
    list(
      x = torch_tensor(x_mat, dtype = torch_float()),
      y = torch_tensor(as.numeric(y_vec), dtype = torch_float())
    )
  }
  
  train_t <- df_to_tensors(train_df)
  val_t   <- df_to_tensors(val_df)
  
  x_train <- train_t$x
  y_train <- train_t$y
  x_val   <- val_t$x
  y_val   <- val_t$y
  
  n_train   <- x_train$size()[1]
  input_dim <- x_train$size()[2]
  
  # ---- CLASS WEIGHTING: make positives more important ----
  n_pos <- as.numeric((y_train == 1)$sum()$item())
  n_neg <- n_train - n_pos
  pos_weight_value <- if (n_pos > 0) n_neg / n_pos else 1
  cat("Training NN with pos_weight =", pos_weight_value, "\n")
  
  Net <- nn_module(
    "Net",
    initialize = function(input_dim) {
      self$fc1  <- nn_linear(input_dim, 64)
      self$fc2  <- nn_linear(64, 32)
      self$fc3  <- nn_linear(32, 1)
      self$drop <- nn_dropout(p = 0.3)
    },
    forward = function(x) {
      x %>%
        self$fc1() %>%
        nnf_relu() %>%
        self$drop() %>%
        self$fc2() %>%
        nnf_relu() %>%
        self$drop() %>%
        self$fc3()       # logits (no sigmoid)
    }
  )
  
  model     <- Net(input_dim)
  optimizer <- optim_adam(model$parameters, lr = lr)
  criterion <- nn_bce_with_logits_loss(
    pos_weight = torch_tensor(pos_weight_value, dtype = torch_float())
  )
  
  for (epoch in seq_len(epochs)) {
    model$train()
    epoch_loss <- 0
    
    idx <- sample.int(n_train)
    for (start in seq(1, n_train, by = batch_size)) {
      end <- min(start + batch_size - 1, n_train)
      batch_idx <- idx[start:end]
      
      x_batch <- x_train[batch_idx, ]
      y_batch <- y_train[batch_idx]$unsqueeze(2)  # shape [B, 1]
      
      optimizer$zero_grad()
      logits <- model(x_batch)
      loss   <- criterion(logits, y_batch)
      loss$backward()
      optimizer$step()
      
      epoch_loss <- epoch_loss + as.numeric(loss$item())
    }
    
    # validation loss
    model$eval()
    with_no_grad({
      val_logits <- model(x_val)
      val_loss   <- as.numeric(
        criterion(val_logits, y_val$unsqueeze(2))$item()
      )
    })
    
    cat(sprintf(
      "Epoch %d/%d - train loss: %.4f - val loss: %.4f\n",
      epoch, epochs, epoch_loss, val_loss
    ))
  }
  
  list(
    model  = model,
    recipe = prep_rec
  )
}

nn_result <- train_nn_model(train_df, val_df)

############################################################
# 10. THRESHOLD TUNING + EVALUATION HELPERS
############################################################

# --- Helper to tune classification threshold on validation set ----
tune_threshold <- function(probs, truth_factor) {
  grid <- seq(0.05, 0.95, by = 0.05)
  
  results <- purrr::map_df(grid, function(th) {
    class_pred <- ifelse(probs >= th, "1", "0") %>%
      factor(levels = c("0", "1"))
    
    metrics_df <- tibble(
      truth       = truth_factor,
      .pred_class = class_pred
    )
    
    f1 <- f_meas(
      metrics_df,
      truth        = truth,
      estimate     = .pred_class,
      event_level  = "second"   # class "1" is the positive event
    )$.estimate
    
    tibble(threshold = th, f1 = f1)
  })
  
  results %>%
    filter(f1 == max(f1, na.rm = TRUE)) %>%
    slice(1)
}

# --- Logistic regression evaluation (with custom threshold) ----
evaluate_logit <- function(fit, df, threshold = 0.5) {
  df_eval <- df %>%
    mutate(meme_alert = factor(meme_alert, levels = c(0, 1)))
  
  probs <- predict(fit, new_data = df_eval, type = "prob")$.pred_1
  class_pred <- ifelse(probs >= threshold, "1", "0") %>%
    factor(levels = c("0", "1"))
  
  metrics_df <- tibble(
    truth       = df_eval$meme_alert,
    .pred_1     = probs,
    .pred_class = class_pred
  )
  
  metrics_list <- list(
    accuracy  = accuracy(metrics_df, truth = truth, estimate = .pred_class),
    precision = precision(metrics_df, truth = truth, estimate = .pred_class,
                          event_level = "second"),
    recall    = recall(metrics_df, truth = truth, estimate = .pred_class,
                       event_level = "second"),
    f_meas    = f_meas(metrics_df, truth = truth, estimate = .pred_class,
                       event_level = "second"),
    roc_auc   = roc_auc(
      metrics_df,
      truth = truth,
      .pred_1,
      event_level = "second"   # class "1" is the event
    )
  )
  
  metrics_tbl <- bind_rows(metrics_list) %>%
    select(.metric, .estimate)
  
  conf <- conf_mat(metrics_df, truth = truth, estimate = .pred_class)
  
  list(metrics = metrics_tbl, confusion = conf)
}

# --- Torch neural net evaluation (with custom threshold) ----
evaluate_nn <- function(nn_result, df, threshold = 0.5) {
  df_proc <- df %>%
    mutate(meme_alert = as.numeric(meme_alert))
  
  baked <- bake(nn_result$recipe, new_data = df_proc)
  
  x <- baked %>% select(-meme_alert) %>% as.matrix()
  y <- baked$meme_alert
  
  x_tensor <- torch_tensor(x, dtype = torch_float())
  
  probs <- nn_result$model(x_tensor) %>%
    torch_sigmoid() %>%
    as_array() %>%
    as.numeric()
  
  class_pred <- ifelse(probs >= threshold, "1", "0") %>%
    factor(levels = c("0", "1"))
  
  metrics_df <- tibble(
    truth       = factor(y, levels = c(0, 1)),
    .pred_1     = probs,
    .pred_class = class_pred
  )
  
  metrics_list <- list(
    accuracy  = accuracy(metrics_df, truth = truth, estimate = .pred_class),
    precision = precision(metrics_df, truth = truth, estimate = .pred_class,
                          event_level = "second"),
    recall    = recall(metrics_df, truth = truth, estimate = .pred_class,
                       event_level = "second"),
    f_meas    = f_meas(metrics_df, truth = truth, estimate = .pred_class,
                       event_level = "second"),
    roc_auc   = roc_auc(metrics_df, truth = truth, .pred_1)
  )
  
  metrics_tbl <- bind_rows(metrics_list) %>%
    select(.metric, .estimate)
  
  conf <- conf_mat(metrics_df, truth = truth, estimate = .pred_class)
  
  list(metrics = metrics_tbl, confusion = conf)
}

############################################################
# 11. TUNE THRESHOLDS ON VALIDATION SET
############################################################

# ---- Logistic: best threshold on val ----
val_eval_logit <- val_df %>%
  mutate(meme_alert = factor(meme_alert, levels = c(0, 1)))

val_probs_logit <- predict(logit_fit, new_data = val_eval_logit, type = "prob")$.pred_1

best_th_logit <- tune_threshold(val_probs_logit, val_eval_logit$meme_alert)
cat("Best logistic threshold (val):\n")
print(best_th_logit)

# ---- NN: best threshold on val ----
val_proc_nn  <- val_df %>%
  mutate(meme_alert = as.numeric(meme_alert))

baked_val_nn <- bake(nn_result$recipe, new_data = val_proc_nn)

x_val_nn <- baked_val_nn %>% select(-meme_alert) %>% as.matrix()
y_val_nn <- baked_val_nn$meme_alert

val_probs_nn <- nn_result$model(torch_tensor(x_val_nn, dtype = torch_float())) %>%
  torch_sigmoid() %>%
  as_array() %>%
  as.numeric()

best_th_nn <- tune_threshold(val_probs_nn, factor(y_val_nn, levels = c(0, 1)))
cat("Best NN threshold (val):\n")
print(best_th_nn)

############################################################
# 12. FINAL EVALUATION ON TEST SET (USING TUNED THRESHOLDS)
############################################################

logit_eval_test <- evaluate_logit(
  logit_fit,
  test_df,
  threshold = best_th_logit$threshold
)

nn_eval_test <- evaluate_nn(
  nn_result,
  test_df,
  threshold = best_th_nn$threshold
)

print("Logistic regression metrics (test):")
print(logit_eval_test$metrics)
print(logit_eval_test$confusion)

print("Neural net (torch) metrics (test):")
print(nn_eval_test$metrics)
print(nn_eval_test$confusion)

############################################################
# 13. VISUAL SUMMARY PLOT (TEST SET)
############################################################

# Combine main metrics (excluding roc_auc) into one table
metrics_plot_df <- bind_rows(
  logit_eval_test$metrics %>%
    mutate(model = "Logistic regression"),
  nn_eval_test$metrics %>%
    mutate(model = "Neural net (torch)")
) %>%
  filter(.metric %in% c("accuracy", "precision", "recall", "f_meas")) %>%
  mutate(
    .metric = factor(
      .metric,
      levels = c("accuracy", "precision", "recall", "f_meas"),
      labels = c("Accuracy", "Precision", "Recall", "F1-score")
    )
  )

# Bar plot comparing models on the test set
ggplot(metrics_plot_df, aes(x = .metric, y = .estimate, fill = model)) +
  geom_col(position = position_dodge(width = 0.7)) +
  geom_text(
    aes(label = round(.estimate, 3)),
    position = position_dodge(width = 0.7),
    vjust = -0.3,
    size = 3
  ) +
  ylim(0, 1) +
  labs(
    title    = "Test-set performance: Logistic vs Neural Net",
    x        = "Metric",
    y        = "Score",
    fill     = "Model"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title   = element_text(hjust = 0.5),
    legend.position = "bottom"
  )

# (Optional) Save to file for your thesis/report
ggsave("model_comparison_test_metrics.png", width = 7, height = 4)

############################################################
# 14. SAVE OUTPUTS
############################################################

readr::write_csv(df_model, "meme_asset_day_dataset_with_llm_scores.csv")
saveRDS(logit_fit, "logistic_meme_model.rds")
saveRDS(nn_result, "nn_meme_model_torch.rds")  # torch model + recipe

# Helper for single-row prediction (logistic) --------------
predict_meme_alert_for_row <- function(logit_fit, new_row) {
  probs <- predict(logit_fit, new_data = new_row, type = "prob")$.pred_1
  tibble(
    prob_meme_alert  = probs,
    class_meme_alert = ifelse(probs >= 0.5, 1, 0)
  )
}

############################################################
# End of script.
############################################################