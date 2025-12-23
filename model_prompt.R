############################################################
# Meme stock detection using Reddit (LLM sentiment/features)
#################### Using Prompt C4 #######################
############################################################

library(tidyverse)
library(lubridate)
library(purrr)
library(tidymodels)
library(slider)
library(torch)

set.seed(123)

reddit_prompt_path <- "Prompts/Datasets/PromptC4.csv"
price_path  <- "dataset_price_yahoo.csv"

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

ensure_col <- function(df, target, alternatives = character()) {
  if (target %in% names(df)) return(df)
  for (alt in alternatives) {
    if (alt %in% names(df)) return(dplyr::rename(df, !!target := !!rlang::sym(alt)))
  }
  stop("Missing column '", target, "'. Looked for: ", paste(c(target, alternatives), collapse = ", "))
}

zscore <- function(x) {
  s <- sd(x, na.rm = TRUE)
  if (is.na(s) || s == 0) return(rep(NA_real_, length(x)))
  (x - mean(x, na.rm = TRUE)) / s
}

# ----------------------------------------------------------
# 1) Prices: standard columns + returns + 20d avg volume
# ----------------------------------------------------------

load_and_prepare_prices <- function(price_path) {
  price_raw <- readr::read_csv(price_path, show_col_types = FALSE)
  names(price_raw) <- tolower(names(price_raw))
  
  price_raw <- price_raw %>%
    ensure_col("ticker", c("symbol", "tick", "ticker")) %>%
    ensure_col("date",   c("timestamp", "datetime", "date")) %>%
    ensure_col("close",  c("adjclose", "adj_close", "close")) %>%
    ensure_col("daily_change", c("ret", "return", "daily_return")) %>%
    ensure_col("rolling_vol_20d", c("rolling_vol_20", "vol_20d", "sigma20"))
  
  price_raw %>%
    mutate(
      ticker = toupper(as.character(ticker)),
      date   = as.Date(date),
      # daily_change in your file looks like percent units (e.g., 18.8 = +18.8%)
      ret    = daily_change / 100
    ) %>%
    arrange(ticker, date) %>%
    select(ticker, date, close, ret, rolling_vol_5d, rolling_vol_10d, rolling_vol_20d)
}

price_df <- load_and_prepare_prices(price_path)
tradeable_tickers <- unique(price_df$ticker)

# ----------------------------------------------------------
# 2) Reddit: clean text + date
# ----------------------------------------------------------

load_and_clean_reddit <- function(reddit_prompt_path) {
  df <- readr::read_csv(reddit_prompt_path, show_col_types = FALSE)
  
  if (!"created_dt" %in% names(df) && "created_utc" %in% names(df)) {
    df <- df %>% mutate(created_dt = as_datetime(created_utc, tz = "UTC"))
  }
  
  df %>%
    mutate(
      created_dt = ymd_hms(created_dt, tz = "UTC"),
      date       = as.Date(created_dt),
      title      = coalesce(title, ""),
      selftext   = coalesce(selftext, ""),
      text       = str_squish(paste(title, selftext, sep = "\n\n"))
    ) %>%
    filter(!is.na(text), nchar(text) >= 10)
}

reddit_df <- load_and_clean_reddit(reddit_prompt_path)

# ----------------------------------------------------------
# 3) Ticker extraction from Reddit posts
# ----------------------------------------------------------

extract_tickers <- function(text, tradeable_tickers) {
  if (is.na(text) || !nzchar(text)) return(character(0))
  txt <- toupper(text)
  
  dollar_syms <- str_extract_all(txt, "\\$[A-Z]{1,5}") %>% unlist() %>% str_sub(2)
  bare_syms   <- str_extract_all(txt, "\\b[A-Z]{2,5}\\b") %>% unlist()
  
  intersect(unique(c(dollar_syms, bare_syms)), tradeable_tickers)
}

reddit_posts_ticker <- reddit_df %>%
  mutate(tickers = map(text, extract_tickers, tradeable_tickers = tradeable_tickers)) %>%
  filter(map_int(tickers, length) > 0) %>%
  unnest(tickers) %>%
  transmute(
    ticker = toupper(tickers),
    date,
    across(everything(), identity)
  )

# ----------------------------------------------------------
# 4) Aggregate LLM features to daily (ticker, date)
# ----------------------------------------------------------

reddit_daily <- reddit_posts_ticker %>%
  group_by(ticker, date) %>%
  summarise(
    num_posts            = n(),
    mean_sent_hype       = mean(sent_hype, na.rm = TRUE),
    max_sent_hype        = ifelse(all(is.na(sent_hype)), NA_real_, max(sent_hype, na.rm = TRUE)),
    mean_sent_fomo       = mean(sent_fomo, na.rm = TRUE),
    mean_sent_fear       = mean(sent_fear, na.rm = TRUE),
    mean_sent_panic      = mean(sent_panic, na.rm = TRUE),
    mean_sent_sarcasm    = mean(sent_sarcasm, na.rm = TRUE),
    mean_sent_confidence = mean(sent_confidence, na.rm = TRUE),
    mean_sent_anger      = mean(sent_anger, na.rm = TRUE),
    mean_sent_regret     = mean(sent_regret, na.rm = TRUE),
    share_rocket_emoji   = mean(has_rocket_emoji == 1, na.rm = TRUE),
    share_moon_emoji     = mean(has_moon_emoji == 1, na.rm = TRUE),
    share_diamond_emoji  = mean(has_diamond_emoji == 1, na.rm = TRUE),
    share_money_emoji    = mean(has_money_emoji == 1, na.rm = TRUE),
    avg_score            = mean(score, na.rm = TRUE),
    avg_num_comments     = mean(num_comments, na.rm = TRUE),
    .groups = "drop"
  )

# ----------------------------------------------------------
# 5) Label: meme_alert based on future return + volume spike
# ----------------------------------------------------------

create_meme_alert_label <- function(price_df,
                                    return_threshold   = 0.08,  # +8%
                                    vol_multiplier     = 1.8,   # volatility spike factor
                                    horizon_days       = 6) {
  
  price_df %>%
    arrange(ticker, date) %>%
    group_by(ticker) %>%
    mutate(
      future_ret_max = slider::slide_dbl(ret, ~ max(.x, na.rm = TRUE),
                                         .after = horizon_days, .complete = TRUE),
      future_vol_max = slider::slide_dbl(rolling_vol_5d, ~ max(.x, na.rm = TRUE),
                                         .after = horizon_days, .complete = TRUE),
      
      meme_alert = dplyr::if_else(
        !is.na(future_ret_max) & !is.na(rolling_vol_20d) &
          future_ret_max > return_threshold &
          future_vol_max > vol_multiplier * rolling_vol_20d,
        1L, 0L
      )
    ) %>%
    ungroup()
}

price_labeled <- create_meme_alert_label(price_df)

# ----------------------------------------------------------
# 6) Merge features + label (final modelling dataset)
# ----------------------------------------------------------

df_model <- inner_join(reddit_daily, price_labeled, by = c("ticker", "date")) %>%
  drop_na()

# ----------------------------------------------------------
# 7) Time-based split (avoid leakage)
# ----------------------------------------------------------

make_time_splits <- function(df, train_prop = 0.7, val_prop = 0.15) {
  df <- df %>% arrange(date)
  all_dates <- sort(unique(df$date))
  n_dates <- length(all_dates)
  
  train_end <- floor(train_prop * n_dates)
  val_end   <- floor((train_prop + val_prop) * n_dates)
  
  list(
    train = df %>% filter(date %in% all_dates[1:train_end]),
    val   = df %>% filter(date %in% all_dates[(train_end + 1):val_end]),
    test  = df %>% filter(date %in% all_dates[(val_end + 1):n_dates])
  )
}

splits <- make_time_splits(df_model)
train_df <- splits$train
val_df   <- splits$val
test_df  <- splits$test

# ----------------------------------------------------------
# 8) Logistic regression (tidymodels)
# ----------------------------------------------------------

train_logistic_model <- function(train_df) {
  id_cols <- c("ticker", "date")
  
  train_df <- train_df %>%
    mutate(meme_alert = factor(meme_alert, levels = c(0, 1)))
  
  rec <- recipe(meme_alert ~ ., data = train_df %>% select(-all_of(id_cols))) %>%
    step_zv(all_predictors()) %>%
    step_normalize(all_numeric_predictors())
  
  workflow() %>%
    add_recipe(rec) %>%
    add_model(logistic_reg(mode = "classification") %>% set_engine("glm")) %>%
    fit(train_df)
}

logit_fit <- train_logistic_model(train_df)

# ----------------------------------------------------------
# 9) Neural net (torch) with pos_weight
# ----------------------------------------------------------

train_nn_model <- function(train_df, val_df, epochs = 30, batch_size = 64, lr = 1e-3) {
  id_cols <- c("ticker", "date")
  
  train_df <- train_df %>% mutate(meme_alert = as.numeric(meme_alert))
  val_df   <- val_df   %>% mutate(meme_alert = as.numeric(meme_alert))
  
  rec <- recipe(meme_alert ~ ., data = train_df %>% select(-all_of(id_cols))) %>%
    step_zv(all_predictors()) %>%
    step_normalize(all_numeric_predictors())
  
  prep_rec <- prep(rec)
  
  df_to_tensors <- function(df) {
    baked <- bake(prep_rec, new_data = df)
    x <- baked %>% select(-meme_alert) %>% as.matrix()
    y <- baked$meme_alert
    
    x[!is.finite(x)] <- 0
    y <- as.numeric(ifelse(y > 0, 1, 0))
    
    list(
      x = torch_tensor(x, dtype = torch_float()),
      y = torch_tensor(y, dtype = torch_float())
    )
  }
  
  tr <- df_to_tensors(train_df)
  va <- df_to_tensors(val_df)
  
  n_train   <- tr$x$size()[1]
  input_dim <- tr$x$size()[2]
  
  n_pos <- as.numeric((tr$y == 1)$sum()$item())
  n_neg <- n_train - n_pos
  pos_weight_value <- if (n_pos > 0) min(n_neg / n_pos, 10) else 1
  
  Net <- nn_module(
    "Net",
    initialize = function(input_dim) {
      self$fc1  <- nn_linear(input_dim, 32)
      self$fc2  <- nn_linear(32, 16)
      self$fc3  <- nn_linear(16, 1)
      self$drop <- nn_dropout(p = 0.5)
    },
    forward = function(x) {
      x %>%
        self$fc1() %>% nnf_relu() %>% self$drop() %>%
        self$fc2() %>% nnf_relu() %>% self$drop() %>%
        self$fc3()
    }
  )
  
  model     <- Net(input_dim)
  optimizer <- optim_adam(model$parameters, lr = lr, weight_decay = 1e-4)
  criterion <- nn_bce_with_logits_loss(
    pos_weight = torch_tensor(pos_weight_value, dtype = torch_float())
  )
  
  best_val <- Inf
  patience <- 5
  pat_left <- patience
  best_state <- NULL
  
  for (epoch in seq_len(epochs)) {
    model$train()
    idx <- sample.int(n_train)
    epoch_loss <- 0
    
    for (start in seq(1, n_train, by = batch_size)) {
      end <- min(start + batch_size - 1, n_train)
      b <- idx[start:end]
      
      x_b <- tr$x[b, ]
      y_b <- tr$y[b]$unsqueeze(2)
      
      optimizer$zero_grad()
      loss <- criterion(model(x_b), y_b)
      loss$backward()
      optimizer$step()
      
      epoch_loss <- epoch_loss + as.numeric(loss$item())
    }
    
    model$eval()
    with_no_grad({
      val_loss <- as.numeric(criterion(model(va$x), va$y$unsqueeze(2))$item())
    })
    
    cat(sprintf("Epoch %d/%d - train loss: %.4f - val loss: %.4f\n",
                epoch, epochs, epoch_loss, val_loss))
    
    if (val_loss < best_val - 1e-3) {
      best_val <- val_loss
      pat_left <- patience
      best_state <- model$state_dict()
    } else {
      pat_left <- pat_left - 1
      if (pat_left <= 0) {
        cat("Early stopping.\n")
        break
      }
    }
  }
  
  if (!is.null(best_state)) model$load_state_dict(best_state)
  
  list(model = model, recipe = prep_rec)
}

nn_result <- train_nn_model(train_df, val_df)

# ----------------------------------------------------------
# 10) Threshold tuning + evaluation
# ----------------------------------------------------------

tune_threshold <- function(probs, truth_factor) {
  grid <- seq(0.01, 0.99, by = 0.01)
  
  purrr::map_df(grid, function(th) {
    pred <- factor(ifelse(probs >= th, "1", "0"), levels = c("0", "1"))
    
    # skip thresholds that predict no positives
    if (sum(pred == "1") == 0) return(tibble(threshold = th, f1 = NA_real_))
    
    f1 <- f_meas(tibble(truth = truth_factor, pred = pred),
                 truth = truth, estimate = pred, event_level = "second")$.estimate
    tibble(threshold = th, f1 = f1)
  }) %>%
    filter(!is.na(f1)) %>%
    slice_max(f1, n = 1, with_ties = FALSE)
}

evaluate_with_threshold <- function(truth, probs, threshold) {
  pred <- factor(ifelse(probs >= threshold, "1", "0"), levels = c("0", "1"))
  dfm <- tibble(truth = truth, .pred_1 = probs, .pred_class = pred)
  
  list(
    metrics = tibble(
      accuracy  = accuracy(dfm, truth, .pred_class)$.estimate,
      precision = precision(dfm, truth, .pred_class, event_level = "second")$.estimate,
      recall    = recall(dfm, truth, .pred_class, event_level = "second")$.estimate,
      f1        = f_meas(dfm, truth, .pred_class, event_level = "second")$.estimate,
      roc_auc   = roc_auc(dfm, truth, .pred_1, event_level = "second")$.estimate
    ),
    confusion = conf_mat(dfm, truth, .pred_class)
  )
}

# Validation probabilities
val_truth <- factor(val_df$meme_alert, levels = c(0, 1))

val_probs_logit <- predict(
  logit_fit,
  new_data = val_df %>% mutate(meme_alert = val_truth),
  type = "prob"
)$.pred_1

baked_val_nn <- bake(nn_result$recipe, new_data = val_df %>% mutate(meme_alert = as.numeric(meme_alert)))
x_val_nn <- baked_val_nn %>% select(-meme_alert) %>% as.matrix()

val_probs_nn <- nn_result$model(torch_tensor(x_val_nn, dtype = torch_float())) %>%
  torch_sigmoid() %>% as_array() %>% as.numeric()

best_th_logit <- tune_threshold(val_probs_logit, val_truth)
best_th_nn    <- tune_threshold(val_probs_nn,    val_truth)

# Test probabilities
test_truth <- factor(test_df$meme_alert, levels = c(0, 1))

test_probs_logit <- predict(
  logit_fit,
  new_data = test_df %>% mutate(meme_alert = test_truth),
  type = "prob"
)$.pred_1

baked_test_nn <- bake(nn_result$recipe, new_data = test_df %>% mutate(meme_alert = as.numeric(meme_alert)))
x_test_nn <- baked_test_nn %>% select(-meme_alert) %>% as.matrix()

test_probs_nn <- nn_result$model(torch_tensor(x_test_nn, dtype = torch_float())) %>%
  torch_sigmoid() %>% as_array() %>% as.numeric()

logit_eval_test <- evaluate_with_threshold(test_truth, test_probs_logit, best_th_logit$threshold)
nn_eval_test    <- evaluate_with_threshold(test_truth, test_probs_nn,    best_th_nn$threshold)

print(logit_eval_test$metrics); print(logit_eval_test$confusion)
print(nn_eval_test$metrics);    print(nn_eval_test$confusion)

# ----------------------------------------------------------
# 11) Model plots (test set)
# ----------------------------------------------------------

test_pred_df <- tibble(
  truth      = factor(test_truth, levels = c("0", "1")),
  prob_logit = test_probs_logit,
  prob_nn    = test_probs_nn
)

# 11.1 Metrics bar
metrics_plot_df <- bind_rows(
  tibble(model = "Logistic", value = as.numeric(logit_eval_test$metrics[1, ]), metric = names(logit_eval_test$metrics)) %>%
    select(model, metric, value),
  tibble(model = "Neural net", value = as.numeric(nn_eval_test$metrics[1, ]), metric = names(nn_eval_test$metrics)) %>%
    select(model, metric, value)
) %>%
  filter(metric %in% c("accuracy", "precision", "recall", "f1"))

p_metrics <- ggplot(metrics_plot_df, aes(x = metric, y = value, fill = model)) +
  geom_col(position = position_dodge(width = 0.7)) +
  geom_text(
    aes(label = sprintf("%.3f", value)),
    position = position_dodge(width = 0.7),
    vjust = -0.35,
    size = 3
  ) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(
    title = "Test performance (threshold tuned on validation)",
    x = NULL, y = "Score", fill = "Model"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5), legend.position = "bottom")

print(p_metrics)
ggsave("plot_test_metrics_bar.png", p_metrics, width = 7, height = 4)

# 11.2 ROC curves
roc_logit <- roc_curve(test_pred_df, truth, prob_logit, event_level = "second") %>% mutate(model = "Logistic")
roc_nn    <- roc_curve(test_pred_df, truth, prob_nn,    event_level = "second") %>% mutate(model = "Neural net")

p_roc <- bind_rows(roc_logit, roc_nn) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_line(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  labs(title = "ROC curves (test set)",
       x = "False Positive Rate", y = "True Positive Rate", color = "Model") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5), legend.position = "bottom")

print(p_roc)
ggsave("plot_test_roc.png", p_roc, width = 7, height = 4)

# 11.3 Precision–Recall curves
pr_logit <- pr_curve(test_pred_df, truth, prob_logit, event_level = "second") %>% mutate(model = "Logistic")
pr_nn    <- pr_curve(test_pred_df, truth, prob_nn,    event_level = "second") %>% mutate(model = "Neural net")

p_pr <- bind_rows(pr_logit, pr_nn) %>%
  ggplot(aes(x = recall, y = precision, color = model)) +
  geom_line(linewidth = 1) +
  labs(title = "Precision–Recall curves (test set)",
       x = "Recall", y = "Precision", color = "Model") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5), legend.position = "bottom")

print(p_pr)
ggsave("plot_test_pr.png", p_pr, width = 7, height = 4)

# 11.4 Score distributions
scores_long <- test_pred_df %>%
  pivot_longer(cols = c(prob_logit, prob_nn), names_to = "model", values_to = "prob") %>%
  mutate(model = recode(model, prob_logit = "Logistic", prob_nn = "Neural net"))

p_dist <- ggplot(scores_long, aes(x = prob, fill = truth)) +
  geom_histogram(bins = 30, position = "identity", alpha = 0.5) +
  facet_wrap(~ model, ncol = 1) +
  labs(title = "Predicted probability distributions (test set)",
       x = "Predicted P(meme_alert = 1)", y = "Count", fill = "True class") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5))

print(p_dist)
ggsave("plot_test_prob_distributions.png", p_dist, width = 7, height = 6)

# ----------------------------------------------------------
# 12) Backtest/directionality (Reddit leads vs reacts)
# ----------------------------------------------------------

run_backtest <- function(df_model,
                         signal_col = "num_posts",
                         z_thresh = 2,
                         price_jump_thresh = 0.05,
                         max_lag = 10) {
  
  df_bt <- df_model %>%
    arrange(ticker, date) %>%
    group_by(ticker) %>%
    mutate(
      reddit_signal = .data[[signal_col]],
      reddit_z      = zscore(reddit_signal),
      
      ret_fwd_1 = lead(ret, 1),
      ret_fwd_3 = slide_dbl(ret, ~ sum(.x, na.rm = TRUE), .before = 1, .after = 3, .complete = TRUE),
      ret_fwd_5 = slide_dbl(ret, ~ sum(.x, na.rm = TRUE), .before = 1, .after = 5, .complete = TRUE)
    ) %>%
    ungroup()
  
  # --- Reddit spike event study
  df_spike <- df_bt %>%
    filter(is.finite(reddit_z)) %>%
    mutate(is_spike = reddit_z >= z_thresh)
  
  event_long <- df_spike %>%
    filter(is_spike) %>%
    summarise(
      `+1d` = mean(ret_fwd_1, na.rm = TRUE),
      `+3d` = mean(ret_fwd_3, na.rm = TRUE),
      `+5d` = mean(ret_fwd_5, na.rm = TRUE)
    ) %>%
    pivot_longer(everything(), names_to = "horizon", values_to = "avg_fwd_ret")
  
  p_event <- ggplot(event_long, aes(x = horizon, y = avg_fwd_ret)) +
    geom_col() +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(
      title = paste0("Event study: Reddit spike (z≥", z_thresh, ") → future returns"),
      subtitle = paste0("Signal = ", signal_col, " | z-scored per ticker"),
      x = "Horizon after spike day",
      y = "Average cumulative return"
    ) +
    theme_minimal(base_size = 12)
  
  # --- Price jump → future Reddit
  df_rev <- df_bt %>%
    group_by(ticker) %>%
    mutate(
      is_price_jump = ret >= price_jump_thresh,
      reddit_z_fwd_1 = lead(reddit_z, 1),
      reddit_z_fwd_3 = slide_dbl(reddit_z, ~ mean(.x, na.rm = TRUE), .before = 1, .after = 3, .complete = TRUE),
      reddit_z_fwd_5 = slide_dbl(reddit_z, ~ mean(.x, na.rm = TRUE), .before = 1, .after = 5, .complete = TRUE)
    ) %>%
    ungroup()
  
  rev_long <- df_rev %>%
    filter(is_price_jump) %>%
    summarise(
      `+1d` = mean(reddit_z_fwd_1, na.rm = TRUE),
      `+3d` = mean(reddit_z_fwd_3, na.rm = TRUE),
      `+5d` = mean(reddit_z_fwd_5, na.rm = TRUE)
    ) %>%
    pivot_longer(everything(), names_to = "horizon", values_to = "avg_reddit_z")
  
  p_reverse <- ggplot(rev_long, aes(x = horizon, y = avg_reddit_z)) +
    geom_col() +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(
      title = paste0("Reverse check: Price jump (ret≥", price_jump_thresh, ") → future Reddit activity"),
      subtitle = paste0("Signal = ", signal_col, " | z-scored per ticker"),
      x = "Horizon after price jump day",
      y = "Average Reddit z-score"
    ) +
    theme_minimal(base_size = 12)
  
  # --- Lead–lag correlation curve
  lags <- -max_lag:max_lag
  corr_df <- purrr::map_df(lags, function(k) {
    tmp <- df_bt %>%
      group_by(ticker) %>%
      mutate(ret_shift = if (k >= 0) lead(ret, k) else lag(ret, -k)) %>%
      ungroup() %>%
      filter(is.finite(reddit_z), is.finite(ret_shift))
    
    tibble(
      lag  = k,
      corr = cor(tmp$reddit_z, tmp$ret_shift, use = "complete.obs")
    )
  })
  
  p_lag <- ggplot(corr_df, aes(x = lag, y = corr)) +
    geom_line(linewidth = 1) +
    geom_point(size = 2) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(
      title = "Lead–lag correlation: Reddit activity vs returns",
      subtitle = paste0("Positive lag = Reddit today vs returns in +k days | Signal = ", signal_col),
      x = "Lag k (days)",
      y = "Correlation corr(reddit_z[t], ret[t+k])"
    ) +
    theme_minimal(base_size = 12)
  
  list(p_event = p_event, p_reverse = p_reverse, p_lag = p_lag)
}

bt <- run_backtest(df_model, signal_col = "num_posts", z_thresh = 2, price_jump_thresh = 0.05, max_lag = 10)

print(bt$p_event)
ggsave("backtest_reddit_spike_future_returns.png", bt$p_event, width = 7, height = 4)

print(bt$p_reverse)
ggsave("backtest_price_jump_future_reddit.png", bt$p_reverse, width = 7, height = 4)

print(bt$p_lag)
ggsave("backtest_lead_lag_correlation.png", bt$p_lag, width = 7, height = 4)
ggsave("backtest_price_jump_future_reddit.png", bt$p_reverse, width = 7, height = 4)

print(bt$p_lag)
ggsave("backtest_lead_lag_correlation.png", bt$p_lag, width = 7, height = 4)