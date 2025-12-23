# Meme Stock Detection Using LLM-Enhanced Sentiment Analysis

A machine learning project that uses Large Language Models (LLMs) to extract sentiment features from Reddit posts and predict meme stock events.

## Project Structure

### Main File
- [model_prompt.R](model_prompt.R) - **Main modeling script** containing all analysis, feature engineering, model training, and evaluation

### Data Files
- `dataset_reddit.csv` - Reddit posts data from meme stock communities
- `dataset_reddit.py` - Python script to collect Reddit data via PullPush API
- `dataset_price_yahoo.csv` - Historical stock price data
- `dataset_price_yahoo.py` - Python script to download stock prices from Yahoo Finance

### Prompts & Results
The `Prompts/` folder contains:
- `PromptA.txt` through `PromptC5.txt` - Different LLM prompting strategies for sentiment analysis
- `Datasets/` subfolder - CSV outputs from each prompt variant (PromptA.csv through PromptC5.csv)

**Currently using:** PromptC4 (multi-dimensional sentiment analysis with 8 emotion scores + 4 emoji indicators)