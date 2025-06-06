# 🚫🧟 StopSlop

The objective of this project is to use AI to avoid slop (low-quality, unoriginal, or spammy material—often AI-generated—that adds noise rather than value) websites. Then, implement a chrome extension to automatically flag websites based on the classification.
<br><br>
Main limitations:
- The model should be very very (very) lightweight
- The model should be blazing fast
- The model should be easily implementable without external libraries
- The model should be trained for final websites

This means (sadly) no transformers models / language models  D:


## Quickstart

### Chrome Extension
Install from
```stop_slop/extension.crx```

### Gradio Demo
```https://huggingface.co/spaces/elalber2000/stop-slop```

### Dataset
- Raw HTML: ```https://huggingface.co/datasets/elalber2000/stop-slop-data-html```
- Parsed text: ```https://huggingface.co/datasets/elalber2000/stop-slop-data```

## Dataset

I created a custom dataset by scrapping websites considered slop vs not-slop by domain.
<br><br>
Methodology:
- Use deep research to generate the list of websites (main url)
- Scrap a list of subwebsites coming from the url (I use a quick heuristic to define if they are worth)

The dataset it's not super clean (on purpose! I swear)
I wanted to do a proper clean up but actually I want some noise to account for edge cases in the extension

To check the dataset you have:
- Dataset with raw HTML: ```https://huggingface.co/datasets/elalber2000/stop-slop-data-html```
- Dataset with parsed text: ```https://huggingface.co/datasets/elalber2000/stop-slop-data```
<br><br>
The previous simple approach considered just doing some feature engineering
with linear regression on top.

The new approach is a bit more ambitious, it involves:
1. Implementing a fasttext model with numpy to handle vocabulary
2. Adding hand-crafted features (trained simultaneously as fasttext)
3. Classifying with a simple NN (linear + softmax) -> This is binary classification so no need for fancy hierarchical-softmax or negative-sampling
4. Train on raw html (after preprocessing it)
5. Do some explainability tests (mainly looking the impact of each n-gram and feature) and keep only the most important and less overfittingy n-grams and features
6. Retraining only with those n-grams and features
7. Implementing the neat optimization trick of collapsing mean-pool + linear step [Zolotov & Kung (2017), Joulin et al. (2016)]
8. Implementing the inference pipeline on javascript for the chrome extension


## Results

I should have probably implemented some test dataset but honestly I wanted to train with as much data as possible, also the task is pretty difficult so still the test wouldn't have been that good to be honest. So I just went for the vibe eval instead (classic nlp engineer thing I guess) and used the extension for a while.

My results? Honestly not super great but gets more than I expected right. Idk, try for yourself but I'm pretty happy with the results.
Still I added a whitelist and blacklist to the extension for convenience jeje


## Wanna Try?

- If you want the extension: Install it from ```stop_slop/extension.crx```
- If you want to try the gradio demo (with some neat explainability): ```https://huggingface.co/spaces/elalber2000/stop-slop```