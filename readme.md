# 🚫🧟 StopSlop

The objective of this project is to use AI to **avoid slop** (low-quality, unoriginal, or spammy material —often AI-generated— that adds noise rather than value) websites. Then, implement a chrome extension to automatically flag websites based on the classification.
<br><br>
Main limitations:
- The model should be very very lightweight (under \<10KB)
- The model should be blazing fast
- The model should be easily implementable without external libraries
- The model should be trained for final websites

This means (sadly) no transformers models / language models  D:

---

<p align="center">
  <b><a href="https://github.com/elalber2000/stop_slop/blob/main/extension.crx" target="_blank">Gradio Demo</a></b>
  .
  <b><a href="https://github.com/elalber2000/stop_slop/blob/main/extension.crx" target="_blank">Chrome Extension</a></b>
  ·
  <b><a href="https://huggingface.co/datasets/elalber2000/stop-slop-data-html" target="_blank">Raw HTML Dataset</a></b>
  ·
  <b><a href="https://huggingface.co/datasets/elalber2000/stop-slop-data" target="_blank">Parsed Text Datasetg</a></b>
</p>

---

## Dataset

I created a custom dataset by scrapping websites considered slop vs not-slop by domain.
<br><br>
Methodology:
- Use deep research to generate the list of websites (main url)
- Scrap a list of subwebsites coming from the url (I use a quick heuristic to define if they are worth)

The dataset it's not super clean (on purpose! I swear)
I wanted to do a proper clean up but actually I want some noise to account for edge cases in the extension

To check the dataset you have:
- [Dataset with raw HTML](https://huggingface.co/datasets/elalber2000/stop-slop-data-html)
- [Dataset with parsed text](https://huggingface.co/datasets/elalber2000/stop-slop-data)

# Model

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

My experience testing this the usual way is not very good, it's hard to get a representative test as there is a ton of variety on websites and the task itself is very qualitative. So I just went for the vibe eval instead (classic nlp engineer thing I guess) and used the extension for a while.

My results? Honestly not super great but for something under 5KB is pretty decent. Idk, try for yourself but I'm pretty happy with the results.
Still I added a whitelist and blacklist to the extension for convenience jeje


## Wanna Try?

- [Gradio demo (with some neat explainability)](https://huggingface.co/spaces/elalber2000/stop-slop)
- [Install the extension](https://github.com/elalber2000/stop_slop/blob/main/extension.crx)


## Structure

```plaintext
. 
├── extension.crx                       // Compiled chrome extension
├── extension                           // Code for the chrome extension
│   ├── background.js
│   ├── contentScript.js
│   ├── icons
│   │   ├── icon_128.png
│   │   ├── icon_16.png
│   │   └── icon_48.png
│   ├── inference.js
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   └── weights.json
├── notebooks                           // Notebooks (mainly for training)
│   ├── data_exploration.ipynb          // Data exploration to generate the dataset
│   ├── fasttext_training.ipynb         // First version of the model (trained on parsed text)
│   ├── fasttext_training-html.ipynb    // Final version of the model (trained on raw html)
│   └── old                             // Deprecated code of the first version
│       └── data_analysis_and_linear_model.ipynb
├── readme.md
├── src
│   ├── app                             // Gradio app
│   │   ├── app.py
│   │   ├── requirements.txt
│   │   └── weights.json
│   ├── config.py
│   ├── model
│   │   ├── tokenizer.py                // Tokenizer
│   │   ├── fasttext_np_nofeatures.py   // Fasttext model without additional features
│   │   ├── fasttext_np.py              // Fasttext model + additional features
│   │   └── fasttext_inference.py       // Direct fasttext model inference
│   ├── scrapping                       // Code to scrap the dataset
│   │   ├── chrome-setup.sh
│   │   ├── readme.md
│   │   ├── scrapping.py
│   │   └── sources.yaml
├── pyproject.toml
└── uv.lock
```