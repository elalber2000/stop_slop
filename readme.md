# 🚫🧟 StopSlop

The objective of this notebook is to create a system to identify slop websites.
The system should be fast, adaptable to js and non-reliable on external libraries. Because of that, the approach will be:
1. Create a slop vs non-slop dataset with web scrapping
2. Identify interesting features that can be ran with regex
3. Train a small linear model to calculate feature weights
4. Implement it in a chrome extension with a whitelist and blacklist

Take in mind that this is not a perfect system to detect AI generated text.
- Slop is not only ai generated, it can also be seo articles made to sell or drive engagement
- The system is made to be fast and simple, there's better SOTA stuff but requires transformers and stuff like that which is not viable for a simple chrome extension
- The data is not super big but this is more of a interesting project than a real app