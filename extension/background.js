console.log("[StopSlop] Background service worker loaded");

// Model parameters
const weights = [
  0.003224097730849752,
  0.041174883598401546,
  -0.08363519093665123,
  0.007118889863322623,
  -0.24675198467522655,
  -0.20711948433976712,
  -0.01923992845820089,
  -0.06415759681531612,
  0.057993917413981785,
  0.023981672786734798,
  -0.012971271988087783,
  0.005762784684722451,
  0.019531599440375018
];
const intercept = 0.4748231644400827;

/**
 * Extracts only paragraph text.
 * Finds all <p>...</p> segments, strips any HTML tags inside.
 */
function extractParagraphText(html) {
  const paragraphMatches = html.match(/<p[^>]*>([\s\S]*?)<\/p>/gi);
  if (!paragraphMatches) return "";
  const paragraphs = paragraphMatches.map((p) =>
    p.replace(/<[^>]+>/g, "").trim()
  );
  return paragraphs.join(" ");
}

// Feature extractor
function extractFeatures(text) {
  return {
    contains_dash: (text.match(/-/g) || []).length,
    contains_em_dash: (text.match(/—/g) || []).length,
    contains_double_quote: (text.match(/"/g) || []).length,
    contains_smart_double_quote: (text.match(/[“”]/g) || []).length,
    contains_single_quote_pair: text.match(/'/g) && text.match(/’/g) ? 1 : 0,
    contains_seamless: (text.match(/\bseamless\b/gi) || []).length,
    contains_elevate: (text.match(/\belevate\b/gi) || []).length,
    contains_pivotal: (text.match(/\bpivotal\b/gi) || []).length,
    contains_align: (text.match(/\balign\b/gi) || []).length,
    contains_leverage: (text.match(/\bleverage\b/gi) || []).length,
    number_postags_vbg: (text.match(/\b\w+ing\b/g) || []).length,
    number_postags_prp: (
      text.match(
        /\b(?:I|me|you|he|she|it|we|they|him|her|us|them|myself|yourself|herself|himself|itself|ourselves|themselves)\b/gi
      ) || []
    ).length,
    number_postags_rb: (
      text.match(
        /\b(\w+ly|here|there|now|then|soon|always|never|often|sometimes|everywhere|nowhere|somewhere|very|too|quite|almost|rather|fast|hard|late|near|far|straight|well)\b/gi
      ) || []
    ).length
  };
}

function predict(text, threshold = 0.3) {
  console.log("[StopSlop] Analyzing truncated text:", text.slice(0, 10000), "...");
  const feats = extractFeatures(text);
  let score = intercept;
  const keys = Object.keys(feats);
  for (let i = 0; i < keys.length; i++) {
    score += feats[keys[i]] * weights[i];
  }
  console.log("[StopSlop] Score computed:", score);
  return score >= threshold ? 1 : 0;
}

// Listen for messages from the content script
chrome.runtime.onConnect.addListener((port) => {
  console.log("[StopSlop] Connected port:", port.name);
  port.onMessage.addListener(async (request) => {
    if (request.action === "analyzeURL") {
      console.log("[StopSlop] analyzeURL for", request.url);

      let domain;
      try {
        domain = new URL(request.url).hostname;
      } catch (e) {
        console.error("[StopSlop] Invalid URL", request.url);
        port.postMessage({ url: request.url, result: 1 });
        return;
      }

      // Retrieve whitelist and blacklist from storage
      chrome.storage.local.get(["whitelist", "blacklist"], async (data) => {
        const whitelist = data.whitelist || [];
        const blacklist = data.blacklist || [];

        // If whitelisted, bypass filtering
        if (whitelist.includes(domain)) {
          console.log(`[StopSlop] ${domain} is whitelisted`);
          port.postMessage({ url: request.url, result: 1 });
          return;
        }

        // If blacklisted, force filter (set result 0)
        if (blacklist.includes(domain)) {
          console.log(`[StopSlop] ${domain} is blacklisted`);
          port.postMessage({ url: request.url, result: 0 });
          return;
        }

        // Skip analyzing Google links directly
        if (request.url.includes("google.com")) {
          port.postMessage({ url: request.url, result: 1 });
          return;
        }

        try {
          const response = await fetch(request.url);
          let html = await response.text();
          let text = extractParagraphText(html);
          const result = predict(text);
          console.log("[StopSlop] Score for", request.url, ":", result);
          port.postMessage({ url: request.url, result });
        } catch (err) {
          console.error("[StopSlop] Fetch error for", request.url, err);
          port.postMessage({ url: request.url, result: 1 });
        }
      });
    }
  });
});
