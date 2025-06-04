let W_NUM, BIAS, MU, SIGMA, U_EMB, initDone = false;

function flatten1d(arr) {
    if (Array.isArray(arr) && Array.isArray(arr[0]) && arr.length === 1)
        return arr[0];
    return arr;
}

async function init() {
    if (initDone) return;
    const url = chrome.runtime.getURL("weights.json");
    const w   = await (await fetch(url)).json();

    // Handle weight shapes
    W_NUM = w.W_num; // 12x2, not Float32Array
    BIAS  = Float32Array.from(w.bias); // [2]
    MU    = Float32Array.from(flatten1d(w.mu)); // [12]
    SIGMA = Float32Array.from(flatten1d(w.sigma)); // [12]
    U_EMB = Object.fromEntries(
        Object.entries(w.U).map(([k, v]) => [k, Float32Array.from(v)])
    );
    initDone = true;

    console.log("[StopSlop:init] loaded sizes",
        { W_NUM: (W_NUM && W_NUM.length), BIAS: BIAS.length,
          MU: MU.length, SIGMA: SIGMA.length,
          U_keys: Object.keys(U_EMB).length });
}

// ─── static tables & helpers (unchanged, collapsed for brevity) ───
const ALLOWED_LENGTHS = new Set([4,5,6,7,8,9,10]);
const NUM_COLUMNS = [
    "as_i_x_i_will_y","i_x_that_is_not_y_but_z","iframe_count","inline_css_ratio",
    "links_per_kb","markup_to_text_ratio","prp_ratio","sentences_per_paragraph",
    "stopword_ratio","straight_apostrophe","type_token_ratio","vbg"
];
const STOPWORDS = new Set(["the","and","is","in","it","of","to","a","with","that","for","on","as","are","this","but","be","at","or","by","an","if","from","about","into","over","after","under"]);
const RE_SCRIPT_STYLE = /<(?:script|style)[^>]*>.*?<\/(?:script|style)>/gis;
const RE_TAG = /<[^>]+>/g, RE_SENT=/[.!?]+/g, RE_PARA=/\n{2,}/g, RE_WORDS=/\w+/g;
const RE_TAG_NAME=/<\s*(\w+)/ig, RE_IFRAME=/<\s*iframe\b/ig, RE_LINK=/href=["']([^"']+)["']/ig;
const EXPRS = {
  i_x_that_is_not_y_but_z:/\bI\s+\w+\s+that\s+is\s+not\s+\w+,\s*but\s+\w+/i,
  as_i_x_i_will_y:/\bAs\s+I\s+\w+,\s*I\s+will\s+\w+/i
};
const meanVec = arr => {
    const out = new Float32Array(arr[0].length);
    arr.forEach(v => { for (let i=0;i<v.length;i++) out[i]+=v[i]; });
    for (let i=0;i<out.length;i++) out[i]/=arr.length;
    return out;
};
const softmax = logits => {
    const m = Math.max(...logits);
    const ex = logits.map(x => Math.exp(x - m));
    const s  = ex.reduce((a,b)=>a+b,0);
    return ex.map(x=>x/s);
};
function featureDict(html){
    const cleaned = html.replace(RE_SCRIPT_STYLE,"");
    const text    = cleaned.replace(RE_TAG," ");
    const tokens  = text.toLowerCase().match(RE_WORDS)||[];
    const paragraphs=text.split(RE_PARA).filter(p=>p.trim());
    const bytesTotal=html.length, bytesText=text.length;
    const nTags=[...html.toLowerCase().matchAll(RE_TAG_NAME)].length||1;
    const iframeCnt=(html.match(RE_IFRAME)||[]).length;
    const hrefs=[...html.matchAll(RE_LINK)];
    const stopCnt=tokens.reduce((c,t)=>c+STOPWORDS.has(t),0);
    const sentPerPara=paragraphs.length?
        paragraphs.reduce((c,p)=>c+p.split(RE_SENT).filter(s=>s.trim()).length,0)/paragraphs.length:0;
    const typeToken=tokens.length?new Set(tokens).size/tokens.length:0;
    const prpCnt=(text.match(/\b(?:I|me|you|he|she|it|we|they|him|her|us|them)\b/ig)||[]).length;
    const vbgCnt=(text.match(/\b\w+ing\b/g)||[]).length;
    return {
        stopword_ratio:tokens.length?stopCnt/tokens.length:0,
        links_per_kb:bytesTotal?hrefs.length/(bytesTotal/1024):0,
        type_token_ratio:typeToken,
        i_x_that_is_not_y_but_z:(text.match(EXPRS.i_x_that_is_not_y_but_z)||[]).length,
        prp_ratio:tokens.length?prpCnt/tokens.length:0,
        sentences_per_paragraph:sentPerPara,
        markup_to_text_ratio:bytesTotal?(bytesTotal-bytesText)/bytesTotal:0,
        inline_css_ratio:(html.match(/style=/gi)||[]).length/nTags,
        iframe_count:iframeCnt,
        as_i_x_i_will_y:(text.match(EXPRS.as_i_x_i_will_y)||[]).length,
        vbg:vbgCnt,
        straight_apostrophe:(text.match(/'/g)||[]).length
    };
}

// ─── main entry ───────────────────────────────────────────
export async function inference(html){
    await init();
    console.log("[StopSlop:infer] html bytes", html.length);

    // ----- token-level embeddings -----
    const toks = (html.toLowerCase().match(/\w+|[^\w\s]+/g)) || [];
    const wEmb=[];
    toks.forEach(word=>{
        const subs=[];
        ALLOWED_LENGTHS.forEach(L=>{
            if(word.length<L) return;
            for(let i=0;i<=word.length-L;i++){
                const sub=word.slice(i,i+L), e=U_EMB[sub];
                if(e) subs.push(e);
            }
        });
        if(subs.length) wEmb.push(meanVec(subs));
    });
    const textScore = wEmb.length?meanVec(wEmb):new Float32Array(BIAS.length);
    console.log("[StopSlop:infer] tokens", toks.length,
                "wEmb", wEmb.length,
                "textScore len", textScore.length);

    // ----- engineered features -----
    const feats   = featureDict(html);
    const numRaw  = NUM_COLUMNS.map(c=>feats[c]||0);
    const numStd  = numRaw.map((v,i)=>(v-MU[i])/SIGMA[i]);

    // Compute per-class feature score
    const numScore = W_NUM[0].map((_, k) => // k = class 0, class 1
        numStd.reduce((sum, v, i) => sum + v * W_NUM[i][k], 0)
    ); // [score0, score1]
    console.log("[StopSlop:infer] numScore", numScore);

    // ----- combine & classify -----
    const logits = textScore.map((t,i)=>t+numScore[i]+BIAS[i]);
    console.log("[StopSlop:infer] logits len", logits.length,
                "sample", logits.slice(0,4));

    if(logits.length===0){
        console.warn("[StopSlop:infer] EMPTY logits – weight file corrupt?");
        return null;
    }
    if(logits.length===1){
        const p = 1/(1+Math.exp(-logits[0]));
        console.log("[StopSlop:infer] binary p", p);
        return p;
    }

    const probs = softmax(logits);
    console.log("[StopSlop:infer] probs", probs.slice(0,4));
    return probs[1];        // “OK” probability
}
