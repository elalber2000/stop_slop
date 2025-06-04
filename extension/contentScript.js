(function () {
    console.log("[StopSlop] Content script loaded");

    const port = chrome.runtime.connect({ name: "analyze" });

    port.onMessage.addListener((msg) => {
        console.log("[StopSlop] Received message", msg, "for", msg.url);
        if (typeof msg.score !== "number") return; // Defensive

        // Find all links matching this URL, not yet labeled
        const links = document.querySelectorAll(`a[href="${msg.url}"]`);
        links.forEach((link) => {
            // Only add if not already labeled
            if (link.dataset.stopslopProcessed === "true") return;

            // Score bands
            let color, label;
            if (msg.score < 0.15) {
                color = "red"; label = "Slop!";
            } else if (msg.score < 0.3) {
                color = "orange"; label = "Risk";
            } else if (msg.score < 0.6) {
                color = "yellow"; label = "Hmm";
            } else {
                color = "green"; label = "No Slop";
            }

            // Create button
            const btn = document.createElement("button");
            btn.textContent = label;
            btn.dir = "ltr";
            btn.style.unicodeBidi = "bidi-override";
            btn.title = `Slop score: ${msg.score.toFixed(3)}`;
            btn.style.background = color;
            btn.style.color = "white";
            btn.style.border = "none";
            btn.style.borderRadius = "4px";
            btn.style.marginRight = "6px";
            btn.style.fontWeight = "bold";
            btn.style.cursor = "pointer";
            btn.className = "stopslop-score-btn";

            link.parentNode.insertBefore(btn, link);
            link.dataset.stopslopProcessed = "true";
        });
    });

    // Watch for new links added to the DOM
    const observer = new MutationObserver(() => {
        console.log("[StopSlop] Scanning links");
        const links = document.querySelectorAll("a[href^='http']");
        links.forEach((link) => {
            if (link.dataset.stopslopChecked === "true") return;
            link.dataset.stopslopChecked = "true";
            // Skip Google links
            if (link.href.includes("google.com")) return;
            port.postMessage({ action: "analyzeURL", url: link.href });
        });
    });

    observer.observe(document.body, { childList: true, subtree: true });

    // Initial scan
    const initialLinks = document.querySelectorAll("a[href^='http']");
    initialLinks.forEach((link) => {
        if (link.dataset.stopslopChecked === "true") return;
        link.dataset.stopslopChecked = "true";
        if (link.href.includes("google.com")) return;
        port.postMessage({ action: "analyzeURL", url: link.href });
    });

    console.log("[StopSlop] Observer set up");
})();
