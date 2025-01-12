(function () {
    console.log("[StopSlop] Content script loaded");

    const port = chrome.runtime.connect({ name: "analyze" });
    port.onMessage.addListener((msg) => {
        console.log("[StopSlop] Received message", msg, "for", msg.url);
        if (msg.result === 0) {
            try {
                const urlObj = new URL(msg.url);
                const bannedDomain = urlObj.hostname;
                // Ban all links belonging to the banned domain
                const bannedLinks = document.querySelectorAll(`a[href*="${bannedDomain}"]`);
                bannedLinks.forEach((link) => {
                    const container = link.closest("div");
                    if (container) {
                        console.log("[StopSlop] Replacing container for:", link.href);
                        const messageDiv = document.createElement("div");
                        messageDiv.textContent = `Removed possible slop in ${link.href}`;
                        messageDiv.style.fontSize = "1.2em"; // Bigger font
                        container.replaceWith(messageDiv);
                        // Mark as processed
                        link.dataset.checked = "true";
                    }
                });
            } catch (e) {
                console.error("[StopSlop] URL parsing error for", msg.url, e);
            }
        }
    });

    const observer = new MutationObserver(() => {
        console.log("[StopSlop] Scanning links");
        const links = document.querySelectorAll("a[href*='http']");
        links.forEach((link) => {
            // Skip Google links
            if (link.href.includes("google.com")) return;
            if (link.dataset.checked === "true") return;
            link.dataset.checked = "true";
            console.log("[StopSlop] Analyzing link:", link.href);
            port.postMessage({ action: "analyzeURL", url: link.href });
        });
    });

    observer.observe(document.body, { childList: true, subtree: true });
    console.log("[StopSlop] Observer set up");
})();
