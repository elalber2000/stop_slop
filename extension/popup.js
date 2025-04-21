document.addEventListener("DOMContentLoaded", () => {
    const whitelistElem = document.getElementById("whitelist");
    const blacklistElem = document.getElementById("blacklist");
    const saveButton = document.getElementById("save");
  
    // Load settings
    chrome.storage.local.get(["whitelist", "blacklist"], (data) => {
      whitelistElem.value = (data.whitelist || []).join("\n");
      blacklistElem.value = (data.blacklist || []).join("\n");
    });
  
    // Save settings
    saveButton.addEventListener("click", () => {
      const whitelist = whitelistElem.value.split("\n").map(item => item.trim()).filter(Boolean);
      const blacklist = blacklistElem.value.split("\n").map(item => item.trim()).filter(Boolean);
      chrome.storage.local.set({ whitelist, blacklist }, () => {
        saveButton.textContent = "Settings Saved!";
        setTimeout(() => (saveButton.textContent = "Save Settings"), 1500);
      });
    });
  });
  