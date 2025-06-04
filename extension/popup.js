document.addEventListener("DOMContentLoaded", () => {
  const wlBox = document.getElementById("whitelist");
  const blBox = document.getElementById("blacklist");
  const save  = document.getElementById("save");

  chrome.storage.local.get(["whitelist", "blacklist"], d => {
    wlBox.value = (d.whitelist || []).join("\n");
    blBox.value = (d.blacklist || []).join("\n");
  });

  save.addEventListener("click", () => {
    const wl = wlBox.value.split("\n").map(x => x.trim()).filter(Boolean);
    const bl = blBox.value.split("\n").map(x => x.trim()).filter(Boolean);
    chrome.storage.local.set({ whitelist: wl, blacklist: bl }, () => {
      save.textContent = "Saved!";
      setTimeout(() => (save.textContent = "Save Settings"), 1200);
    });
  });
});
