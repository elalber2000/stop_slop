import { inference } from "./inference.js";

/* seed built-in whitelist once */
const DEFAULT_WHITELIST = `
adobe.com
acrobat.com
pdf.com
office.net
wetransfer.com
dropbox.com
box.com
onedrive.com
docs.google.com
drive.google.com
icloud.com
docsend.com
scribd.com
file.io
sendgb.com
transfer.sh
zippyshare.com
mediafire.com
mega.nz
filemail.com
khanacademy.org
coursera.org
edx.org
udemy.com
duolingo.com
openai.com
code.org
alison.com
futurelearn.com
udacity.com
mit.edu
stanford.edu
harvard.edu
ocw.mit.edu
academicearth.org
open.edu
skillshare.com
brilliant.org
codecademy.com
lynda.com
pluralsight.com
sololearn.com
study.com
chegg.com
quizlet.com
byjus.com
edmodo.com
github.com
githubusercontent.com
gitlab.com
bitbucket.org
sourceforge.net
npmjs.com
pypi.org
rubygems.org
docker.com
hub.docker.com
python.org
nodejs.org
perl.org
cpan.org
cran.r-project.org
r-project.org
anaconda.com
readthedocs.io
maven.apache.org
packagist.org
nuget.org
dotnet.microsoft.com
jetbrains.com
vscode.dev
visualstudio.com
eclipse.org
gnome.org
kde.org
slackbuilds.org
freecodecamp.org
f-droid.org
osdn.net
wikipedia.org
wikimedia.org
archive.org
arxiv.org
nature.com
sciencedirect.com
ncbi.nlm.nih.gov
nih.gov
pubmed.ncbi.nlm.nih.gov
springer.com
sciencemag.org
newscientist.com
encyclopedia.com
britannica.com
infoplease.com
wolframalpha.com
stackexchange.com
stackoverflow.com
quora.com
snopes.com
factcheck.org
politifact.com
reuters.com
bbc.com
bbc.co.uk
nytimes.com
ft.com
economist.com
wsj.com
bloomberg.com
apnews.com
npr.org
theguardian.com
theatlantic.com
forbes.com
techcrunch.com
wired.com
arstechnica.com
engadget.com
cnet.com
zdnet.com
pcmag.com
slashdot.org
hackernews.com
producthunt.com
vox.com
vice.com
nationalgeographic.com
smithsonianmag.com
time.com
newsweek.com
usnews.com
usatoday.com
cnn.com
abcnews.go.com
cbsnews.com
nbcnews.com
pbs.org
aljazeera.com
dw.com
cbc.ca
globalnews.ca
sciencenews.org
journals.aps.org`
  .trim()
  .split("\n")
  .map(s => s.trim())
  .filter(Boolean);

chrome.runtime.onInstalled.addListener(() =>
  chrome.storage.local.get("whitelist", r => {
    if (!Array.isArray(r.whitelist) || !r.whitelist.length)
      chrome.storage.local.set({ whitelist: DEFAULT_WHITELIST });
  })
);

/* LIKE-style matcher */
const likeMatch = (val, pats) => {
  val = val.toLowerCase();
  return pats.some(raw => {
    const p = raw.trim().toLowerCase();
    if (!p) return false;
    if (p.includes("%")) {
      const re = new RegExp(
        p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/%/g, ".*"),
        "i"
      );
      return re.test(val);
    }
    return val.includes(p);
  });
};

/* main port */
chrome.runtime.onConnect.addListener(port => {
  let live = true;
  port.onDisconnect.addListener(() => (live = false));
  const post = m => { if (live) try { port.postMessage(m); } catch {} };

  port.onMessage.addListener(async req => {
    if (req.action !== "analyzeURL") return;

    let u;
    try { u = new URL(req.url); } catch { post({ url: req.url, score: 1 }); return; }
    const host = u.hostname.toLowerCase();
    const href = req.url.toLowerCase();

    chrome.storage.local.get(["whitelist", "blacklist"], async d => {
      const wl = d.whitelist?.length ? d.whitelist : DEFAULT_WHITELIST;
      const bl = d.blacklist || [];

      if (likeMatch(href, wl) || likeMatch(host, wl)) { post({ url: req.url, score: 1 }); return; }
      if (likeMatch(href, bl) || likeMatch(host, bl)) { post({ url: req.url, score: 0 }); return; }
      if (href.includes("google.com"))                { post({ url: req.url, score: 1 }); return; }

      const ctrl = new AbortController();
      port.onDisconnect.addListener(() => ctrl.abort());
      try {
        const html  = await (await fetch(req.url, { signal: ctrl.signal })).text();
        const score = await inference(html);
        post({ url: req.url, score });
      } catch {
        post({ url: req.url, score: 1 });
      }
    });
  });
});
