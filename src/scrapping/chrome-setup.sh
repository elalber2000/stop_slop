#!/bin/bash
set -e  # exit on error

echo "[*] Updating packages"
sudo apt-get update

echo "[*] Installing wget and unzip"
sudo apt-get install -y wget unzip

echo "[*] Installing Google Chrome"
wget -q -O chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i chrome.deb || sudo apt-get -f install -y
sudo apt-get -f install -y
rm -f chrome.deb

echo "[*] Installing ChromeDriver"
CHROME_VERSION=$(google-chrome --version | grep -oP '\d+\.\d+\.\d+')
CHROMEDRIVER_VERSION=$(curl -s "https://chromedriver.storage.googleapis.com/LATEST_RELEASE_$CHROME_VERSION")
wget -q "https://chromedriver.storage.googleapis.com/${CHROMEDRIVER_VERSION}/chromedriver_linux64.zip"
unzip -o chromedriver_linux64.zip
sudo mv -f chromedriver /usr/local/bin/
sudo chmod +x /usr/local/bin/chromedriver
rm -f chromedriver_linux64.zip

echo "[+] Chrome and ChromeDriver installed"
