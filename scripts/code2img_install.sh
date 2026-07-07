#!/bin/bash

set -e

echo "Updating package lists..."
apt-get update

echo "Installing Chromium..."
apt-get install -y chromium-browser

echo "Installing Chromium dependencies..."
apt-get install -y \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpangocairo-1.0-0 \
    libpango-1.0-0 \
    libcairo2 \
    libgtk-3-0 \
    libx11-xcb1 \
    libxshmfence1 \
    libnss3 \
    libxss1 \
    libxtst6

pip install playwright jinja2

echo "Installing Playwright browser..."
playwright install chromium

echo "Installation complete!"