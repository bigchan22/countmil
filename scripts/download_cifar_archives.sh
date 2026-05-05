#!/usr/bin/env bash
set -euo pipefail

mkdir -p data logs

download_archive() {
  local name="$1"
  local url="$2"
  local md5="$3"
  local final="data/${name}.tar.gz"
  local part="${final}.part"

  if [ -f "$final" ] && tar -tzf "$final" >/dev/null 2>&1; then
    if echo "${md5}  ${final}" | md5sum -c - >/dev/null 2>&1; then
      echo "${name}: already complete"
      return 0
    fi
    echo "${name}: existing archive has unexpected md5; keeping it and redownloading to ${part}"
  fi

  if [ ! -f "$part" ] && [ -f "$final" ]; then
    echo "${name}: seeding resumable .part from existing partial archive"
    cp "$final" "$part"
  fi

  echo "${name}: downloading ${url}"
  curl -L --fail --retry 200 --retry-delay 20 --connect-timeout 120 \
    --speed-time 300 --speed-limit 1024 --continue-at - \
    -o "$part" "$url"

  echo "${name}: verifying tar"
  tar -tzf "$part" >/dev/null
  echo "${name}: verifying md5"
  echo "${md5}  ${part}" | md5sum -c -
  mv "$part" "$final"
  echo "${name}: complete at ${final}"
}

download_archive \
  "cifar-10-python" \
  "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" \
  "c58f30108f718f92721af3b95e74349a"

download_archive \
  "cifar-100-python" \
  "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz" \
  "eb9058c3a382ffc7106e4002c42a8d85"
