#!/usr/bin/env bash
# Keep CI's apt steps working when the runner cannot resolve Ubuntu's package hosts.
#
# The hosted runner images list three mirrors in /etc/apt/apt-mirrors.txt, all under
# ubuntu.com (azure.archive, archive, security). When those names do not resolve (as in
# actions/runner-images#15296) every apt-get install fails before a test runs. This adds
# retries and a full mirror on another domain: last while the Ubuntu hosts resolve, so a
# healthy job never uses it, and first when they do not, so apt does not wait out three
# dead hosts per file. apt checks every index against the Ubuntu archive key whichever
# host served it, so the mirror cannot change what is installed.
set -euo pipefail

echo 'Acquire::Retries "3";' | sudo tee /etc/apt/apt.conf.d/zzz-sdca-retries >/dev/null

mirrors=/etc/apt/apt-mirrors.txt
fallback=https://mirrors.edge.kernel.org/ubuntu/
[ -f "$mirrors" ] || exit 0
grep -qF "$fallback" "$mirrors" && exit 0

if getent hosts azure.archive.ubuntu.com >/dev/null 2>&1; then
  printf '%s\tpriority:%s\n' "$fallback" "$(($(grep -c . "$mirrors") + 1))" |
    sudo tee -a "$mirrors" >/dev/null
else
  echo "::warning title=apt mirror::Ubuntu's package hosts do not resolve; apt uses $fallback first."
  {
    printf '%s\tpriority:1\n' "$fallback"
    awk -F'\t' 'NF { print $1 "\tpriority:" NR + 1 }' "$mirrors"
  } | sudo tee "$mirrors.new" >/dev/null
  sudo mv "$mirrors.new" "$mirrors"
fi
cat "$mirrors"
