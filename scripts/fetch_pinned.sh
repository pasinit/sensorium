#!/usr/bin/env bash
# Fetch a few directories of a git repo at one pinned commit into a directory.
#
# Usage: scripts/fetch_pinned.sh <repo-url> <commit-sha> <dest-dir> <path>...
#
# Idempotent: if <dest-dir> is already at <commit-sha> it does nothing. Only the
# listed paths (plus top-level files such as LICENSE) are checked out, and blobs
# are fetched lazily, so large repos stay small on disk.
set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "usage: $0 <repo-url> <commit-sha> <dest-dir> <path>..." >&2
    exit 2
fi
url=$1
sha=$2
dest=$3
shift 3

if [ -d "$dest/.git" ] && [ "$(git -C "$dest" rev-parse HEAD 2>/dev/null || true)" = "$sha" ]; then
    missing=0
    for path in "$@"; do
        [ -e "$dest/$path" ] || missing=1
    done
    if [ "$missing" = 0 ]; then
        echo "$dest already at $sha"
        exit 0
    fi
fi

mkdir -p "$dest"
git -C "$dest" init -q .
git -C "$dest" remote remove origin 2>/dev/null || true
git -C "$dest" remote add origin "$url"
git -C "$dest" sparse-checkout init --cone
git -C "$dest" sparse-checkout set "$@"
git -C "$dest" fetch -q --depth 1 --filter=blob:none origin "$sha"
git -C "$dest" checkout -q --force FETCH_HEAD
echo "$dest at $(git -C "$dest" rev-parse HEAD)"
