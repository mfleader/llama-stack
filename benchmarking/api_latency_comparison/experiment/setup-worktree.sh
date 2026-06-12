#!/bin/bash
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -euo pipefail

# Sets up OGX worktrees for the CI regression benchmark.
# Creates isolated worktrees for baseline and comparison versions,
# applies the brave-search base_url patch so the mock server can
# serve search results locally.
#
# Usage:
#   ./setup-worktree.sh --baseline v1.0.2 --baseline-label baseline \
#     --comparison HEAD --comparison-label comparison

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_BASE="/tmp/ci-validation"
SKIP_DEPS=false
MOCK_PORT=8080
STACK_PORT=8321

log_msg() {
    echo "[$(date +%H:%M:%S)] $*"
}

REPO=""
BASELINE_REF=""
BASELINE_LABEL=""
COMPARISON_REF=""
COMPARISON_LABEL=""

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Sets up OGX worktrees for the CI benchmark experiment.

Required:
    --baseline REF          Git ref for baseline (tag, branch, or commit)
    --baseline-label NAME   Label for baseline (e.g., v1.0.2)
    --comparison REF        Git ref for comparison (tag, branch, or commit)
    --comparison-label NAME Label for comparison (e.g., HEAD)

Options:
    --repo DIR              Path to OGX git repository (default: auto-detect)
    --worktree-base DIR     Base directory for worktrees (default: $WORKTREE_BASE)
    --skip-deps             Skip dependency install
    --mock-port PORT        Mock server port (default: $MOCK_PORT)
    --stack-port PORT       OGX server port (default: $STACK_PORT)
    --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --repo)              REPO="$2"; shift 2 ;;
        --baseline)          BASELINE_REF="$2"; shift 2 ;;
        --baseline-label)    BASELINE_LABEL="$2"; shift 2 ;;
        --comparison)        COMPARISON_REF="$2"; shift 2 ;;
        --comparison-label)  COMPARISON_LABEL="$2"; shift 2 ;;
        --worktree-base)     WORKTREE_BASE="$2"; shift 2 ;;
        --skip-deps)         SKIP_DEPS=true; shift ;;
        --mock-port)         MOCK_PORT="$2"; shift 2 ;;
        --stack-port)        STACK_PORT="$2"; shift 2 ;;
        --help)              usage; exit 0 ;;
        *) echo "Error: unknown option $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "$REPO" ]]; then
    REPO="$(git rev-parse --show-toplevel 2>/dev/null || true)"
fi

if [[ ! -d "$REPO/.git" ]]; then
    echo "Error: $REPO is not a git repository" >&2
    echo "Use --repo to specify the OGX repo path" >&2
    exit 1
fi

if [[ -z "$BASELINE_REF" || -z "$BASELINE_LABEL" || -z "$COMPARISON_REF" || -z "$COMPARISON_LABEL" ]]; then
    echo "Error: --baseline, --baseline-label, --comparison, and --comparison-label are required" >&2
    usage >&2
    exit 1
fi

log_msg "=== CI Regression Worktree Setup ==="
log_msg "  Repo: $REPO"
log_msg "  Worktree base: $WORKTREE_BASE"
echo ""

BASELINE_DIR="$WORKTREE_BASE/$BASELINE_LABEL"
COMPARISON_DIR="$WORKTREE_BASE/$COMPARISON_LABEL"

resolve_ref() {
    local ref="$1"
    # Resolve a user-supplied ref (tag, branch, remote branch, or SHA)
    # to a form that git worktree add --detach can use.
    if git rev-parse --verify "$ref^{commit}" >/dev/null 2>&1; then
        echo "$ref"
        return
    fi
    # Bare branch name (e.g., "main") with no local branch but origin/main exists.
    # actions/checkout only creates a local branch for the checked-out ref.
    if git rev-parse --verify "origin/$ref^{commit}" >/dev/null 2>&1; then
        echo "origin/$ref"
        return
    fi
    echo "Error: cannot resolve git ref '$ref'" >&2
    echo "  Tried: '$ref', 'origin/$ref'" >&2
    return 1
}

setup_worktree() {
    local version="$1"
    local git_ref="$2"
    local wt_dir="$3"

    log_msg "Setting up worktree: $version (ref: $git_ref) -> $wt_dir"

    if [[ -d "$wt_dir" ]]; then
        log_msg "  Worktree exists, resetting to clean state..."
        cd "$wt_dir"
        git checkout HEAD -- .
    else
        cd "$REPO"
        git worktree prune
        local resolved
        resolved=$(resolve_ref "$git_ref")
        log_msg "  Resolved ref: $git_ref -> $resolved"
        git worktree add "$wt_dir" "$resolved" --detach
    fi

    cd "$wt_dir"

    if [[ "$SKIP_DEPS" == false ]]; then
        log_msg "  Installing dependencies..."
        uv sync 2>&1 | tail -3
    fi

    # Patch brave-search provider to accept a configurable base_url.
    # Uses sed instead of git apply because the surrounding context lines
    # differ between versions (e.g., type annotations added in later releases).
    local bs_impl="$wt_dir/src/ogx/providers/remote/tool_runtime/brave_search/brave_search.py"
    local bs_conf="$wt_dir/src/ogx/providers/remote/tool_runtime/brave_search/config.py"

    if ! grep -q 'self.config.base_url' "$bs_impl" 2>/dev/null; then
        if grep -q 'url = "https://api.search.brave.com/res/v1/web/search"' "$bs_impl"; then
            log_msg "  Patching brave-search base_url..."
            sed -i 's|url = "https://api.search.brave.com/res/v1/web/search"|url = (self.config.base_url or "https://api.search.brave.com") + "/res/v1/web/search"|' "$bs_impl"
            sed -i '/^    max_results.*Field(/,/^    )/{/^    )/a\    base_url: str | None = Field(\n        default=None,\n        description="Override base URL for the search API (e.g., http://localhost:8080 for mock)",\n    )
            }' "$bs_conf"
        else
            log_msg "ERROR: Cannot find brave-search URL line to patch in $version"
            log_msg "  Expected: url = \"https://api.search.brave.com/res/v1/web/search\""
            exit 1
        fi
    else
        log_msg "  brave-search base_url already patched"
    fi

    # Verify the patch took effect
    if ! grep -q 'self.config.base_url' "$bs_impl"; then
        log_msg "ERROR: brave-search patch verification failed for $version"
        exit 1
    fi

    local commit_hash
    commit_hash=$(git rev-parse HEAD)
    log_msg "  OK: $version at $commit_hash"

    cat > "$wt_dir/.benchmark-env" <<ENVEOF
export WORKTREE_DIR="$wt_dir"
export OPENAI_API_KEY="fake-token"
export MOCK_PORT=$MOCK_PORT
export STACK_PORT=$STACK_PORT
ENVEOF
}

# Step 1: Baseline worktree
setup_worktree "$BASELINE_LABEL" "$BASELINE_REF" "$BASELINE_DIR"
echo ""

# Step 2: Comparison worktree
setup_worktree "$COMPARISON_LABEL" "$COMPARISON_REF" "$COMPARISON_DIR"
echo ""

# Verify worktree hashes match expected refs
BASELINE_HASH=$(cd "$BASELINE_DIR" && git rev-parse HEAD)
COMPARISON_HASH=$(cd "$COMPARISON_DIR" && git rev-parse HEAD)

log_msg "=== Setup Complete ==="
echo ""
echo "  $BASELINE_LABEL:    $BASELINE_DIR ($BASELINE_HASH)"
echo "  $COMPARISON_LABEL:  $COMPARISON_DIR ($COMPARISON_HASH)"
echo ""
echo "Cleanup:"
echo "  git -C $REPO worktree remove $BASELINE_DIR"
echo "  git -C $REPO worktree remove $COMPARISON_DIR"
