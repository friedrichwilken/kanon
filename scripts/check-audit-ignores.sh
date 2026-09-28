#!/usr/bin/env bash
# Fails when an advisory ignored in .cargo/audit.toml is no longer reported for Cargo.lock.
#
# `cargo audit` says nothing about an ignore that is no longer needed, so without this a fixed
# advisory keeps its entry for good and would go on hiding a later advisory with the same id. It
# audits a copy of Cargo.lock in an empty directory, where no .cargo/audit.toml applies, and
# requires every id the file ignores to be among the findings. Needs cargo-audit and jq.
#
#   scripts/check-audit-ignores.sh
#
# AUDIT_TOML and LOCK override the two files, which is how the script itself is tested.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
audit_toml="${AUDIT_TOML:-$root/.cargo/audit.toml}"
lock="${LOCK:-$root/Cargo.lock}"

# The ids in the `ignore = [...]` list: one quoted id per line, as the file's header asks.
ignored="$(grep -oE '^[[:space:]]*"RUSTSEC-[0-9]{4}-[0-9]{4}"' "$audit_toml" | tr -d ' "' || true)"
if [[ -z "$ignored" ]]; then
  echo "no advisories ignored in $audit_toml"
  exit 0
fi

scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
cp "$lock" "$scratch/Cargo.lock"
# cargo audit exits 1 when it finds a vulnerability: the findings are what is wanted here.
report="$(cd "$scratch" && cargo audit --json 2>/dev/null || true)"
if ! jq -e '.vulnerabilities' <<<"$report" >/dev/null 2>&1; then
  echo "cargo audit produced no report: is cargo-audit installed and the advisory database reachable?" >&2
  exit 1
fi
reported="$(jq -r '[.vulnerabilities.list[]?.advisory.id, (.warnings // {} | to_entries[].value[]?.advisory.id)] | .[]' <<<"$report")"

status=0
while read -r id; do
  if grep -qx "$id" <<<"$reported"; then
    echo "$id: still reported, the ignore is still needed"
  else
    echo "$id: ignored in $audit_toml but no longer reported for $lock; delete the entry (its exit condition holds)" >&2
    status=1
  fi
done <<<"$ignored"
exit "$status"
