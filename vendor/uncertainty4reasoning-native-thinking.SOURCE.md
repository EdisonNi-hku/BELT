# Native-thinking UHead source snapshot

This is an immutable tracked snapshot for Qwen3-1.7B DAPO annotation, ReProbe/UHead training, layer sweep, and BoN evaluation. It is a Git-tree import of the source commit and tree below with exactly two reviewed credential redactions.

- Source URL: https://github.com/cant-access-rediska0123/uncertainty4reasoning
- Source commit: `8356374b6b04e9f409d0038c0d864a432a1dcae9`
- Source tree: `f1d06fb7043ab69e1757313457ebf1fcb2019579`
- Exported path count: 600
- Target prefix: `vendor/uncertainty4reasoning-native-thinking/`
- Sanitized snapshot tree: `b52957d598a65105be5d6b66d2f7b4f59852971d`

The only snapshot edits are these reviewed credential redactions:

- `synthetic_dataset_generation/scripts/extract_claim_qwen3_1.7b_thinking.sh`
- `synthetic_dataset_generation/scripts/extract_claim_qwen3_8b_thinking.sh`

In each file, the source credential export is replaced with:

```sh
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export HF_TOKEN
```

No other target-specific snapshot edits are allowed. The ignored root `uncertainty4reasoning/` is a separate runtime clone that may contain local controller patches. The source commit is not currently reachable from the stated remote; this target branch is the durable review snapshot. No top-level upstream license was present, so redistribution rights should be confirmed before reuse outside this project.

Update policy: import a new reviewed source commit atomically, apply only reviewed credential redactions, and update all identities in this manifest.

Verify this snapshot with:

```sh
test "$(git rev-parse HEAD:vendor/uncertainty4reasoning-native-thinking)" = "b52957d598a65105be5d6b66d2f7b4f59852971d"
if ! command -v rg >/dev/null 2>&1; then
  echo "HF-token-shaped scan failed: rg is required" >&2
  exit 2
fi
if rg -l 'hf_[A-Za-z0-9]{20,}' vendor/uncertainty4reasoning-native-thinking >/dev/null; then
  echo "HF-token-shaped string found; stop and review privately" >&2
  exit 1
else
  hf_scan_status=$?
  case "${hf_scan_status}" in
    1) : ;;
    *) echo "HF-token-shaped scan failed with rg status ${hf_scan_status}" >&2; exit "${hf_scan_status}" ;;
  esac
fi
```
