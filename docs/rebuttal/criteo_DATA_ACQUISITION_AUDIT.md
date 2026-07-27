# Criteo_x1 Data Acquisition Audit

- Hugging Face repo: `reczoo/Criteo_x1`
- Resolved revision: `4ecadde5eb8473e6ed6c570d9fb57e71079c0ef4`
- ZIP: `/home/chanhomin/datasets/criteo_x1/download/Criteo_x1.zip`
- ZIP size: `2892380492`
- ZIP SHA256: `ad87602d1a0c855a234da10f4510ba058990247380e961083f7d60b8bff62608`
- Download command: `curl -L --fail --retry 10 --retry-delay 10 --continue-at - -o /home/chanhomin/datasets/criteo_x1/download/Criteo_x1.zip https://huggingface.co/datasets/reczoo/Criteo_x1/resolve/main/Criteo_x1.zip?download=true`
- Caveat: RecZoo Criteo_x1 public preprocessing/partitions are used; this is LLP-Bench-style, not an exact original LLP-Bench five-fold reproduction.

## CSV Verification

| Split | Rows | MD5 | Path |
|---|---:|---|---|
| train | 33003326 | `30b89c1c7213013b92df52ec44f52dc5` | `/home/chanhomin/datasets/criteo_x1/extracted/train.csv` |
| valid | 8250124 | `f73c71fb3c4f66b6ebdfa032646bea72` | `/home/chanhomin/datasets/criteo_x1/extracted/valid.csv` |
| test | 4587167 | `2c48b26e84c04a69b948082edae46f8c` | `/home/chanhomin/datasets/criteo_x1/extracted/test.csv` |

## Disk

```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme1n1p2  879G  207G  627G  25% /
```
