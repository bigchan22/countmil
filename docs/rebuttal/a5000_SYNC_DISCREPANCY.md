# A5000 synchronization note

The synchronized branch `neurips26-rebuttal` was fetched and checked out on the A5000 server.

- Checked-out branch tip: `1872e6c1fa7c3defdc894053fe5eb9f599270703`
- Recorded 4090 baseline: `5f7cc250a398df7df38668e58a4b36861293dea4`

The checked-out branch tip does not equal the recorded baseline SHA because it contains one additional bookkeeping commit:

```text
1872e6c rebuttal: record 4090 baseline branch SHA
```

The diff from `5f7cc250a398df7df38668e58a4b36861293dea4` to `1872e6c1fa7c3defdc894053fe5eb9f599270703` contains only:

```text
docs/rebuttal/BASELINE_SHA_4090.txt
```

No executable source, configuration, scripts, tests, or experiment code differ across these two SHAs. Therefore the executable code baseline for A5000 experiments is recorded as `5f7cc250a398df7df38668e58a4b36861293dea4`, with the checked-out documentation tip recorded separately.
