# Rebuttal Tables



All `±` values are sample standard deviations with `ddof=1`. Tables only include completed runs and compatible protocols.



## Table R1: Digit Sum

| method | bag_size_mean | train_bags | seeds | expected_sum_mae | rounded_sum_acc | rounded_sum_mae | instance_digit_acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FS-Conv | 10 | 1000 | 3 | 0.7039 ± 0.0287 | 0.8463 ± 0.0015 | 0.6120 ± 0.0079 | 0.9822 ± 0.0003 |
| FS-Conv | 50 | 1000 | 3 | 4.8643 ± 1.0032 | 0.0793 ± 0.0185 | 4.9010 ± 0.9734 | 0.6927 ± 0.2279 |
| Gaussian-AMLE | 10 | 1000 | 3 | 2.0325 ± 0.3931 | 0.2037 ± 0.0656 | 2.0117 ± 0.4020 | 0.6492 ± 0.1640 |



## Table R2: Signed Count

| method | bag_size_mean | train_bags | cancellation_heavy | seeds | expected_signed_count_mae | rounded_signed_count_acc | instance_acc | instance_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FS-Conv | 10 | 1000 | False | 3 | 0.1090 ± 0.0111 | 0.9073 ± 0.0111 | 0.9901 ± 0.0009 | 0.9949 ± 0.0012 |
| FS-Conv | 10 | 1000 | True | 3 | 0.0792 ± 0.0066 | 0.9300 ± 0.0062 | 0.9923 ± 0.0003 | 0.9978 ± 0.0001 |
| FS-Conv | 50 | 1000 | False | 3 | 0.2730 ± 0.0544 | 0.7743 ± 0.0485 | 0.9943 ± 0.0015 | 0.9985 ± 0.0009 |
| FS-Conv | 50 | 1000 | True | 3 | 1.6878 ± 0.0176 | 0.1880 ± 0.0026 | 0.1222 ± 0.0129 | 0.9689 ± 0.0034 |
| Gaussian-AMLE | 10 | 1000 | False | 3 | 0.2211 ± 0.0226 | 0.8423 ± 0.0289 | 0.9818 ± 0.0039 | 0.9898 ± 0.0014 |
| Gaussian-AMLE | 10 | 1000 | True | 3 | 0.2463 ± 0.0131 | 0.8237 ± 0.0199 | 0.9794 ± 0.0036 | 0.9900 ± 0.0030 |
| MSE | 10 | 1000 | False | 3 | 0.3296 ± 0.2583 | 0.7533 ± 0.2376 | 0.9585 ± 0.0517 | 0.9731 ± 0.0332 |
| MSE | 10 | 1000 | True | 3 | 0.1307 ± 0.0097 | 0.9180 ± 0.0095 | 0.9909 ± 0.0008 | 0.9961 ± 0.0008 |
| MSE | 50 | 1000 | False | 3 | 0.4712 ± 0.0259 | 0.6320 ± 0.0308 | 0.9914 ± 0.0009 | 0.9955 ± 0.0018 |
| MSE | 50 | 1000 | True | 3 | 0.3881 ± 0.0198 | 0.7120 ± 0.0221 | 0.9930 ± 0.0005 | 0.9984 ± 0.0003 |



## Table R3: Fixed-Bag CIFAR-10

| method | bag_size | train_bags | alpha | seeds | hist_count_mae | hist_proportion_mae | composite_count_nll | instance_acc | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |



## Table R4: FS-Conv vs LLP-PVC Count Component Equivalence

| batch_size | bag_size | classes | max_abs_forward_loss_diff | max_abs_per_class_probability_diff | max_abs_logit_gradient_diff |
| --- | --- | --- | --- | --- | --- |
| 5 | 13 | 7 | 4.440892098500626e-16 | 2.220446049250313e-16 | 2.1163626406917047e-16 |
