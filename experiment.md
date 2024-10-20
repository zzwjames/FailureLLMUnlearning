1. Given the learned model, 'cd baselines' and then run 'bash scripts/unlearn_news.sh' to get unlearned model.
    (i) In unlearn_news.sh, change out_dir to store the unlearned model

2. run '/muse_bench/bash eval.sh' to eval the unlearned model.
    (i) In eval.sh, 'model_dirs' is the address of unlearned model
    (ii) When test method 'whp', remmeber to change the 'reinforced_name_or_path' in utils.py
