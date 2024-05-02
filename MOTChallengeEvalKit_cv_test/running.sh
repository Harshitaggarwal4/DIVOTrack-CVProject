#!/bin/bash

folder="/media/bhavb/E Volume/Dev_Linux/CV/CV Project/Delta_Ablation"

for entry in "$folder"/*/; do
    # Check if the entry is a directory
    if [ -d "$entry" ]; then
        # Print the full path to the directory
        fullpath="$(realpath "$entry")"
        name="$(basename "$entry")"
        echo "$name"
        # echo "$fullpath"
        cp cv_test/prepare_cross_view_eval.py cv_test/prepare_cross_view_eval1.py
        sed -i "s|^track_dir = .*|track_dir = \"$fullpath\"|" cv_test/prepare_cross_view_eval1.py

        python cv_test/prepare_cross_view_eval1.py
        echo "Running for F1"
        python MOT/evalMOT.py --name bhav > "./results/$name.txt"
        echo "Running for CVMA"
        python MOT/evalMOT.py --name bhav --type Acc > "./results/${name}_cvma.txt"
    fi
done
