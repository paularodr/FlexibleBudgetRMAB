#!/bin/bash
for T in 2 3 5
do
        (for i in {0..29}
        do
                echo "T $T seed $i"
                python3 run_experiments.py --seed $i --domain dropOutState --T $T --H 30 --N 10 --S 3
        done)
        echo "T $T DONE"
done