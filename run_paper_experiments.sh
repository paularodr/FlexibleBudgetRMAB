#!/bin/bash
for F in 2 3 5;do
        for seed in {0..29}; do
                python3 python3 run_experiments.py --seed $seed --domain dropOutState --F $F --H 30 --N 10 --S 3 --niters 50 100 200
        done
        echo "dropOutState F:$F .... DONE"
done

for F in 2 5 10;do
        for seed in {0..29}; do
                python3 python3 run_experiments.py --seed $seed --domain immediateRecovery --F $F --H 10 --N 10 --S 5 --niters 50 100 200
        done
        echo "immediateRecovery F:$F .... DONE"
done

for F in 2 3 6;do
        for seed in {0..29}; do
                python3 python3 run_experiments.py --seed $seed --domain twoStateProcess --F $F --H 6 --N 10 --S 2 --niters 50 100 200
        done
        echo "twoStateProcess F:$F .... DONE"
done

