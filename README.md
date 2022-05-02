steps to run code:
# use greedy in solve_all 
python python/solve_all.py inputs test

# use greedy2 in solve_all
python python/solve_all.py inputs test1

python python/merge.py --inputs inputs test test1 test

# use greedy3 in solve_all
python python/solve_all.py inputs test2

python python/merge.py --inputs inputs test test2 test

# use greedy4 in solve_all
python python/solve_all.py inputs test3

python python/merge.py --inputs inputs test test3 test

# use greedy5 in solve_all
python python/solve_all.py inputs test4

python python/merge.py --inputs inputs test test4 test

python3 -m tarfile -c outputs.tar test
