# How to run

To reproduce the simulation of the paper,

    python3 bayesbai_lb.py -m full > full.txt

, which takes more than 1 day(s) in a decent computer.

For a quick simulation, 

    python3 bayesbai_lb.py -m debug > debug.txt

For a middle-scale simulation (~ hour(s)), 

    python3 bayesbai_lb.py > default.txt

## Output

A simulation run yields two pdfs

    varK.pdf
    varT.pdf
