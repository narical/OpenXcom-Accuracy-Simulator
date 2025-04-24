# OpenXcom-Accuracy-Simulator
Tool for testing accuracy algorithms

# How to use
There are two algorithms: vanilla and new.

Vanilla represents the very exact algorithm which is used by OXC/E and by original DOS game.

On first tab, you could switch between them to compare.

You can hide legend if you don't need it.

"Run simulation sweep" runs simulation for both algorithms, for shot accuracy in range 0 - 120.

Setup from first tab is used for simulation, including:
* shooter and target positions (tiles)
* number of shots for each accuracy value
* accuracy increment (5% recommended)

Shots count 3.000-10.000 gives good chart precision

# Function editing
On third tab, you can edit "New" function code, and immediately apply it.

Switch to New algorithm on first tab to see the change.

Accuracy Simulator doesn't save any settings or changed code.

If you got something meaningful - copy&paste the function to external editor and save it.
