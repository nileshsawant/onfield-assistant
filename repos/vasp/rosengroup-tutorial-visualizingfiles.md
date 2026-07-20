<!-- JC note: This tutorial is copied from: https://rosengroup.slite.page/p/h1rRlrGzpqV14t/Visualizing-Files -->
<!-- I have reviewed this and agree with the info here. I have slightly modified certain text to be specific to Kestrel. -->

# Structure

There is no GUI on the cluster itself from a standard. To view the outputs below, it is currently best to copy the necessary files to your local machine and visualize them there. You can use Kestrel's DAV nodes to accomplish this for applications that require VirtualGL: https://natlabrockies.github.io/HPC/Documentation/Viz_Analytics/virtualgl_fastx/

## Trajectory

To view an optimization trajectory (structure, energy, and forces vs. geometry step) using ASE:

```
from ase.io import read
```

Alternatively, from the command-line on your local machine or a visualization node:

```
ase gui OUTCAR
```

## Final Structure

The easiest option is to simply click-and-drag the structure into VESTA!

To view the final structure via Python using ASE:

```
from ase.io import read
```

Alternatively, from the command-line:

```
ase gui CONTCAR
```

# Charge Density

Click-and-drag the `CHGCAR` into VESTA. That's it!

# Density of States

Using [sumo](https://github.com/SMTG-UCL/sumo), run `sumo-dosplot` in a folder containing the `vasprun.xml`.

Alternatively, in Pymatgen, do the following:

```
import matplotlib.pyplot as plt
```

or a slightly more stylized version: ✨

```
from pymatgen.io.vasp import Vasprun
```

