# Meshes-in-dealii
﻿Mesh using C++ library dealii. In order to run code dealii library should be installed. Further cmake creates build files with Eclipse.



To build from the repository, execute the following commands first:

$ mkdir mesh

$ cd mesh

$ git clone https://github.com/Heena008/Meshes-in-dealii.git msfem

$ cmake -DDEAL_II_DIR=/path/to/dealii -G"Eclipse CDT4 - Unix Makefiles" ../mesh

$ make debug

$ make run


License:
Please see the file ./LICENSE.md for details


Further information:
For further information have a look at ./doc/index.html and ./doc/users/cmake.html.

Continuous Integration Status:
