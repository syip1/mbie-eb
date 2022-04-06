#!/usr/bin/env python3
import numpy as np

a = np.zeros((5, 2, 3, 2))
print(a[tuple([0, 1, 2])])
