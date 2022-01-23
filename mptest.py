# importing libraries
import numpy as np
import time
import matplotlib.pyplot as plt

plt.ion()
 
# here we are creating sub plots
figure, ax = plt.subplots()

plt.rcParams["figure.figsize"] = [10.0, 10.0]
plt.rcParams["figure.autolayout"] = True
xx = [(1,3), (3,7), (5,2)]
x1, y1 = zip(*[[x-0.5 for x in t] for t in xx])

x2 = [3,2,1]
y2 = [4,2,1]

plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xticks([x for x in range(11)])
plt.yticks([y  for y in range(11)])
plt.grid()

set1 = plt.scatter(x1, y1, marker="o", s=100, c=['red', 'green', 'blue'])
set2 = plt.scatter([x-0.5 for x in x2], [y-.5 for y in y2], marker="o", s=100, c="grey")

figure.suptitle('step 1')

figure.canvas.draw()
figure.canvas.flush_events()

time.sleep(5)

set1.set_offsets(np.c_[[x-.5 for x in x2],[y-.5 for y in y2]])
set1.set_facecolors(['yellow', 'yellow', 'yellow'])
set2.set_offsets(np.c_[[x-.5 for x in x1],[y-.5 for y in y1]])
figure.suptitle('step 2')

figure.canvas.draw()
 
figure.canvas.flush_events()
 
time.sleep(5)