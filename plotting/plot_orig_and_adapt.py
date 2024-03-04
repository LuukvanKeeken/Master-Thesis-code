import numpy as np
import matplotlib.pyplot as plt


orig = [199.96800000000002, 199.91199999999998, 198.356, 198.66, 199.38400000000001, 198.644, 199.98, 198.702, 199.408]
training_length_evals = [199.34, 200.0, 200.0, 200.0, 200.0, 200.0, 184.2, 200.0, 200.0]
training_length_evals_more = [199.13, 196.61, 194.162, 181.454, 196.302, 196.672, 196.332, 195.72, 192.02]
length = [173.8717142857143, 183.9702857142857, 178.13914285714287, 162.58314285714283, 168.09514285714286, 181.42257142857142, 170.4797142857143, 182.2577142857143, 180.95342857142856]
mass = [119.90240000000001, 116.52559999999998, 110.66239999999998, 117.4808, 114.4156, 114.3828, 114.44680000000001, 111.57719999999999, 112.51399999999998]
force_mag = [199.6645, 187.8435, 195.433, 198.50100000000003, 182.9605, 195.75050000000002, 199.8135, 192.061, 191.3195]

print(f"Best original: {np.max(orig)} at {np.argmax(orig)}")
print(f"all originals sorted: {np.argsort(orig)[::-1]}")

print(f"Best length: {np.max(length)} at {np.argmax(length)}")
print(f"all lengths sorted: {np.argsort(length)[::-1]}")

print(f"Best mass: {np.max(mass)} at {np.argmax(mass)}")
print(f"all mass sorted: {np.argsort(mass)[::-1]}")

print(f"Best force_mag: {np.max(force_mag)} at {np.argmax(force_mag)}")
print(f"all force_mag sorted: {np.argsort(force_mag)[::-1]}")

print(f"Best training_length_evals: {np.max(training_length_evals)} at {np.argmax(training_length_evals)}")
print(f"all training_length_evals sorted: {np.argsort(training_length_evals)[::-1]}")

print(f"Best training_length_evals_more: {np.max(training_length_evals_more)} at {np.argmax(training_length_evals_more)}")
print(f"all training_length_evals_more sorted: {np.argsort(training_length_evals_more)[::-1]}")

points = [0]*len(orig)
for i in range(len(orig)):
    points[i] += length[i]
    points[i] += mass[i]
    points[i] += force_mag[i]

points = np.asarray(points)
print(f"Best model: {np.argmax(points)} with total {np.max(points)}")
print(np.argsort(points)[::-1])

plt.scatter(training_length_evals_more, length)
plt.ylabel('Mean avg pole length adapt performance')
print(np.corrcoef(training_length_evals_more, length))
# plt.scatter(training_length_evals_more, mass)
# plt.ylabel('Mean avg pole mass adapt performance')
# print(np.corrcoef(training_length_evals_more, mass))
# plt.scatter(training_length_evals_more, force_mag)
# plt.ylabel('Mean avg force magnitude adapt performance')
# print(np.corrcoef(training_length_evals_more, force_mag))

# plt.scatter(training_length_evals, length)
# plt.ylabel('Mean avg pole length adapt performance')
# print(np.corrcoef(training_length_evals, length))
# plt.scatter(training_length_evals, mass)
# plt.ylabel('Mean avg pole mass adapt performance')
# print(np.corrcoef(training_length_evals, mass))
# plt.scatter(training_length_evals, force_mag)
# plt.ylabel('Mean avg force magnitude adapt performance')
# print(np.corrcoef(training_length_evals, force_mag))

plt.xlabel('Avg training selection performance')

# plt.scatter(orig, length)
# plt.ylabel('Mean avg pole length adapt performance')
# print(np.corrcoef(orig, length))
# plt.scatter(orig, mass)
# plt.ylabel('Mean avg pole mass adapt performance')
# print(np.corrcoef(orig, mass))
# plt.scatter(orig, force_mag)
# plt.ylabel('Mean avg force magnitude adapt performance')
# print(np.corrcoef(orig, force_mag))

# plt.xlabel('Mean avg performance original env')
plt.show()


