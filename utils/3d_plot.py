import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

SIZE = 100

# create dataset
x1 = np.random.randint(100,size = SIZE)
y1 = np.random.randint(80,size = SIZE)
z1 = np.random.randint(60,size = SIZE)

x2 = np.random.randint(100,size = SIZE)
y2 = np.random.randint(80,size = SIZE)
z2 = np.random.randint(60,size = SIZE)

print('x ',x1.shape,'y ',y1.shape,'z ',z1.shape)

# create figure
fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection='3d')

# creating plot
ax.scatter3D(x1,y1,z1,color='red')
ax.scatter3D(x2,y2,z2,color='green')
plt.title('simple 3D scatter plot')
ax.set_xlim(0,100)
ax.set_ylim(0,100)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# show plot
plt.show()
