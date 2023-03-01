import matplotlib.pyplot as plt
import numpy as np
import sympy as sym


# a)

X_coord = np.linspace(-10, 10, 100)
Y_coord = 1 / (1 + np.exp(-X_coord))

plt.plot(X_coord, Y_coord)
plt.ylabel("f(x)")
plt.xlabel("x")
plt.show()

# b)

X_coord = np.linspace(-10, 10, 100)
Y_coord = 1 / (1 + np.exp(-X_coord))
df = np.exp(-X_coord) / (1 + np.exp(-X_coord) ** 2)
plt.plot(X_coord, df)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()

# c)
#  a)

X_coord = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
Y_coord = np.sin(np.sin(2 * X_coord))
plt.plot(X_coord, Y_coord)
plt.show()

#  b)
X_coord = np.linspace(-10, 10, 100)
Y_coord = -X_coord ** 3 - 2 * X_coord * 2 + 3 * X_coord + 10
plt.plot(X_coord, Y_coord)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()

# c)

X_coord = np.linspace(-10, 10, 100)
Y_coord = np.exp(-0.8 * X_coord)
plt.plot(X_coord, Y_coord)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# d)
X_coord = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
Y_coord = X_coord ** 2 * np.cos(np.cos(2 * X_coord)) - 2 * np.sin(np.sin(X_coord - (np.pi / 3)))
plt.plot(X_coord, Y_coord)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()

# e)
def g(x):
    if -np.pi <= x < 0:
        return 2 * np.cos(x + np.pi / 6)
    if 0 <= x < np.pi:
        return x * np.exp(-0.4 * x ** 2)
    if x <= -np.pi:
        m = x + (2 * np.pi)
        result = g(m)
        return result
    if x > np.pi:
        m = x - (2 * np.pi)
        result = g(m)
        return result


x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
y = [g(l) for l in x]
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('g(x)')
plt.show()


# d)

# 1)

def logistic_function(x):
    return 1 / (1 + np.exp(-x))

def f(x):
    return np.sin(np.sin(2 * x))


x = np.linspace(-10, 10, 100)
y = logistic_function(f(x))
plt.plot(x,y)
plt.show()

# 2)
#
def f(x):
    return -x ** 3 - 2 * x ** 2 + 3 * x + 10


x = np.linspace(-10, 10, 100)
y = logistic_function(f(x))
plt.plot(x,y)
plt.show()

# 3)

def f(x):
    return np.exp(-0.8 * x)


x = np.linspace(-10, 10, 100)
y = logistic_function(f(x))
plt.plot(x, y)
plt.show()

# 4)

def f(x):
    return x ** 2 * np.cos(np.cos(2 * x)) - 2 * np.sin(np.sin(x - np.pi / 3))


x = np.linspace(-10, 10, 100)
y = logistic_function(f(x))

plt.plot(x, y)
plt.show()


"""
# 5)

x_valuesList = []
y_valuesList = []
for i in np.arange(-np.pi, np.pi):
    x_valuesList.append(i * 0.1)


for x in x_valuesList:
    if -np.pi <= x < 0:
        y = 2 * np.cos(x + np.pi / 6)
    else:
        y = x * np.exp(-0.4 * x ** 2)


x = np.linspace(-10, 10, 100)
plt.plot(x, y)
plt.show() 
"""

