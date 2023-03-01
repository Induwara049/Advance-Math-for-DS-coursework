import math
import matplotlib.pyplot as plt
import numpy as np
import sympy
import sympy as sym

## part(a) ##
x = np.linspace(-5 * np.pi, 7 * np.pi)

# y = f(x)
y = x * np.cos(x / 2)
plt.ylabel("f(x)")
plt.xlabel("x")
plt.title("function")
plt.plot(x, y)
plt.show()


## part (b) ##
def cosineTaylorSeries(x, rangeValues):
    x = x % (2 * np.pi)
    total = 0
    for i in range(0, rangeValues + 1): # for first 10 terms
        total += ((-1) ** i) * (x ** (2 * i) / math.factorial(2 * i)) #general formulae to get the summation of co efficients from google
    return total


x = np.pi / 2
n = 10
print(cosineTaylorSeries(x, n))


## part (c) ##
def function(x):
    return sympy.cos(x)


x = sympy.Symbol('x')
x0 = np.pi / 2
count = 60

Series = sympy.series(function(x), x0, count)
function_range = np.linspace(-15, 10, 100)  #defining x range
y = [Series.evalf(subs={x: xValues}) for xValues in function_range] #defining y range depending on x range

plt.title("Taylor Series")
plt.plot(function_range, y)
plt.show()


## part (d) ##

def func_cos(x, n):
    cos_approx = 0
    for i in range(n):
        coef = (-1) ** i
        num = x ** (2 * i)
        denom = math.factorial(2 * i)
        cos_approx += (coef) * ((num) / (denom))

    return cos_approx


radius_angel = (math.radians(45))
coefficent = math.radians(60)
out = coefficent * func_cos(radius_angel, 5)
print(out)

# discussion part

angles = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
p_cos = np.cos(angles)
t_cos = [func_cos(angle, 3) for angle in angles]

fig, ax = plt.subplots()
ax.plot(angles, p_cos)
ax.plot(angles, t_cos)
ax.set_ylim([-5, 5])
ax.legend(['cos() function', 'Taylor Series - 3 terms'])

plt.show()