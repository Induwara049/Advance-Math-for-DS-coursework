import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
import math


## part (a)  ##
def f(x):
    if (0 > x >= -np.pi):
        return x ** 2 + 1

    if (np.pi >= x >= 0):
        return x * np.exp(-x)

    if (x < -np.pi):
        z = x+(2*np.pi)
        result = f(z)

    if (x > np.pi):
        z = x - (2 * np.pi)
        result = f(z)

    return result

# Creating sequence between -4.pi < x < +4.pi
X_values = np.linspace(-4*np.pi, 4*np.pi, 1000)
period = 2 * np.pi + X_values

Y_values = [f(x) for x in period]

# Plot the function
plt.title("Periodic function f(x)")
plt.xlabel=("y")
plt.ylabel =("x")
plt.plot(period, Y_values)
plt.show()

## part (b) ##
x = sym.symbols('x')
n = sym.symbols('n', integer=True, positive=True)

ms = np.empty(150, dtype=object)
xrange = np.linspace(-4 * np.pi, +4 * np.pi, 1000)
y = np.zeros([151, 1000])

eq = (x ** 2) + 1
eq2 = x*sym.exp(-1*x)

a0 = (1 / (2 * sym.pi)) * (eq.integrate((x, -1*sym.pi, 0)) + eq2.integrate((x, 0, sym.pi)))
print("a0")
print(a0)
print()


an = (1 / sym.pi) * (sym.integrate((eq * sym.cos(n * x)), (x, -1*sym.pi, 0)) + sym.integrate((eq2 * sym.cos(n * x)), (x, 0, sym.pi)))
print("an")
print(an)
print()

bn = (1 / sym.pi) * (sym.integrate((eq * sym.sin(n * x)), (x, -1*sym.pi, 0)) + sym.integrate((eq2 * sym.sin(n * x)), (x, 0, sym.pi)))
print("bn")
print(bn)
print()

ms[0] = a0

f = sym.lambdify(x, ms[0], 'numpy')
y[0, :] = f(xrange)


for m in range(1, 150):
    ms[m] = ms[m - 1] + an.subs(n, m) * sym.cos(m * x) + bn.subs(n, m) * sym.sin(m * x)
    f = sym.lambdify(x, ms[m], 'numpy')
    y[m, :] = f(xrange)

print("Fourier series")
print(ms[1])

## part (c) ##

plt.plot(xrange, y[1, :])
plt.plot(xrange, y[4, :])
plt.plot(xrange, y[149, :])
plt.plot(xrange, y[150, :])

plt.legend(["1", "5", "150", "func"])
plt.show()

## part (d) ##

real_value = [y[1, :]]
predict_value = [Y_values]
MSE = np.square(np.subtract(real_value, predict_value)).mean()
RMV = math.sqrt(MSE)
print("Root mean error value for 1st harmonic :", RMV)


real_value = [y[4, :]]
MSE = np.square(np.subtract(real_value, predict_value)).mean()
RMV = math.sqrt(MSE)
print("Root Mean Square Error for 5th harmonic:", RMV)

real_value = [y[149, :]]
MSE = np.square(np.subtract(real_value, predict_value)).mean()
RMV = math.sqrt(MSE)
print("Root Mean Square Error for 150th harmonic:", RMV)