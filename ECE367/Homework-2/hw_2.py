import numpy as np
import matplotlib.pyplot as plt

# Function definitions for f1, f2, f3
f1 = lambda X, Y: 2*X + 3*Y + 1
f2 = lambda X, Y: X**2 + Y**2 - X*Y - 5
f3 = lambda X, Y: (X - 5) * np.cos(Y - 5) - (Y - 5) * np.sin(X - 5)

def grad_f1(X, Y):
    # Returns gradient of f1
    return [2, 3]

def grad_f2(x, y):
    # Returns gradient of f2
    return [2*x - y, 2*y - x]

def grad_f3(x, y):
    # Returns gradient of f3
    return [np.cos(y - 5) - (y - 5) * np.cos(x - 5), 
            (5 - x) * np.sin(y - 5) - np.sin(x - 5)]

def hess_f1(X, Y):
    # Returns hessian of f1
    return [[0,0],
            [0,0]]

def hess_f2(x, y):
    # Returns hessian of f2
    return [[2, -1], 
            [-1, 2]]

def hess_f3(x, y):
    # Returns hessian of f3
    f_xx = (y - 5) * np.sin(x - 5)
    f_xy = - (x - 5) * np.sin(y - 5) - np.cos(x - 5)
    f_yy = - (x - 5) * np.cos(y - 5) - np.sin(x - 5)

    return [[f_xx, f_xy],
            [f_xy, f_yy]]

def tangent_surface(X, Y, function, grad_function, hess_fcn=None, point = (1, 0)):
    """
    Computes the tangent surface (linear or second-order Taylor approximation) of function at point.

    Parameters:
    - X, Y: Meshgrid arrays over which the function is evaluated.
    - function: The original function f(x, y).
    - grad_function: Function that returns the gradient of f [f_x, f_y] at (x, y).
    - hess_fcn: Function that returns the Hessian matrix [[f_xx, f_xy], [f_yx, f_yy]]  of f
                at (x, y). If None, only the linear approximation is computed.
    - point: Tuple containing the point [x0, y0] where the approximation is centered.

    Returns:
    - aproxx: The approximated surface values over the meshgrid.
    """
    x0, y0 = point
    f0 = function(x0, y0)
    grad_f = grad_function(x0, y0)
    f_x, f_y = grad_f

    # Linear approximation
    aproxx = f0 + f_x * (X - x0) + f_y * (Y - y0)

    # Second Order terms
    if hess_fcn:
        hess_f = hess_fcn(x0, y0)
        f_xx, f_xy, _, f_yy = hess_f[0][0], hess_f[0][1], hess_f[1][0], hess_f[1][1]

        aproxx += (0.5 * f_xx * (X - x0)**2) + (f_xy * (X - x0) * (Y - y0)) + (0.5 * f_yy * (Y - y0)**2)

    return aproxx

def plot_3d(X, Y, function, grad_function, hess_fcn=None, equation=None, point=(1, 0)):
    """
    Plots the original surface and its tangent (linear or second-order) Taylor approximation.
    Parameters:
    - X, Y: Meshgrid arrays over which the function is evaluated.
    - function: The original function f(x, y).
    - grad_function: Function that returns the gradient of f [f_x, f_y] at (x, y).
    - equation: Equation being plotted.
    - hess_fcn: Function that returns the Hessian matrix [[f_xx, f_xy], [f_yx, f_yy]]  of f
                at (x, y). If None, only the linear approximation is computed.
    Returns:
    - None
    """
    x0, y0 = point
    
    # Compute f(x,y)
    Z = function(X, Y)

    # Compute the approximation
    H = tangent_surface(X, Y, function, grad_function, hess_fcn=hess_fcn, point=point)

    # Plot them on the same axes
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z,  alpha=0.8) # The surface f
    ax.plot_surface(X, Y, H,  alpha=0.5) # The approximation of f

    # contour
    ax.contour(X, Y, Z, zdir='z', offset=ax.get_zlim()[0], levels=50, cmap='viridis') 

    # The point of interest
    ax.scatter(x0, y0, function(x0, y0), c='r', marker='o', s=100, label=f"Point ({x0}, {y0}), f({x0}, {y0}))")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if equation:
        order = "First" if hess_fcn == None else "Second"
        ax.set_title(f'{order}-order Approximation of {equation} at {point}')
    plt.show()


def plot(X, Y, function, grad_function, equation, point=(1,0)):
    """
    Plots the contour of the function with its tangent line and gradient vector at the given point.
    Parameters:
    - X, Y: Meshgrid arrays over which the function is evaluated.
    - function: The original function f(x, y).
    - grad_function: Function that returns the gradient of f [f_x, f_y] at (x, y).
    - equation: String representation of the function for labeling.
    - hess_fcn: Function that returns the Hessian matrix [[f_xx, f_xy], [f_yx, f_yy]]  of f
                at (x, y). If None, only the linear approximation is computed.
    Returns:
    - None
    """
    x0, y0 = point

    # Compute f(x,y)
    Z = function(X, Y)
    grad = grad_function(x0, y0)

    # Normalie gradient so it is easier to see
    normalization = np.sqrt(grad[0]**2 + grad[1]**2)
    x = grad[0]/normalization
    y = grad[1]/normalization

    # Compute equation of the tangent line
    tangent_slope = - x/y
    x_tangent = np.linspace(-2, 3.5, 100)
    y_tangent = tangent_slope * (x_tangent - x0) + y0

    # Set bounds on y for the tangent
    y_min, y_max = -2, 3.5  
    mask = (y_tangent >= y_min) & (y_tangent <= y_max)
    x_tangent_bounded = x_tangent[mask]
    y_tangent_bounded = y_tangent[mask]
    
    plt.title(fr'Contour plot of ${equation}$')
    plt.contour(X, Y, Z, levels=50, cmap='Blues')  # Plot contour
    plt.plot(x0, y0, 'ro', label="Point (1, 0)") # Plot point
    plt.plot(x_tangent_bounded, y_tangent_bounded, 'b-', label=f"Tangent Line at ({x0}, {y0})", zorder=10) # Plot tangent
    plt.quiver(x0, y0, x, y, angles='xy', scale_units='xy', scale=1, color='blue', zorder=3) # Plot quiver    
    plt.axis('equal') # Keep axes proportional to show orthogonality of gradient
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()


def main():
    x_vals = np.linspace(-2, 3.5, 400)
    y_vals = np.linspace(-2, 3.5, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    point = (1, 0)

    # Question 1 Part b
    plot(X, Y, f1, grad_f1, f'f_1 = 2x + 3y + 1', point=point)
    plot(X, Y, f2, grad_f2, f'f_2 = x^2 + y^2 - xy - 5', point=point)
    plot(X, Y, f3, grad_f3, f'f_3 = (x - 5)\cos(y - 5) - (y - 5)\sin(x - 5)', point=point)

    # Question 1 Part c
    plot_3d(X, Y, f1, grad_f1, equation=f'$f_1 = 2x + 3y + 1$', point=point)
    plot_3d(X, Y, f2, grad_f2, equation=f'$f_2 = x^2 + y^2 - xy - 5$', point=point)
    plot_3d(X, Y, f3, grad_f3, equation=f'$f_2(x, y) = x^2 + y^2 - xy - 5$', point=point)

    # Question 2 Part b and c
    for pt in  ((1, 0), (-0.7, 2), (2.5, -1)):
        plot_3d(X, Y, f1, grad_f1, hess_fcn=hess_f1, equation=f'$f_1 = 2x + 3y + 1$', point=pt)
        plot_3d(X, Y, f2, grad_f2, hess_fcn=hess_f2, equation=f'$f_2 = x^2 + y^2 - xy - 5$', point=pt)
        plot_3d(X, Y, f3, grad_f3, hess_fcn=hess_f3, equation=f'$f_2(x, y) = x^2 + y^2 - xy - 5$', point=pt)

if __name__ == "__main__":
    main()

"""
Comment on where your approximations are accurate and where they are not (if anywhere) for 
the three functions. Discuss what the reason is behind your observations.

The first order approximation is only accurate for f1. This makes sence as f2 and f3 are nonlinear
while f1 is linear allowing it to be modelled by 1 degree of of x. 

The second order approximate is accurate for f1 and f2 as f1 and f2 are of degree 1 and 2 respectively, 
again meaning they can be modelled by a second order approximation. f3 is a interesting sinusiod which 
is periodic and has infitite turning points. It is not possible to get an accurate estimation for such a
funcion with a second order approximation - which can only support a max of 1 turning point. This idea is
clearer when examining the plot of f3 at (2.5, -1) (this figure is attached as './q2_plots2d.png'). This plot 
shows how the point is in a lower area of the function and as the sinusiod reaches its max, the second order
approximatin does an alright job at approximating it. But once the sinusiod peaks and begins the next period, 
the second order approximation cannot keep up as it is only able to model a single turning point.

"""