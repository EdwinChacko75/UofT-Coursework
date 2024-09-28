%%  Introduction to Plotting and Vectorization
% This lab will provide an introduction to two dimensional plots.  At
% the end, you should feel comfortable plotting functions of the form f(x).
% You should find this useful not only for visualizing the solutions of
% ODEs, but also for plotting general data that might appear in other
% courses and work.
%
% In the process we will also introduce the concept of vectorization, which
% is helpful for writing efficient and "clean" code.  Vectorization is a 
% concept found not only in MATLAB but also C/C++, Fortran, and other
% languages.
%
% Opening the m-file lab1.m in the MATLAB editor, step through each
% part using cell mode to see the results.  Compare the output with the
% PDF, which was generated from this m-file.
%
% There are twelve (12) exercises in this lab that will be graded.  
% Write your solutions in the template, including appropriate descriptions 
% in each step. Save the .m file as lab1_<UTORid>.m and submit it online on
% Quercus. Also, submit a PDF of the output by generating html (by going to
% 'File', 'Publish') and then saving to a PDF called lab1_<UTORid>.pdf. 
% Substitute your UTORid for <UTORid>.
%
%
%
%% Student Information
%
% Student Name: Edwin Chacko
%
% Student Number: 1009149716
%
%% Exercise 1
% Objective: Observe a MATLAB error when you try to access an invalid
% index.
%
% Details: Try to access elements of the vector |x|, created in the
% previous step at the command prompt |>>|.  Try values of |i| outside the 
% valid range.  Try negative values, zero, and large positive values.  Copy
% down the error message in each case, and report this in your submitted 
% work as:
%
% Error for negative index:
% Array indices must be positive integers or logical values.

% Error for zero index:
% Array indices must be positive integers or logical values.

% Error for positive index outside of the bounds:
% Index exceeds the number of array elements. Index must not exceed 6.

% Only include the error message as a comment (with the percent signs), 
% DO NOT INCLUDE THE CODE IN YOUR SUBMISSION.


%% Exercise 2
% Objective:  Create two vectors and plot one against the other.
%
% Details:  Explicitly create the vectors 
%
% x = [-5, -3, -1, 0, 2, 4, 6] 
%
% and 
%
% y = [-25, -9, -1, 0, 4, 16, 36]
%
% And plot |x| against |y|.  Your submission should show both the
% creation of the vectors and the figure, as in the example.

x = [-5, -3, -1, 0, 2, 4, 6]; % creating x vector
 
y = [-25, -9, -1, 0, 4, 16, 36]; % creating y vector
 
plot(x,y); % plotting y against x
title('Plotting Y against X');
xlabel('X');
ylabel('Y');

%% Exercise 3
% Objective:  Plot |x|.x
%
% Details:  Using a for loop, create a vector x of 101 equally spaced
% points between -3 and 7, inclusive.  Then create a vector y, such that
% |y(i) = abs(x(i))*x(i)| using a for loop.  Plot the results.  
%
% Your submission should include the code, as in the example, and 
% appropriate comments.

% Set the number of points
N = 101; 
% determine x distance between points
delta = (7-(-3))/(N-1);
% preallocate x and y vectors for efficency
x = zeros(N,1);
y = zeros(N,1);

% loop through length of x and y vectors, set x(j) to -3 + appropriate
% partion, set y(j) as |x|x
for j = 1:N
    x(j) = -3 + (j-1) * delta;
    y(j) = x(j)*abs(x(j));
end
% plot the graph
plot(x,y);


%% Exercise 4
% Objective:  Compare the speedup from vectorization for a larger vector.
%
% Details:  Create a vector |x| of 5000001 equally space points between -2 and
% 3, inclusive, using a for loop.  Then time the construction of vector |y|,
% whose values are arctan of x, by a for loop and by vectorization.  Be sure to clear 
% |y| between the two runs.  (Hint: use the help file to find the command for arctan!)
%
% Your submission should include the code, as in the example, with
% appropriate comments.
    
clear y;
% Initialize partitions, partion difference, and x array
N = 5000001;
dx = (3 - (-2))/(N - 1);
x = zeros(N,1);

% Loop through x and assign each position its appropriate value between -2
% and 3
for i = 1:N
    x(i) = -2 + dx * (i - 1);
end

tic; % Start timer loop
y = zeros(N,1);
for j = 1:N
    y(j) = atan(x(j));
end
toc; % End timer loop
clear y; % Clear y

tic; % Start timer vectorization
y = atan(x);
toc; % End timer vectorization
plot(x,y);


%% Exercise 5
% Objective:  Solve an ODE and plot three particular solutions.
%
% Details:  Manually find the general solution of the ODE:
%
% dy/dt + (2*t)*y = 1/exp(t^2)
%
% and the particular solutions for initial conditions y(1) = -1, y(1) = 0,
% and y(1) = 1. Plot these three particular solutions on the same axes
% lettting the independent variable range from 1 to 5. 
%
% Once you have the three particular solutions you will
% need to:
%
%  1. Create a vector for the independent variable, t.  Use linspace with
%  a reasonable number of points (100 is fine).
%  2. Create vectors for the three solutions.  Use vectorized operations.
%  3. Plot three solutions on the same axes, annotating the figure
%   appropriately.
%
% Your submission should include, in the description section, the general 
% solution of the ODE and the three particular solutions.  It should also 
% have the appropriately commented code.

% GENERAL SOLUTION:
% y(t) = (t + c) / e^(t^2)

% Initialize t with linspace
t = linspace(1, 5, 100);

% For y(1) = 1, vectorize
yInitialIsOne = (t + exp(1) - 1)./exp(t.^2);

% For y(1) = 0, vectorize
yInitialIsZero = (t - 1)./exp(t.^2);

% For y(1) = -1. vectorize
yInitialIsMOne = (t - exp(1) - 1)./exp(t.^2);

% Plot the three solutions on the same axes
plot(t, yInitialIsOne, '-r', t, yInitialIsZero, '-b', t, yInitialIsMOne, '-g');

% Add readability features to plot
legend('y(1) = 1', 'y(1) = 0', 'y(1) = -1');
xlabel('t');
ylabel('y(t)');
title('Solutions to the Differential Equation for Different Initial Conditions');
grid on;


%% Exercise 6
%
% Objective: Write and use an inline function of one variable.
%
% Details: Define the inline function for
%
% |f(x) = (x^3 - abs(x)) * ln(x^2+1)|
%
% Use this function to compute |f(0)|, |f(1)|, and |f(-1)| and plot the
% function with 100 grid points from -5 to 5. Make sure to
% use vectorization in your definition, and label your axes.
%
% Your submission should show the definition of the function, and the
% computation of these values.

% Initialize x with linspace
x = linspace(-5, 5, 100);

% Define the function f
f = @(x) (x.^3 - abs(x)) .* log(x.^2 + 1);

% Display |f(0)|, |f(1)|, and |f(-1)|
values = f([0, 1, -1]);
disp(['f(0) = ', num2str(values(1))])
disp(['f(1) = ', num2str(values(2))])
disp(['f(-1) = ', num2str(values(3))])

% Plot the function
plot(x, f(x))

% Add readability features to plot
xlabel('x')
ylabel('f(x)')
title('Plot of f(x) = (x^3 - abs(x)) * ln(x^2+1)')
grid on;

%% Exercise 7
%
% Objective: Write and use an inline function of two variables.
%
% Details: Define the inline function for
%
% |f(x,y) = y + y^3 - cos x + x^2 - x^4|
%
% Note that |f(x,y) = c| is the implicit solution to 
%
% |dy/dx = (4x^3 - 2x - sin x)/(1+3y^2)|.
%
% See Examples 1-3 of section 2.1 in Brannan and Boyce for similar examples.
%
% Use this function to compute |f(0, 0)|, |f(pi/2,1)|, and |f(-1,-1)|.  Make
% sure to use vectorization in your definition.
%
% Your submission should show the definition of the function, and the
% computation of these values.
%

% Define the inline function
f = @(x,y) y + y.^3 - cos(x) + x.^2 - x.^4;

% Compute the values for the specified inputs
result1 = f(0, 0);
result2 = f(pi/2, 1);
result3 = f(-1, -1);

% Display the results
fprintf('f(0,0) = %g\n', result1);
fprintf('f(pi/2,1) = %g\n', result2);
fprintf('f(-1,-1) = %g\n', result3);



%% Exercise 8
%
% Objective: Use fzero to solve an equation |f(x,y)=0| for y at different
% values of x.
%
% Details: Consider the function
%
% |f(x,y) = y + y^3 - cos x + x^2 - x^4|
%
% Define the appropriate inline function and compute the single solution 
% at |x = 0| and the two positive solutions at |y = 1/2|.  
%
% You will need to experiment with different guesses to get these three
% solutions.
%
% Your submission should show the definition of the function, and the
% computation of these three roots.

% Define the function f(x, y)
f = @(x,y) y + y.^3 - cos(x) + x.^2 - x.^4;

% Define given values for x and y
x0 = 0;
y0 = 1/2;

% Define initial guesses
guessY0 = 1;
guessX01 = 1;
guessX02 = 0.5;

% Find y such that f(0,y) = 0 using the initial guess guessY0
y = fzero(@(y) f(x0, y), guessY0);
fprintf('f(%g, y) = 0; y = %g\n', x0, y);

% Find x such that f(x, 1/2) = 0 using the initial guess guessX01
x1 = fzero(@(x) f(x, y0), guessX01);
fprintf('f(x, %g) = 0; x = %g\n', y0, x1);

% Find x such that f(x, 1/2) = 0 using another initial guess guessX02
x2 = fzero(@(x) f(x, y0), guessX02);
fprintf('f(x, %g) = 0; x = %g\n', y0, x2);


%% Exercise 9
%
% Objective: Plot a portion of an implicit equation.
%
% Details: Plot the portion of the solution to
%
% |f(x,y) = y + y^3 - cos x + x^2 - x^4|
%
% passing through the point |(0,0)| for x from -2 to 2.  Use 100 grid
% points.
%
% You will need to be careful with the initial guess you use in solving the
% equation.
%
% Your submission should show the definition of the function, the
% construction of the arrays, the for loop, and the resultant figure.
%
% Label your axes.
% 

% Define the function f(x, y)
% constant added to ensure initial condition is met
f = @(x,y) y + y.^3 - cos(x) + x.^2 - x.^4 + 1; 

% Guess close to the origin (known point)
guess = 0;

% Allcoate x and yValues
x = linspace(-2, 2, 100);
yValues = zeros(size(x));

% Loop through, updating yValues with values that zero f for the given x
for i = 1:100
    yValues(i) = fzero(@(y) f(x(i), y), guess);
end

% Plot the result
plot(x,yValues);
hold on;

% Add a point marker at (0,0)
plot(0, 0, 'ro'); % 'ro' specifies a red circle

hold off;

% Label the axes + readability
title('x and y such that f(x, y) = 0');
xlabel('x');
ylabel('y');
legend('y + y^3 - cos x + x^2 - x^4 + 1 = 0');
grid on;


%% Exercise 10
%
% Objective: Solve a differential equation and plot a portion of it.
%
% Details: Find the general implicit solution of the differential equation
%
% |dy/dx = (-cos x + 3x^2) y|
%
% Plot the particular solution passing through y(0) = 1 using 100 grid
% points for x in [-1.5, 1.25].
%
% Be careful on your choice of guess. You will be penalized if MATLAB shows
% an error.
%
% Your submission should show the general and particular solutions, in
% implicit form, in the comments, the definition of the appropriate inline
% functions, arrays, the for loop, and the figure.
%
% Label your axes.
%

% GENERAL SOLUTION:
% f(x, y) = C*exp(x^3 - sin(x)) - y
% PARTICULAR SOLUTION:
% exp(x^3 - sin(x)) - y     

% Define f
f = @(x, y) exp(x.^3 - sin(x)) - y;

% Allocate x and y
x = linspace(-1.5, 1.25, 100);
yValues = zeros(size(x));

% Make guess
guess = 0.2;

% Set the initial y value using the guess
yValues(1) = fzero(@(y) f(x(1), y), guess);

% Loop through, updating yValues
% Guess is updated to the previously used value to prevent search errors
for i = 2:100
    guess = yValues(i - 1);
    yValues(i) = fzero(@(y) f(x(i), y), guess);
end

% Plot the result
plot(x, yValues);

% Labelling axes + readability
xlabel('x');
ylabel('y');
title('Plot of C * exp(x^3 - sin(x)) - y');
grid on;

%% Exercise 11
%
% Objective: use iode a plot the direction field of a differential equation
%
% Details: After starting iode, click the "direction fields" buttom from 
% the main menu. The default is the direction field for
%
% |dy/dx = sin(y-x)|
%
% Click any point in the x-y plane to see a (numerically computed) solution
% to the differential equation passing through that point.
%
% Starting from the "Equation" menu, plot the direction field of 
%
% |dy/dx = xy^2|
%
% Describe, in words, the behaviour of the solutions. What happens as x
% approaches infinity? Explain why solutions are symmetric about the 
% y-axis, i.e. |y(x)=y(-x)| for any solution.

% The positive solutions appear to be parabolic
% and as x increases in magnitude, the solutions
% become increasingly wide and flat.
% The negative solutions near the origin are
% smooth v shaped divot and appear to converge
% to zero at infinity. The divot deepens as
% solutions depart the y-axis.
% Direction fields along the x and y-axes are horizontal.
% The symmetry about the y-axis is a result
% of the ‘x^2’ term in the general solution -
% leading to the same y values for -x and +x.

%% Exercise 12
%
% Objective: study the long-run behaviour of a differential equation from 
% its direction field
%
% Details: Using iode, plot the direction field of 
%
% |dy/dx = y^4 - y^3 - 3*y^2 + y + 2|
%
% By clicking on various initial conditions |y(0) = y_0| (iode plots the
% solution), determine the long-run behaviour of solutions as a function of
% |y_0|. In other words, list the constant solutions that are approached as
% x goes to infinity and how |y_0| determines which constant solution is 
% approached.
%
% If your numerically computed solutions ever appear to cross then change
% the solution method from Euler to the, much better, Runge-Kutta method.
% Think about why solutions are not permitted to cross. If they do then
% this indicates a problem with the numerical solver.

% y_0 = 0
% As x -> -infinity, y -> -1
% As x -> +infinity, y -> 1

% y_0 = 1
% Solution appears to be a horizontal line at y = 1.

% y_0 = -1
% Solution appears to be a horizontal line at y = -1.

% y_0 = 1.5
% As x -> -infinity, y -> 2
% As x -> +infinity, y -> 1

% y_0 = -2
% As x -> +infinity, y -> -1
% Solution has a vertical asymptote that approaches -infinity around x = 0

% y_0 = 3
% As x -> -infinity, y -> 2
% Solution has a vertical asymptote approaching +infinity around x = 0

% y_0 = 4, 5.5, 5.6
% As x -> -infinity, y -> 2
% Solutions appear to make a sudden change from a horizontal to a vertical 
% line around x = 0

% y_0 = -4, -5.5, -6
% As x -> +infinity, y -> -1 or 1
% Solutions appear to be a vertical line around x = 0 or have a vertical
% asymptote

% As y_0 increases, the resemblance of the solution to a vertical line on 
% the y-axis also increases.
