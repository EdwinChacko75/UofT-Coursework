%% Laplace Transform Lab: Solving ODEs using Laplace Transform in MATLAB
%
% This lab will teach you to solve ODEs using a built in MATLAB Laplace 
% transform function |laplace|.
%
% There are five (5) exercises in this lab that are to be handed in.  
% Write your solutions in a separate file, including appropriate descriptions 
% in each step.
%
% Include your name and student number in the submitted file.
%
%% Student Information
%
%  Student Name: Edwin Chacko
%
%  Student Number: 1009149716
%

%% Using symbolic variables to define functions
% 
% In this exercise we will use symbolic variables and functions.

syms t s x y

f = cos(t)
h = exp(2*x)


%% Laplace transform and its inverse

% The routine |laplace| computes the Laplace transform of a function

F=laplace(f)

%%
% By default it uses the variable |s| for the Laplace transform
% But we can specify which variable we want:

H=laplace(h)
laplace(h,y)

% Observe that the results are identical: one in the variable |s| and the
% other in the variable |y|

%% 
% We can also specify which variable to use to compute the Laplace
% transform:

j = exp(x*t)
laplace(j)
laplace(j,x,s)

% By default, MATLAB assumes that the Laplace transform is to be computed
% using the variable |t|, unless we specify that we should use the variable
% |x|

%% 
% We can also use inline functions with |laplace|. When using inline
% functions, we always have to specify the variable of the function.

l = @(t) t^2+t+1
laplace(l(t))

%% 
% MATLAB also has the routine |ilaplace| to compute the inverse Laplace
% transform

ilaplace(F)
ilaplace(H)
ilaplace(laplace(f))

%% 
% If |laplace| cannot compute the Laplace transform, it returns an
% unevaluated call.

g = 1/sqrt(t^2+1)
G = laplace(g)

%% 
% But MATLAB "knows" that it is supposed to be a Laplace transform of a
% function. So if we compute the inverse Laplace transform, we obtain the
% original function

ilaplace(G)

%%
% The Laplace transform of a function is related to the Laplace transform 
% of its derivative:

syms g(t)
laplace(diff(g,t),t,s)


%% Exercise 1
%
% Objective: Compute the Laplace transform and use it to show that MATLAB
% 'knows' some of its properties.
%
% Details:  
%
% (a) Define the function |f(t)=exp(2t)*t^3|, and compute its Laplace
%   transform |F(s)|.
% (b) Find a function |f(t)| such that its Laplace transform is
%   |(s - 1)*(s - 2))/(s*(s + 2)*(s - 3)|
% (c) Show that MATLAB 'knows' that if |F(s)| is the Laplace transform of
%   |f(t)|, then the Laplace transform of |exp(at)f(t)| is |F(s-a)| 
% 
% (in your answer, explain part (c) using comments).      
%
% Observe that MATLAB splits the rational function automatically when
% solving the inverse Laplace transform.

% Initiatins symbolics variables
syms t s a

% (a)
f = @(t) exp(2*t) * t^3;
F = laplace(f(t));
disp(F);

% (b)
G = (s - 1)*(s - 2)/(s*(s + 2)*(s - 3));
g = ilaplace(G);
disp(g);

% (c) 

% Define symbolic variables and function
syms a t
syms f(t) % Symbolic function f(t)

% Compute the Laplace Transform of f(t)
h = laplace(f(t)); % h represents the Laplace Transform F(s) of f(t)
disp(h); % Display the Laplace Transform of f(t)

% Compute the Laplace Transform of exp(at)*f(t)
j = laplace(exp(a*t)*f(t)); % j represents the Laplace Transform of exp(at)*f(t)
disp(j); % Display the Laplace Transform of exp(at)*f(t)

% Explanation:
% In this part, we are demonstrating that MATLAB recognizes the property of the Laplace Transform:
% If h is the Laplace Transform of f(t), then the Laplace Transform of exp(at)*f(t) should be h evaluated at s-a.
% This property is confirmed by comparing the output of h with that of j.
% h is the Laplace Transform F(s) of f(t), and j should be F(s-a) when f(t) is multiplied by exp(at).
% The MATLAB code computes these transforms and displays them, allowing for a comparison to verify this property.



%% Heaviside and Dirac functions
%
% These two functions are builtin to MATLAB: |heaviside| is the Heaviside
% function |u_0(t)| at |0|
%
% To define |u_2(t)|, we need to write

f=heaviside(t-2)
ezplot(f,[-1,5])

% The Dirac delta function (at |0|) is also defined with the routine |dirac|

g = dirac(t-3)

% MATLAB "knows" how to compute the Laplace transform of these functions

laplace(f)
laplace(g)


%% Exercise 2
%
% Objective: Find a formula comparing the Laplace transform of a 
%   translation of |f(t)| by |t-a| with the Laplace transform of |f(t)|
%
% Details:  
%
% * Give a value to |a|
% * Let |G(s)| be the Laplace transform of |g(t)=u_a(t)f(t-a)| 
%   and |F(s)| is the Laplace transform of |f(t)|, then find a 
%   formula relating |G(s)| and |F(s)|
%
% In your answer, explain the 'proof' using comments.

% Clearing all variables 
clear all;

% Define symbolic variables and function
syms a s t;
syms f(t); % Symbolic function f(t)

% Assign a specific value to 'a' for demonstration
a = 5; % Time shift value

% Compute the Laplace Transform of the time-shifted function g(t)
G(s) = laplace(heaviside(t-a)*f(t-a), t, s); % Calculating the Laplace Transform of g(t)
disp(G(s));

% Compute the Laplace Transform of the original function f(t)
F(s) = laplace(f(t), t, s); % Calculating the Laplace Transform of f(t)
disp(F(s));

% Display the ratio of G(s) to F(s)
% The time-shifting property in Laplace Transforms indicates that 
% G(s) should be F(s) multiplied by exp(-as)
disp(G(s)/F(s)); % This simplifies to exp(-5*s) for a = 5

% Explanation:
% MATLAB's calculation shows that G(s)/F(s) = exp(-5*s). By rearranging,
% we find that G(s) = exp(-5*s)*F(s). While a specific value of a = 5 was 
% chosen for this demonstration, the relationship holds for any value of 'a',
% yielding G(s) = exp(-a*s)*F(s) in general. This result was obtained by 
% evaluating F(s) and G(s) for a generic function f(t), where F and G were defined
% according to the exercise's instructions. The division of these two functions 
% of s led to a result independent of the specifics of f(t), G, and F, resulting 
% in exp(-5*s). This expression was then rearranged to derive the general 
% formula for the effect of time shifting in Laplace Transforms.

%% Solving IVPs using Laplace transforms
%
% Consider the following IVP, |y''-3y = 5t| with the initial
% conditions |y(0)=1| and |y'(0)=2|.
% We can use MATLAB to solve this problem using Laplace transforms:

% First we define the unknown function and its variable and the Laplace
% tranform of the unknown

syms y(t) t Y s

% Then we define the ODE

ODE=diff(y(t),t,2)-3*y(t)-5*t == 0

% Now we compute the Laplace transform of the ODE.

L_ODE = laplace(ODE)

% Use the initial conditions

L_ODE=subs(L_ODE,y(0),1)
L_ODE=subs(L_ODE,subs(diff(y(t), t), t, 0),2)

% We then need to factor out the Laplace transform of |y(t)|

L_ODE = subs(L_ODE,laplace(y(t), t, s), Y)
Y=solve(L_ODE,Y)

% We now need to use the inverse Laplace transform to obtain the solution
% to the original IVP

y = ilaplace(Y)

% We can plot the solution

ezplot(y,[0,20])

% We can check that this is indeed the solution

diff(y,t,2)-3*y


%% Exercise 3
%
% Objective: Solve an IVP using the Laplace transform
%
% Details: Explain your steps using comments
%
%
% * Solve the IVP
% *   |y'''+2y''+y'+2*y=-cos(t)|
% *   |y(0)=0|, |y'(0)=0|, and |y''(0)=0|
% * for |t| in |[0,10*pi]|
% * Is there an initial condition for which |y| remains bounded as |t| goes to infinity? If so, find it.

% Clearing all variables 
clear all;

% Symbol Definiton
syms y(t) t Y s;

% Definition of the ODE: y''' + 2y'' + y' + 2y = -cos(t)
ODE = diff(y(t), t, 3) + 2*diff(y(t), t, 2) + diff(y(t), t) + 2*y(t) == -cos(t);

% Computation of the Laplace transform 
L_ODE = laplace(ODE, t, s);

% Initial conditions: y(0) = 0, y'(0) = 0, y''(0) = 0
L_ODE = subs(L_ODE, {y(0), subs(diff(y(t), t), t, 0), subs(diff(y(t), t, 2), t, 0)}, {0, 0, 0});

% Factor out the Laplace transform of y(t) (Y)
L_ODE = subs(L_ODE, laplace(y(t), t, s), Y);

% Solve for Y, the Laplace transform of y(t)
Y = solve(L_ODE, Y);

% Get the inverse Laplace transform
y(t) = ilaplace(Y, s, t);

% Plot the solution for t in [0, 10*pi]
fplot(y, [0, 10*pi]);
title("Solution of y''' + 2y'' + y' + 2y = -cos(t)");
ylabel('y(t)');
xlabel('t');

% To check if y remains bounded as t goes to infinity
poles = poles(Y);
if all(real(poles) < 0)
    disp('y(t) remains bounded as t goes to infinity');
else
    disp('y(t) does not remain bounded as t goes to infinity');
end

%% Exercise 4
%
% Objective: Solve an IVP using the Laplace transform
%
% Details:  
% 
% * Define 
% *   |g(t) = 3 if 0 < t < 2|
% *   |g(t) = t+1 if 2 < t < 5|
% *   |g(t) = 5 if t > 5|
%
% * Solve the IVP
% *   |y''+2y'+5y=g(t)|
% *   |y(0)=2 and y'(0)=1|
%
% * Plot the solution for |t| in |[0,12]| and |y| in |[0,2.25]|.
%
% In your answer, explain your steps using comments.

% Clearing all previous work
clc; close all; clear;

% Declaring symbolic variables and functions
syms t y(t) Y s;

% Defining the piecewise function g(t) using Heaviside functions
% g(t) = 3 for 0 < t < 2, g(t) = t + 1 for 2 < t < 5, and g(t) = 5 for t > 5
g(t) = 3*heaviside(t) + (t-2)*heaviside(t-2) + (4-t)*heaviside(t-5);

% Defining the second-order linear non-homogeneous ODE
% y'' + 2y' + 5y = g(t)
ODE = diff(y(t), t, 2) + 2*diff(y(t), t) + 5*y(t) == g(t);

% Taking the Laplace transform of the ODE
L_ODE = laplace(ODE, t, s);

% Applying the initial conditions: y(0) = 2, y'(0) = 1
L_ODE = subs(L_ODE, y(0), 2);
L_ODE = subs(L_ODE, subs(diff(y(t), t), t, 0), 1);

% Solving for Y, the Laplace transform of y(t)
L_ODE = subs(L_ODE, laplace(y(t), t, s), Y);
Y = solve(L_ODE, Y);

% Finding the inverse Laplace transform of Y to get y(t)
y(t) = ilaplace(Y, s, t);

% Plotting the solution for t in [0,12] and y in [0,2.25]
fplot(y, [0, 12]);
ylim([0, 2.25]);
title('Solution of y'''' + 2y'' + 5y = g(t)');
xlabel('t');
ylabel('y(t)');

% Explanation:
% Expressing the piecewise function g(t) in terms of shifted Heaviside unit step functions.
% The differential equation is then transformed into the s-domain using the Laplace transform,
% where it becomes an algebraic equation. The initial conditions are applied to determine the 
% particular solution of the system. Finally, the inverse Laplace transform is used 
% to convert the solution back into the time domain, providing the solution to the original ODE.


%% Exercise 5
%
% Objective: Use the Laplace transform to solve an integral equation
% 
% Verify that MATLAB knowns about the convolution theorem by explaining why the following transform is computed correctly.
syms t tau y(tau) s
I=int(exp(-2*(t-tau))*y(tau),tau,0,t);
L_I = laplace(I,t,s);
disp(L_I);

% Output:
% ans =
% laplace(y(t), t, s)/(s + 2)

% Explanation:
% The given integral I is equivalent to the convolution of exp(-2t) and y(t),
% noted as (exp(-2t)*y(t))(t). According to the convolution theorem, the 
% Laplace transform of a convolution of two functions (f*g)(t) is the product 
% of their individual Laplace transforms: Laplace{(f*g)(t)} = F(s)G(s). The 
% Laplace transform of exp(-2t) is known to be 1/(s+2) because 
% Laplace{exp(-at)} = 1/(s+a). Therefore, the Laplace transform of the 
% convolution of exp(-2t) and y(t) should be the product of 1/(s+2) and 
% Laplace{y(t)}. This is precisely what MATLAB computes and shows as 
% laplace(y(t), t, s)/(s + 2).This confirms MATLAB's correct application of
% the convolution theorem in the context of Laplace transforms.

