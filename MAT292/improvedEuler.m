% This file defines a function euler that perfoms improved euler's method
% and returns the array of the calculated y values

function [t, y] = euler(t0, tN, y0, h, func)
    N = round(abs((tN-t0)/h) )+1;
    t = linspace(t0, tN, N);
    y = zeros(1, N);
    
    y(1) = y0;

    for i = 2:N
        tn = t(i-1);
        yn = y(i-1);
        fn = func(tn, yn);
        y(i) = yn + 0.5*h*(fn + func(tn + h, yn + h* fn));
    end
    
end
