function [t,y] = adaptiveEuler(t0, tN, y0, h, func)
    tol = 1e-8;
    t = [t0];
    y = [y0];
    i = 2;
    while (t(end)<tN)
        while 1
            tn = t(i-1);
            yn = y(i-1);
            fn = func(tn, yn);
            Z0 = fn*0.5*h + yn;
            Z = func(tn + h/2, Z0)*0.5*h + Z0;

            Y = yn + fn*h;
            if (abs(Z-Y)<tol)
                y = [y Z+(Z-Y)];
                t = [t t(i-1)+h];
                i = i + 1;                
                break
            else
                h = 0.9*h*min(max(tol/abs(Z-Y),0.3),2);
            end
        end
   
    end
