%% problem 1
% algorithm3.4 (cholesky with added multiple of identity) 
A = [1 4 3;4 2 5;3 5 3];
beta = 1e-3;
choleskey = false;
% test using chol function
try
    L = chol(A);
catch 
    disp('A is not positive definite');
    disp('lets make it positive definite');
end
%make it Positive definite
if min(diag(A))> 0
    taw = 0;
else 
    taw = -min(diag(A)) + beta;
end
while~choleskey
    try 
        E = taw*eye(3);
        A = A + E;
        L = chol(A);
        choleskey = true;
        disp(taw)
    catch 
        disp('Still not positive definite');
    end
    taw_new = max(2*taw,beta);
    taw = taw_new;
end
disp("The matrix E is: ");
disp(E);

%% problem 2
[x1,x2] = meshgrid(linspace(-3, 3, 400), linspace(-3, 3, 400));
%calculating gradient and hessain at x = [0,-1]
x = [0 -1];
[f,grad,H] = rosenbrock(1,10,x);
mk = zeros(size(x1));
%quadratic model mk(p)
for i = 1:length(x1(:,1))
    for j = 1:length(x1(:,1))
        p = [x1(i,j) - x(1);x2(i,j) - x(2)];
        mk(i,j) = f + grad'*p + 0.5*p'*H*p;
    end
end
% Plotting the contour of the quadratic model
figure;
contour(x1, x2, mk, 25);
title('Contour Lines of the Quadratic Model');
% Plotting trust region different radii
hold on;
for Delta = 0:0.2:2
    theta = linspace(0, 2*pi, 100);
    x_circle = Delta * cos(theta) + x(1);
    y_circle = Delta * sin(theta) + x(2);
    plot(x_circle, y_circle, '--');
end
legend('Quadratic model contours', 'Trust region radii');
hold off;

%% problem 3
delta_max = 2;
delta = 0.1;
etta = 0.25;
x = [-1.2;1];
tol = 1e-6;
iter = 0;
error = 1;
x_history = x;
[X1, X2] = meshgrid(-1.5:0.01:1.5, -0.5:0.01:1.5);
F = (1 - X1).^2 + 100.*(X2 - X1.^2).^2;
figure;hold on;
contour(X1, X2, log(F + 2), 45);
xlabel('x_1');
ylabel('x_2');

while error>tol
    %solve for diretion
    [f, grad, H] = rosenbrock(1,100,x);
    newton_step = -H\grad;
    S_D_step = - (grad'*H*grad\grad'*grad)*grad;
    taw = delta/norm(S_D_step);
    if norm(newton_step) <= delta
        p = newton_step;
    elseif norm(S_D_step) >= delta 
        p = taw*S_D_step;
    else
        taw = find_taw(newton_step,S_D_step,delta);
        p = S_D_step + taw*(newton_step - S_D_step);
    end
    %evaluate the fit 
    x_new = x + p;
    x_history = [x_history, x_new];
    plot(x_history(1,:), x_history(2,:), 'r-o', 'LineWidth', 1, 'MarkerFaceColor', 'r','MarkerSize',2);
    title(['number of iterations to converge: ',num2str(iter)]);
%     pause(0.2);

    [f_new,grad_new,~] = rosenbrock(1,100,x_new);
    mk = @(p) f + grad'*p + 0.5*p'*H*p; 
    rho = (f - f_new)/(f - mk(p));

    %decide if the delta size is ok or not/adjust it
    if rho< 0.25
        delta_new = 0.25*delta;

    elseif rho >3/4 && norm(p) == delta
        delta_new = min(2*delta,delta_max);
    else 
        delta_new = delta;
    end

    %decide to take the step
    if rho > etta
        x = x_new;
        error = norm(grad_new);
    else
        error = norm(grad);
    end
    delta = delta_new;
    iter = iter + 1;
end
hold off; 
disp(['Convereged to:',num2str(x_new(1)),', ',num2str(x_new(2))]);
% function [f, grad, H] = rosenbrock(a,b,x)
%     f = (a - x(1))^2 + b*(x(2) - x(1)^2)^2;
% % Gradient of the Rosenbrock function
%     if nargout > 1
%         grad = [-2*(a - x(1)) - 4*b*x(1)*(x(2) - x(1)^2);
%             2*b*(x(2) - x(1)^2)];
%     end
% % Hessian of the Rosenbrock function
%     if nargout > 2
%         H = [-4*b*(x(2) - x(1)^2) + 8*b*x(1)^2 + 2, -4*b*x(1);
%             -4*b*x(1), 2*b];
%     end
% end
% function taw = find_taw(pU, pB, delta)
%         a = norm(pU - pB)^2;
%         b = 2*pU'*(pU - pB);
%         c = norm(pU)^2 - delta^2;
%         taw1 = (-b + sqrt(b^2 - 4*a*c))/(2*a);
%         taw2 = (-b - sqrt(b^2 - 4*a*c))/(2*a);
%         taw = max(taw1,taw2);
%         taw = max(0,min(2,taw));
% end
% function taw = find_taw(pU, pB, delta)
%     fun = @(taw) norm(pU + taw*(pB - pU)) - delta;
%     taw_initial = 0.5;
%     taw = fzero(fun, taw_initial);
%     taw = max(0, min(2, taw));
% end

%% problem 4
delta_max = 2;
delta = 0.1;
etta = 0.25;
x = randi(60,60,1);
tol = 1e-6;
iter = 0;
error = 1;
while error>tol
    %solve for diretion
    [f, grad, H] = n_dimensional_rosenbrock(x);
    newton_step = -H\grad;
    S_D_step = - (grad'*H*grad\grad'*grad)*grad;
    taw = delta/norm(S_D_step);
    if norm(newton_step) <= delta
        p = newton_step;
    elseif norm(S_D_step) >= delta 
        p = taw*S_D_step;
        disp("Reached the Trust region boundary")
    else
        taw = find_taw(newton_step,S_D_step,delta);
        p = S_D_step + taw*(newton_step - S_D_step);
        disp('Reached the trust region boundary')
    end
    %evaluate the fit 
    x_new = x + p;
    [f_new,grad_new,~] = n_dimensional_rosenbrock(x_new);
    mk = @(p) f + grad'*p + 0.5*p'*H*p; 
    rho = (f - f_new)/(f - mk(p));

    %decide if the delta size is ok or not/ adjust it
    if rho< 0.25
        delta_new = 0.25*delta;
    elseif (rho >3/4 && norm(p) == delta)
        delta_new = min(2*delta,delta_max);
    else 
        delta_new = delta;
    end

    %decide to take the step
    if rho > etta
        x = x_new;
        error = norm(grad_new);
        [~,test] = chol(H);
        if test~=0
            disp("Negative curvature");
        end
    else
        error = norm(grad);
        
    end
    delta = delta_new;
    iter = iter + 1;
    text = sprintf('%.0f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t', ...
        iter,x(1),x(2),f_new,error,rho,delta);
    disp(text);
end
disp("Algorithm Converged");
disp(['number of iterations to converge: ',num2str(iter)]);

function taw = find_taw(pU, pB, delta)
        a = norm(pU - pB)^2;
        b = 2*pU'*(pU - pB);
        c = norm(pU)^2 - delta^2;
        taw1 = (-b + sqrt(b^2 - 4*a*c))/(2*a);
        taw2 = (-b - sqrt(b^2 - 4*a*c))/(2*a);
        taw = max(taw1,taw2);
        taw = max(0,min(2,taw));
end
% function [f, grad, H] = n_dimensional_rosenbrock(x)
%     n = length(x);
%     f = 0;
%     grad = zeros(n, 1);
%     H = zeros(n, n);
%     for i = 1:2:n-1
%         f = f + (1 - x(i))^2 + 10*(x(i+1) - x(i)^2)^2;
%         grad(i) = -2 * (1 - x(i)) - 40* x(i) * (x(i+1) - x(i)^2);
%         grad(i+1) = 20* (x(i+1) - x(i)^2);
%         %Diagonal Hessian elements
%         H(i, i) =  120*x(i)^2 - 40*(x(i+1) - x(i)^2) + 2;
%         H(i+1, i+1) = 20;
%         %Off-diagonal Hessian elements
%         H(i, i+1) = - 40*x(i);
%         H(i+1, i) = - 40*x(i);
%     end
% end



