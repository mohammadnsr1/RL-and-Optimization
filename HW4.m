%% Problem 1
dimension = [5,8,12,20];
iter_max = 1000;
tol = 1e-6;
for n= dimension
    A = hilb(n);
    b = ones(n,1);
    iter = 0;
    x = zeros(n,1);
    r = b - A*x;
    p = r;
    while norm(r) > tol && iter < iter_max
        Ap = A*p;
        alpha = (p'*r)/(p'*Ap);
        x = x + alpha*p;
        r = r - alpha*Ap;
        beta = (Ap'*r)/(Ap'*p);
        p = r - beta*p;
        iter = iter + 1;
    end
    disp(['Number of iterations to converge:', num2str(iter)]);
    disp('Vector X:');
    x
end

%% Problem 4
% Parameters
alpha0 = 1;
rho = 0.5;
c = 1e-4;
tol = 1e-6;
iter = 0;
iter_max = 2000;
%Initial points
x0_1 = [1.2; 1.2];
x = x0_1;
[~, grad, ~] = rosenbrock(x);
p = grad;
while norm(grad) > tol && iter < iter_max
    alpha = backtracking_line_search(x, p, grad, alpha0, rho, c);
    x_new = x + alpha*p;
    [~, grad_new, ~] = rosenbrock(x_new);
    beta = (grad_new'*grad_new)/(grad'*grad);
    p = -grad_new + beta*p;
    grad = grad_new;
    x = x_new;
    iter = iter + 1;
end
    disp(['Number of iterations to converge:', num2str(iter)]);
    disp(['Vector X:', num2str(x(1)),',',num2str(x(2))]);


%% Problem 5
% Parameters
alpha0 = 1;
rho = 0.9;
c = 1e-4;
tol = 1e-6;
iter = 0;
%Initial point
x0_1 = [1.2; 1.2];
x = x0_1;
[~, grad, ~] = rosenbrock(x);
p = grad;
H = eye(2); %initial guess for hessian 
while norm(grad) > tol
    p = -H*grad;
    alpha = backtracking_line_search(x, p, grad, alpha0, rho, c);
    x_new = x + alpha*p;
    [~, grad_new, ~] = rosenbrock(x_new);
    s = x_new - x;
    y = grad_new - grad;
    % update hessian matrix
    H_new = (eye(2) - (s * y')/ (y' * s))*H*(eye(2) -(y * s')/(y' * s))+...
        (s * s')/ (y' * s);
    grad = grad_new;
    x = x_new;
    H = H_new;
    iter = iter + 1;
end
disp(['Number of iterations to converge:', num2str(iter)]);
disp(['Vector X:', num2str(x(1)),',',num2str(x(2))]);

% function [f, grad, H] = rosenbrock(x)
% a = 1; b = 100;
% f = (a - x(1))^2 + b*(x(2) - x(1)^2)^2;
% % Gradient of the Rosenbrock function
% if nargout > 1
%     grad = [-2*(a - x(1)) - 4*b*x(1)*(x(2) - x(1)^2);
%         2*b*(x(2) - x(1)^2)];
% end
% % Hessian of the Rosenbrock function
% if nargout > 2
%     H = [-4*b*(x(2) - x(1)^2) + 8*b*x(1)^2 + 2, -4*b*x(1);
%         -4*b*x(1), 2*b];
% end
% end
% % backtracking_algorithm
% function alpha = backtracking_line_search(x, p, grad, alpha0, rho, c)
% alpha = alpha0;
% while rosenbrock(x + alpha*p) > rosenbrock(x) + c*alpha*grad'*p
%     alpha = rho*alpha;
% end
% end


%% problem 6
t1 = [1, 2, 4, 5,8,4.1]';
y = [3, 4, 6, 11,20,46]';
x0 = [3, 0.1]';
tol = 1e-10;
iter = 0;
iter_max = 1000;
alpha0 = 1;
rho = 0.5;
c = 1e-4;
text = sprintf('\titer\t \tx1 \tx2 \t   grad(res) \t  residual');
disp(text);
syms fit_function(x1,x2,x3,t)
fit_function(x1,x2,t) = x1*exp(x2*t);
jacobian_matrix = jacobian(fit_function, [x1, x2]);
jacobian_matrix = matlabFunction(jacobian_matrix);
f(x1,x2)= norm((fit_function(x1,x2,t1) - y),2)^2;
x = x0;
r = double(fit_function(x(1), x(2),t1) - y);
grad = jacobian_matrix(x(1), x(2),t1)'*r;
H = jacobian_matrix(x(1), x(2),t1)'*jacobian_matrix(x(1), x(2),t1);
error = norm(grad)^2;
while error > tol && iter < iter_max
    p = -H\grad;
    % backtracking_algorithm
    alpha = alpha0;
    while double(f(x(1)+alpha*p(1), x(2)+alpha*p(2)))>...
            double(f(x(1),x(2)) + c*alpha*grad'*p)
        alpha = rho*alpha;
    end
    x_new = x + alpha*p;
    r_new = double(fit_function(x_new(1), x_new(2),t1) - y);
    grad = jacobian_matrix(x_new(1), x_new(2),t1)'*r_new;
    H = jacobian_matrix(x_new(1), x_new(2),t1)'*jacobian_matrix(x_new(1), x_new(2),t1);
    x = x_new;
    iter = iter + 1;
    error = norm(grad)^2;
    disp(double([iter x(1) x(2) double(error) norm(r_new)^2]))
end
% %plot the data and the fitted curve
figure(1);
t2 = 1:0.01:8;
plot(t1,y,'ko',t2,fit_function(x(1),x(2),t2),'r-');hold on;
xlabel('t');
ylabel('y');
%%
% t2 = 1:0.01:10;
% fun = @(x,t) x(1)*exp(x(2)*t); 
% options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt','Display','iter');
% [x,resnorm,residual,exitflag,output] = lsqcurvefit(fun,x0,t1,y,[],[],options);
% figure(2);
% plot(t1,y,'ko',t2,fun(x,t2),'r-');


