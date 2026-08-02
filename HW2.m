%% Problem 2
% part 1, Newtons method
syms x;
g = 7*sin(x)*exp(-x) -1;
g_d = diff(g,1);

%plot the function
figure(1);
fplot(g,[-7,2]);hold on;
plot(-7:0.1:2,zeros(91,1),LineStyle="--",LineWidth=1);
xlabel("x");ylabel('g(x)');title('Plot of g(x) = 7sin(x)exp(-x) - 1');
grid on;

% implement the Newton's method
g = matlabFunction(g);
g_d = matlabFunction(g_d);
max_relative_error = 1e-6;
max_iteration = 1000;
iter = 1;
initial_guess = -2;
x = initial_guess;
x_new = x - (g(x)/g_d(x));
error_rates = [];
while (abs(x_new - x)/abs(x) >max_relative_error && iter< max_iteration )
    x= x_new;
    x_new = x - (g(x)/g_d(x));
    iter = iter + 1;
    error = x_new - x;
    error_rates = [error_rates,error];
    if(length(error_rates)>2)
        error_rates = error_rates(end-2:end);
    end
end
disp(["root found at:",num2str(x_new)]);
disp(["Number of iteratoins:",num2str(iter)]);
rate_of_convergence = log(error_rates(2))/log(error_rates(1));
disp(['Rate of convergence is:', num2str(rate_of_convergence)]);
%%
%part2: using fzero
fun = @(x) 7*sin(x)*exp(-x) -1; 
x0 = [-4 -2]; 
options = optimset('Display','iter'); 
[x fval exitflag output] = fzero(fun,x0,options);

%% Problem 3
syms x1 x2
f = 0.5*(x1^2 - x2)^2 + 0.5*(1 - x1)^2;
f_d = gradient(f,[x1,x2]);
f_dd = hessian(f,[x1,x2]);
f = matlabFunction(f,'Vars',{[x1; x2]});
f_d = matlabFunction(f_d, 'Vars', {[x1; x2]});
f_dd = matlabFunction(f_dd, 'Vars', {[x1; x2]});
initial_guess = [2;2];
max_relative_error = 1e-6;
max_iteration = 1000;
iter = 1;
x = initial_guess;
x_new = -f_dd(x)\f_d(x) + x;
step = x_new - x;

%computing values at initial value and after the first step
first_value = f(x);
second_value = f(x + step);
disp(['After 1 iteration, the new values for x1,x2 are: ',num2str(x_new(1)),', ', num2str(x_new(2))]);
disp(['the function values after one iteration are:', num2str(first_value),', ',num2str(second_value)]);
disp("we see large decrease in the function value, so that it is a good step");
while (norm(x_new - x)/norm(x) >max_relative_error && iter< max_iteration )
    x= x_new;
    x_new = -f_dd(x)\f_d(x) + x;
    iter = iter +1;
end
disp(['Exact minimizer: ',num2str(x_new(1)),',',num2str(x_new(2)) ])
disp(['Total iteration: ',num2str(iter)]);

%% Problem 5
% Parameters
alpha0 = 1;
rho = 0.5;
c = 1e-4;
tol = 1e-6;

% Initial points
x0_1 = [1.2; 1.2];
x0_2 = [-1.2; 1];

error = 1;
iter = 0;
alpha_history_sd = [];
x = x0_1;
while error>tol
    %steepest-descent
    [f, grad] = rosenbrock(x);
    p = -grad;
    alpha = backtracking_line_search(x, p, grad, alpha0, rho, c);
    alpha_history_sd = [alpha_history_sd; alpha];
    x_new = x + alpha*p;
    error = norm(x_new -x);
    x = x_new;
    iter = iter + 1;
end
disp(["Total number of iterations for steepest_descent:",num2str(iter)]);
disp(['converged values for x1,x2 are: ',num2str(x(1)),', ', num2str(x(2))]);

% Newton_method
iter = 0;
error = 1;
alpha_history_n = [];
x_n = x0_2;
while error > tol
    [~, grad, H] = rosenbrock(x_n);
    p = -H\grad;
    alpha = backtracking_line_search(x_n, p, grad, alpha0, rho, c);
    alpha_history_n = [alpha_history_n; alpha];
    x_n_new = x_n + alpha*p;
    error = norm(x_n_new - x_n);
    x_n = x_n_new;
    iter = iter + 1;
end
disp(["Total number of iterations for newton_method:",num2str(iter)]);
disp(['converged values for x1,x2 are: ',num2str(x_n(1)),', ', num2str(x_n(2))]);

function [f, grad, H] = rosenbrock(x)
    a = 1; b = 100;
    f = (a - x(1))^2 + b*(x(2) - x(1)^2)^2;
% Gradient of the Rosenbrock function
    if nargout > 1
        grad = [-2*(a - x(1)) - 4*b*x(1)*(x(2) - x(1)^2);
            2*b*(x(2) - x(1)^2)];
    end
% Hessian of the Rosenbrock function
    if nargout > 2
        H = [-4*b*(x(2) - x(1)^2) + 8*b*x(1)^2 + 2, -4*b*x(1);
            -4*b*x(1), 2*b];
    end
end

% backtracking_algorithm
function alpha = backtracking_line_search(x, p, grad, alpha0, rho, c)
alpha = alpha0;
while rosenbrock(x + alpha*p) > rosenbrock(x) + c*alpha*grad'*p
    alpha = rho*alpha;
end
end




