
mu_values = [1, 10, 100, 1000];
% Initial guess for starting point
x_current = [0 0];
alpha0 = 1;
rho = 0.5;
C = 1e-4;
for i = 1:length(mu_values)
    mu = mu_values(i);
    fprintf('\nOptimizing with mu = %d\n', mu);
    % Using steepest descent with backtracking line search to find the minimum
    x_current = steepest_method(@penalty_function, x_current, mu, alpha0, rho, C);
    fprintf('Approximate solution for mu = %d: (%f, %f)\n', mu, x_current(1), x_current(2));
end


function [f, grad] = penalty_function(x, mu)
    f0 = x(1) + x(2);
    c = x(1)^2 + x(2)^2 - 2;
    f = f0 + (mu/2) * c^2;
    grad_f0 = [1; 1];
    grad_c = [2*x(1); 2*x(2)];
    if nargout > 1
        grad = grad_f0 + mu * c * grad_c; 
    end
end

function x_min = steepest_method(penaltyFunc, x_start, mu, alpha0, rho, C)
    x = x_start;
    tolerance = 1/mu;
    [f, grad] = penaltyFunc(x, mu);
    while norm(grad) > tolerance
        p = -grad;
        alpha = alpha0;
        while true
            x_new = x + alpha * p;
            f_new = penaltyFunc(x_new, mu);
            if f_new <= f + C * alpha * grad' * p
                break;
            end
            alpha = rho * alpha;
        end
        x = x_new;
        [f, grad] = penaltyFunc(x, mu);
    end
    x_min = x;
end
