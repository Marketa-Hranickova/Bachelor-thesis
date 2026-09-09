%% ROBOT_CHECK  Off-hooked trailer robot: coefficients + control feasibility
%
% Companion script for:
%   Zuyev & Grushkovskaya, "Stabilization of a Nonholonomic Car Model with
%   Off-Hooked Trailers", arXiv:2403.03341v1
%
% What this script does:
%   1) self-test:  reproduce the paper's parameter sets (26) and (27)
%   2) coefficients alpha, kappa, theta, beta from the mechanical
%      parameters d0, d1, l1, l2   (eqs. (9), (11)-(13))
%   3) physical initial pose  ->  privileged coordinates z   (eq. (6))
%   4) worst-case amplitude of the controls (16) vs. TurtleBot3 limits
%   5) sweep over (gamma, epsilon): largest gain that still fits the robot
%   6) sweep over d0: how the coefficients collapse as the hitch offset -> 0
%
% Usage:  robot_check
% -------------------------------------------------------------------------

clc; clear; close all;

%% ================================================================ 1. SELF-TEST
% Reference values are the ones hard-coded in params.m (sets (26) and (27)).
% If this block fails, the closed-form coefficients below contain a typo --
% do not trust anything downstream until it passes.

[a26, k26, t26, ~] = offhooked_coeffs(0.1, 0.1, 1, 1);
[a27, k27, t27, ~] = offhooked_coeffs(1.0, 1.0, 1, 1);

fprintf('=== 1. Self-test against the paper ===\n');
fprintf('  set (26):  alpha=%.5f  kappa=%.5f  theta=%.5f   expected 0.10495 / 0.0024 / 0.0112\n', a26, k26, t26);
fprintf('  set (27):  alpha=%.5f  kappa=%.5f  theta=%.5f   expected 1 / 1.5 / 4\n', a27, k27, t27);

tol = 1e-4;
if abs(a26-0.10495)<tol && abs(k26-0.0024)<tol && abs(t26-0.0112)<tol && ...
   abs(a27-1)<tol && abs(k27-1.5)<tol && abs(t27-4)<tol
    fprintf('  --> PASS\n\n');
else
    error('Self-test FAILED: coefficient formulas do not reproduce the paper.');
end

%% ============================================================== 2. THE ROBOT
% TODO: replace these with the values measured on the printed hardware
%       (or read off the CAD model).
%
%   d0 = TB3 wheel axle  ->  first hitch point          [m]
%   l1 = first hitch     ->  first trailer axle         [m]
%   d1 = first trailer axle -> second hitch point       [m]
%   l2 = second hitch    ->  second trailer axle        [m]

d0 = 0.10;
d1 = 0.10;
l1 = 0.15;
l2 = 0.15;

[alpha, kappa, theta, beta] = offhooked_coeffs(d0, d1, l1, l2);

fprintf('=== 2. Robot coefficients ===\n');
fprintf('  d0=%.3f  d1=%.3f  l1=%.3f  l2=%.3f  [m]\n', d0, d1, l1, l2);
fprintf('  alpha=%.6f  kappa=%.6f  theta=%.6f  beta=%.6f\n\n', alpha, kappa, theta, beta);

%% ================================================== 3. CONTROL PARAMETERS
% TurtleBot3 Burger limits (Sebastian's paper, Sec. 4.1)
U1_MAX = 0.22;    % [m/s]   translational
U2_MAX = 2.84;    % [rad/s] angular

% Controller parameters (defaults taken from params.m)
gamma   = 1.0;
epsilon = 0.1;            % omega = 2*pi/epsilon
k12     = 8;
k112    = 6;
k1112   = 5;
h       = 1.0;            % h = 1 for the thesis; |log(||z||^2)| is the other project

% Initial pose in PHYSICAL units:  x = [xC0, yC0, phi, theta1, theta2]
x0 = [-0.5; -0.5; 0; 0; 0];

z0 = x_to_z(x0, d0, d1, l1, l2);

fprintf('=== 3. Initial condition ===\n');
fprintf('  x0 = [%.3f %.3f %.3f %.3f %.3f]\n', x0);
fprintf('  z0 = [%.5f %.5f %.5f %.5f %.5f]\n\n', z0);

%% ============================================ 4. AMPLITUDE FEASIBILITY CHECK
omega = 2*pi/epsilon;
[u1max, u2max, b1, b2] = amp_bounds(z0, alpha, kappa, theta, gamma, omega, k12, k112, k1112, h);

fprintf('=== 4. Worst-case control amplitudes (epsilon=%.2f, gamma=%.3g) ===\n', epsilon, gamma);
fprintf('  |u1| <= %10.3f  m/s     (limit %.2f)   %s\n', u1max, U1_MAX, verdict(u1max, U1_MAX));
fprintf('  |u2| <= %10.3f  rad/s   (limit %.2f)   %s\n', u2max, U2_MAX, verdict(u2max, U2_MAX));
fprintf('  u1 term breakdown:  |a1|=%.4g  bracket-2=%.4g  bracket-3=%.4g  bracket-4=%.4g\n', b1);
fprintf('  u2 term breakdown:  |a2|=%.4g  bracket-2=%.4g  bracket-3=%.4g  bracket-4=%.4g\n\n', b2);

%% ================================================ 5. FEASIBLE (gamma, epsilon)
% For a fixed epsilon, find the largest gamma for which BOTH limits hold.
% Note how weakly this scales: the length-4 term grows like gamma^(1/4),
% so shrinking gamma buys very little.

eps_list = [0.1 0.5 1 2 4 8];
fprintf('=== 5. Largest admissible gain per epsilon ===\n');
fprintf('  %8s %10s %14s\n', 'epsilon', 'omega', 'max gamma');
for e = eps_list
    w  = 2*pi/e;
    lo = 1e-14;  hi = 10;
    for it = 1:200                       % bisection in log-space
        mid = sqrt(lo*hi);
        [p, q] = amp_bounds(z0, alpha, kappa, theta, mid, w, k12, k112, k1112, h);
        if p <= U1_MAX && q <= U2_MAX
            lo = mid;
        else
            hi = mid;
        end
    end
    fprintf('  %8.2f %10.2f %14.3e\n', e, w, lo);
end
fprintf('\n');

%% ======================================================== 6. SWEEP OVER d0
% Thesis question: how does the behaviour depend on the hitch offset?
% All three coefficients carry a factor d0, so they collapse as d0 -> 0.

d0_list = linspace(0.005, 0.30, 200);
A = zeros(size(d0_list)); K = A; T = A; B = A;
for i = 1:numel(d0_list)
    [A(i), K(i), T(i), B(i)] = offhooked_coeffs(d0_list(i), d1, l1, l2);
end

figure('Name', 'Coefficients vs hitch offset', 'Color', 'w');
subplot(1,2,1);
plot(d0_list, A, 'LineWidth', 1.5); hold on;
plot(d0_list, K, 'LineWidth', 1.5);
plot(d0_list, T, 'LineWidth', 1.5);
xlabel('d_0  [m]'); ylabel('coefficient value');
legend('\alpha', '\kappa', '\theta', 'Location', 'northwest');
title('Linear scale'); grid on;

subplot(1,2,2);
loglog(d0_list, abs(A), 'LineWidth', 1.5); hold on;
loglog(d0_list, abs(K), 'LineWidth', 1.5);
loglog(d0_list, abs(T), 'LineWidth', 1.5);
xlabel('d_0  [m]'); ylabel('|coefficient|');
legend('\alpha', '\kappa', '\theta', 'Location', 'northwest');
title('Log-log: slope = order of vanishing'); grid on;

fprintf('=== 6. d0 sweep plotted. ===\n');
fprintf('  On the log-log axes the slopes show how fast each coefficient\n');
fprintf('  vanishes as the hitch offset goes to zero.\n');

%% ========================================================= LOCAL FUNCTIONS

function [alpha, kappa, theta, beta] = offhooked_coeffs(d0, d1, l1, l2)
% Closed-form coefficients of the nilpotent approximation, eqs. (11)-(13),
% plus beta from eq. (9).

    alpha = ( d0^3*( l1^3 + l1^2*l2 - l1*l2^2 - d1*l2*(d1 + l2) ) ...
            + d0*l1^3*( d0*(l1 + 2*l2) + l2*(l1 + l2) ) ) ...
            / ( 2*(d0 + l2)*l1^3 );

    kappa = d0^2*( 1.5*l2^2*(d0 + l1)*(d1 + l1) ...
            - l2*( l1^3 + 0.5*d0*l1^2 - d1*l1*(d0 + 1.5*d1) - 1.5*d0*d1^2 ) ...
            + 0.5*d0*d1^2*l1 ) ...
            / ( 3*(d0 + l2)*l1^3 );

    theta = d0^2*( l2^2*(d1 + l1)*(d0 + l1) ...
            + ( d0*d1 + (d0 + d1)*l1 )*d1*l2 + d0*d1^2*l1 ) ...
            / ( (d0 + l2)*l1^3 );

    beta  = ( d0*l2*( d0^2*(d1 + l1)*(l1 - l2 - d1) + l1^3*(d0 + l2) ) ) ...
            / ( 6*(d0 + l2)*l1^2 );
end


function z = x_to_z(x, d0, d1, l1, l2)
% Physical state x = [xC0; yC0; phi; theta1; theta2]  ->  privileged z.
% Equations (6) and (9) of the paper. Same as the x_to_z.m in original_system/.

    c4_x2 = (l1 + l2);
    c4_x3 = -( d0*( d0*(l1^2 + l1*l2 + l2^2) + l1*l2*(l1 + l2) ) ) ...
            / ( (d0 + l1)*(d0 + l2) );
    c4_x4 = -( l2^2*(l1 + l2)*(d1 + l1) / ((d0 + l2)*(d1 + l2)) ...
             + ( l1^2*l2*(d0*d1 + (d0 + d1)*l1) + d0*d1*l1^3 ) ...
               / ((d0 + l1)*(d0 + l2)*(d1 + l2)) );
    c4_x5 = -l2^4 / ( (d0 + l2)*(d1 + l2) );

    c5_x2 = -l1*l2;
    c5_x3 = ( d0*l1*l2*( l1*l2 + d0*(l1 + l2) ) ) / ( (d0 + l1)*(d0 + l2) );
    c5_x4 = ( l1*l2^3*(d1 + l1) / ((d0 + l2)*(d1 + l2)) ...
            + ( l1^2*l2*( l2*(d0*d1 + (d0 + d1)*l1) + d0*d1*l1 ) ) ...
              / ((d0 + l1)*(d0 + l2)*(d1 + l2)) );
    c5_x5 = ( l1*l2^4 ) / ( (d0 + l2)*(d1 + l2) );

    y4 = c4_x2*x(2) + c4_x3*x(3) + c4_x4*x(4) + c4_x5*x(5);
    y5 = c5_x2*x(2) + c5_x3*x(3) + c5_x4*x(4) + c5_x5*x(5);

    [~, ~, ~, beta] = offhooked_coeffs(d0, d1, l1, l2);

    z    = zeros(5,1);
    z(1) =  x(1);
    z(2) =  x(3);
    z(3) = -x(2);
    z(4) =  y4;
    z(5) =  y5 - beta*x(3)^3;
end


function [u1max, u2max, br1, br2] = amp_bounds(z, alpha, kappa, theta, gamma, omega, k12, k112, k1112, h)
% Worst-case (triangle-inequality) bound on the controls (16). The true
% maximum over one period is somewhat smaller, but of the same order.

    a1    = -gamma*h*z(1);
    a2    = -gamma*h*z(2);
    a12   = -gamma*h*( z(1)*z(2) + z(3) );
    a112  = -gamma*h*( alpha*z(2)^3 + z(1)*z(3) + z(4) );
    a1112 = -gamma*h*( (alpha + 2*kappa - theta)*z(1)*z(2)^3 ...
                     + (alpha + 3*kappa)*z(2)^2*z(3) ...
                     + z(1)*z(4) + z(5) );

    T12   = ( 2*k12*omega*abs(a12) )^(1/2);
    T112  = ( 2*k112*omega )^(2/3) * abs(a112)^(1/3);
    T1112 = ( k1112*omega )^(3/4) * abs(a1112)^(1/4);

    br1 = [abs(a1), T12,   T112, 2*T1112];   % u1 contributions
    br2 = [abs(a2), T12, 2*T112, 6*T1112];   % u2 contributions

    u1max = sum(br1);
    u2max = sum(br2);
end


function s = verdict(val, lim)
    if val <= lim
        s = 'OK';
    else
        s = sprintf('OVER by %.0fx', val/lim);
    end
end
