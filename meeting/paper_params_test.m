% Exact model with the design parameters of the paper (Zuyev & Grushkovskaya):
% quadratic V (h = 1), no limit on u1. Our geometry and our design start.
cd('/Users/marke/Documents/MATLAB/turtlebot/bachelor thesis/2 trailers');
P = params();  geo = struct('d0', P.d0, 'l1', P.l1, 'd1', P.d1, 'l2', P.l2);
[alpha, kappa, theta] = offhooked_coeffs(geo.d0, geo.d1, geo.l1, geo.l2);
x0 = P.x0(:);  z0 = x_to_z(x0, geo.d0, geo.d1, geo.l1, geo.l2);  r0 = norm(z0);
h1 = @(z) 1;
% name, epsilon, gamma, k
cases = { 'paper: k=(8,6,5) eps=0.1 gamma=1',  0.1, 1,   [8 6 5];
          'k=(14,5,1) eps=0.1 gamma=1',        0.1, 1,   [14 5 1];
          'k=(8,6,5) eps=0.5 gamma=1',         0.5, 1,   [8 6 5];
          'template: k=(8,6,5) eps=2 gamma=0.1', 2,  0.1, [8 6 5] };
T = 12;
for c = 1:size(cases, 1)
    nm = cases{c,1};  ep = cases{c,2};  gam = cases{c,3};  k = cases{c,4};
    dt = min(0.02, 2*pi/(max(k)*2*pi/ep)/40);
    fprintf('\n##### %s\n', nm);
    [ta, za] = trailer2('simulate_case', z0, alpha, kappa, theta, gam, ep, 2*pi/ep, k(1), k(2), k(3), 0, T, dt, h1);
    [te, xe, ze, u1e] = trailer2('simulate_case_physical', x0, geo.d0, geo.d1, geo.l1, geo.l2, ...
                                 alpha, kappa, theta, gam, ep, 2*pi/ep, k(1), k(2), k(3), 0, T, dt, h1);
    na = vecnorm(za, 2, 2)/r0;  ne = vecnorm(ze, 2, 2)/r0;
    for tq = [1 3 6 12]
        ja = find(ta >= tq, 1);  je = find(te >= tq, 1);
        fprintf('  t = %4g s:  approx %.4g   exact %.4g\n', tq, na(ja), ne(je));
    end
    fprintf('  max |u1| = %.2f m/s, largest |angle| = %.0f deg, min exact = %.4g\n', ...
            max(abs(u1e)), rad2deg(max(max(abs(xe(:,3:5))))), min(ne));
end
