clear; clc; close all;

%% 1. COMMON CONSTANTS & SYSTEM
rng(7);   % reproducible comparison

n = 0.00104218449;   % rad/s
T_final = 800;       % s
dt = 1;              % s
t_vec = 0:dt:T_final;
N_steps = length(t_vec);

options_ode = odeset('RelTol', 1e-10, 'AbsTol', 1e-12);
options_fsolve = optimoptions('fsolve', ...
    'Display', 'none', ...
    'TolFun', 1e-12, ...
    'TolX', 1e-12, ...
    'MaxIterations', 500, ...
    'Algorithm', 'levenberg-marquardt');

% CW dynamics
A = zeros(6,6);
A(1,4) = 1;   A(2,5) = 1;   A(3,6) = 1;
A(4,1) = 3*n^2;   A(4,5) = 2*n;
A(5,4) = -2*n;
A(6,3) = -n^2;

B = zeros(6,3);
B(4,1) = 1;
B(5,2) = 1;
B(6,3) = 1;

% Discrete-time model for estimator
Phi = expm(A*dt);
Gamma = integral(@(tau) expm(A*tau)*B, 0, dt, 'ArrayValued', true);

%% 2. BOUNDARY CONDITIONS & TUNING
r0 = [0.03031809; 0; 31.16639];         % km
v0 = [-0.02963377; 0.04570523; 0];      % km/s
x0 = [r0; v0];
xf = zeros(6,1);

u_max = 8e-4;   % km/s^2

% KF settings
H = [eye(3), zeros(3,3)];
R_nominal = diag([1e-3, 1e-3, 1e-3].^2);
Q_proc    = diag([1e-6, 1e-6, 1e-6, 1e-7, 1e-7, 1e-7].^2);
P0        = diag([1e-3, 1e-3, 1e-3, 1e-5, 1e-5, 1e-5].^2);

% Adaptive homotopy settings
eps_min   = 0.00;
eps_max   = 1.00;
beta      = 0.25;

% Composite innovation score tuning
m_meas    = 3;       % measurement dimension
rho_score = 0.85;    % low-pass filtering on composite score
k_mtf     = 12.0;    % weight on MTF severity
score_cap = 50.0;    % saturation to avoid huge spikes
eps_blend = 0.10;    % epsilon smoothing
reg_eps   = 1e-12;   % numerical regularization

% Receding horizon re-solve
resolve_period = 10;   % s
resolve_min_rem = 20;  % s

%% 3. OPEN-LOOP BASELINES
disp('--- Solving open-loop baselines ---');

% A) Energy-optimal baseline
disp('...Energy-optimal baseline');
A_aug = [A, -B*B'; zeros(6,6), -A'];
Phi_T = expm(A_aug*T_final);
lambda0_energy = Phi_T(1:6, 7:12) \ (xf - Phi_T(1:6,1:6)*x0);

[T_energy, Z_energy] = ode45(@(t,z) odefun_energy(t, z, A_aug), ...
    t_vec, [x0; lambda0_energy], options_ode);

u_energy = zeros(length(T_energy),1);
for k = 1:length(T_energy)
    u_vec = -B' * Z_energy(k,7:12)';
    u_energy(k) = norm(u_vec);
end

% B) Fuel-optimal baseline via continuation
disp('...Fuel-optimal baseline via continuation');
lambda0_guess = lambda0_energy;
eps_schedule = 1.0:-0.1:0.0;

for i = 1:length(eps_schedule)
    eps_val = eps_schedule(i);
    [lambda_sol, fval, exitflag] = fsolve( ...
        @(l) residualfun_fuel(l, x0, xf, A, B, u_max, T_final, eps_val, options_ode), ...
        lambda0_guess, options_fsolve);

    fprintf('   continuation step %2d | eps = %.2f | exitflag = %2d | residual = %.3e\n', ...
        i, eps_val, exitflag, norm(fval));

    if exitflag <= 0
        warning('Continuation step at eps = %.2f did not fully converge.', eps_val);
    end
    lambda0_guess = lambda_sol;
end
lambda0_fuel = lambda0_guess;

[T_fuel, Z_fuel] = ode45(@(t,z) odefun_fuel(t, z, A, B, u_max, 0), ...
    t_vec, [x0; lambda0_fuel], options_ode);

u_fuel = zeros(length(T_fuel),1);
for k = 1:length(T_fuel)
    lambda_v = Z_fuel(k,10:12)';
    [u_cmd, ~, ~] = control_from_costate(lambda_v, u_max, 0);
    u_fuel(k) = norm(u_cmd);
end

%% 4. COMMON SENSOR-DEGRADATION REALIZATION
% Same random realization for all closed-loop cases -> fair comparison
noise_seq = randn(3, N_steps);

R_scale_hist = ones(1, N_steps);
bias_hist    = zeros(3, N_steps);

for k = 1:N_steps
    t = t_vec(k);

    if t >= 250 && t < 325
        % severe noise inflation
        R_scale_hist(k) = 300;
        bias_hist(:,k)  = [0;0;0];
    elseif t >= 325 && t <= 400
        % severe noise + transient measurement bias
        R_scale_hist(k) = 300;
        bias_hist(:,k)  = [0.010; -0.008; 0.012];  % km
    else
        R_scale_hist(k) = 1;
        bias_hist(:,k)  = [0;0;0];
    end
end

%% 5. CLOSED-LOOP CASES
disp('--- Running closed-loop cases ---');

% Case 1: Plain KF + fixed eps = 0
cfg_plain.name            = 'Plain KF + Fixed Bang-Bang';
cfg_plain.use_mtf         = false;
cfg_plain.use_adaptive_eps= false;
cfg_plain.fixed_eps       = 0.0;
cfg_plain.init_eps        = 0.0;
cfg_plain.beta            = beta;
cfg_plain.m_meas          = m_meas;
cfg_plain.rho_score       = rho_score;
cfg_plain.k_mtf           = k_mtf;
cfg_plain.score_cap       = score_cap;
cfg_plain.eps_blend       = eps_blend;
cfg_plain.eps_min         = eps_min;
cfg_plain.eps_max         = eps_max;
cfg_plain.reg_eps         = reg_eps;

% Case 2: MTF-only + fixed eps = 0
cfg_mtf = cfg_plain;
cfg_mtf.name     = 'MTF KF + Fixed Bang-Bang';
cfg_mtf.use_mtf  = true;

% Case 3: Proposed method: MTF + adaptive epsilon
cfg_adapt = cfg_plain;
cfg_adapt.name             = 'MTF KF + Adaptive Homotopy';
cfg_adapt.use_mtf          = true;
cfg_adapt.use_adaptive_eps = true;
cfg_adapt.fixed_eps        = 0.0;
cfg_adapt.init_eps         = eps_max;

res_plain = simulate_case(cfg_plain, A, B, Phi, Gamma, H, ...
    R_nominal, Q_proc, P0, x0, xf, u_max, ...
    lambda0_fuel, lambda0_energy, ...
    t_vec, dt, options_ode, options_fsolve, ...
    noise_seq, R_scale_hist, bias_hist, ...
    resolve_period, resolve_min_rem);

res_mtf = simulate_case(cfg_mtf, A, B, Phi, Gamma, H, ...
    R_nominal, Q_proc, P0, x0, xf, u_max, ...
    lambda0_fuel, lambda0_energy, ...
    t_vec, dt, options_ode, options_fsolve, ...
    noise_seq, R_scale_hist, bias_hist, ...
    resolve_period, resolve_min_rem);

res_adapt = simulate_case(cfg_adapt, A, B, Phi, Gamma, H, ...
    R_nominal, Q_proc, P0, x0, xf, u_max, ...
    lambda0_fuel, lambda0_energy, ...
    t_vec, dt, options_ode, options_fsolve, ...
    noise_seq, R_scale_hist, bias_hist, ...
    resolve_period, resolve_min_rem);

disp('Simulation complete.');

%% 6. METRICS
dV_energy = trapz(T_energy, u_energy);
dV_fuel   = trapz(T_fuel,   u_fuel);

fprintf('\n================== OPEN-LOOP BASELINES ==================\n');
fprintf('Energy-optimal open-loop   : dV = %.6e km/s | miss = %.6e km\n', ...
    dV_energy, norm(Z_energy(end,1:3)));
fprintf('Fuel-optimal   open-loop   : dV = %.6e km/s | miss = %.6e km\n', ...
    dV_fuel,   norm(Z_fuel(end,1:3)));

fprintf('\n================== CLOSED-LOOP COMPARISON ==================\n');
print_case_metrics(res_plain, dV_fuel);
print_case_metrics(res_mtf,   dV_fuel);
print_case_metrics(res_adapt, dV_fuel);

fprintf('\n================== COMPUTATIONAL BURDEN ==================\n');
fprintf('Proposed MTF-Adaptive Solver Real-Time Viability:\n');
fprintf('   Mean Solve Time : %.2f milliseconds\n', 1000 * mean(res_adapt.solve_times));
fprintf('   Max Solve Time  : %.2f milliseconds\n', 1000 * max(res_adapt.solve_times));
fprintf('   Solve Success %% : %.2f %%\n', 100 * (sum(res_adapt.solve_success) / length(res_adapt.solve_success)));

%% 7. FIGURE 1: THRUST / DELTA-V / SCORE / EPSILON
figure('Color','w');
sgtitle('Closed-Loop Comparison Under the Same Sensor Degradation');

subplot(2,2,1);
plot(T_energy, u_energy, 'g:', 'LineWidth', 1.5); hold on;
plot(T_fuel, u_fuel, 'k--', 'LineWidth', 1.2); 
plot(t_vec, res_plain.u_mag, 'LineWidth', 1.5);
plot(t_vec, res_mtf.u_mag,   'LineWidth', 1.5);
plot(t_vec, res_adapt.u_mag, 'LineWidth', 2.0);
xline(250, 'r--'); xline(325, 'r--'); xline(400, 'g--');
xlabel('Time [s]'); ylabel('||u|| [km/s^2]');
title('Instantaneous Thrust');
legend('Energy OL', 'Fuel OL', 'Plain KF + Fixed \epsilon', 'MTF + Fixed \epsilon', 'MTF + Adaptive \epsilon', 'Location', 'best');
grid on; ylim([-0.05*u_max, 1.10*u_max]);

subplot(2,2,2);
plot(T_energy, cumtrapz(T_energy, u_energy), 'g:', 'LineWidth', 1.5); hold on;
plot(T_fuel, cumtrapz(T_fuel, u_fuel), 'k--', 'LineWidth', 1.2); 
plot(t_vec, cumtrapz(t_vec, res_plain.u_mag), 'LineWidth', 1.5);
plot(t_vec, cumtrapz(t_vec, res_mtf.u_mag),   'LineWidth', 1.5);
plot(t_vec, cumtrapz(t_vec, res_adapt.u_mag), 'LineWidth', 2.0);
xline(250, 'r--'); xline(325, 'r--'); xline(400, 'g--');
xlabel('Time [s]'); ylabel('\Delta v [km/s]');
title('Cumulative \Delta v');
legend('Energy OL', 'Fuel OL', 'Plain KF + Fixed \epsilon', 'MTF + Fixed \epsilon', 'MTF + Adaptive \epsilon', 'Location', 'best');
grid on;

subplot(2,2,3);
plot(t_vec, res_adapt.NIS_history,       'k-',  'LineWidth', 1.2); hold on;
plot(t_vec, res_adapt.MTF_ratio_history, 'b--', 'LineWidth', 1.5);
plot(t_vec, res_adapt.Score_history,     'r-',  'LineWidth', 2.0);
xline(250, 'r--', 'Degrades'); xline(325, 'r--'); xline(400, 'g--', 'Recovers');
xlabel('Time [s]'); ylabel('Metric');
title('Adaptive Scheduler Inputs');
legend('Raw NIS', 'MTF Ratio', 'Composite Score', 'Location', 'best');
grid on;

subplot(2,2,4);
plot(t_vec, res_plain.Eps_history, 'LineWidth', 1.5); hold on;
plot(t_vec, res_mtf.Eps_history,   'LineWidth', 1.5);
plot(t_vec, res_adapt.Eps_history, 'LineWidth', 2.0);
yline(eps_min, 'k--', '\epsilon_{min}');
yline(eps_max, 'k:',  '\epsilon_{max}');
xline(250, 'r--'); xline(325, 'r--'); xline(400, 'g--');
xlabel('Time [s]'); ylabel('\epsilon');
title('Homotopy Parameter');
legend('Plain KF + Fixed \epsilon', 'MTF + Fixed \epsilon', 'MTF + Adaptive \epsilon', 'Location', 'best');
grid on; ylim([-0.02, 1.05]);

%% 8. FIGURE 2: TRAJECTORY COMPARISON
figure('Color','w');
sgtitle('Trajectory Comparison');

subplot(2,2,1);
plot3(Z_energy(:,1), Z_energy(:,2), Z_energy(:,3), 'g:', 'LineWidth', 1.5); hold on;
plot3(Z_fuel(:,1), Z_fuel(:,2), Z_fuel(:,3), 'k--', 'LineWidth', 1.2); 
plot3(res_plain.X_true(1,:), res_plain.X_true(2,:), res_plain.X_true(3,:), 'LineWidth', 1.5);
plot3(res_mtf.X_true(1,:),   res_mtf.X_true(2,:),   res_mtf.X_true(3,:),   'LineWidth', 1.5);
plot3(res_adapt.X_true(1,:), res_adapt.X_true(2,:), res_adapt.X_true(3,:), 'LineWidth', 2.0);
plot3(r0(1), r0(2), r0(3), 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 8);
plot3(0,0,0, 'kx', 'LineWidth', 2, 'MarkerSize', 10);
xlabel('x [km]'); ylabel('y [km]'); zlabel('z [km]');
title('3D Relative Trajectory');
legend('Energy OL', 'Fuel OL', 'Plain KF', 'MTF Fixed \epsilon', 'MTF Adaptive \epsilon', 'Start', 'Target', 'Location', 'best');
grid on; view(-35,20);

subplot(2,2,2);
plot(Z_energy(:,1), Z_energy(:,3), 'g:', 'LineWidth', 1.5); hold on;
plot(Z_fuel(:,1), Z_fuel(:,3), 'k--', 'LineWidth', 1.2); 
plot(res_plain.X_true(1,:), res_plain.X_true(3,:), 'LineWidth', 1.5);
plot(res_mtf.X_true(1,:),   res_mtf.X_true(3,:),   'LineWidth', 1.5);
plot(res_adapt.X_true(1,:), res_adapt.X_true(3,:), 'LineWidth', 2.0);
plot(r0(1), r0(3), 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 8);
plot(0,0, 'kx', 'LineWidth', 2, 'MarkerSize', 10);
xlabel('x [km]'); ylabel('z [km]');
title('X-Z Plane');
legend('Energy OL', 'Fuel OL', 'Plain KF', 'MTF Fixed \epsilon', 'MTF Adaptive \epsilon', 'Start', 'Target', 'Location', 'best');
grid on;

subplot(2,2,3);
plot(t_vec, res_plain.pos_err, 'LineWidth', 1.5); hold on;
plot(t_vec, res_mtf.pos_err,   'LineWidth', 1.5);
plot(t_vec, res_adapt.pos_err, 'LineWidth', 2.0);
xline(250, 'r--'); xline(325, 'r--'); xline(400, 'g--');
xlabel('Time [s]'); ylabel('||r|| [km]');
title('Position Error to Target');
legend('Plain KF', 'MTF Fixed \epsilon', 'MTF Adaptive \epsilon', 'Location', 'best');
grid on;

subplot(2,2,4);
plot(t_vec, res_plain.est_err, 'LineWidth', 1.5); hold on;
plot(t_vec, res_mtf.est_err,   'LineWidth', 1.5);
plot(t_vec, res_adapt.est_err, 'LineWidth', 2.0);
xline(250, 'r--'); xline(325, 'r--'); xline(400, 'g--');
xlabel('Time [s]'); ylabel('||x_{true}-x_{hat}||');
title('State Estimation Error');
legend('Plain KF', 'MTF Fixed \epsilon', 'MTF Adaptive \epsilon', 'Location', 'best');
grid on;

%% 9. LOCAL FUNCTIONS

function res = simulate_case(cfg, A, B, Phi, Gamma, H, R_nominal, Q_proc, P0, ...
                             x0, xf, u_max, lambda0_fuel, lambda0_energy, ...
                             t_vec, dt, options_ode, options_fsolve, ...
                             noise_seq, R_scale_hist, bias_hist, ...
                             resolve_period, resolve_min_rem)
    N_steps = length(t_vec);
    I6 = eye(6);
    
    % Histories
    X_true = zeros(6, N_steps);  X_true(:,1) = x0;
    X_hat  = zeros(6, N_steps);  X_hat(:,1)  = x0;
    U_hist = zeros(3, N_steps);
    NIS_history       = zeros(1, N_steps);
    MTF_ratio_history = zeros(1, N_steps);
    Score_history     = zeros(1, N_steps);
    Eps_history       = zeros(1, N_steps);
    pos_err = zeros(1, N_steps);
    est_err = zeros(1, N_steps);
    
    % INITIALIZE TIMING ARRAYS OUTSIDE THE LOOP
    solve_times = [];
    solve_success = [];
    
    % Initial estimator state
    x_true = x0;
    x_hat  = x0;
    P      = P0;
    
    % Initial costate / epsilon
    if cfg.use_adaptive_eps
        eps_current = cfg.init_eps;
        [lambda_current, ~, exitflag] = fsolve( ...
            @(l) residualfun_fuel(l, x0, xf, A, B, u_max, t_vec(end), eps_current, options_ode), ...
            lambda0_energy, options_fsolve);
        if exitflag <= 0
            warning('Initial adaptive solve at eps_max did not fully converge. Falling back to lambda0_energy.');
            lambda_current = lambda0_energy;
        end
    else
        eps_current = cfg.fixed_eps;
        lambda_current = lambda0_fuel;
    end
    Eps_history(1) = eps_current;
    
    for k = 1:N_steps-1
        t_current = t_vec(k);
        t_rem = t_vec(end) - t_current;
        
        % Current control
        [u_cmd, ~, ~] = control_from_costate(lambda_current(4:6), u_max, eps_current);
        U_hist(:,k) = u_cmd;
        
        % Truth propagation
        [~, x_next_true] = ode45(@(t,x) A*x + B*u_cmd, [0, dt], x_true, options_ode);
        x_true = x_next_true(end,:)';
        X_true(:,k+1) = x_true;
        
        % Common measurement realization
        R_true = R_scale_hist(k) * R_nominal;
        L_true = chol(R_true, 'lower');
        y_meas = H*x_true + bias_hist(:,k) + L_true*noise_seq(:,k);
        
        % KF prediction
        x_hat_minus = Phi*x_hat + Gamma*u_cmd;
        P_minus     = Phi*P*Phi' + Q_proc;
        
        % Innovation
        nu    = y_meas - H*x_hat_minus;
        S_cov = H*P_minus*H' + R_nominal;
        s_raw_nis = nu' * (S_cov \ nu);
        
        % MTF term
        S_raw = nu*nu' - S_cov;
        S_MTF = diag(max(diag(S_raw), 0));
        mtf_ratio = trace(S_MTF) / max(trace(S_cov), cfg.reg_eps);
        
        % Measurement-side covariance used in update
        if cfg.use_mtf
            R_eff = R_nominal + S_MTF;
            % BUG FIXED: Do NOT set mtf_ratio to 0 here!
        else
            R_eff = R_nominal;
            S_MTF = zeros(3,3);
            mtf_ratio = 0;
        end
        
        S_eff = H*P_minus*H' + R_eff;
        K = P_minus * H' / (S_eff + cfg.reg_eps*eye(3));
        
        % KF update (Joseph form)
        x_hat = x_hat_minus + K*nu;
        P = (I6 - K*H)*P_minus*(I6 - K*H)' + K*R_eff*K';
        X_hat(:,k+1) = x_hat;
        
        % Composite adaptation score
        score_inst = max(s_raw_nis - cfg.m_meas, 0) + cfg.k_mtf * mtf_ratio;
        score_inst = min(score_inst, cfg.score_cap);
        if k == 1
            score_filt = score_inst;
        else
            score_filt = cfg.rho_score*Score_history(k) + (1 - cfg.rho_score)*score_inst;
        end
        NIS_history(k+1)       = s_raw_nis;
        MTF_ratio_history(k+1) = mtf_ratio;
        Score_history(k+1)     = score_filt;
        
        % Epsilon scheduling
        if cfg.use_adaptive_eps
            eps_target = cfg.eps_min + (cfg.eps_max - cfg.eps_min) * (1 - exp(-cfg.beta * score_filt));
            eps_current = (1 - cfg.eps_blend)*eps_current + cfg.eps_blend*eps_target;
        else
            eps_current = cfg.fixed_eps;
        end
        Eps_history(k+1) = eps_current;
        
        % Receding-horizon re-solve
        if mod(t_current, resolve_period) == 0 && t_rem > resolve_min_rem
            eps_for_solve = eps_current;
            
            tic;
            [lambda_new, fval, exitflag] = fsolve( ...
                @(l) residualfun_fuel(l, x_hat, xf, A, B, u_max, t_rem, eps_for_solve, options_ode), ...
                lambda_current, options_fsolve);
            t_solve = toc;
            
            solve_times(end+1) = t_solve;
            solve_success(end+1) = (exitflag > 0 && norm(fval) < 1e-4);
            
            if exitflag > 0 && norm(fval) < 1e-4
                lambda_current = lambda_new;
            end
        end
        
        % Costate propagation over one step
        [~, lam_next] = ode45(@(t,l) -A'*l, [0, dt], lambda_current, options_ode);
        lambda_current = lam_next(end,:)';
        pos_err(k+1) = norm(x_true(1:3) - xf(1:3));
        est_err(k+1) = norm(x_true - x_hat);
    end
    
    % Package results
    res.name              = cfg.name;
    res.X_true            = X_true;
    res.X_hat             = X_hat;
    res.U_hist            = U_hist;
    res.u_mag             = sqrt(sum(U_hist.^2, 1));
    res.NIS_history       = NIS_history;
    res.MTF_ratio_history = MTF_ratio_history;
    res.Score_history     = Score_history;
    res.Eps_history       = Eps_history;
    res.pos_err           = pos_err;
    res.est_err           = est_err;
    res.dV                = trapz(t_vec, res.u_mag);
    res.miss_distance     = norm(X_true(1:3,end) - xf(1:3));
    res.solve_times       = solve_times;
    res.solve_success     = solve_success;
end

function print_case_metrics(res, dV_fuel)
    fuel_penalty = 100*(res.dV - dV_fuel)/max(dV_fuel, 1e-15);
    fprintf('%-30s : dV = %.6e km/s | miss = %.6e km | fuel penalty = %+7.3f %%\n', ...
        res.name, res.dV, res.miss_distance, fuel_penalty);
end

function err = residualfun_fuel(lambda0_guess, x0, xf, A, B, u_max, ToF, eps, options_ode)
    z0 = [x0; lambda0_guess];
    [~, Z] = ode45(@(t,z) odefun_fuel(t, z, A, B, u_max, eps), [0, ToF], z0, options_ode);
    x_final = Z(end,1:6)';
    err = x_final - xf;
end

function z_dot = odefun_energy(~, z, A_aug)
    z_dot = A_aug * z;
end

function z_dot = odefun_fuel(~, z, A, B, u_max, eps)
    x      = z(1:6);
    lambda = z(7:12);

    [u_cmd, ~, ~] = control_from_costate(lambda(4:6), u_max, eps);

    x_dot      = A*x + B*u_cmd;
    lambda_dot = -A'*lambda;

    z_dot = [x_dot; lambda_dot];
end

function [u_cmd, u_mag, alpha] = control_from_costate(lambda_v, u_max, eps)
    norm_lv = norm(lambda_v);
    S = 1 - norm_lv;

    if norm_lv > 1e-12
        alpha = -lambda_v / norm_lv;
    else
        alpha = zeros(3,1);
    end

    if eps <= 1e-12
        % Exact bang-bang limit
        if S > 0
            u_mag = 0;
        else
            u_mag = u_max;
        end
    else
        % Smoothed homotopy law
        if S > eps
            u_mag = 0;
        elseif S < -eps
            u_mag = u_max;
        else
            u_mag = (eps - S) * u_max / (2*eps);
        end
    end

    u_cmd = u_mag * alpha;
end