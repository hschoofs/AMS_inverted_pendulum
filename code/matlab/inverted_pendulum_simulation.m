%% Inverted Pendulum Simulation with LQR Control and Animation

clc; clear; close all;

% Physical parameters
M = 0.5;      % Mass of the cart (kg)
m = 0.2;      % Mass of the pendulum (kg)
l = 0.3;      % Length to pendulum center of mass (m)
g = 9.81;     % Gravity (m/s^2)

% Linearized state-space matrices
A = [0 1 0 0;
     0 0 (m*g)/M 0;
     0 0 0 1;
     0 0 (M+m)*g/(M*l) 0];

B = [0;
     1/M;
     0;
     1/(M*l)];

% LQR design: cost matrices
Q = diag([10 1 100 1]);  % Heavily penalize theta
R = 0.001;
K = lqr(A, B, Q, R);

% Closed-loop dynamics
Acl = A - B * K;

% Simulation time
tspan = [0 10];
x0 = [0.0; 0.0; 0.2; 0.0];  % Initial state (0.2 rad ≈ 11.5 deg)

% Solve the ODE
[t, X] = ode45(@(t,x) Acl * x, tspan, x0);

% Extract states
x = X(:,1);      % Cart position
theta = X(:,3);  % Pendulum angle

%% Plot the states
figure;
subplot(2,2,1); plot(t, x); title('Cart Position'); ylabel('x (m)'); grid on;
subplot(2,2,2); plot(t, X(:,2)); title('Cart Velocity'); ylabel('dx (m/s)'); grid on;
subplot(2,2,3); plot(t, theta); title('Pendulum Angle'); ylabel('\theta (rad)'); grid on;
subplot(2,2,4); plot(t, X(:,4)); title('Pendulum Angular Velocity'); ylabel('d\theta (rad/s)'); grid on;
xlabel('Time (s)');
sgtitle('Inverted Pendulum States with LQR Control');

%% Animation of Cart and Pendulum
figure;
axis equal;
axis([-1.5 1.5 -0.5 1.2]);
grid on;
title('Inverted Pendulum Animation');
xlabel('X (m)');
ylabel('Y (m)');

cart_width = 0.3;
cart_height = 0.2;
pendulum_length = l;

for i = 1:5:length(t)
    clf;
    hold on;
    % Draw track
    plot([-2 2], [0 0], 'k', 'LineWidth', 2);
    
    % Cart position
    xc = x(i);
    cart_x = [xc - cart_width/2, xc + cart_width/2, xc + cart_width/2, xc - cart_width/2];
    cart_y = [0, 0, cart_height, cart_height];
    fill(cart_x, cart_y, [0.2 0.6 1]);
    
    % Pendulum position
    pend_x = xc + pendulum_length * sin(theta(i));
    pend_y = cart_height + pendulum_length * cos(theta(i));
    
    % Draw pendulum
    plot([xc, pend_x], [cart_height, pend_y], 'r-', 'LineWidth', 3);
    plot(pend_x, pend_y, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    
    % Frame settings
    axis([-1.5 1.5 -0.5 1.2]);
    title(sprintf('Time: %.2f s', t(i)));
    drawnow;
end
