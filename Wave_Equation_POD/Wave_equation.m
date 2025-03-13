% Define the domain

% Space parameters
Lx = 10;   % Length in the x-direction
Ly = 10;   % Length in the y-direction
dx = 0.1;  % Grid spacing in the x-direction
dy = 0.1;  % Grid spacing in the y-direction

% Compute the number of grid points
nx = fix(Lx / dx);  % Number of points in the x-direction
ny = fix(Ly / dy);  % Number of points in the y-direction

% Generate spatial grid
x = linspace(0, Lx, nx);  % x-coordinates
y = linspace(0, Ly, ny);  % y-coordinates

% Time parameter
T = 10;  % Total simulation time

%% Field Variable Definition

% Initialize field variables
wn = zeros(nx, ny);  % Field variable at current time step (n)
wnm1 = wn;           % Field variable at previous time step (n-1)
wnp1 = wn;           % Field variable at next time step (n+1)

%% Parameters

CFL = 0.5;         % Courant-Friedrichs-Lewy (CFL) number, CFL = c * dt / dx
c = 1;             % Wave speed (assumed to be 1)
dt = CFL * dx / c; % Time step size based on CFL condition

t = 0;

while (t < T)

    % Reflecting Boundary Conditions
    wn(:, [1 end]) = 0;
    wn([1 end], :) = 0;

    % Solution update
    t = t + dt;



% Update time step
t = t + dt;

% Save current and previous arrays
wnm1 = wn;  
wn = wnp1;



% Update field variable using finite difference scheme
for i = 2:nx-1
    for j = 2:ny-1
        wnp1(i,j) = 2 * wn(i,j) - wnm1(i,j) ...
                  + CFL^2 * ( wn(i+1,j) + wn(i,j+1) - 4 * wn(i,j) + wn(i-1,j) + wn(i,j-1) );
        % Apply source term at the midpoint (50,50)
        if i == 50 && j == 50
            wnp1(i,j) = wnp1(i,j) + dt^2 * 20 * sin(30 * pi * t / 20);
        end
    end
end

% Visualize at selected steps
% Visualize at selected steps
clf;
subplot(2,1,1);
imagesc(x, y, wn'); colorbar; caxis([-0.02 0.02]);
title(sprintf('t = %.2f', t));
subplot(2,1,2);
mesh(x, y, wn'); colorbar; caxis([-0.02 0.02]);
axis([0 Lx 0 Ly -0.05 0.05]);
shg; pause(0.01);
end

