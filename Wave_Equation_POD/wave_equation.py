import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Define the domain
Lx, Ly = 10, 10
dx, dy = 0.1, 0.1
nx, ny = int(Lx / dx), int(Ly / dy)
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
print(x[0])

# Parameters
T = 100
CFL = 0.5
c = 1
dt = (CFL * dx / c)

# Initialize field variables
current = np.zeros((nx, ny))
print(f"The shape of the current flow is {current.shape}")
previous = current.copy()
next = current.copy()
t = 0

# Store snapshots for POD
snapshots = []

# Set up the figure and axes
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

# Set spacing to make subplots evenly spaced
fig.subplots_adjust(hspace=0.3)  # Adjust space between plots

ax1 = axes[0]  # First subplot
ax2 = fig.add_subplot(2, 1, 2, projection='3d', anchor='C')  # 3D plot centered

# Initialize plots
im = ax1.imshow(current.T, extent=[0, Lx, 0, Ly], origin='lower', aspect='auto', cmap='jet')
cbar1 = plt.colorbar(im, ax=ax1)
ax1.set_title(f't = {t:.2f}')
im.set_clim(-0.02, 0.02)

X, Y = np.meshgrid(x, y)
surf = ax2.plot_surface(X, Y, current.T, cmap='jet')
cbar2 = fig.colorbar(surf, ax=ax2, shrink=0.5)  # Adjust colorbar size
ax2.set_zlim(-0.05, 0.05)

# Adjust layout for better alignment
fig.tight_layout()

# Update function for animation
def update(frame):
    global current, previous, next, t, snapshots
    
    # Apply boundary conditions
    current[:, [0, -1]] = 0  # Left and right boundaries
    current[[0, -1], :] = 0  # Top and bottom boundaries


    # Finite difference update
    for i in range(1, nx-1):
     for j in range(1, ny-1):
        next[i, j] = (2 * current[i, j] - previous[i, j] +
                      CFL**2 * (current[i+1, j] + current[i, j+1] - 4 * current[i, j] +
                                current[i-1, j] + current[i, j-1]) +
                      dt**2 * 20 * np.sin(30 * np.pi * t / 20) * (i == 50 and j == 50))  # Add forcing term
   

    # Update field values
    previous, current = current.copy(), next.copy()

    print(next[:, [0, -1]] )  # Left and right boundaries
    print(next[[0, -1], :])
    t += dt

    # Store snapshots for POD (flatten each time step)
    snapshots.append(current.flatten())

    # Update plots
    im.set_array(current.T)  # Update image plot
    ax1.set_title(f'Full fluid flow at last time step = {t:.2f}')

    ax2.clear()
    ax2.plot_surface(X, Y, current.T, cmap='jet')  # Update 3D surface plot
    ax2.set_zlim(-0.05, 0.05)

# Run animation
ani = FuncAnimation(fig, update, frames=199, interval=50, repeat=False )  # 200 steps, 50ms interval
plt.show()

# ---- POD ANALYSIS ----
# Convert snapshots list to a NumPy array (each column is a time snapshot)
snapshots = np.array(snapshots).T  # Shape: (space, time)
print(f"The shape of snapshots after simulation is {snapshots.shape}")


# Compute Singular Value Decomposition (POD)
U, S, Vt = np.linalg.svd(snapshots, full_matrices=False)
V_plot= Vt.T
print(f"Vt shape is {Vt.shape}")
print(f"The shape of U is {U.shape}")
print(f"The shape of S is {S.shape}")

# Plot the first few POD temporal modes (V)
plt.figure(figsize=(8, 4))
plt.plot(Vt[0, :], label='1st POD mode')
plt.plot(Vt[1, :], label='2nd POD mode')
plt.plot(Vt[3, :], label='3rd POD mode')
plt.xlabel('Time Step')
plt.ylabel('Amplitude')
plt.title('Temporal POD Modes')
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(Vt[100, :], label='100th POD mode')
plt.plot(Vt[-50, :], label='150th POD mode')
plt.plot(Vt[-1, :], label='200th POD mode')
plt.xlabel('Time Step')
plt.ylabel('Amplitude')
plt.title('Temporal POD Modes')
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(Vt[0, :], label='1st POD mode')
plt.plot(Vt[1, :], label='2nd POD mode')
plt.plot(Vt[2, :], label='2nd POD mode')
plt.plot(Vt[100, :], label='100th POD mode')
plt.plot(Vt[-50, :], label='150th POD mode')
plt.plot(Vt[-1, :], label='200th POD mode')
plt.xlabel('Time Step')
plt.ylabel('Amplitude')
plt.title('Temporal POD Modes')
plt.legend()
plt.show()

plt.figure(figsize=(6,3))
plt.figure(1)
plt.semilogy(S) 
plt.title("Singular Values")
plt.xlabel("Singular values j in descending order")
plt.ylabel("Energy (Singular values)")
plt.show()

# Compute cumulative energy to find k_optimal
energy_contributions = np.cumsum(S) / np.sum(S)  
k_optimal = np.argmax(energy_contributions >= 0.9) + 1  # First k that reaches 90%

print(f"Number of modes needed to capture 90% of the energy: {k_optimal}")

# Extract first k_optimal singular values and vectors
sigma_k_optimal = S[:k_optimal]
u_k_optimal = U[:, :k_optimal]
v_k_optimal = Vt[:k_optimal, :]

# Reconstruct the fluid flow using k_optimal modes
k_optimal_full_matrix = u_k_optimal @ np.diag(sigma_k_optimal) @ v_k_optimal  # Rank-21 approximation


error = np.linalg.norm(snapshots[:, -1] - k_optimal_full_matrix[:, -1]) / np.linalg.norm(snapshots[:, -1])
print(f"Relative error at the last time step: {error:.4f}")


k_optimal_lastsnapshot = k_optimal_full_matrix[:, 199].reshape(100, 100)  # Taking first time step for comparison

# Plot the reconstructed structure
plt.figure(figsize=(8, 6))
im=plt.imshow(k_optimal_lastsnapshot, cmap='jet', origin='lower')
im.set_clim(-0.02, 0.02)
plt.colorbar(label="Mode Amplitude")
plt.xlabel("X-axis (grid points)")
plt.ylabel("Y-axis (grid points)")
plt.title(f"Rank k ={k_optimal} Approximation of the Flow at last time Step")
plt.show()


diff_matrix = snapshots[:, -1] - k_optimal_full_matrix[:, -1]  # Difference at last time step
im2=plt.imshow(diff_matrix.reshape(100, 100), cmap='coolwarm', origin='lower')
im2.set_clim(-0.02, 0.02)
plt.colorbar(label="Difference Magnitude")
plt.title("Difference Between Full Flow and Rank-k = 21 Approximation")
plt.show()



