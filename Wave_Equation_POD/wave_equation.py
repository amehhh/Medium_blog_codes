import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# -----------------------
# Define the simulation domain
# -----------------------
Lx, Ly = 10, 10                   # Domain dimensions
dx, dy = 0.1, 0.1                 # Grid spacing
nx, ny = int(Lx / dx), int(Ly / dy)  # Number of grid points
x = np.linspace(0, Lx, nx)       # X-axis grid
y = np.linspace(0, Ly, ny)       # Y-axis grid
print(x[0])                      # Print first x value (for check)

# -----------------------
# Set simulation parameters
# -----------------------
T = 100                          # Total simulation time
CFL = 0.5                        # Courant number
c = 1                            # Wave speed
dt = (CFL * dx / c)              # Time step based on CFL

# -----------------------
# Initialize field variables
# -----------------------
current = np.zeros((nx, ny))     # Current time step values
print(f"The shape of the current flow is {current.shape}")
previous = current.copy()        # Previous time step values
next = current.copy()            # Next time step values
t = 0                            # Initialize time

# -----------------------
# Store snapshots for POD
# -----------------------
snapshots = []

# -----------------------
# Set up the figure and axes for plotting
# -----------------------
fig, axes = plt.subplots(2, 1, figsize=(8, 8))           # Create 2 subplots
fig.subplots_adjust(hspace=0.3)                          # Adjust vertical spacing

ax1 = axes[0]                                            # Top subplot for 2D view
ax2 = fig.add_subplot(2, 1, 2, projection='3d', anchor='C')  # Bottom subplot for 3D view

# Initialize 2D and 3D plots
im = ax1.imshow(current.T, extent=[0, Lx, 0, Ly], origin='lower', aspect='auto', cmap='jet')
cbar1 = plt.colorbar(im, ax=ax1)
ax1.set_title(f't = {t:.2f}')
im.set_clim(-0.02, 0.02)

X, Y = np.meshgrid(x, y)                                 # Grid for surface plot
surf = ax2.plot_surface(X, Y, current.T, cmap='jet')
cbar2 = fig.colorbar(surf, ax=ax2, shrink=0.5)
ax2.set_zlim(-0.05, 0.05)

fig.tight_layout()  # Adjust overall layout

# -----------------------
# Animation update function
# -----------------------
def update(frame):
    global current, previous, next, t, snapshots

    # Apply boundary conditions (edges set to zero)
    current[:, [0, -1]] = 0
    current[[0, -1], :] = 0

    # Finite difference update (with source at center)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            next[i, j] = (2 * current[i, j] - previous[i, j] +
                          CFL**2 * (current[i+1, j] + current[i, j+1] - 4 * current[i, j] +
                                    current[i-1, j] + current[i, j-1]) +
                          dt**2 * 20 * np.sin(30 * np.pi * t / 20) * (i == 50 and j == 50))

    # Update field values
    previous, current = current.copy(), next.copy()

    print(next[:, [0, -1]])  # Print left and right boundaries
    print(next[[0, -1], :])  # Print top and bottom boundaries

    t += dt  # Increment time

    # Save snapshot for POD
    snapshots.append(current.flatten())

    # Update 2D heatmap
    im.set_array(current.T)
    ax1.set_title(f'Full fluid flow at last time step = {t:.2f}')

    # Update 3D surface plot
    ax2.clear()
    ax2.plot_surface(X, Y, current.T, cmap='jet')
    ax2.set_zlim(-0.05, 0.05)

# -----------------------
# Run the animation
# -----------------------
ani = FuncAnimation(fig, update, frames=199, interval=50, repeat=False)
plt.show()

# -----------------------
# POD (Proper Orthogonal Decomposition) Analysis
# -----------------------

# Convert list of snapshots to NumPy array (columns = time steps)
snapshots = np.array(snapshots).T
print(f"The shape of snapshots after simulation is {snapshots.shape}")

# Compute SVD (Singular Value Decomposition)
U, S, Vt = np.linalg.svd(snapshots, full_matrices=False)
V_plot = Vt.T
print(f"Vt shape is {Vt.shape}")
print(f"The shape of U is {U.shape}")
print(f"The shape of S is {S.shape}")

# -----------------------
# Plot the first few temporal POD modes
# -----------------------
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

# -----------------------
# Plot singular values (energy of each mode)
# -----------------------
plt.figure(figsize=(6, 3))
plt.figure(1)
plt.semilogy(S)
plt.title("Singular Values")
plt.xlabel("Singular values j in descending order")
plt.ylabel("Energy (Singular values)")
plt.show()

# -----------------------
# Determine optimal number of modes for 90% energy
# -----------------------
energy_contributions = np.cumsum(S) / np.sum(S)
k_optimal = np.argmax(energy_contributions >= 0.9) + 1
print(f"Number of modes needed to capture 90% of the energy: {k_optimal}")

# Extract top-k components
sigma_k_optimal = S[:k_optimal]
u_k_optimal = U[:, :k_optimal]
v_k_optimal = Vt[:k_optimal, :]

# -----------------------
# Reconstruct the flow using k_optimal modes
# -----------------------
k_optimal_full_matrix = u_k_optimal @ np.diag(sigma_k_optimal) @ v_k_optimal

# Compute relative error for last time step
error = np.linalg.norm(snapshots[:, -1] - k_optimal_full_matrix[:, -1]) / np.linalg.norm(snapshots[:, -1])
print(f"Relative error at the last time step: {error:.4f}")

# -----------------------
# Plot reconstructed flow at last time step
# -----------------------
k_optimal_lastsnapshot = k_optimal_full_matrix[:, 199].reshape(100, 100)

plt.figure(figsize=(8, 6))
im = plt.imshow(k_optimal_lastsnapshot, cmap='jet', origin='lower')
im.set_clim(-0.02, 0.02)
plt.colorbar(label="Mode Amplitude")
plt.xlabel("X-axis (grid points)")
plt.ylabel("Y-axis (grid points)")
plt.title(f"Rank k = {k_optimal} Approximation of the Flow at Last Time Step")
plt.show()

# -----------------------
# Plot difference between full and approximated flow
# -----------------------
diff_matrix = snapshots[:, -1] - k_optimal_full_matrix[:, -1]
im2 = plt.imshow(diff_matrix.reshape(100, 100), cmap='coolwarm', origin='lower')
im2.set_clim(-0.02, 0.02)
plt.colorbar(label="Difference Magnitude")
plt.title("Difference Between Full Flow and Rank-k = 21 Approximation")
plt.show()

# -----------------------
# Plot first spatial POD mode
# -----------------------
first_spatial = U[:, 0]
im2 = plt.imshow(first_spatial.reshape(100, 100), cmap='coolwarm', origin='lower')
im2.set_clim(-0.02, 0.02)
plt.colorbar(label="Difference Magnitude")
plt.title('First Spatial Mode')
plt.show()
