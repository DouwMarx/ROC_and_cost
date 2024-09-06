# This will show ROC curve in x-y space and the corresponding cost in the z-axis.
# This is based on the cost = fpr*cost_fpr + tpr*cost_tpr + tnr*cost_tnr + fnr*cost_fnr
# Then, assuming that cost_tpr =0 and cost tnr = 0, the cost = fpr*cost_fpr + fnr*cost_fnr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import cm

# Make a simulated ROC curve using a parameriic function

# name = "vary_ratio"
name = "vary_performance"

class ROC():
    def __init__(self, m, n,cost_fnr=0.3):
        self.m = m
        self.n = n
        self.cost_fnr = cost_fnr
        self.cost_fpr = 1 - cost_fnr

    def generate_roc_with_cost(self,n_points=1000):
        t = np.linspace(0,1,n_points)
        fpr = -(-t + 1)**self.m + 1
        tpr = (t)**self.n

        cost = fpr*self.cost_fpr + (1-tpr)*self.cost_fnr
        return fpr, tpr, cost

    def generate_cost_surface(self,n_points=1000):
        # Plot the cost curve
        n_grid = 100

        # Make a grid of FPR and TPR values
        fpr_grid = np.linspace(0,1,n_grid)
        tpr_grid = np.linspace(0,1,n_grid)
        fpr_grid, tpr_grid = np.meshgrid(fpr_grid, tpr_grid)

        # Calculate the cost
        cost_grid = fpr_grid*self.cost_fpr + (1-tpr_grid)*self.cost_fnr

        return fpr_grid, tpr_grid, cost_grid

# Plot the ROC curve


# Show a 2D version of the ROC curve to demonstrate what it is doing
roca = ROC(0.05,1)
rocb = ROC(1,0.05)

plt.figure()

for roc_obj in [roca, rocb]:
    x_fpr, y_tpr, z_cost = roc_obj.generate_roc_with_cost()

    roc_auc = np.trapz(y_tpr, x_fpr)
    print(f"ROC AUC: {roc_auc}")

    plt.plot(x_fpr, y_tpr, label='ROC curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
plt.title('ROC curves with the same AUC')

# Save the ROC curve as pdf
plt.savefig('roc_curve_2d.pdf')
# plt.show()


fig = plt.figure()
# Show the ROC curve in the x-y plane
ax = fig.add_subplot(111, projection='3d')


# Global counter
count = 0

def animate(i):
    n_view_sweeps = 1

    # Set the axis limits
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_zlim([0,1])

    ax.view_init(elev=10 + np.sin(n_view_sweeps*i/100*np.pi)*3 , azim=np.sin(i/100*np.pi)*3 - 90 + 6)
    ax.clear()


    if name== "vary_performance":
        m = 0.5*(1 + np.sin(3*i/100*np.pi + 0.342*np.pi))*0.9 + 0.05
        n = 0.5*(1 + np.sin(-2*i/100*np.pi))*0.9 + 0.05
        roc = ROC(m, n, cost_fnr=0.5)
    elif name == "vary_ratio":
        n_cost_sweeps = 2
        roc = ROC(1,0.3, cost_fnr=0.2 + 0.6*(0.5+0.5*np.sin(n_cost_sweeps*i/100*np.pi)))
    else:
        raise ValueError("Invalid name")

    x_fpr, y_tpr, z_cost = roc.generate_roc_with_cost()
    # Get the optimal point
    optimal_cost_index = np.argmin(z_cost)
    optimal_cost = z_cost[optimal_cost_index]
    optimal_fpr = x_fpr[optimal_cost_index]
    optimal_tpr = y_tpr[optimal_cost_index]

    # Plot the Roc curve in 3D, with the roc curve in the x-y plane

    # Plot the trace in 3D
    ax.plot(x_fpr, y_tpr, np.zeros_like(x_fpr), label='ROC curve', color='blue')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_zlabel('Cost')

    # Show the random guessing line
    ax.plot([0,1],[0,1],[0,0], color='black', linestyle='--', label='Random guessing')

    # Plot a vertical line from the ROC curve to the for the optimal point
    ax.plot([optimal_fpr,optimal_fpr],[optimal_tpr,optimal_tpr],[0,optimal_cost],color='red')
    ax.scatter(optimal_fpr,optimal_tpr,optimal_cost, color='red', label='Optimal cost')
    # Show the optimal point as a marker

    # Show the associated cost in the z-axis
    ax.plot(x_fpr, y_tpr, z_cost, label='Cost', color='blue', alpha=0.5)

    fpr_grid, tpr_grid, cost_grid = roc.generate_cost_surface()

    # norm = plt.Normalize(z_cost.min(), z_cost.max())
    # colors = cm.viridis(norm(z_cost))
    # rcount, ccount, _ = colors.shape

    # Plot the cost surface as a grid with no fill
    # ax.plot_surface(fpr_grid, tpr_grid, cost_grid, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
    # ax.plot_surface(fpr_grid, tpr_grid, cost_grid, alpha=0.1, cmap='viridis', rstride=10, cstride=10)
    ax.plot_wireframe(fpr_grid, tpr_grid, cost_grid, alpha=0.3,rstride=10, cstride=10, color='black')
    # plt.show()

    # print(i)
    # plt.savefig('./' + name + '/roc_cost' + name +  '-' + str(count) + '.png', dpi=300)


ani = FuncAnimation(fig, animate, frames=np.linspace(0, 100, 30), repeat=True)

ani.save('roc_cost' + name + '.gif', writer='imagemagick', fps=5, dpi=300)
