"""
strandbeest
"""

import numpy as np
import matplotlib.pyplot as plt
from strandbeest_genetic import StrandbeestOptimizer, LegConfiguration

def plot_leg_path(leg_config, ax=None):
    """Plot the path traced by the foot point"""
    if ax is None:
        _, ax = plt.subplots()
    
    points = leg_config.calculate_path_points()
    x_coords, y_coords = zip(*points)
    
    ax.plot(x_coords, y_coords, 'b-', label='Foot Path')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('X Position (mm)')
    ax.set_ylabel('Y Position (mm)')
    ax.legend()

def run_optimization(generations=1000, population_size=100, plot_interval=100):
    """Run the optimization process and visualize results"""
    # Initialize optimizer
    optimizer = StrandbeestOptimizer(population_size=population_size)
    
    # Track best fitness over generations
    best_fitness_history = []
    avg_fitness_history = []
    
    # Create figure for visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for gen in range(generations):
        # Evolve population
        optimizer.evolve(1)
        
        # Track fitness metrics
        current_fitness = [leg.fitness_score() for leg in optimizer.population]
        best_fitness_history.append(max(current_fitness))
        avg_fitness_history.append(np.mean(current_fitness))
        
        # Periodically update visualization
        if gen % plot_interval == 0:
            print(f"Generation {gen}:")
            print(f"Best Fitness: {best_fitness_history[-1]:.4f}")
            print(f"Average Fitness: {avg_fitness_history[-1]:.4f}")
            
            # Clear and update plots
            ax1.clear()
            ax2.clear()
            
            # Plot fitness history
            ax1.plot(best_fitness_history, 'b-', label='Best Fitness')
            ax1.plot(avg_fitness_history, 'r-', label='Average Fitness')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness')
            ax1.set_title('Optimization Progress')
            ax1.legend()
            ax1.grid(True)
            
            # Plot best leg configuration
            best_leg = max(optimizer.population, 
                          key=lambda x: x.fitness_score())
            plot_leg_path(best_leg, ax2)
            ax2.set_title('Best Leg Path')
            
            plt.pause(0.01)
    
    # Get final best solution
    best_leg = max(optimizer.population, key=lambda x: x.fitness_score())
    
    print("\nOptimization Complete!")
    print("Final Best Configuration:")
    for i, length in enumerate(best_leg.lengths):
        print(f"Bar {i+1}: {length:.2f}mm")
    
    plt.show()
    return best_leg

def save_results(leg_config, filename="best_leg_config.txt"):
    """Save the optimized leg configuration to a file"""
    with open(filename, 'w') as f:
        f.write("Optimized Strandbeest Leg Configuration\n")
        f.write("=====================================\n\n")
        for i, length in enumerate(leg_config.lengths):
            f.write(f"Bar {i+1}: {length:.2f}mm\n")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(0)
    
    # Run optimization with visualization
    best_leg = run_optimization(
        generations=1000,
        population_size=100,
        plot_interval=10
    )
    
    # Save results
    save_results(best_leg)
