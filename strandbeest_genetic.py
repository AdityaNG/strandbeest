import numpy as np
from dataclasses import dataclass
import random

@dataclass
class LegConfiguration:
    # The 11 lengths that define a Strandbeest leg mechanism
    lengths: list[float]
    
    def compute_foot_position(self, crank_angle):
        """Compute the (x,y) position of the foot given a crank angle"""
        # Define fixed pivot points (relative to crank center)
        fixed_pivot_x = 0
        fixed_pivot_y = 38.0  # From the diagram
        
        # Crank point (point a in the diagram)
        crank_x = self.lengths[0] * np.cos(crank_angle)
        crank_y = self.lengths[0] * np.sin(crank_angle)
        
        try:
            # This is a simplified version of the full linkage calculation
            # Following the triangle formations shown in the diagram
            
            # Triangle formed by the first three bars
            triangle1_angle = np.arctan2(crank_y - fixed_pivot_y, crank_x - fixed_pivot_x)
            triangle1_base = np.sqrt((crank_x - fixed_pivot_x)**2 + (crank_y - fixed_pivot_y)**2)
            
            # Use cosine law to find angles in the triangles
            cos_alpha = (self.lengths[1]**2 + triangle1_base**2 - self.lengths[2]**2) / (2 * self.lengths[1] * triangle1_base)
            cos_alpha = np.clip(cos_alpha, -1, 1)  # Prevent domain errors
            alpha = np.arccos(cos_alpha)
            
            # Calculate position of the connecting point
            connector_angle = triangle1_angle + alpha
            connector_x = fixed_pivot_x + self.lengths[1] * np.cos(connector_angle)
            connector_y = fixed_pivot_y + self.lengths[1] * np.sin(connector_angle)
            
            # Finally calculate foot position using the last bar
            # This is greatly simplified - the real mechanism has more intermediate points
            foot_angle = connector_angle + np.pi/4  # Approximate angle based on diagram
            foot_x = connector_x + self.lengths[10] * np.cos(foot_angle)
            foot_y = connector_y + self.lengths[10] * np.sin(foot_angle)
            
            return foot_x, foot_y
            
        except (ValueError, RuntimeWarning):
            # Return a highly penalized position if the geometry is impossible
            return 1000, 1000  # This will result in a very poor fitness score

    def calculate_path_points(self, num_points=100):
        """Calculate the path traced by the foot point over one full rotation"""
        points = []
        angles = np.linspace(0, 2*np.pi, num_points)
        
        for angle in angles:
            x, y = self.compute_foot_position(angle)
            # Only add point if it's valid (not our error case)
            if x != 1000 and y != 1000:
                points.append((x, y))
        
        return points if points else [(0, 0)]  # Ensure we always return at least one point

    def evaluate_ground_flatness(self, points):
        """Evaluate how flat the ground-contact portion of the path is"""
        # Find lowest points (ground contact phase)
        y_coords = [p[1] for p in points]
        min_y = min(y_coords)
        ground_points = [p for p in points if p[1] < min_y + 5]  # 5mm threshold
        
        if not ground_points:
            return 0
        
        # Calculate variance of y-coordinates during ground contact
        y_variance = np.var([p[1] for p in ground_points])
        
        # Convert variance to score (lower variance = higher score)
        return 1.0 / (1.0 + y_variance)

    def evaluate_step_height(self, points):
        """Evaluate if the foot lifts to appropriate height between steps"""
        y_coords = [p[1] for p in points]
        step_height = max(y_coords) - min(y_coords)
        
        # Optimal step height is between 15-25% of mechanism size
        optimal_min = 15
        optimal_max = 25
        
        if step_height < optimal_min:
            return step_height / optimal_min
        elif step_height > optimal_max:
            return optimal_max / step_height
        else:
            return 1.0

    def evaluate_complexity(self):
        """Evaluate the mechanical complexity based on bar lengths"""
        scores = []
        
        # Check each bar length is within reasonable bounds
        for length in self.lengths:
            if length < 10:  # Too short
                scores.append(length / 10)
            elif length > 100:  # Too long
                scores.append(100 / length)
            else:
                scores.append(1.0)
        
        # Check ratios between consecutive bars
        for i in range(len(self.lengths) - 1):
            ratio = self.lengths[i] / self.lengths[i + 1]
            if ratio < 0.2 or ratio > 5:  # Extreme ratios are penalized
                scores.append(0.5)
            else:
                scores.append(1.0)
        
        return sum(scores) / len(scores)
    
    def fitness_score(self):
        """Calculate how well this leg configuration performs"""
        path_points = self.calculate_path_points()
        
        # Criteria for scoring:
        ground_flatness = self.evaluate_ground_flatness(path_points)
        step_height = self.evaluate_step_height(path_points)
        mechanism_complexity = self.evaluate_complexity()
        
        # Weighted combination of scores
        return (0.5 * ground_flatness + 
                0.3 * step_height + 
                0.2 * mechanism_complexity)


class StrandbeestOptimizer:
    def __init__(self, population_size=100, mutation_rate=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()
    
    def initialize_population(self):
        """Create initial population with random leg configurations"""
        population = []
        for _ in range(self.population_size):
            # Generate random lengths within reasonable bounds
            lengths = [random.uniform(10, 100) for _ in range(11)]
            population.append(LegConfiguration(lengths))
        return population
    
    def select_parents(self):
        """Select parents using tournament selection"""
        tournament_size = 5
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness_score())
    
    def crossover(self, parent1, parent2):
        """Create child configuration by combining parents' lengths"""
        crossover_point = random.randint(0, len(parent1.lengths))
        child_lengths = (parent1.lengths[:crossover_point] + 
                        parent2.lengths[crossover_point:])
        return LegConfiguration(child_lengths)
    
    def mutate(self, leg_config):
        """Randomly modify some lengths to maintain diversity"""
        mutated_lengths = leg_config.lengths.copy()
        for i in range(len(mutated_lengths)):
            if random.random() < self.mutation_rate:
                # Add or subtract up to 10% of current length
                delta = random.uniform(-0.1, 0.1) * mutated_lengths[i]
                mutated_lengths[i] += delta
        return LegConfiguration(mutated_lengths)
    
    def evolve(self, generations=1000):
        """Run the genetic algorithm for specified generations"""
        for generation in range(generations):
            new_population = []
            
            # Keep the best solutions (elitism)
            elite_count = self.population_size // 10
            sorted_pop = sorted(self.population, 
                              key=lambda x: x.fitness_score(),
                              reverse=True)
            new_population.extend(sorted_pop[:elite_count])
            
            # Create rest of new population
            while len(new_population) < self.population_size:
                parent1 = self.select_parents()
                parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            self.population = new_population
            
            if generation % 100 == 0:
                best = max(self.population, key=lambda x: x.fitness_score())
                print(f"Generation {generation}: Best fitness = {best.fitness_score()}")