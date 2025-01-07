import numpy as np
import random
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ConwayGame:
    def __init__(self, toroidal=True, width=32, height=32, grid_size=1, order_mean=0.5, order_std=0.1):
        self.WIDTH, self.HEIGHT = width, height
        self.GRID_SIZE = grid_size
        self.GRID_WIDTH, self.GRID_HEIGHT = self.WIDTH // self.GRID_SIZE, self.HEIGHT // self.GRID_SIZE
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=bool)
        self.toroidal = toroidal
        self.entropies = []
        self.region_indices = []

    def randomize_grid_uniform(self, probability_of_one):
        """Randomize the grid with a uniform probability of ones."""
        self.grid = np.random.rand(self.GRID_WIDTH, self.GRID_HEIGHT) < probability_of_one

    def update_grid(self):
        """Update the grid based on Conway's Game of Life rules."""
        new_grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=bool)
        for i in range(self.GRID_WIDTH):
            for j in range(self.GRID_HEIGHT):
                neighbors = sum(
                    self.grid[(i + x) % self.GRID_WIDTH, (j + y) % self.GRID_HEIGHT]
                    for x in [-1, 0, 1] for y in [-1, 0, 1]
                ) - self.grid[i, j]
                if self.grid[i, j]:
                    new_grid[i, j] = neighbors in [2, 3]
                else:
                    new_grid[i, j] = neighbors == 3
        self.grid = new_grid

    def compute_number_of_ones_per_region(self, region_size=3):
        """
        Compute the number of ones in each region of the grid.

        Args:
            region_size (int): Size of the square region.

        Returns:
            list: List containing the number of ones in each region.
        """
        probs = []
        for i in range(self.GRID_WIDTH - region_size + 1):
            for j in range(self.GRID_HEIGHT - region_size + 1):
                region = self.grid[i:i + region_size, j:j + region_size]
                probs.append(np.sum(region) / (region_size ** 2))
        return probs

    def compute_heterogeneity(self, region_size=3):
        """
        Compute heterogeneity as the standard deviation of the number of ones per region.

        Args:
            region_size (int): Size of the square region.

        Returns:
            float: Standard deviation representing heterogeneity.
        """
        probs = self.compute_number_of_ones_per_region(region_size=region_size)
        if len(probs) == 0:
            return 0.0
        # Modified to ensure distinct treatment of all 0 and all 1 regions
        heterogeneity = np.std(probs)
        return heterogeneity


    def precompute_affected_regions(self, kernel_size, toroidal):
        """
        Precompute and store the regions affected by each cell.

        Args:
            kernel_size (int): Size of the kernel used for heterogeneity calculations.
            toroidal (bool): Whether to use toroidal boundaries.

        Returns:
            dict: A dictionary mapping each cell to a set of region indices that include it.
        """
        affected_regions_map = {}
        for i in range(self.GRID_WIDTH):
            for j in range(self.GRID_HEIGHT):
                affected_regions_map[(i, j)] = set()

        for region_idx, (i, j) in enumerate(self.region_indices):
            for x in range(kernel_size):
                for y in range(kernel_size):
                    cell_i = (i + x) % self.GRID_WIDTH if toroidal else i + x
                    cell_j = (j + y) % self.GRID_HEIGHT if toroidal else j + y
                    if 0 <= cell_i < self.GRID_WIDTH and 0 <= cell_j < self.GRID_HEIGHT:
                        affected_regions_map[(cell_i, cell_j)].add(region_idx)
        return affected_regions_map

    def compute_entropy(self, region):
        """
        Calculate the entropy of a given region of the grid.

        Args:
            region (np.ndarray): A region of the grid.

        Returns:
            float: Entropy value of the region.
        """
        prob = np.mean(region)
        if prob == 0 or prob == 1:
            return 0
        return -prob * np.log2(prob) - (1 - prob) * np.log2(1 - prob)

    def compute_initial_entropies(self, kernel_size=3, toroidal=True):
        """
        Compute and store the initial entropy values for all regions.

        Args:
            kernel_size (int): Size of the kernel for heterogeneity calculation.
            toroidal (bool): Whether to use periodic boundary conditions.
        """
        self.entropies = []
        self.region_indices = []

        for i in range(self.GRID_WIDTH):
            for j in range(self.GRID_HEIGHT):
                idx = [(i + x) % self.GRID_WIDTH for x in range(kernel_size)]
                idy = [(j + y) % self.GRID_HEIGHT for y in range(kernel_size)]
                region = self.grid[np.ix_(idx, idy)]
                entropy_value = self.compute_entropy(region)
                self.entropies.append(entropy_value)
                self.region_indices.append((i, j))  # Store the top-left corner

    def generate_tunable_IC(self, target_prob_ones, prob_tolerance, target_heterogeneity, heterogeneity_tolerance, 
                            max_iterations=1000, heterogeneity_kernel_size=3, toroidal=True, debug=False):
        """
        Generate an initial condition with tunable probability of ones and heterogeneity.

        Args:
            target_prob_ones (float): Target probability of ones.
            prob_tolerance (float): Tolerance for the probability of ones.
            target_heterogeneity (float): Target heterogeneity (standard deviation of entropy values).
            heterogeneity_tolerance (float): Tolerance for the target heterogeneity.
            max_iterations (int): Maximum number of iterations allowed.
            heterogeneity_kernel_size (int): The size of the kernel for heterogeneity calculation.
            toroidal (bool): Whether to consider toroidal boundary conditions for heterogeneity.
            debug (bool): If True, prints detailed information about each iteration for troubleshooting.

        Returns:
            bool: True if successful, False if unable to reach targets within tolerance.
        """
        # Precompute affected regions map
        affected_regions_map = self.precompute_affected_regions(kernel_size=heterogeneity_kernel_size, toroidal=toroidal)

        # Start with a grid that has the exact number of ones corresponding to the target probability
        num_cells = self.GRID_WIDTH * self.GRID_HEIGHT
        num_ones = int(round(target_prob_ones * num_cells))
        # Initialize grid with exact number of ones
        flat_grid = np.zeros(num_cells, dtype=bool)
        flat_grid[:num_ones] = True
        np.random.shuffle(flat_grid)
        self.grid = flat_grid.reshape((self.GRID_WIDTH, self.GRID_HEIGHT))

        # Compute initial entropies
        self.compute_initial_entropies(kernel_size=heterogeneity_kernel_size, toroidal=toroidal)
        current_heterogeneity = self.compute_heterogeneity()

        for iteration in range(max_iterations):
            # Display debug information if enabled
            if debug:
                print(f"Iteration {iteration + 1}: Probability of Ones = {target_prob_ones:.4f}, "
                      f"Heterogeneity = {current_heterogeneity:.4f}")

            # Check if the current grid meets the target values within tolerances
            if abs(current_heterogeneity - target_heterogeneity) <= heterogeneity_tolerance:
                if debug:
                    print(f"Iteration {iteration + 1}: IC generated with Probability of Ones = {target_prob_ones:.4f} "
                          f"and Heterogeneity = {current_heterogeneity:.4f}")
                return True

            # Calculate the difference in heterogeneity
            heterogeneity_diff = target_heterogeneity - current_heterogeneity

            # Determine the number of swaps based on the difference
            k = 400  # Adjust this factor based on performance and convergence
            num_swaps = int(k * abs(heterogeneity_diff))
            num_swaps = max(1, min(num_swaps, 1000))  # Limit swaps between 1 and 1000

            # Perform swaps proportional to the heterogeneity difference
            self.dynamic_monte_carlo_adjust_heterogeneity(
                target_heterogeneity, heterogeneity_kernel_size, heterogeneity_tolerance,
                toroidal, heterogeneity_diff, affected_regions_map, num_swaps, current_heterogeneity, debug
            )

            # Update the current heterogeneity
            current_heterogeneity = self.compute_heterogeneity()

        if debug:
            print("Failed to generate IC within target ranges after max iterations.")
        return False

    def dynamic_monte_carlo_adjust_heterogeneity(self, target_heterogeneity, kernel_size, heterogeneity_tolerance,
                                             toroidal, heterogeneity_diff, affected_regions_map, num_swaps,
                                             current_heterogeneity, debug=False):
        """
        Adjust heterogeneity by swapping cells to redistribute ones and zeros
        without changing the overall probability of ones. The number of swaps is 
        proportional to the difference in heterogeneity.

        Args:
            target_heterogeneity (float): Target heterogeneity value.
            kernel_size (int): Size of the kernel used for heterogeneity calculations.
            heterogeneity_tolerance (float): Tolerance for the target heterogeneity.
            toroidal (bool): Whether to use toroidal boundaries.
            heterogeneity_diff (float): Current difference in heterogeneity.
            affected_regions_map (dict): Mapping of cells to affected regions.
            num_swaps (int): Number of swaps to perform in this adjustment.
            current_heterogeneity (float): Current heterogeneity value.
            debug (bool): If True, prints detailed information about each swap.
        """
        # Scale the number of swaps based on the heterogeneity difference
        scaling_factor = max(1, int(abs(heterogeneity_diff) * 10))  # Scale number of swaps based on heterogeneity difference
        total_swaps = min(1000, scaling_factor)  # Cap the number of swaps for performance

        for swap_num in range(total_swaps):
            # Randomly select multiple pairs of cells to swap at once
            for _ in range(num_swaps):
                # Randomly select cells with value 1 and 0
                ones_indices = np.argwhere(self.grid)
                zeros_indices = np.argwhere(~self.grid)

                if len(ones_indices) == 0 or len(zeros_indices) == 0:
                    # Can't swap if we don't have both ones and zeros
                    if debug:
                        print("No more cells to swap.")
                    break

                idx_one = tuple(ones_indices[np.random.randint(len(ones_indices))])
                idx_zero = tuple(zeros_indices[np.random.randint(len(zeros_indices))])

                # Identify affected regions
                regions_one = affected_regions_map[idx_one]
                regions_zero = affected_regions_map[idx_zero]
                affected_regions = regions_one.union(regions_zero)

                # Compute initial energy (sum of entropies) for affected regions
                initial_energy = sum([self.entropies[region_idx] for region_idx in affected_regions])

                # Swap the cells
                self.grid[idx_one], self.grid[idx_zero] = self.grid[idx_zero], self.grid[idx_one]

                # Recompute entropy for affected regions
                new_energy = 0
                for region_idx in affected_regions:
                    region_pos = self.region_indices[region_idx]
                    i, j = region_pos
                    idx = [(i + x) % self.GRID_WIDTH for x in range(kernel_size)]
                    idy = [(j + y) % self.GRID_HEIGHT for y in range(kernel_size)]
                    region = self.grid[np.ix_(idx, idy)]
                    new_entropy = self.compute_entropy(region)
                    new_energy += new_entropy
                    # Update the entropy list
                    self.entropies[region_idx] = new_entropy

                # Calculate the change in energy
                delta_energy = new_energy - initial_energy

                # Calculate new heterogeneity after swap
                new_heterogeneity = self.compute_heterogeneity()

                # Decide whether to accept or revert the swap
                if abs(new_heterogeneity - target_heterogeneity) < abs(current_heterogeneity - target_heterogeneity):
                    # Accept the swap
                    if debug:
                        print(f"Swap {swap_num + 1}/{total_swaps}: Accepted (Delta Energy: {delta_energy:.4f})")
                    current_heterogeneity = new_heterogeneity
                else:
                    # Revert the swap and entropy changes
                    self.grid[idx_one], self.grid[idx_zero] = self.grid[idx_zero], self.grid[idx_one]
                    for region_idx in affected_regions:
                        region_pos = self.region_indices[region_idx]
                        i, j = region_pos
                        idx = [(i + x) % self.GRID_WIDTH for x in range(kernel_size)]
                        idy = [(j + y) % self.GRID_HEIGHT for y in range(kernel_size)]
                        region = self.grid[np.ix_(idx, idy)]
                        reverted_entropy = self.compute_entropy(region)
                        self.entropies[region_idx] = reverted_entropy
                    if debug:
                        print(f"Swap {swap_num + 1}/{total_swaps}: Reverted (Delta Energy: {delta_energy:.4f})")



    def get_state_as_string(self, grid):
        """Convert the grid to a string representation."""
        return ''.join(['1' if cell else '0' for row in grid for cell in row])
    
    def randomize_grid(self):
        probability_of_one = np.clip(np.random.normal(self.order_mean, self.order_std), 0, 1)
        for i in range(0, self.GRID_WIDTH):
            for j in range(0, self.GRID_HEIGHT):
                self.grid[i, j] = random.random() < probability_of_one
    
    def update_grid_zeros_BC(self):
        new_grid = np.zeros_like(self.grid)  # Initialize new grid with zeros
        for i in range(self.GRID_WIDTH):
            for j in range(self.GRID_HEIGHT):
                neighbors = sum(
                    self.grid[x, y]  # Access neighbors directly
                    for x in range(max(0, i - 1), min(self.GRID_WIDTH, i + 2))
                    for y in range(max(0, j - 1), min(self.GRID_HEIGHT, j + 2))
                ) - self.grid[i, j]
                if self.grid[i, j]:
                    if neighbors < 2 or neighbors > 3:
                        new_grid[i, j] = False
                    else:
                        new_grid[i, j] = True
                else:
                    if neighbors == 3:
                        new_grid[i, j] = True
        self.grid = new_grid
    
    def run(self, num_iterations=2):
        self.metagrid = []
        for _ in range(num_iterations):
            self.metagrid.append(np.copy(self.grid))
            if self.toroidal:
                self.update_grid()
            else:
                self.update_grid_zeros_BC()
        return np.array(self.metagrid)

    def get_state_as_string(self, grid):
        return ''.join(str(int(cell)) for row in grid for cell in row)
    
    def set_grid_from_string(self, grid_string):
        if len(grid_string) != self.GRID_WIDTH * self.GRID_HEIGHT:
            raise ValueError("The length of the input string does not match the grid dimensions.")
        grid_array = np.array([int(char) for char in grid_string]).reshape((self.GRID_WIDTH, self.GRID_HEIGHT))
        self.grid = grid_array.astype(bool)

    def generate_sets(self, A=100, N=32, I=2, s=.5, e=.5):
        self.WIDTH = self.HEIGHT = N
        self.GRID_WIDTH, self.GRID_HEIGHT = self.WIDTH // self.GRID_SIZE, self.HEIGHT // self.GRID_SIZE
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=bool)

        data = []
        probabilities = np.linspace(s, e, A)
        for probability_of_one in probabilities:
            self.randomize_grid_uniform(probability_of_one)
            self.run(num_iterations=I)
            metagrid = self.metagrid
            states = [self.get_state_as_string(metagrid[i]) for i in range(I)]
            data.append(states)
        return np.array(data)

    def initialize_pattern(self, pattern_name):
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=bool)
        if pattern_name == "glider_gun":
            self.place_glider_gun()
        elif pattern_name == "cloverleaf":
            self.place_cloverleaf()
        elif pattern_name == "hammerhead_spaceship":
            self.place_hammerhead_spaceship()
        elif pattern_name == "blinkers":
            self.place_blinkers()
        elif pattern_name == "r_pentomino":
            self.place_r_pentomino()
        elif pattern_name == "gliders":
            self.place_gliders()
        else:
            print("inputted pattern not found")
            exit()

    def place_glider_gun(self):
        pattern = [
            "........................O...........",
            "......................O.O...........",
            "............OO......OO............OO",
            "...........O...O....OO............OO",
            "OO........O.....O...OO..............",
            "OO........O...O.OO....O.O...........",
            "..........O.....O.......O...........",
            "...........O...O....................",
            "............OO......................",
        ]
        for i, row in enumerate(pattern):
            for j, cell in enumerate(row):
                self.grid[i + 5][j + 5] = (cell == 'O')

    def place_cloverleaf(self):
        pattern = [
            ".......................",
            "........OO...OO........",
            ".......O..O.O..O.......",
            ".......O.OO.OO.O.......",
            "......OO.......OO......",
            "........O.O.O.O.......",
            "......OO.......OO......",
            ".......O.OO.OO.O.......",
            ".......O..O.O..O.......",
            "........OO...OO........",
            ".......................",
        ]
        for i, row in enumerate(pattern):
            for j, cell in enumerate(row):
                self.grid[i + 5][j + 5] = (cell == 'O')

    def place_hammerhead_spaceship(self):
        pattern = [
            "......................",
            "..OOOOO...............",
            "..O....O.......OO.....",
            "..O...........OO.OOO..",
            "...O.........OO.OOOO..",
            ".....OO...OO.OO..OO...",
            ".......O....O..O......",
            "........O.O.O.O.......",
            ".........O............",
            ".........O............",
            "........O.O.O.O.......",
            ".......O....O..O......",
            ".....OO...OO.OO..OO...",
            "...O.........OO.OOOO..",
            "..O...........OO.OOO..",
            "..O....O.......OO.....",
            "..OOOOO...............",
        ]
        for i, row in enumerate(pattern):
            for j, cell in enumerate(row):
                self.grid[i + 5][j + 5] = (cell == 'O')

    def place_blinkers(self):
        pattern = [
            "OOO..............",
            ".................",
            ".................",
            ".......OOO.......",
            ".................",
            ".................",
            ".................",
            ".......OOO.......",
            "...............O.",
            "...............O.",
            "...............O.",
        ]
        for i, row in enumerate(pattern):
            for j, cell in enumerate(row):
                self.grid[i][j] = (cell == 'O')
    def place_r_pentomino(self):
        pattern = [
            ".OO..............",
            "OO...............",
            ".O...............",
            ".................",
            ".................",
            ".................",
            ".................",
            ".................",
            ".................",
            ".................",
            ".................",
        ]
        for i, row in enumerate(pattern):
            for j, cell in enumerate(row):
                self.grid[i][j] = (cell == 'O')
    def place_gliders(self):
        pattern = [
            ".................",
            ".................",
            ".................",
            ".................",
            ".......OO........",
            "........OO.......",
            ".......O.........",
            ".................",
            "..........O......",
            "..........O.O....",
            "..........OO.....",
        ]
        for i, row in enumerate(pattern):
            for j, cell in enumerate(row):
                self.grid[i + 5][j + 5] = (cell == 'O')

    def run_and_save(self, num_iterations, save_folder):
        for index in range(num_iterations):
            if self.toroidal:
                self.update_grid()
            else:
                self.update_grid_zeros_BC()
            save_path = os.path.join(save_folder, f'grid_{index}.npy')
            np.save(save_path, self.grid)

    def create_animation(self, num_iterations, interval=8.35):
        fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            ax.imshow(self.metagrid[frame], cmap='binary')
            ax.set_title(f"Iteration {frame+1}")

        ani = FuncAnimation(fig, update, frames=num_iterations, interval=interval, repeat=False)
        plt.show()
    def flip_bits(self, grid, num_bits_to_flip):
        """Flip random bits in the grid to generate new variations of ICs."""
        indices = np.random.choice(grid.size, num_bits_to_flip, replace=False)
        flat_grid = grid.flatten()
        flat_grid[indices] = np.logical_not(flat_grid[indices])  # Flip the bits
        return flat_grid.reshape(grid.shape)



# Example usage of the ConwayGame class with the new tunable IC generation
game = ConwayGame(toroidal=True, width=32, height=32)
game.initialize_pattern("hammerhead_spaceship")
game.run(num_iterations=300)
game.create_animation(num_iterations=300, interval=16.7)


#####################################################################################################
#####################################   OLD CODE  ###################################################
#####################################################################################################

# class ConwayGame:
#     def __init__(self, toroidal=True, width=32, height=32, grid_size=1, order_mean=0.5, order_std=0.1):
#         self.WIDTH, self.HEIGHT = width, height
#         self.GRID_SIZE = grid_size
#         self.GRID_WIDTH, self.GRID_HEIGHT = self.WIDTH // self.GRID_SIZE, self.HEIGHT // self.GRID_SIZE
#         self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=bool)
#         self.metagrid = []
#         self.order_mean = order_mean
#         self.order_std = order_std
#         self.toroidal = toroidal

#     def randomize_grid(self):
#         probability_of_one = np.clip(np.random.normal(self.order_mean, self.order_std), 0, 1)
#         for i in range(0, self.GRID_WIDTH):
#             for j in range(0, self.GRID_HEIGHT):
#                 self.grid[i, j] = random.random() < probability_of_one

#     def randomize_grid_uniform(self, probability_of_one):
#         for i in range(0, self.GRID_WIDTH):
#             for j in range(0, self.GRID_HEIGHT):
#                 self.grid[i, j] = random.random() < probability_of_one

#     def update_grid(self):
#         new_grid = np.copy(self.grid)
#         for i in range(self.GRID_WIDTH):
#             for j in range(self.GRID_HEIGHT):
#                 neighbors = sum(
#                     self.grid[(i + x) % self.GRID_WIDTH, (j + y) % self.GRID_HEIGHT]
#                     for x in [-1, 0, 1] for y in [-1, 0, 1]
#                 ) - self.grid[i, j]
#                 if self.grid[i, j]:
#                     if neighbors < 2 or neighbors > 3:
#                         new_grid[i, j] = False
#                 else:
#                     if neighbors == 3:
#                         new_grid[i, j] = True
#         self.grid = new_grid

#     def update_grid_zeros_BC(self):
#         new_grid = np.zeros_like(self.grid)  # Initialize new grid with zeros
#         for i in range(self.GRID_WIDTH):
#             for j in range(self.GRID_HEIGHT):
#                 neighbors = sum(
#                     self.grid[x, y]  # Access neighbors directly
#                     for x in range(max(0, i - 1), min(self.GRID_WIDTH, i + 2))
#                     for y in range(max(0, j - 1), min(self.GRID_HEIGHT, j + 2))
#                 ) - self.grid[i, j]
#                 if self.grid[i, j]:
#                     if neighbors < 2 or neighbors > 3:
#                         new_grid[i, j] = False
#                     else:
#                         new_grid[i, j] = True
#                 else:
#                     if neighbors == 3:
#                         new_grid[i, j] = True
#         self.grid = new_grid

#     def run(self, num_iterations=2):
#         self.metagrid = []
#         for _ in range(num_iterations):
#             self.metagrid.append(np.copy(self.grid))
#             if self.toroidal:
#                 self.update_grid()
#             else:
#                 self.update_grid_zeros_BC()
#         return np.array(self.metagrid)

#     def get_state_as_string(self, grid):
#         return ''.join(str(int(cell)) for row in grid for cell in row)
    
#     def set_grid_from_string(self, grid_string):
#         if len(grid_string) != self.GRID_WIDTH * self.GRID_HEIGHT:
#             raise ValueError("The length of the input string does not match the grid dimensions.")
#         grid_array = np.array([int(char) for char in grid_string]).reshape((self.GRID_WIDTH, self.GRID_HEIGHT))
#         self.grid = grid_array.astype(bool)

#     def generate_sets(self, A=100, N=32, I=2, s=.5, e=.5):
#         self.WIDTH = self.HEIGHT = N
#         self.GRID_WIDTH, self.GRID_HEIGHT = self.WIDTH // self.GRID_SIZE, self.HEIGHT // self.GRID_SIZE
#         self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=bool)

#         data = []
#         probabilities = np.linspace(s, e, A)
#         for probability_of_one in probabilities:
#             self.randomize_grid_uniform(probability_of_one)
#             self.run(num_iterations=I)
#             metagrid = self.metagrid
#             states = [self.get_state_as_string(metagrid[i]) for i in range(I)]
#             data.append(states)
#         return np.array(data)

#     def initialize_pattern(self, pattern_name):
#         self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=bool)
#         if pattern_name == "glider_gun":
#             self.place_glider_gun()
#         elif pattern_name == "cloverleaf":
#             self.place_cloverleaf()
#         elif pattern_name == "hammerhead_spaceship":
#             self.place_hammerhead_spaceship()
#         elif pattern_name == "blinkers":
#             self.place_blinkers()
#         elif pattern_name == "r_pentomino":
#             self.place_r_pentomino()
#         elif pattern_name == "gliders":
#             self.place_gliders()
#         else:
#             print("inputted pattern not found")
#             exit()

#     def place_glider_gun(self):
#         pattern = [
#             "........................O...........",
#             "......................O.O...........",
#             "............OO......OO............OO",
#             "...........O...O....OO............OO",
#             "OO........O.....O...OO..............",
#             "OO........O...O.OO....O.O...........",
#             "..........O.....O.......O...........",
#             "...........O...O....................",
#             "............OO......................",
#         ]
#         for i, row in enumerate(pattern):
#             for j, cell in enumerate(row):
#                 self.grid[i + 5][j + 5] = (cell == 'O')

#     def place_cloverleaf(self):
#         pattern = [
#             ".......................",
#             "........OO...OO........",
#             ".......O..O.O..O.......",
#             ".......O.OO.OO.O.......",
#             "......OO.......OO......",
#             "........O.O.O.O.......",
#             "......OO.......OO......",
#             ".......O.OO.OO.O.......",
#             ".......O..O.O..O.......",
#             "........OO...OO........",
#             ".......................",
#         ]
#         for i, row in enumerate(pattern):
#             for j, cell in enumerate(row):
#                 self.grid[i + 5][j + 5] = (cell == 'O')

#     def place_hammerhead_spaceship(self):
#         pattern = [
#             "......................",
#             "..OOOOO...............",
#             "..O....O.......OO.....",
#             "..O...........OO.OOO..",
#             "...O.........OO.OOOO..",
#             ".....OO...OO.OO..OO...",
#             ".......O....O..O......",
#             "........O.O.O.O.......",
#             ".........O............",
#             ".........O............",
#             "........O.O.O.O.......",
#             ".......O....O..O......",
#             ".....OO...OO.OO..OO...",
#             "...O.........OO.OOOO..",
#             "..O...........OO.OOO..",
#             "..O....O.......OO.....",
#             "..OOOOO...............",
#         ]
#         for i, row in enumerate(pattern):
#             for j, cell in enumerate(row):
#                 self.grid[i + 5][j + 5] = (cell == 'O')

#     def place_blinkers(self):
#         pattern = [
#             "OOO..............",
#             ".................",
#             ".................",
#             ".......OOO.......",
#             ".................",
#             ".................",
#             ".................",
#             ".......OOO.......",
#             "...............O.",
#             "...............O.",
#             "...............O.",
#         ]
#         for i, row in enumerate(pattern):
#             for j, cell in enumerate(row):
#                 self.grid[i][j] = (cell == 'O')
#     def place_r_pentomino(self):
#         pattern = [
#             ".OO..............",
#             "OO...............",
#             ".O...............",
#             ".................",
#             ".................",
#             ".................",
#             ".................",
#             ".................",
#             ".................",
#             ".................",
#             ".................",
#         ]
#         for i, row in enumerate(pattern):
#             for j, cell in enumerate(row):
#                 self.grid[i][j] = (cell == 'O')
#     def place_gliders(self):
#         pattern = [
#             ".................",
#             ".................",
#             ".................",
#             ".................",
#             ".......OO........",
#             "........OO.......",
#             ".......O.........",
#             ".................",
#             "..........O......",
#             "..........O.O....",
#             "..........OO.....",
#         ]
#         for i, row in enumerate(pattern):
#             for j, cell in enumerate(row):
#                 self.grid[i + 5][j + 5] = (cell == 'O')

#     def run_and_save(self, num_iterations, save_folder):
#         for index in range(num_iterations):
#             if self.toroidal:
#                 self.update_grid()
#             else:
#                 self.update_grid_zeros_BC()
#             save_path = os.path.join(save_folder, f'grid_{index}.npy')
#             np.save(save_path, self.grid)

#     def create_animation(self, num_iterations, interval=8.35):
#         fig, ax = plt.subplots()

#         def update(frame):
#             ax.clear()
#             ax.imshow(self.metagrid[frame], cmap='binary')
#             ax.set_title(f"Iteration {frame+1}")

#         ani = FuncAnimation(fig, update, frames=num_iterations, interval=interval, repeat=False)
#         plt.show()
#     def flip_bits(self, grid, num_bits_to_flip):
#         """Flip random bits in the grid to generate new variations of ICs."""
#         indices = np.random.choice(grid.size, num_bits_to_flip, replace=False)
#         flat_grid = grid.flatten()
#         flat_grid[indices] = np.logical_not(flat_grid[indices])  # Flip the bits
#         return flat_grid.reshape(grid.shape)

#     def compute_entropy(self, region):
#         """
#         Calculate the entropy of a given region of the grid.

#         Args:
#             region (np.ndarray): A region of the grid.

#         Returns:
#             float: Entropy value of the region.
#         """
#         prob = np.mean(region)
#         if prob == 0 or prob == 1:
#             return 0
#         return -prob * np.log2(prob) - (1 - prob) * np.log2(1 - prob)

#     def compute_initial_entropies(self, kernel_size=3, toroidal=True):
#         """
#         Compute and store the initial entropy values for all regions.

#         Args:
#             kernel_size (int): Size of the kernel for heterogeneity calculation.
#             toroidal (bool): Whether to use periodic boundary conditions.
#         """
#         self.entropies = []
#         self.region_indices = []

#         for i in range(self.GRID_WIDTH):
#             for j in range(self.GRID_HEIGHT):
#                 # Extract a kernel region with periodic boundaries
#                 idx = [(i + x) % self.GRID_WIDTH for x in range(kernel_size)]
#                 idy = [(j + y) % self.GRID_HEIGHT for y in range(kernel_size)]
#                 region = self.grid[np.ix_(idx, idy)]
#                 entropy_value = self.compute_entropy(region)
#                 self.entropies.append(entropy_value)
#                 self.region_indices.append((i, j))  # Store the top-left corner

    
#     def compute_number_of_ones_per_region(self, region_size=3):
#         """
#         Compute the number of ones in each region of the grid.

#         Args:
#             region_size (int): Size of the square region.

#         Returns:
#             list: List containing the number of ones in each region.
#         """
#         probs = []
#         for i in range(self.GRID_WIDTH - region_size + 1):
#             for j in range(self.GRID_HEIGHT - region_size + 1):
#                 region = self.grid[i:i + region_size, j:j + region_size]
#                 probs.append(np.sum(region)/(region_size**2))
#         return probs

#     def compute_heterogeneity(self, region_size=3):
#         """
#         Compute heterogeneity as the standard deviation of the number of ones per region.

#         Args:
#             region_size (int): Size of the square region.

#         Returns:
#             float: Standard deviation representing heterogeneity.
#         """
#         probs = self.compute_number_of_ones_per_region(region_size=region_size)
#         if len(probs) == 0:
#             return 0.0
#         return np.std(probs)


#     def precompute_affected_regions(self, kernel_size, toroidal):
#         """
#         Precompute and store the regions affected by each cell.

#         Args:
#             kernel_size (int): Size of the kernel used for heterogeneity calculations.
#             toroidal (bool): Whether to use toroidal boundaries.

#         Returns:
#             dict: A dictionary mapping each cell to a set of region indices that include it.
#         """
#         affected_regions_map = {}
#         for i in range(self.GRID_WIDTH):
#             for j in range(self.GRID_HEIGHT):
#                 affected_regions_map[(i, j)] = set()

#         for region_idx, (i, j) in enumerate(self.region_indices):
#             for x in range(kernel_size):
#                 for y in range(kernel_size):
# #                     cell_i = (i + x) % self.GRID_WIDTH if toroidal else i + x
# #                     cell_j = (j + y) % self.GRID_HEIGHT if toroidal else j + y
# #                     if 0 <= cell_i < self.GRID_WIDTH and 0 <= cell_j < self.GRID_HEIGHT:
#                         affected_regions_map[(cell_i, cell_j)].add(region_idx)
#         return affected_regions_map

#     def generate_tunable_IC(self, target_prob_ones, prob_tolerance, target_heterogeneity, heterogeneity_tolerance, 
#                             max_iterations=1000, heterogeneity_kernel_size=3, toroidal=True, debug=False):
#         """
#         Generate an initial condition with tunable probability of ones and heterogeneity.

#         Args:
#             target_prob_ones (float): Target probability of ones.
#             prob_tolerance (float): Tolerance for the probability of ones.
#             target_heterogeneity (float): Target heterogeneity (standard deviation of entropy values).
#             heterogeneity_tolerance (float): Tolerance for the target heterogeneity.
#             max_iterations (int): Maximum number of iterations allowed.
#             heterogeneity_kernel_size (int): The size of the kernel for heterogeneity calculation.
#             toroidal (bool): Whether to consider toroidal boundary conditions for heterogeneity.
#             debug (bool): If True, prints detailed information about each iteration for troubleshooting.

#         Returns:
#             bool: True if successful, False if unable to reach targets within tolerance.
#         """
#         # Precompute affected regions map
#         affected_regions_map = self.precompute_affected_regions(kernel_size=heterogeneity_kernel_size, toroidal=toroidal)

#         # Start with a grid that has the exact number of ones corresponding to the target probability
#         num_cells = self.GRID_WIDTH * self.GRID_HEIGHT
#         num_ones = int(round(target_prob_ones * num_cells))
#         # Initialize grid with exact number of ones
#         flat_grid = np.zeros(num_cells, dtype=bool)
#         flat_grid[:num_ones] = True
#         np.random.shuffle(flat_grid)
#         self.grid = flat_grid.reshape((self.GRID_WIDTH, self.GRID_HEIGHT))

#         # Compute initial entropies
#         self.compute_initial_entropies(kernel_size=heterogeneity_kernel_size, toroidal=toroidal)
#         current_heterogeneity = self.compute_heterogeneity()

#         for iteration in range(max_iterations):
#             # Display debug information if enabled
#             if debug:
#                 print(f"Iteration {iteration + 1}: Probability of Ones = {target_prob_ones:.4f}, "
#                       f"Heterogeneity = {current_heterogeneity:.4f}")

#             # Check if the current grid meets the target values within tolerances
#             if abs(current_heterogeneity - target_heterogeneity) <= heterogeneity_tolerance:
#                 if debug:
#                     print(f"Iteration {iteration + 1}: IC generated with Probability of Ones = {target_prob_ones:.4f} "
#                           f"and Heterogeneity = {current_heterogeneity:.4f}")
#                 return True

#             # Calculate the difference in heterogeneity
#             heterogeneity_diff = target_heterogeneity - current_heterogeneity

#             # Determine the number of swaps based on the difference
#             # Scaling factor k determines how aggressive the tuning is
#             k = 100  # Adjust this factor based on performance and convergence
#             num_swaps = int(k * abs(heterogeneity_diff))
#             num_swaps = max(1, min(num_swaps, 1000))  # Limit swaps between 1 and 1000

#             # Perform swaps proportional to the heterogeneity difference
#             self.dynamic_monte_carlo_adjust_heterogeneity(
#                 target_heterogeneity, heterogeneity_kernel_size, heterogeneity_tolerance, 
#                 toroidal, heterogeneity_diff, affected_regions_map, num_swaps, debug
#             )

#             # Update the current heterogeneity
#             current_heterogeneity = self.compute_heterogeneity()

#         if debug:
#             print("Failed to generate IC within target ranges after max iterations.")
#         return False

#     def dynamic_monte_carlo_adjust_heterogeneity(self, target_heterogeneity, kernel_size, heterogeneity_tolerance, 
#                                                 toroidal, heterogeneity_diff, affected_regions_map, num_swaps, debug=False):
#         """
#         Adjust heterogeneity by swapping cells to redistribute ones and zeros
#         without changing the overall probability of ones. The number of swaps is 
#         proportional to the difference in heterogeneity.

#         Args:
#             target_heterogeneity (float): Target heterogeneity value.
#             kernel_size (int): Size of the kernel used for heterogeneity calculations.
#             heterogeneity_tolerance (float): Tolerance for the target heterogeneity.
#             toroidal (bool): Whether to use toroidal boundaries.
#             heterogeneity_diff (float): Current difference in heterogeneity.
#             affected_regions_map (dict): Mapping of cells to affected regions.
#             num_swaps (int): Number of swaps to perform in this adjustment.
#             debug (bool): If True, prints detailed information about each swap.
#         """
#         for swap_num in range(num_swaps):
#             # Randomly select two cells: one with value 1 and one with value 0
#             ones_indices = np.argwhere(self.grid)
#             zeros_indices = np.argwhere(~self.grid)

#             if len(ones_indices) == 0 or len(zeros_indices) == 0:
#                 # Can't swap if we don't have both ones and zeros
#                 if debug:
#                     print("No more cells to swap.")
#                 break

#             idx_one = tuple(ones_indices[np.random.randint(len(ones_indices))])
#             idx_zero = tuple(zeros_indices[np.random.randint(len(zeros_indices))])

#             # Identify affected regions
#             regions_one = affected_regions_map[idx_one]
#             regions_zero = affected_regions_map[idx_zero]
#             affected_regions = regions_one.union(regions_zero)

#             # Compute initial energy (sum of entropies) for affected regions
#             initial_energy = sum([self.entropies[region_idx] for region_idx in affected_regions])

#             # Swap the cells
#             self.grid[idx_one], self.grid[idx_zero] = self.grid[idx_zero], self.grid[idx_one]

#             # Recompute entropy for affected regions
#             new_energy = 0
#             for region_idx in affected_regions:
#                 region_pos = self.region_indices[region_idx]
#                 i, j = region_pos
#                 idx = [(i + x) % self.GRID_WIDTH for x in range(kernel_size)]
#                 idy = [(j + y) % self.GRID_HEIGHT for y in range(kernel_size)]
#                 region = self.grid[np.ix_(idx, idy)]
#                 new_entropy = self.compute_entropy(region)
#                 new_energy += new_entropy
#                 # Update the entropy list
#                 self.entropies[region_idx] = new_entropy

#             # Calculate the change in energy
#             delta_energy = new_energy - initial_energy

#             # Calculate new heterogeneity after swap
#             # Since computing entire heterogeneity is expensive, we use incremental updates
#             # However, for acceptance, we need the new heterogeneity
#             # Hence, we compute it now
#             # Note: This is a simplification; for larger grids, consider maintaining a running sum and sum of squares
#             # to compute std without full recomputation
#             # Here, we'll recompute it
#             # Alternatively, keep track of how delta_energy affects heterogeneity
#             # But for simplicity, we'll recompute
#             # (This can be optimized further)
#             new_heterogeneity = self.compute_heterogeneity()
#             new_diff = target_heterogeneity - new_heterogeneity
#             old_diff = abs(target_heterogeneity - (new_heterogeneity + delta_energy))

#             # Decide whether to accept or revert the swap
#             # If the swap reduces the absolute difference to target, accept
#             # Else, revert
#             if abs(new_heterogeneity - target_heterogeneity) < abs(new_heterogeneity + delta_energy - target_heterogeneity):
#                 # Accept the swap
#                 if debug:
#                     print(f"Swap {swap_num + 1}/{num_swaps}: Accepted (Delta Energy: {delta_energy:.4f})")
#                 continue
#             else:
#                 # Revert the swap and entropy changes
#                 self.grid[idx_one], self.grid[idx_zero] = self.grid[idx_zero], self.grid[idx_one]
#                 for region_idx in affected_regions:
#                     # Recompute entropy to revert
#                     region_pos = self.region_indices[region_idx]
#                     i, j = region_pos
#                     idx = [(i + x) % self.GRID_WIDTH for x in range(kernel_size)]
#                     idy = [(j + y) % self.GRID_HEIGHT for y in range(kernel_size)]
#                     region = self.grid[np.ix_(idx, idy)]
#                     reverted_entropy = self.compute_entropy(region)
#                     self.entropies[region_idx] = reverted_entropy
#                 if debug:
#                     print(f"Swap {swap_num + 1}/{num_swaps}: Reverted (Delta Energy: {delta_energy:.4f})")

#     def get_state_as_string(self, grid):
#         """Convert the grid to a string representation."""
#         return ''.join(['1' if cell else '0' for row in grid for cell in row])

#     def flip_random_bits(self, num_flips):
#         """Flip random bits in the grid to generate new variations of ICs."""
#         indices = np.random.choice(self.GRID_WIDTH * self.GRID_HEIGHT, size=num_flips, replace=False)
#         for idx in indices:
#             i, j = divmod(idx, self.GRID_HEIGHT)
#             self.grid[i, j] = not self.grid[i, j]

#     def update_grid(self):
#         """
#         Placeholder method for updating the grid to the next state.
#         Implement the logic based on Conway's Game of Life rules or your specific rules.
#         """
#         # Example implementation (Conway's Game of Life)
#         new_grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=bool)
#         for i in range(self.GRID_WIDTH):
#             for j in range(self.GRID_HEIGHT):
#                 # Define neighbors with toroidal wrapping
#                 neighbors = self.grid[
#                     (i-1)%self.GRID_WIDTH:(i+2)%self.GRID_WIDTH or None,
#                     (j-1)%self.GRID_HEIGHT:(j+2)%self.GRID_HEIGHT or None
#                 ]
#                 count = np.sum(neighbors) - self.grid[i, j]
#                 if self.grid[i, j]:
#                     new_grid[i, j] = count in [2, 3]
#                 else:
#                     new_grid[i, j] = count == 3
#         self.grid = new_grid







#     def get_state_as_string(self, grid):
#         return ''.join(str(int(cell)) for row in grid for cell in row)
# # Example usage of the ConwayGame class with the new tunable IC generation
# # game = ConwayGame(toroidal=True, width=32, height=32)
# # target_entropy = 0.6
# # entropy_tolerance = 0.05
# # target_heterogeneity = 0.4
# # heterogeneity_tolerance = 0.05

# # success = game.generate_tunable_IC(target_entropy, entropy_tolerance, target_heterogeneity, heterogeneity_tolerance)
# # if success:
# #     initial_state = game.get_state_as_string(game.grid)
# #     print(f"Generated IC: {initial_state}")

# save_path = "C:\\Users\\jaime\\ML_Playground_1\\Game_Testing\\"

# game = ConwayGame()
# game.__init__(width=32,height=32)
# game.initialize_pattern("hammerhead_spaceship")
# game.run(num_iterations=300)
# game.create_animation(num_iterations=300,interval=16.7)
