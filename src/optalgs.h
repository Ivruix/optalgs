#ifndef OPTALG_H_INCLUDED
#define OPTALG_H_INCLUDED

#include <iostream>
#include <vector>
#include <cassert>
#include <random>
#include <limits>

enum class VerbosityLevels {
    Silent, Low, Medium, High, Maximum
};

/// DE implementation
class DifferentialEvolution {
private:
    double (*cost_function_)(std::vector<double>);
    std::vector<std::pair<double, double>> bounds_;
    unsigned int dimensionality_;
    unsigned int population_size_;
    bool ensure_bounds_;
    double crossover_probability_;
    double differential_weight_;

    std::vector<std::vector<double>> population_;
    std::vector<double> cost_per_agent_;

    std::mt19937 generator_;
    std::uniform_int_distribution<unsigned int> index_distribution_;
    std::uniform_int_distribution<unsigned int> dimension_index_distribution_;
    std::uniform_real_distribution<double> r_distribution_;

    unsigned int best_agent_index_;
    double best_cost_;

    unsigned int iteration_ = 0;

public:
    /**
     * Constructs differential evolution optimizer.
     *
     * @param cost_function Cost function to be minimized.
     * @param bounds Bounds used to initialize agents. First and second elements of the pair are lower and upper bounds, respectively.
     * Number of pairs must be equal to the number of function arguments.
     * @param population_size Number of agents in population. Must be set to 4 or more. High values may result in reduced performance.
     * @param seed Seed used by pseudo-random number generator.
     * @param ensure_bounds Determines whether bounds should be maintained.
     * @param crossover_probability Recombination constant (CR), must be in the range [0, 1]. High values may result in slower convergence.
     * @param differential_weight Mutation constant (F), must be in the range [0, 2]. High values make population less stable.
     */
    DifferentialEvolution(double (*cost_function)(std::vector<double>),
                          std::vector<std::pair<double, double>> bounds,
                          unsigned int population_size = 15,
                          int seed = 0,
                          bool ensure_bounds = true,
                          double crossover_probability = 0.7,
                          double differential_weight = 0.5)
        : cost_function_(cost_function),
          bounds_(bounds),
          dimensionality_(bounds_.size()),
          population_size_(population_size),
          ensure_bounds_(ensure_bounds),
          crossover_probability_(crossover_probability),
          differential_weight_(differential_weight) {

        assert(population_size_ >= 4);
        assert((crossover_probability_ >= 0.0) && (crossover_probability_ <= 1.0));
        assert((differential_weight_ >= 0.0) && (differential_weight_ <= 2.0));

        population_.resize(population_size_);
        for (auto& agent: population_)
            agent.resize(dimensionality_);
        cost_per_agent_.resize(population_size_);

        generator_.seed(seed);
        index_distribution_ = std::uniform_int_distribution<unsigned int>(0, population_size_ - 1);
        r_distribution_ = std::uniform_real_distribution<double>(0.0, 1.0);
        dimension_index_distribution_ = std::uniform_int_distribution<unsigned int>(0, dimensionality_ - 1);

        // Initialize population with random vectors
        std::uniform_real_distribution<double> distribution;
        for (unsigned int i = 0; i < dimensionality_; i++) {
            distribution = std::uniform_real_distribution<double>(bounds_[i].first, bounds_[i].second);
            for (auto& agent: population_)
                agent[i] = distribution(generator_);
        }

        // Calculate cost function for all agents and start tracking best agent
        best_cost_ = std::numeric_limits<double>::infinity();
        for (unsigned int i = 0; i < population_size_; i++) {
            cost_per_agent_[i] = (*cost_function_)(population_[i]);
            if (cost_per_agent_[i] < best_cost_) {
                best_cost_ = cost_per_agent_[i];
                best_agent_index_ = i;
            }
        }
    }

    /**
     * Optimizes cost function.
     *
     * @param iterations Number of optimization iterations to be performed.
     * @param verbosity Higher verbosity levels include information from lower levels. Log formats by levels:\n
     * VerbosityLevels | Format
     * --------------- | ---------------------------
     * Silent          | Nothing
     * Low             | Current iteration
     * Medium          | Mean cost and best cost
     * High            | Best agent
     * Maximum         | All agents with their costs
     * @param logging_interval Interval between logging. If set to 1, logging will be performed for all iterations.
     */
    void Optimize(unsigned int iterations, VerbosityLevels verbosity = VerbosityLevels::Silent, unsigned int logging_interval = 25) {
        for (unsigned int iteration = 0; iteration < iterations; iteration++) {
            for (unsigned int agent_index = 0; agent_index < population_size_; agent_index++) {
                unsigned int a = index_distribution_(generator_); // base vector index
                unsigned int b = index_distribution_(generator_);
                unsigned int c = index_distribution_(generator_);

                //Make agent_index, a, b, c different
                while (a == agent_index || b == agent_index || c == agent_index || a == b || a == c || b == c) {
                    a = index_distribution_(generator_);
                    b = index_distribution_(generator_);
                    c = index_distribution_(generator_);
                }

                std::vector<double> new_vector(dimensionality_);
                unsigned int R = dimension_index_distribution_(generator_);
                for (unsigned int i = 0; i < dimensionality_; i++) {
                    if (r_distribution_(generator_) < crossover_probability_ || i == R)
                        new_vector[i] = population_[a][i] + differential_weight_ * (population_[b][i] - population_[c][i]);
                    else
                        new_vector[i] = population_[agent_index][i];
                    }

                if (ensure_bounds_)
                    for (unsigned int i = 0; i < dimensionality_; i++)
                        new_vector[i] = EnsureBounds(new_vector[i], i);

                double new_cost = (*cost_function_)(new_vector);
                if (new_cost <= cost_per_agent_[agent_index]) {
                    population_[agent_index] = new_vector;
                    cost_per_agent_[agent_index] = new_cost;
                    if (new_cost < best_cost_) {
                        best_cost_ = new_cost;
                        best_agent_index_ = agent_index;
                    }
                }
            }

            iteration_++;

            if (iteration_ % logging_interval == 0)
                LogIteration(verbosity);
        }
    }

    void LogIteration(VerbosityLevels verbosity) const {
        if (verbosity == VerbosityLevels::Silent)
            return;
        if (verbosity >= VerbosityLevels::Low)
            std::cout << "Iteration: " << iteration_ << '\n';
        if (verbosity >= VerbosityLevels::Medium) {
            std::cout << "Mean cost: " << std::accumulate(cost_per_agent_.begin(), cost_per_agent_.end(), 0.0) / population_size_ << '\n';
            std::cout << "Best cost: " << best_cost_ << '\n';
        }
        if (verbosity >= VerbosityLevels::High) {
            std::cout << "Best agent: ";
            PrintBestAgent();
        }
        if (verbosity >= VerbosityLevels::Maximum) {
            std::cout << "All agents: \n";
            PrintPopulation(true);
        }
        std::cout << '\n';
    }

    /**
     * @param print_cost If true, additionally prints cost of all agents.
     */
    void PrintPopulation(bool print_cost = false) const {
        for (unsigned int agent_index = 0; agent_index < population_size_; agent_index++) {
            std::cout << '{';
            for (unsigned int i = 0; i < dimensionality_ - 1; i++) {
                std::cout << population_[agent_index][i] << ", ";
            }
            std::cout << population_[agent_index][dimensionality_ - 1] << '}';

            if (print_cost)
                std::cout << " : " << cost_per_agent_[agent_index];
            std::cout << '\n';
        }
    }

    /**
     * @param print_cost If true, additionally prints cost of the best agent.
     */
    void PrintBestAgent(bool print_cost = false) const {
        std::cout << '{';
        for (unsigned int i = 0; i < dimensionality_ - 1; i++) {
            std::cout << population_[best_agent_index_][i] << ", ";
        }
        std::cout << population_[best_agent_index_][dimensionality_ - 1] << '}';

        if (print_cost)
            std::cout << " : " << cost_per_agent_[best_agent_index_];
        std::cout << '\n';
    }

    /// Returns all agents.
    std::vector<std::vector<double>> GetPopulation() const {
        return population_;
    }

    /// Returns agent with lowest cost.
    std::vector<double> GetBestAgent() const {
        return population_[best_agent_index_];
    }

    /// Returns current best cost.
    double GetBestCost() const {
        return best_cost_;
    }

    /// Returns number of passed optimization iterations.
    unsigned int GetCurrentIteration() const {
        return iteration_;
    }

    /// Returns costs of all agents in the same order as `GetPopulation`.
    std::vector<double> GetCosts() const {
        return cost_per_agent_;
    }

private:
    double EnsureBounds(double value, unsigned int dimension_index) {
        if (value > bounds_[dimension_index].second)
            return bounds_[dimension_index].second;
        else if (value < bounds_[dimension_index].first)
            return bounds_[dimension_index].first;
        else
            return value;
    }
};

/// PSO implementation
class ParticleSwarmOptimization {
private:
    double (*cost_function_)(std::vector<double>);
    std::vector<std::pair<double, double>> bounds_;
    unsigned int dimensionality_;
    unsigned int swarm_size_;
    bool ensure_bounds_;
    double omega_;
    double phi_1_;
    double phi_2_;

    std::vector<std::vector<double>> particle_postitions_;
    std::vector<std::vector<double>> particle_velocities_;
    std::vector<std::vector<double>> particle_best_known_positions_;
    std::vector<double> particle_best_known_costs_;
    std::vector<double> swarm_best_known_position_;
    double swarm_best_known_cost_;

    std::mt19937 generator_;
    std::uniform_real_distribution<double> r_distribution_;

    unsigned int iteration_ = 0;

public:
    /**
     * Constructs differential evolution optimizer.
     *
     * @param cost_function Cost function to be minimized.
     * @param bounds Bounds used to initialize agents. First and second elements of the pair are lower and upper bounds, respectively.
     * Number of pairs must be equal to the number of function arguments.
     * @param population_size Number of particles in swarm. High values may result in reduced performance.
     * @param seed Seed used by pseudo-random number generator.
     * @param ensure_bounds Determines whether bounds should be maintained.
     * @param omega Inertia weight of particles. Must be in range [0, 1].
     * @param phi_1 Multiplier of acceleration applied in the direction of previous best particle position. Must be in range [0, 4].
     * @param phi_2 Multiplier of acceleration applied in the direction of the best known position of the entire swarm. Must be in range [0, 4].
     * @param initial_velocity_multiplier Maximum initial velocity of particle as a fraction of bounds. Must be in range [0, 1].
     */
    ParticleSwarmOptimization(double (*cost_function)(std::vector<double>),
                              std::vector<std::pair<double, double>> bounds,
                              unsigned int swarm_size = 30,
                              int seed = 0,
                              bool ensure_bounds = true,
                              double omega = 0.729,
                              double phi_1 = 2.05,
                              double phi_2 = 2.05,
                              double initial_velocity_multiplier = 0.5)
        : cost_function_(cost_function),
          bounds_(bounds),
          dimensionality_(bounds_.size()),
          swarm_size_(swarm_size),
          ensure_bounds_(ensure_bounds),
          omega_(omega),
          phi_1_(phi_1),
          phi_2_(phi_2) {

        assert((omega_ >= 0.0) && (omega_ <= 1.0));
        assert((phi_1_ >= 0.0) && (phi_1_ <= 4.0));
        assert((phi_2_ >= 0.0) && (phi_2_ <= 4.0));
        assert((initial_velocity_multiplier >= 0.0) && (initial_velocity_multiplier <= 1.0));

        particle_postitions_.resize(swarm_size_);
        for (auto& position: particle_postitions_)
            position.resize(dimensionality_);
        particle_velocities_.resize(swarm_size_);
        for (auto& velocity: particle_velocities_)
            velocity.resize(dimensionality_);
        particle_best_known_positions_.resize(swarm_size_);
        for (auto& position: particle_best_known_positions_)
            position.resize(dimensionality_);
        particle_best_known_costs_.resize(swarm_size_);
        swarm_best_known_position_.resize(dimensionality_);

        generator_.seed(seed);
        r_distribution_ = std::uniform_real_distribution<double>(0, 1.0);

        // Initialize particles positions
        std::uniform_real_distribution<double> distribution;
        for (unsigned int i = 0; i < dimensionality_; i++) {
            distribution = std::uniform_real_distribution<double>(bounds_[i].first, bounds_[i].second);
            for (auto& position: particle_postitions_)
                position[i] = distribution(generator_);
        }

        // Initialize best known positions to initial positions
        for (unsigned int i = 0; i < swarm_size; i++)
            particle_best_known_positions_[i] = particle_postitions_[i];

        // Initialize particles velocities
        for (unsigned int i = 0; i < dimensionality_; i++) {
            distribution = std::uniform_real_distribution<double>(-initial_velocity_multiplier * (bounds_[i].second - bounds_[i].first),
                                                                   initial_velocity_multiplier * (bounds_[i].second - bounds_[i].first));
            for (auto& velocity: particle_velocities_)
                velocity[i] = distribution(generator_);
        }

        // Initialize best known cost per particle and for the entire swarm
        swarm_best_known_cost_ = std::numeric_limits<double>::infinity();
        for (unsigned int i = 0; i < swarm_size; i++) {
            particle_best_known_costs_[i] = (*cost_function)(particle_best_known_positions_[i]);
            if (particle_best_known_costs_[i] < swarm_best_known_cost_) {
                swarm_best_known_cost_ = particle_best_known_costs_[i];
                swarm_best_known_position_ = particle_best_known_positions_[i];
            }
        }
    }

    /**
     * Optimizes cost function.
     *
     * @param iterations Number of optimization iterations to be performed.
     * @param verbosity Higher verbosity levels include information from lower levels. Log formats by levels:\n
     * VerbosityLevels | Format
     * --------------- | -----------------------------------------------
     * Silent          | Nothing
     * Low             | Current iteration
     * Medium          | Mean velocity, mean local and best global costs
     * High            | Best found position
     * Maximum         | All current positions of particles
     * @param logging_interval Interval between logging. If set to 1, logging will be performed for all iterations.
     */
    void Optimize(unsigned int iterations, VerbosityLevels verbosity = VerbosityLevels::Silent, unsigned int logging_interval = 25) {
        for (unsigned int iteration = 0; iteration < iterations; iteration++) {
            for (unsigned int agent_index = 0; agent_index < swarm_size_; agent_index++) {

                // Update velocity of particle
                double r_1, r_2;
                for (unsigned int dimension = 0; dimension < dimensionality_; dimension++) {
                    r_1 = r_distribution_(generator_);
                    r_2 = r_distribution_(generator_);
                    particle_velocities_[agent_index][dimension] *= omega_;
                    particle_velocities_[agent_index][dimension] += phi_1_ * r_1 * (particle_best_known_positions_[agent_index][dimension] - particle_postitions_[agent_index][dimension]);
                    particle_velocities_[agent_index][dimension] += phi_2_ * r_2 * (swarm_best_known_position_[dimension] - particle_postitions_[agent_index][dimension]);
                }

                // Update position of particle
                for (unsigned int dimension = 0; dimension < dimensionality_; dimension++)
                    particle_postitions_[agent_index][dimension] += particle_velocities_[agent_index][dimension];

                if (ensure_bounds_)
                    for (unsigned int dimension = 0; dimension < dimensionality_; dimension++)
                        particle_postitions_[agent_index][dimension] = EnsureBounds(particle_postitions_[agent_index][dimension], dimension);

                double new_cost = (*cost_function_)(particle_postitions_[agent_index]);
                if (new_cost < particle_best_known_costs_[agent_index]) {
                    particle_best_known_costs_[agent_index] = new_cost;
                    particle_best_known_positions_[agent_index] = particle_postitions_[agent_index];
                    if (new_cost < swarm_best_known_cost_) {
                        swarm_best_known_cost_ = new_cost;
                        swarm_best_known_position_ = particle_postitions_[agent_index];
                    }
                }
            }

            iteration_++;

            if (iteration_ % logging_interval == 0)
                LogIteration(verbosity);
        }
    }

    void LogIteration(VerbosityLevels verbosity) const {
        if (verbosity == VerbosityLevels::Silent)
            return;
        if (verbosity >= VerbosityLevels::Low)
            std::cout << "Iteration: " << iteration_ << '\n';
        if (verbosity >= VerbosityLevels::Medium) {
            std::cout << "Mean velocity: " << GetMeanVelocity() << '\n';
            std::cout << "Mean maximum known cost: " << std::accumulate(particle_best_known_costs_.begin(), particle_best_known_costs_.end(), 0.0) / swarm_size_ << '\n';
            std::cout << "Best global cost: " << swarm_best_known_cost_ << '\n';
        }
        if (verbosity >= VerbosityLevels::High) {
            std::cout << "Best agent: ";
            PrintBestAgent();
        }
        if (verbosity >= VerbosityLevels::Maximum) {
            std::cout << "All current agents: \n";
            PrintPopulation();
        }
        std::cout << '\n';
    }

    void PrintPopulation() const {
        for (unsigned int agent_index = 0; agent_index < swarm_size_; agent_index++) {
            std::cout << '{';
            for (unsigned int i = 0; i < dimensionality_ - 1; i++) {
                std::cout << particle_postitions_[agent_index][i] << ", ";
            }
            std::cout << particle_postitions_[agent_index][dimensionality_ - 1] << '}';

            std::cout << '\n';
        }
    }

    /**
     * @param print_cost If true, additionally prints cost of the agent (best found position of swarm).
     */
    void PrintBestAgent(bool print_cost = false) const {
        std::cout << '{';
        for (unsigned int i = 0; i < dimensionality_ - 1; i++) {
            std::cout << swarm_best_known_position_[i] << ", ";
        }
        std::cout << swarm_best_known_position_[dimensionality_ - 1] << '}';

        if (print_cost)
            std::cout << " : " << swarm_best_known_cost_;
        std::cout << '\n';
    }

    /// Returns current positions of particles.
    std::vector<std::vector<double>> GetPopulation() const {
        return particle_postitions_;
    }

    /// Returns best known position of each particle.
    std::vector<std::vector<double>> GetBestKnownPositions() const {
        return particle_best_known_positions_;
    }

    /// Returns best known position of the entire swarm.
    std::vector<double> GetBestAgent() const {
        return swarm_best_known_position_;
    }

    /// Returns cost of the best known position.
    double GetBestCost() const {
        return swarm_best_known_cost_;
    }

    /// Returns number of passed optimization iterations.
    unsigned int GetCurrentIteration() const {
        return iteration_;
    }

    /// Returns costs of the best known positions of each particle in the same order as `GetBestKnownPositions`.
    std::vector<double> GetBestKnownCosts() const {
        return particle_best_known_costs_;
    }

    /// Returns mean velocity of particles.
    double GetMeanVelocity() const {
        double velocity_sum = 0.0;
        for (auto& velocity: particle_velocities_) {
            double s = 0.0;
            for (auto &component: velocity)
                s += pow(component, 2);
            velocity_sum += sqrt(s);
        }
        return velocity_sum / swarm_size_;
    }

private:
    double EnsureBounds(double value, unsigned int dimension_index) {
        if (value > bounds_[dimension_index].second)
            return bounds_[dimension_index].second;
        else if (value < bounds_[dimension_index].first)
            return bounds_[dimension_index].first;
        else
            return value;
    }
};

/// PS implementation
class PatternSearch {
private:
    double (*cost_function_)(std::vector<double>);
    std::vector<std::pair<double, double>> bounds_;
    unsigned int dimensionality_;
    unsigned int population_size_;
    bool ensure_bounds_;
    double step_reduction_;

    std::vector<std::vector<double>> population_;
    std::vector<double> step_per_agent_;
    std::vector<double> cost_per_agent_;

    unsigned int best_agent_index_;
    double best_cost_;

    unsigned int iteration_ = 0;

public:
    /**
     * Constructs pattern search optimizer.
     *
     * @param cost_function Cost function to be minimized.
     * @param bounds Bounds used to initialize agents. First and second elements of the pair are lower and upper bounds, respectively.
     * Number of pairs must be equal to the number of function arguments.
     * @param population_size Number of agents in population. High values may result in reduced performance.
     * @param seed Seed used by pseudo-random number generator.
     * @param ensure_bounds Determines whether bounds should be maintained.
     * @param initial_step Initial step size. Must be greater than 0.
     * @param step_reduction Step reduction coefficient, must be in the range (0, 1). High values may result in slower convergence.
     */
    PatternSearch(double (*cost_function)(std::vector<double>),
                  std::vector<std::pair<double, double>> bounds,
                  unsigned int population_size = 50,
                  int seed = 0,
                  bool ensure_bounds = true,
                  double initial_step = 1.0,
                  double step_reduction = 0.5)
        : cost_function_(cost_function),
          bounds_(bounds),
          dimensionality_(bounds_.size()),
          population_size_(population_size),
          ensure_bounds_(ensure_bounds),
          step_reduction_(step_reduction) {

        assert(initial_step > 0);
        assert((step_reduction_ > 0.0) && (step_reduction_ < 1.0));

        population_.resize(population_size_);
        for (auto& agent: population_)
            agent.resize(dimensionality_);
        step_per_agent_.resize(population_size_);
        cost_per_agent_.resize(population_size_);

        std::mt19937 generator;
        generator.seed(seed);

        // Initialize population
        std::uniform_real_distribution<double> distribution;
        for (unsigned int i = 0; i < dimensionality_; i++) {
            distribution = std::uniform_real_distribution<double>(bounds_[i].first, bounds_[i].second);
            for (auto& agent: population_)
                agent[i] = distribution(generator);
        }
        for (auto& step: step_per_agent_)
            step = initial_step;

        // Calculate cost function for all agents and start tracking best agent
        best_cost_ = std::numeric_limits<double>::infinity();
        for (unsigned int i = 0; i < population_size_; i++) {
            cost_per_agent_[i] = (*cost_function_)(population_[i]);
            if (cost_per_agent_[i] < best_cost_) {
                best_cost_ = cost_per_agent_[i];
                best_agent_index_ = i;
            }
        }
    }

    /**
     * Optimizes cost function.
     *
     * @param iterations Number of optimization iterations to be performed.
     * @param verbosity Higher verbosity levels include information from lower levels. Log formats by levels:\n
     * VerbosityLevels | Format
     * --------------- | ---------------------------------------
     * Silent          | Nothing
     * Low             | Current iteration
     * Medium          | Mean step size, mean cost and best cost
     * High            | Best agent
     * Maximum         | All agents with their costs
     * @param logging_interval Interval between logging. If set to 1, logging will be performed for all iterations.
     */
    void Optimize(unsigned int iterations, VerbosityLevels verbosity = VerbosityLevels::Silent, unsigned int logging_interval = 25) {
        for (unsigned int iteration = 0; iteration < iterations; iteration++) {
            for (unsigned int agent_index = 0; agent_index < population_size_; agent_index++) {

                double best_new_cost = std::numeric_limits<double>::infinity();
                std::vector<double> best_new_position(dimensionality_);

                for (unsigned int dimension = 0; dimension < dimensionality_; dimension++) {
                    std::vector<double> new_position_1 = population_[agent_index];
                    if (ensure_bounds_)
                        new_position_1[dimension] = EnsureBounds(new_position_1[dimension] + step_per_agent_[agent_index], dimension);
                    else
                        new_position_1[dimension] += step_per_agent_[agent_index];

                    std::vector<double> new_position_2 = population_[agent_index];
                    if (ensure_bounds_)
                        new_position_2[dimension] = EnsureBounds(new_position_2[dimension] - step_per_agent_[agent_index], dimension);
                    else
                        new_position_2[dimension] -= step_per_agent_[agent_index];

                    double new_cost_1 = cost_function_(new_position_1);
                    double new_cost_2 = cost_function_(new_position_2);

                    if (new_cost_1 < best_new_cost) {
                        best_new_cost = new_cost_1;
                        best_new_position = new_position_1;
                    }
                    if (new_cost_2 < best_new_cost) {
                        best_new_cost = new_cost_2;
                        best_new_position = new_position_2;
                    }
                }

                if (best_new_cost < cost_per_agent_[agent_index]) {
                    population_[agent_index] = best_new_position;
                    cost_per_agent_[agent_index] = best_new_cost;
                    if (best_new_cost < best_cost_) {
                        best_cost_ = best_new_cost;
                        best_agent_index_ = agent_index;
                    }
                } else {
                    step_per_agent_[agent_index] *= step_reduction_;
                }
            }

            iteration_++;

            if (iteration_ % logging_interval == 0)
                LogIteration(verbosity);
        }
    }

    void LogIteration(VerbosityLevels verbosity) const {
        if (verbosity == VerbosityLevels::Silent)
            return;
        if (verbosity >= VerbosityLevels::Low)
            std::cout << "Iteration: " << iteration_ << '\n';
        if (verbosity >= VerbosityLevels::Medium) {
            std::cout << "Mean step size: "  << std::accumulate(step_per_agent_.begin(), step_per_agent_.end(), 0.0) / population_size_ << '\n';
            std::cout << "Mean cost: " << std::accumulate(cost_per_agent_.begin(), cost_per_agent_.end(), 0.0) / population_size_ << '\n';
            std::cout << "Best cost: " << best_cost_ << '\n';
        }
        if (verbosity >= VerbosityLevels::High) {
            std::cout << "Best agent: ";
            PrintBestAgent();
        }
        if (verbosity >= VerbosityLevels::Maximum) {
            std::cout << "All agents: \n";
            PrintPopulation(true);
        }
        std::cout << '\n';
    }

    /**
     * @param print_cost If true, additionally prints cost of all agents.
     */
    void PrintPopulation(bool print_cost = false) const {
        for (unsigned int agent_index = 0; agent_index < population_size_; agent_index++) {
            std::cout << '{';
            for (unsigned int i = 0; i < dimensionality_ - 1; i++) {
                std::cout << population_[agent_index][i] << ", ";
            }
            std::cout << population_[agent_index][dimensionality_ - 1] << '}';

            if (print_cost)
                std::cout << " : " << cost_per_agent_[agent_index];
            std::cout << '\n';
        }
    }

    /**
     * @param print_cost If true, additionally prints cost of the best agent.
     */
    void PrintBestAgent(bool print_cost = false) const {
        std::cout << '{';
        for (unsigned int i = 0; i < dimensionality_ - 1; i++) {
            std::cout << population_[best_agent_index_][i] << ", ";
        }
        std::cout << population_[best_agent_index_][dimensionality_ - 1] << '}';

        if (print_cost)
            std::cout << " : " << cost_per_agent_[best_agent_index_];
        std::cout << '\n';
    }

    /// Returns all agents.
    std::vector<std::vector<double>> GetPopulation() const {
        return population_;
    }

    /// Returns agent with lowest cost.
    std::vector<double> GetBestAgent() const {
        return population_[best_agent_index_];
    }

    /// Returns current best cost.
    double GetBestCost() const {
        return best_cost_;
    }

    /// Returns number of passed optimization iterations.
    unsigned int GetCurrentIteration() const {
        return iteration_;
    }

    /// Returns costs of all agents in the same order as `GetPopulation`.
    std::vector<double> GetCosts() const {
        return cost_per_agent_;
    }

private:
    double EnsureBounds(double value, unsigned int dimension_index) {
        if (value > bounds_[dimension_index].second)
            return bounds_[dimension_index].second;
        else if (value < bounds_[dimension_index].first)
            return bounds_[dimension_index].first;
        else
            return value;
    }
};

#endif // OPTALG_H_INCLUDED
