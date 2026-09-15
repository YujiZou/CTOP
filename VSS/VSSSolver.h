#ifndef VSS_SOLVER_H
#define VSS_SOLVER_H

#include "Instance.h"

#include <chrono>
#include <cstdint>
#include <random>
#include <vector>

class VSSSolver {
public:
    enum class Mode { Tabu, SimulatedAnnealing };
    struct Result {
        double profit = 0.0, distance = 0.0, timeToBest = 0.0, totalTime = 0.0;
        std::vector<std::vector<int>> routes;
    };

    VSSSolver(const Instance &instance, Mode mode, std::uint32_t seed);
    Result solve();

private:
    struct Solution {
        std::vector<std::vector<int>> routes;
        std::vector<unsigned char> selected;
        double profit = 0.0, distance = 0.0;
    };
    struct Parameters { double alpha = .5, beta = .5, gamma = .5; };
    struct ConstructionCandidate {
        Solution solution;
        Parameters parameters;
    };
    struct AdaptiveState {
        Parameters center;
        int destructionMax = 3;
    };

    const Instance &data;
    Mode mode;
    std::mt19937 rng;
    int n, vehicles;
    double capacity, duration;
    std::chrono::steady_clock::time_point start;
    double bestTime = 0.0;
    double observedBestProfit = 0.0;
    double aidchScoreMin = 0.0;
    double aidchScoreMax = 0.0;
    bool diagnostics = false;
    mutable bool splitValidated = false;

    double routeTime(const std::vector<int> &route) const;
    double routeLoad(const std::vector<int> &route) const;
    bool feasible(const std::vector<int> &route) const;
    void rebuild(Solution &solution) const;
    bool profitBetter(const Solution &a, const Solution &b) const;
    void observeProfit(double profit);

    Solution bestInsertion(Solution solution, Parameters p);
    std::vector<ConstructionCandidate> constructionCandidates(
            const Solution &partial, const Parameters &center);
    Solution adaptiveConstruction(const Solution &partial, Parameters &center,
                                  bool recordScores = false);
    void destroy(Solution &solution, int count);
    void twoOpt(Solution &solution);
    Solution fullAIDCH(Solution initial);
    ConstructionCandidate fastAIDCHNeighbor(
            const Solution &solution, AdaptiveState &state);
    void diagnostic(const char *stage, const Solution &solution,
                    int first = -1, int second = -1) const;

    std::vector<int> concat(const Solution &solution);
    Solution split(const std::vector<int> &tour) const;
    std::vector<int> giantTourSearch(std::vector<int> tour);
    Solution routeSearch(Solution solution);
    Solution tabuSearch(Solution solution);
    Solution annealingSearch(Solution solution);
};

#endif
