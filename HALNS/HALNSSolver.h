#ifndef HALNS_SOLVER_H
#define HALNS_SOLVER_H

#include "Instance.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

class HALNSSolver {
public:
    struct Result {
        double profit = 0;
        double distance = 0;
        double timeToBestCpu = 0;
        double totalCpuTime = 0;
        double timeToBestWall = 0;
        double totalWallTime = 0;
        std::size_t routePoolSize = 0;
        std::vector<std::vector<int>> routes;
    };

    HALNSSolver(const Instance &, std::uint32_t seed, int segments = 150,
                int iterationsPerSegment = 0);
    Result solve();

private:
    struct Solution {
        std::vector<std::vector<int>> routes;
        std::vector<unsigned char> selected;
        double profit = 0;
        double distance = 0;
    };

    struct Adaptive {
        std::vector<double> weight;
        std::vector<double> score;
        std::vector<int> calls;

        explicit Adaptive(int n) : weight(n, 1), score(n), calls(n) {}
    };

    struct Place {
        int route = -1;
        std::size_t pos = 0;
        double delta = 0;
    };

    struct Diagnostics {
        std::uint64_t iterations = 0;
        std::uint64_t accepted = 0;
        std::uint64_t rejected = 0;
        std::uint64_t newBest = 0;
        std::uint64_t sppCalls = 0;
        std::uint64_t sppImprovements = 0;
        std::uint64_t sequenceCalls = 0;
        std::uint64_t sequenceRequested = 0;
        std::uint64_t sequenceRemoved = 0;
        std::uint64_t sequenceShortened = 0;
        std::array<std::uint64_t, 5> strategyCalls{};
        std::array<std::uint64_t, 7> removalCalls{};
        std::array<std::uint64_t, 5> insertionCalls{};
        std::array<std::uint64_t, 5> strategyBest{};
        std::array<std::uint64_t, 7> removalBest{};
        std::array<std::uint64_t, 5> insertionBest{};
        std::array<std::uint64_t, 5> localAttempts{};
        std::array<std::uint64_t, 5> localImprovements{};
    };

    const Instance &data;
    std::mt19937 rng;
    int n;
    int vehicles;
    int segments;
    int iterationsPerSegment;
    double capacity;
    double duration;
    std::clock_t cpuStart;
    std::chrono::steady_clock::time_point wallStart;
    double bestCpuTime = 0;
    double bestWallTime = 0;
    std::vector<std::vector<int>> routePool;
    std::unordered_set<std::string> routeKeys;
    bool diagnosticsEnabled = false;
    Diagnostics diagnostic;

    double routeTime(const std::vector<int> &) const;
    double routeLoad(const std::vector<int> &) const;
    double elapsedCpu() const;
    double elapsedWall() const;
    bool feasible(const std::vector<int> &) const;
    void rebuild(Solution &) const;
    bool better(const Solution &, const Solution &) const;

    Solution initialSolution();
    std::vector<Place> feasiblePlaces(const Solution &, int) const;
    int roulette(const std::vector<double> &);
    int choose(Adaptive &);

    void removeNodes(Solution &, int op, int beta);
    bool insertNode(Solution &, int node, int op);
    int selectNode(const Solution &, int strategy, int insertionOp);
    void repair(Solution &, int strategy, int insertionOp);

    void localSearch(Solution &);
    void twoOpt(Solution &);
    void addToPool(const Solution &);
    Solution solveSetPacking(const Solution &incumbent,
                             double &timeFoundCpu,
                             double &timeFoundWall) const;

    static void update(Adaptive &);
    void reward(Adaptive &, int, double);
    void printDiagnostics(int finalStale, double finalTemperature) const;
};

#endif
