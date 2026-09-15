#include "HALNSSolver.h"

#include <ilcplex/ilocplex.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace {
constexpr double EPS = 1e-9;

// CPLEX may discover the final SPP solution before it finishes proving
// optimality.  Record each strictly improving integer incumbent at the
// instant at which CPLEX reports it, instead of assigning the end of the
// complete SPP solve as the time to best.
class SppIncumbentTimer final : public IloCplex::Callback::Function {
public:
    SppIncumbentTimer(double incumbentProfit, std::clock_t cpuOrigin,
                      std::chrono::steady_clock::time_point wallOrigin)
        : highestProfit(incumbentProfit), cpuStart(cpuOrigin),
          wallStart(wallOrigin) {}

    void invoke(const IloCplex::Callback::Context &context) override {
        if (!context.inCandidate() || !context.isCandidatePoint())
            return;

        const double objective = context.getCandidateObjective();
        std::lock_guard<std::mutex> lock(mutex);
        if (objective > highestProfit + EPS) {
            highestProfit = objective;
            timeCpu = static_cast<double>(std::clock() - cpuStart) /
                      CLOCKS_PER_SEC;
            timeWall = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - wallStart)
                           .count();
            observedImprovement = true;
        }
    }

    bool observed(double finalProfit, double &cpu, double &wall) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!observedImprovement ||
            std::abs(highestProfit - finalProfit) > EPS)
            return false;
        cpu = timeCpu;
        wall = timeWall;
        return true;
    }

private:
    std::mutex mutex;
    double highestProfit;
    double timeCpu = 0;
    double timeWall = 0;
    bool observedImprovement = false;
    std::clock_t cpuStart;
    std::chrono::steady_clock::time_point wallStart;
};
}

HALNSSolver::HALNSSolver(const Instance &instance, std::uint32_t seed,
                         int segmentCount, int iterationCount)
    : data(instance), rng(seed), n(instance.nbClients),
      vehicles(instance.nbVehicles), segments(segmentCount),
      iterationsPerSegment(iterationCount),
      capacity(instance.vehicleCapacity), duration(instance.durationLimite) {
    const char *diagnosticsValue = std::getenv("HALNS_DIAGNOSTICS");
    diagnosticsEnabled = diagnosticsValue &&
                         std::string(diagnosticsValue) != "0";

    if (iterationsPerSegment <= 0)
        iterationsPerSegment = n > 250 ? 3500 : 1500;

    if (n <= 0 || vehicles <= 0 || segments <= 0 ||
        iterationsPerSegment <= 0 || capacity <= 0 || duration <= 0 ||
        !std::isfinite(capacity) || !std::isfinite(duration) ||
        data.profits.size() != static_cast<std::size_t>(n + 1) ||
        data.demands.size() != static_cast<std::size_t>(n + 1) ||
        data.serviceTime.size() != static_cast<std::size_t>(n + 1) ||
        data.dist_mtx.size() != static_cast<std::size_t>(n + 1)) {
        throw std::runtime_error("invalid or incomplete data returned by Instance");
    }
}

double HALNSSolver::routeTime(const std::vector<int> &route) const {
    double time = 0;
    int previous = 0;
    for (int customer : route) {
        time += data.dist_mtx[previous][customer] + data.serviceTime[customer];
        previous = customer;
    }
    return time + data.dist_mtx[previous][0];
}

double HALNSSolver::routeLoad(const std::vector<int> &route) const {
    double load = 0;
    for (int customer : route)
        load += data.demands[customer];
    return load;
}

double HALNSSolver::elapsedCpu() const {
    return static_cast<double>(std::clock() - cpuStart) / CLOCKS_PER_SEC;
}

double HALNSSolver::elapsedWall() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         wallStart)
        .count();
}

bool HALNSSolver::feasible(const std::vector<int> &route) const {
    return routeLoad(route) <= capacity + EPS &&
           routeTime(route) <= duration + EPS;
}

void HALNSSolver::rebuild(Solution &solution) const {
    if (solution.routes.size() != static_cast<std::size_t>(vehicles))
        throw std::runtime_error("invalid route count");

    solution.selected.assign(n + 1, 0);
    solution.profit = 0;
    solution.distance = 0;

    for (const auto &route : solution.routes) {
        if (!feasible(route))
            throw std::runtime_error("HALNS generated an infeasible route");

        solution.distance += routeTime(route);
        for (int customer : route) {
            if (customer <= 0 || customer > n || solution.selected[customer])
                throw std::runtime_error(
                    "HALNS generated duplicate/invalid customer");
            solution.selected[customer] = 1;
            solution.profit += data.profits[customer];
        }
    }
}

bool HALNSSolver::better(const Solution &lhs, const Solution &rhs) const {
    // 论文Algorithm 1中的评价函数f(s)是CTOP的收集利润。
    // 距离只在2-opt和LS4内部用于优化路线，不作为主目标的第二层标准。
    return lhs.profit > rhs.profit + EPS;
}

std::vector<HALNSSolver::Place>
HALNSSolver::feasiblePlaces(const Solution &solution, int customer) const {
    std::vector<Place> places;
    for (int routeIndex = 0; routeIndex < vehicles; ++routeIndex) {
        const auto &route = solution.routes[routeIndex];
        if (routeLoad(route) + data.demands[customer] > capacity + EPS)
            continue;

        const double oldTime = routeTime(route);
        for (std::size_t position = 0; position <= route.size(); ++position) {
            const int predecessor = position ? route[position - 1] : 0;
            const int successor = position < route.size() ? route[position] : 0;
            const double delta =
                data.dist_mtx[predecessor][customer] + data.serviceTime[customer] +
                data.dist_mtx[customer][successor] -
                data.dist_mtx[predecessor][successor];
            if (oldTime + delta <= duration + EPS)
                places.push_back({routeIndex, position, delta});
        }
    }
    return places;
}

HALNSSolver::Solution HALNSSolver::initialSolution() {
    Solution solution;
    solution.routes.resize(vehicles);
    rebuild(solution);

    for (int routeIndex = 0; routeIndex < vehicles; ++routeIndex) {
        while (true) {
            const int last = solution.routes[routeIndex].empty()
                                 ? 0
                                 : solution.routes[routeIndex].back();
            int bestCustomer = -1;
            double nearest = std::numeric_limits<double>::infinity();

            for (int customer = 1; customer <= n; ++customer) {
                if (solution.selected[customer])
                    continue;
                auto candidate = solution.routes[routeIndex];
                candidate.push_back(customer);
                if (feasible(candidate) &&
                    data.dist_mtx[last][customer] < nearest) {
                    nearest = data.dist_mtx[last][customer];
                    bestCustomer = customer;
                }
            }

            if (bestCustomer < 0)
                break;
            solution.routes[routeIndex].push_back(bestCustomer);
            rebuild(solution);
        }
    }
    return solution;
}

int HALNSSolver::roulette(const std::vector<double> &weights) {
    double sum = 0;
    for (double weight : weights)
        sum += std::max(weight, 0.0);

    if (sum <= EPS) {
        std::uniform_int_distribution<int> distribution(
            0, static_cast<int>(weights.size()) - 1);
        return distribution(rng);
    }

    std::uniform_real_distribution<double> distribution(0, sum);
    double value = distribution(rng);
    for (int index = 0; index < static_cast<int>(weights.size()); ++index) {
        value -= std::max(weights[index], 0.0);
        if (value <= 0)
            return index;
    }
    return static_cast<int>(weights.size()) - 1;
}

int HALNSSolver::choose(Adaptive &adaptive) {
    const int selected = roulette(adaptive.weight);
    ++adaptive.calls[selected];
    return selected;
}

void HALNSSolver::removeNodes(Solution &solution, int operation, int beta) {
    const int requestedBeta = beta;
    std::vector<std::pair<int, int>> positions;
    for (int route = 0; route < vehicles; ++route)
        for (int position = 0;
             position < static_cast<int>(solution.routes[route].size());
             ++position)
            positions.push_back({route, position});

    if (positions.empty())
        return;

    std::vector<int> removed;
    if (operation == 5) {
        std::vector<int> nonemptyRoutes;
        for (int route = 0; route < vehicles; ++route)
            if (!solution.routes[route].empty())
                nonemptyRoutes.push_back(route);
        std::uniform_int_distribution<int> distribution(
            0, static_cast<int>(nonemptyRoutes.size()) - 1);
        removed = solution.routes[nonemptyRoutes[distribution(rng)]];
    } else if (operation == 6) {
        ++diagnostic.sequenceCalls;
        diagnostic.sequenceRequested +=
            static_cast<std::uint64_t>(requestedBeta);
        std::vector<int> nonemptyRoutes;
        for (int route = 0; route < vehicles; ++route)
            if (static_cast<int>(solution.routes[route].size()) >= beta)
                nonemptyRoutes.push_back(route);

        // Section 3.2.7 removes beta linked nodes.  Select a route capable of
        // supplying the complete sequence instead of silently shortening it.
        // The fallback is only needed when beta is larger than every route.
        if (nonemptyRoutes.empty()) {
            int longestRoute = -1;
            for (int route = 0; route < vehicles; ++route) {
                if (!solution.routes[route].empty() &&
                    (longestRoute < 0 ||
                     solution.routes[route].size() >
                         solution.routes[longestRoute].size()))
                    longestRoute = route;
            }
            if (longestRoute >= 0)
                nonemptyRoutes.push_back(longestRoute);
        }
        std::uniform_int_distribution<int> routeDistribution(
            0, static_cast<int>(nonemptyRoutes.size()) - 1);
        const int route = nonemptyRoutes[routeDistribution(rng)];
        const int length =
            std::min(beta, static_cast<int>(solution.routes[route].size()));
        std::uniform_int_distribution<int> startDistribution(
            0, static_cast<int>(solution.routes[route].size()) - length);
        const int first = startDistribution(rng);
        removed.assign(solution.routes[route].begin() + first,
                       solution.routes[route].begin() + first + length);
        diagnostic.sequenceRemoved +=
            static_cast<std::uint64_t>(removed.size());
        if (static_cast<int>(removed.size()) < requestedBeta)
            ++diagnostic.sequenceShortened;
    } else {
        beta = std::min(beta, static_cast<int>(positions.size()));
        std::vector<double> weights(positions.size(), 1);
        for (std::size_t index = 0; index < positions.size(); ++index) {
            const int route = positions[index].first;
            const int position = positions[index].second;
            const int customer = solution.routes[route][position];
            const int predecessor =
                position ? solution.routes[route][position - 1] : 0;
            const int successor =
                position + 1 < static_cast<int>(solution.routes[route].size())
                    ? solution.routes[route][position + 1]
                    : 0;
            const double saving =
                data.dist_mtx[predecessor][customer] +
                data.serviceTime[customer] +
                data.dist_mtx[customer][successor] -
                data.dist_mtx[predecessor][successor];

            if (operation == 1)
                weights[index] = std::max(saving, EPS);
            else if (operation == 2)
                weights[index] = std::max(data.demands[customer], EPS);
            else if (operation == 3)
                weights[index] = 1.0 / std::max(data.profits[customer], EPS);
            else if (operation == 4)
                weights[index] = std::max(data.serviceTime[customer], EPS);
        }

        for (int count = 0; count < beta && !positions.empty(); ++count) {
            int selected;
            if (operation == 0) {
                std::uniform_int_distribution<int> distribution(
                    0, static_cast<int>(positions.size()) - 1);
                selected = distribution(rng);
            } else if (operation == 1) {
                selected = static_cast<int>(
                    std::max_element(weights.begin(), weights.end()) -
                    weights.begin());
            } else {
                selected = roulette(weights);
            }

            removed.push_back(solution.routes[positions[selected].first]
                                              [positions[selected].second]);
            positions.erase(positions.begin() + selected);
            weights.erase(weights.begin() + selected);
        }
    }

    for (auto &route : solution.routes) {
        route.erase(std::remove_if(route.begin(), route.end(),
                                   [&](int customer) {
                                       return std::find(removed.begin(),
                                                        removed.end(),
                                                        customer) != removed.end();
                                   }),
                    route.end());
    }
    rebuild(solution);
}

bool HALNSSolver::insertNode(Solution &solution, int customer,
                             int operation) {
    // 只能插入当前解中尚未访问的合法客户。除防止调用方误用外，
    // 这也保证所有插入算子都不会构造出重复客户。
    if (customer <= 0 || customer > n || solution.selected[customer])
        return false;

    const auto places = feasiblePlaces(solution, customer);
    if (places.empty())
        return false;

    Place selected;
    if (operation == 0) {
        selected = *std::min_element(
            places.begin(), places.end(),
            [](const auto &lhs, const auto &rhs) {
                return lhs.delta < rhs.delta;
            });
    } else if (operation == 1) {
        double minimumLoad = std::numeric_limits<double>::infinity();
        int bestRoute = -1;
        for (const auto &place : places) {
            const double load = routeLoad(solution.routes[place.route]);
            if (load < minimumLoad) {
                minimumLoad = load;
                bestRoute = place.route;
            }
        }
        selected = *std::min_element(
            places.begin(), places.end(),
            [&](const auto &lhs, const auto &rhs) {
                if (lhs.route == bestRoute && rhs.route != bestRoute)
                    return true;
                if (lhs.route != bestRoute && rhs.route == bestRoute)
                    return false;
                return lhs.delta < rhs.delta;
            });
    } else if (operation == 2 || operation == 3) {
        std::vector<int> routeOrder(vehicles);
        std::iota(routeOrder.begin(), routeOrder.end(), 0);
        std::shuffle(routeOrder.begin(), routeOrder.end(), rng);

        bool found = false;
        for (int route : routeOrder) {
            std::vector<Place> routePlaces;
            for (const auto &place : places)
                if (place.route == route)
                    routePlaces.push_back(place);
            if (routePlaces.empty())
                continue;

            if (operation == 2) {
                selected = *std::min_element(
                    routePlaces.begin(), routePlaces.end(),
                    [](const auto &lhs, const auto &rhs) {
                        return lhs.pos < rhs.pos;
                    });
            } else {
                selected = *std::max_element(
                    routePlaces.begin(), routePlaces.end(),
                    [](const auto &lhs, const auto &rhs) {
                        return lhs.pos < rhs.pos;
                    });
            }
            found = true;
            break;
        }
        if (!found)
            return false;
    } else {
        // 论文第3.3.5节：先均匀随机选择一条具有可行位置的路线，
        // 再在该路线的可行位置中均匀随机选择一个位置。
        std::vector<int> feasibleRoutes;
        for (const auto &place : places) {
            if (std::find(feasibleRoutes.begin(), feasibleRoutes.end(),
                          place.route) == feasibleRoutes.end())
                feasibleRoutes.push_back(place.route);
        }
        std::uniform_int_distribution<int> routeDistribution(
            0, static_cast<int>(feasibleRoutes.size()) - 1);
        const int route = feasibleRoutes[routeDistribution(rng)];

        std::vector<Place> routePlaces;
        for (const auto &place : places)
            if (place.route == route)
                routePlaces.push_back(place);
        std::uniform_int_distribution<int> positionDistribution(
            0, static_cast<int>(routePlaces.size()) - 1);
        selected = routePlaces[positionDistribution(rng)];
    }

    auto &route = solution.routes[selected.route];
    route.insert(route.begin() + static_cast<std::ptrdiff_t>(selected.pos),
                 customer);
    rebuild(solution);
    return true;
}

int HALNSSolver::selectNode(const Solution &solution, int strategy,
                            int insertionOperation) {
    std::vector<int> unserved;
    for (int customer = 1; customer <= n; ++customer) {
        if (!solution.selected[customer] &&
            !feasiblePlaces(solution, customer).empty())
            unserved.push_back(customer);
    }
    if (unserved.empty())
        return -1;

    if (strategy == 4) {
        std::uniform_int_distribution<int> distribution(
            0, static_cast<int>(unserved.size()) - 1);
        return unserved[distribution(rng)];
    }

    std::vector<double> weights(unserved.size());
    double maximumUnservedProfit = 0.0;
    for (int customer = 1; customer <= n; ++customer)
        if (!solution.selected[customer])
            maximumUnservedProfit =
                std::max(maximumUnservedProfit, data.profits[customer]);

    std::uniform_real_distribution<double> randomFactor(
        std::numeric_limits<double>::epsilon(), 1.0);
    for (std::size_t index = 0; index < unserved.size(); ++index) {
        const int customer = unserved[index];
        if (strategy == 1) {
            weights[index] = 1 / std::max(data.demands[customer], EPS);
        } else if (strategy == 2) {
            weights[index] = std::max(data.profits[customer], EPS);
        } else if (strategy == 3) {
            weights[index] =
                std::max(data.profits[customer] /
                             std::max(data.demands[customer], EPS),
                         EPS);
        } else {
            auto places = feasiblePlaces(solution, customer);
            std::sort(places.begin(), places.end(),
                      [](const auto &lhs, const auto &rhs) {
                          return lhs.delta < rhs.delta;
                      });
            const double preference =
                places.size() == 1
                    ? 1e6 - places[0].delta
                    : places[1].delta - places[0].delta;
            // Algorithm 2: each preference is multiplied by an independent
            // U(0,1] variate and by p_i / max_{j in V_s^-} p_j.
            weights[index] =
                preference * randomFactor(rng) *
                (data.profits[customer] /
                 std::max(maximumUnservedProfit, EPS));
        }
    }

    (void)insertionOperation;
    if (strategy == 0) {
        // 动态策略选择Algorithm 2所定义的最大偏好值客户；随机因子用于
        // 打破规律并避免循环。论文仅对策略2--4明确规定轮盘赌。
        return unserved[static_cast<std::size_t>(
            std::max_element(weights.begin(), weights.end()) - weights.begin())];
    }
    return unserved[roulette(weights)];
}

void HALNSSolver::repair(Solution &solution, int strategy, int operation) {
    while (true) {
        const int customer = selectNode(solution, strategy, operation);
        if (customer < 0 || !insertNode(solution, customer, operation))
            break;
    }
}

void HALNSSolver::twoOpt(Solution &solution) {
    for (auto &route : solution.routes) {
        bool improved = true;
        while (improved) {
            improved = false;
            const double oldTime = routeTime(route);
            for (std::size_t first = 0;
                 first + 1 < route.size() && !improved; ++first) {
                for (std::size_t last = first + 1; last < route.size(); ++last) {
                    std::reverse(route.begin() + first, route.begin() + last + 1);
                    if (routeTime(route) + EPS < oldTime) {
                        improved = true;
                        break;
                    }
                    std::reverse(route.begin() + first, route.begin() + last + 1);
                }
            }
        }
    }
    rebuild(solution);
}

void HALNSSolver::localSearch(Solution &solution) {
    ++diagnostic.localAttempts[0];
    const double distanceBeforeTwoOpt = solution.distance;
    twoOpt(solution);
    if (solution.distance + EPS < distanceBeforeTwoOpt)
        ++diagnostic.localImprovements[0];
    // 保存每个局部搜索阶段实际接受的完整可行解路线。相同客户集合的
    // 2-opt路线会由addToPool自动去重，不会向SPP加入等价列。
    addToPool(solution);

    std::vector<int> served;
    for (int customer = 1; customer <= n; ++customer)
        if (solution.selected[customer])
            served.push_back(customer);
    if (served.empty())
        return;

    auto eraseNode = [&](Solution &candidate, int customer) {
        for (auto &route : candidate.routes)
            route.erase(std::remove(route.begin(), route.end(), customer),
                        route.end());
        rebuild(candidate);
    };

    // LS2: remove one, then greedily fill using dynamic preference and
    // best-time insertion.
    {
        ++diagnostic.localAttempts[1];
        Solution candidate = solution;
        std::shuffle(served.begin(), served.end(), rng);
        eraseNode(candidate, served.front());
        repair(candidate, 0, 0);
        if (better(candidate, solution)) {
            ++diagnostic.localImprovements[1];
            solution = std::move(candidate);
            addToPool(solution);
        }
    }

    // LS3: one-for-one random profitable replacement.
    served.clear();
    for (int customer = 1; customer <= n; ++customer)
        if (solution.selected[customer])
            served.push_back(customer);
    if (!served.empty()) {
        ++diagnostic.localAttempts[2];
        std::uniform_int_distribution<int> distribution(
            0, static_cast<int>(served.size()) - 1);
        const int oldCustomer = served[distribution(rng)];
        Solution candidate = solution;
        eraseNode(candidate, oldCustomer);

        std::vector<int> unserved;
        for (int customer = 1; customer <= n; ++customer) {
            if (!candidate.selected[customer] &&
                data.profits[customer] > data.profits[oldCustomer] + EPS &&
                !feasiblePlaces(candidate, customer).empty())
                unserved.push_back(customer);
        }
        if (!unserved.empty()) {
            // 论文LS3只随机选择一个利润更高的未访问客户并尝试一次；
            // 插入失败时不继续改选其他客户。
            std::uniform_int_distribution<int> customerDistribution(
                0, static_cast<int>(unserved.size()) - 1);
            const int customer = unserved[customerDistribution(rng)];
            insertNode(candidate, customer, 4);
        }
        if (better(candidate, solution)) {
            ++diagnostic.localImprovements[2];
            solution = std::move(candidate);
            addToPool(solution);
        }
    }

    // LS4: swap two visited customers only when total travel time decreases.
    // The paper does not require the two customers to belong to different
    // routes, so both intra-route and inter-route swaps are considered.
    std::vector<int> nonemptyRoutes;
    for (int route = 0; route < vehicles; ++route)
        if (!solution.routes[route].empty())
            nonemptyRoutes.push_back(route);
    if (!nonemptyRoutes.empty()) {
        ++diagnostic.localAttempts[3];

        // Inspect intra-route and inter-route swaps in random order and apply
        // the first feasible move reducing total travel time. Delta checks are
        // O(1), so no complete candidate solution is rebuilt per pair.
        std::shuffle(nonemptyRoutes.begin(), nonemptyRoutes.end(), rng);
        std::vector<double> routeTimes(vehicles);
        std::vector<double> routeLoads(vehicles);
        for (int route : nonemptyRoutes) {
            routeTimes[route] = routeTime(solution.routes[route]);
            routeLoads[route] = routeLoad(solution.routes[route]);
        }

        auto replacementDelta = [&](int routeIndex, int position,
                                    int replacement) {
            const auto &route = solution.routes[routeIndex];
            const int original = route[static_cast<std::size_t>(position)];
            const int predecessor = position ? route[position - 1] : 0;
            const int successor =
                position + 1 < static_cast<int>(route.size())
                    ? route[position + 1]
                    : 0;
            return data.dist_mtx[predecessor][replacement] +
                       data.serviceTime[replacement] +
                       data.dist_mtx[replacement][successor] -
                   data.dist_mtx[predecessor][original] -
                   data.serviceTime[original] -
                   data.dist_mtx[original][successor];
        };

        // Swapping two customers in the same route needs a dedicated delta:
        // treating the two replacements independently would double-count
        // shared arcs when the positions are adjacent.
        auto intraRouteSwapDelta = [&](int routeIndex, int firstPosition,
                                       int secondPosition) {
            if (firstPosition > secondPosition)
                std::swap(firstPosition, secondPosition);

            const auto &route = solution.routes[routeIndex];
            const int firstCustomer = route[firstPosition];
            const int secondCustomer = route[secondPosition];
            const int firstPredecessor =
                firstPosition ? route[firstPosition - 1] : 0;
            const int secondSuccessor =
                secondPosition + 1 < static_cast<int>(route.size())
                    ? route[secondPosition + 1]
                    : 0;

            if (secondPosition == firstPosition + 1) {
                return data.dist_mtx[firstPredecessor][secondCustomer] +
                           data.dist_mtx[secondCustomer][firstCustomer] +
                           data.dist_mtx[firstCustomer][secondSuccessor] -
                       data.dist_mtx[firstPredecessor][firstCustomer] -
                           data.dist_mtx[firstCustomer][secondCustomer] -
                           data.dist_mtx[secondCustomer][secondSuccessor];
            }

            const int firstSuccessor = route[firstPosition + 1];
            const int secondPredecessor = route[secondPosition - 1];
            return data.dist_mtx[firstPredecessor][secondCustomer] +
                       data.dist_mtx[secondCustomer][firstSuccessor] +
                       data.dist_mtx[secondPredecessor][firstCustomer] +
                       data.dist_mtx[firstCustomer][secondSuccessor] -
                   data.dist_mtx[firstPredecessor][firstCustomer] -
                       data.dist_mtx[firstCustomer][firstSuccessor] -
                       data.dist_mtx[secondPredecessor][secondCustomer] -
                       data.dist_mtx[secondCustomer][secondSuccessor];
        };

        bool applied = false;
        for (std::size_t firstRouteIndex = 0;
             firstRouteIndex < nonemptyRoutes.size() && !applied;
             ++firstRouteIndex) {
            for (std::size_t secondRouteIndex = firstRouteIndex;
                 secondRouteIndex < nonemptyRoutes.size() && !applied;
                 ++secondRouteIndex) {
                const int firstRoute = nonemptyRoutes[firstRouteIndex];
                const int secondRoute = nonemptyRoutes[secondRouteIndex];
                std::vector<int> firstPositions(
                    solution.routes[firstRoute].size());
                std::vector<int> secondPositions(
                    solution.routes[secondRoute].size());
                std::iota(firstPositions.begin(), firstPositions.end(), 0);
                std::iota(secondPositions.begin(), secondPositions.end(), 0);
                std::shuffle(firstPositions.begin(), firstPositions.end(), rng);
                std::shuffle(secondPositions.begin(), secondPositions.end(), rng);

                for (int firstPosition : firstPositions) {
                    const int firstCustomer =
                        solution.routes[firstRoute][firstPosition];
                    for (int secondPosition : secondPositions) {
                        if (firstRoute == secondRoute &&
                            firstPosition >= secondPosition)
                            continue;

                        const int secondCustomer =
                            solution.routes[secondRoute][secondPosition];

                        if (firstRoute == secondRoute) {
                            const double delta = intraRouteSwapDelta(
                                firstRoute, firstPosition, secondPosition);
                            if (delta >= -EPS)
                                continue;

                            std::swap(
                                solution.routes[firstRoute][firstPosition],
                                solution.routes[firstRoute][secondPosition]);
                            rebuild(solution);
                            ++diagnostic.localImprovements[3];
                            addToPool(solution);
                            applied = true;
                            break;
                        }

                        const double firstDelta = replacementDelta(
                            firstRoute, firstPosition, secondCustomer);
                        const double secondDelta = replacementDelta(
                            secondRoute, secondPosition, firstCustomer);
                        const double firstLoad =
                            routeLoads[firstRoute] -
                            data.demands[firstCustomer] +
                            data.demands[secondCustomer];
                        const double secondLoad =
                            routeLoads[secondRoute] -
                            data.demands[secondCustomer] +
                            data.demands[firstCustomer];

                        if (firstLoad > capacity + EPS ||
                            secondLoad > capacity + EPS ||
                            routeTimes[firstRoute] + firstDelta >
                                duration + EPS ||
                            routeTimes[secondRoute] + secondDelta >
                                duration + EPS ||
                            firstDelta + secondDelta >= -EPS)
                            continue;

                        std::swap(solution.routes[firstRoute][firstPosition],
                                  solution.routes[secondRoute][secondPosition]);
                        rebuild(solution);
                        ++diagnostic.localImprovements[3];
                        addToPool(solution);
                        applied = true;
                        break;
                    }
                    if (applied)
                        break;
                }
            }
        }
    }

    // LS5: remove two, insert one unserved, then reinsert removed nodes.
    served.clear();
    for (int customer = 1; customer <= n; ++customer)
        if (solution.selected[customer])
            served.push_back(customer);
    if (served.size() >= 2) {
        ++diagnostic.localAttempts[4];
        std::shuffle(served.begin(), served.end(), rng);
        const int first = served[0];
        const int second = served[1];
        Solution candidate = solution;

        // 论文中的“another non-inserted node”是执行本次移除之前就未被
        // 服务的客户，因此必须在移除first和second之前保存该集合。
        std::vector<int> originallyUnserved;
        for (int customer = 1; customer <= n; ++customer)
            if (!solution.selected[customer])
                originallyUnserved.push_back(customer);

        eraseNode(candidate, first);
        eraseNode(candidate, second);

        if (!originallyUnserved.empty()) {
            std::uniform_int_distribution<int> distribution(
                0, static_cast<int>(originallyUnserved.size()) - 1);
            const int customer = originallyUnserved[distribution(rng)];
            // 按论文描述随机选择一个客户并尝试插入；若它没有可行位置，
            // 本次新客户插入失败，不继续改选其他客户。
            insertNode(candidate, customer, 4);
        }
        insertNode(candidate, first, 0);
        insertNode(candidate, second, 0);
        if (better(candidate, solution)) {
            ++diagnostic.localImprovements[4];
            solution = std::move(candidate);
            addToPool(solution);
        }
    }
    rebuild(solution);
}

void HALNSSolver::addToPool(const Solution &solution) {
    for (auto route : solution.routes) {
        if (route.empty())
            continue;
        std::vector<int> canonical = route;
        std::sort(canonical.begin(), canonical.end());
        std::ostringstream key;
        for (int customer : canonical)
            key << customer << ',';
        if (routeKeys.insert(key.str()).second)
            routePool.push_back(std::move(route));
    }
}

HALNSSolver::Solution
HALNSSolver::solveSetPacking(const Solution &incumbent,
                             double &timeFoundCpu,
                             double &timeFoundWall) const {
    IloEnv environment;
    try {
        IloModel model(environment);
        const int realRoutes = static_cast<int>(routePool.size());

        // 论文SPP只在HALNS实际生成的路线集合R上定义变量。
        IloBoolVarArray selectedRoutes(environment, realRoutes);
        IloExpr objective(environment);
        for (int route = 0; route < realRoutes; ++route)
            for (int customer : routePool[route])
                objective += data.profits[customer] * selectedRoutes[route];
        model.add(IloMaximize(environment, objective));
        objective.end();

        for (int customer = 1; customer <= n; ++customer) {
            IloExpr served(environment);
            for (int route = 0; route < realRoutes; ++route) {
                if (std::find(routePool[route].begin(), routePool[route].end(),
                              customer) != routePool[route].end())
                    served += selectedRoutes[route];
            }
            model.add(served <= 1);
            served.end();
        }

        IloExpr routeCount(environment);
        for (int route = 0; route < realRoutes; ++route)
            routeCount += selectedRoutes[route];
        model.add(routeCount == vehicles);
        routeCount.end();

        IloCplex cplex(model);
        cplex.setOut(environment.getNullStream());
        cplex.setWarning(environment.getNullStream());

        SppIncumbentTimer incumbentTimer(incumbent.profit, cpuStart, wallStart);
        cplex.use(&incumbentTimer,
                  IloCplex::Callback::Context::Id::Candidate);

        // The article initializes the SPP with the current ALNS best solution.
        IloNumVarArray startVariables(environment);
        IloNumArray startValues(environment);
        std::vector<unsigned char> matched(realRoutes, 0);
        for (const auto &route : incumbent.routes) {
            if (route.empty())
                continue;
            std::vector<int> key = route;
            std::sort(key.begin(), key.end());
            for (int poolIndex = 0; poolIndex < realRoutes; ++poolIndex) {
                if (matched[poolIndex])
                    continue;
                std::vector<int> candidate = routePool[poolIndex];
                std::sort(candidate.begin(), candidate.end());
                if (candidate == key) {
                    startVariables.add(selectedRoutes[poolIndex]);
                    startValues.add(1);
                    matched[poolIndex] = 1;
                    break;
                }
            }
        }
        if (startVariables.getSize() > 0) {
            cplex.addMIPStart(startVariables, startValues,
                              IloCplex::MIPStartAuto, "HALNS incumbent");
        }
        startVariables.end();
        startValues.end();

        if (!cplex.solve() || cplex.getStatus() != IloAlgorithm::Optimal) {
            throw std::runtime_error(
                "CPLEX did not solve the HALNS set-packing model to optimality");
        }

        Solution result;
        result.routes.resize(vehicles);
        int target = 0;
        for (int route = 0; route < realRoutes && target < vehicles; ++route) {
            if (cplex.getValue(selectedRoutes[route]) > .5)
                result.routes[target++] = routePool[route];
        }
        rebuild(result);

        if (better(result, incumbent) &&
            !incumbentTimer.observed(result.profit, timeFoundCpu,
                                     timeFoundWall)) {
            // Presolve can occasionally establish the solution without
            // issuing a candidate callback.  Falling back to the return time
            // is conservative and is still more accurate than losing the
            // improvement timestamp entirely.
            timeFoundCpu = elapsedCpu();
            timeFoundWall = elapsedWall();
        }
        environment.end();
        return better(result, incumbent) ? result : incumbent;
    } catch (const IloException &error) {
        const std::string message = error.getMessage();
        environment.end();
        throw std::runtime_error("CPLEX SPP error: " + message);
    } catch (...) {
        environment.end();
        throw;
    }
}

void HALNSSolver::reward(Adaptive &adaptive, int operation, double value) {
    adaptive.score[operation] += value;
}

void HALNSSolver::update(Adaptive &adaptive) {
    for (std::size_t operation = 0; operation < adaptive.weight.size();
         ++operation) {
        if (adaptive.calls[operation]) {
            adaptive.weight[operation] =
                .8 * (adaptive.score[operation] / adaptive.calls[operation]) +
                .2 * adaptive.weight[operation];
        }
        adaptive.score[operation] = 0;
        adaptive.calls[operation] = 0;
    }
}

void HALNSSolver::printDiagnostics(int finalStale,
                                   double finalTemperature) const {
    if (!diagnosticsEnabled)
        return;

    static const std::array<const char *, 5> strategyNames{
        "dynamic", "lowest_volume", "highest_profit",
        "profit_per_volume", "random"};
    static const std::array<const char *, 7> removalNames{
        "random", "largest_saving", "largest_demand", "lowest_profit",
        "largest_service", "random_route", "sequence"};
    static const std::array<const char *, 5> insertionNames{
        "best_overall", "least_loaded", "first", "last", "random"};
    static const std::array<const char *, 5> localNames{
        "two_opt", "remove_one_fill", "one_for_one", "swap_positions",
        "remove_two_insert_one"};

    std::cerr << "HALNS_DIAG summary iterations=" << diagnostic.iterations
              << " accepted=" << diagnostic.accepted
              << " rejected=" << diagnostic.rejected
              << " alns_new_best=" << diagnostic.newBest
              << " final_stale=" << finalStale
              << " final_temperature=" << finalTemperature
              << " route_pool=" << routePool.size()
              << " spp_calls=" << diagnostic.sppCalls
              << " spp_improvements=" << diagnostic.sppImprovements << '\n';

    std::cerr << "HALNS_DIAG sequence calls=" << diagnostic.sequenceCalls
              << " requested=" << diagnostic.sequenceRequested
              << " removed=" << diagnostic.sequenceRemoved
              << " shortened_calls=" << diagnostic.sequenceShortened << '\n';

    for (std::size_t i = 0; i < localNames.size(); ++i) {
        const double rate = diagnostic.localAttempts[i]
                                ? 100.0 * diagnostic.localImprovements[i] /
                                      diagnostic.localAttempts[i]
                                : 0.0;
        std::cerr << "HALNS_DIAG local name=" << localNames[i]
                  << " attempts=" << diagnostic.localAttempts[i]
                  << " improvements=" << diagnostic.localImprovements[i]
                  << " rate_percent=" << rate << '\n';
    }

    for (std::size_t i = 0; i < strategyNames.size(); ++i)
        std::cerr << "HALNS_DIAG strategy name=" << strategyNames[i]
                  << " calls=" << diagnostic.strategyCalls[i]
                  << " new_best=" << diagnostic.strategyBest[i] << '\n';
    for (std::size_t i = 0; i < removalNames.size(); ++i)
        std::cerr << "HALNS_DIAG removal name=" << removalNames[i]
                  << " calls=" << diagnostic.removalCalls[i]
                  << " new_best=" << diagnostic.removalBest[i] << '\n';
    for (std::size_t i = 0; i < insertionNames.size(); ++i)
        std::cerr << "HALNS_DIAG insertion name=" << insertionNames[i]
                  << " calls=" << diagnostic.insertionCalls[i]
                  << " new_best=" << diagnostic.insertionBest[i] << '\n';
}

HALNSSolver::Result HALNSSolver::solve() {
    // 论文实验报告CPU time，因此这里记录进程CPU时间而非墙钟时间。
    cpuStart = std::clock();
    wallStart = std::chrono::steady_clock::now();
    Solution admissible = initialSolution();  //初始化
    Solution best = admissible;
    bestCpuTime = elapsedCpu();
    bestWallTime = elapsedWall();
    addToPool(admissible);  //加入到池中

    double temperature = 100;
    int stale = 0;
    Adaptive strategies(5);  //5中策略
    Adaptive removals(7);  //7个移除算子
    Adaptive insertions(5);  //5个插入算子
    std::uniform_real_distribution<double> uniform(0, 1);      
    for (int segment = 0; segment < segments; ++segment) {
        for (int iteration = 0;
             iteration < iterationsPerSegment && stale < 5000; ++iteration) {
            ++diagnostic.iterations;
            Solution current = admissible;
            int selectedCount = 0;
            for (const auto &route : current.routes)
                selectedCount += static_cast<int>(route.size());

            const int maximumRemoval =
                std::max(1, static_cast<int>(std::floor(.15 * selectedCount))); // 15% of selected nodes
            std::uniform_int_distribution<int> removalCount(1, maximumRemoval); // 随机选择移除客户数量
            const int beta = removalCount(rng);  // 随机选择移除客户数量
            const int strategy = choose(strategies); 
            const int removal = choose(removals);
            const int insertion = choose(insertions);
            ++diagnostic.strategyCalls[static_cast<std::size_t>(strategy)];
            ++diagnostic.removalCalls[static_cast<std::size_t>(removal)];
            ++diagnostic.insertionCalls[static_cast<std::size_t>(insertion)];

            removeNodes(current, removal, beta);
            repair(current, strategy, insertion);
            addToPool(current);

            const double difference = current.profit - admissible.profit;
            const bool accepted =
                difference >= -EPS ||
                uniform(rng) <=
                    std::exp(difference / std::max(temperature, 1e-12));
            double points = 1;

            if (accepted) {
                ++diagnostic.accepted;
                if (difference > EPS)
                    localSearch(current);
                addToPool(current);

                if (better(current, best)) {
                    ++diagnostic.newBest;
                    ++diagnostic.strategyBest[static_cast<std::size_t>(strategy)];
                    ++diagnostic.removalBest[static_cast<std::size_t>(removal)];
                    ++diagnostic.insertionBest[static_cast<std::size_t>(insertion)];
                    best = current;
                    bestCpuTime = elapsedCpu();
                    bestWallTime = elapsedWall();
                    stale = 0;
                    points = 8;
                } else {
                    ++stale;
                    points = better(current, admissible) ? 4 : 1;
                }
                admissible = std::move(current);
            } else {
                ++diagnostic.rejected;
                ++stale;
            }

            reward(strategies, strategy, points);
            reward(removals, removal, points);
            reward(insertions, insertion, points);
            temperature *= .9997;

            if (temperature <= .0001) {
                temperature = 100;
                ++diagnostic.sppCalls;
                const double profitBeforeSpp = best.profit;
                const std::size_t poolBeforeSpp = routePool.size();
                double packedCpuTime = 0;
                double packedWallTime = 0;
                Solution packed = solveSetPacking(
                    best, packedCpuTime, packedWallTime);
                if (better(packed, best)) {
                    ++diagnostic.sppImprovements;
                    best = packed;
                    admissible = packed;
                    stale = 0;
                    bestCpuTime = packedCpuTime;
                    bestWallTime = packedWallTime;
                }
                if (diagnosticsEnabled)
                    std::cerr << "HALNS_DIAG spp call="
                              << diagnostic.sppCalls
                              << " pool=" << poolBeforeSpp
                              << " before=" << profitBeforeSpp
                              << " after=" << best.profit << '\n';
            }
        }

        update(strategies);
        update(removals);
        update(insertions);
        // The no-improvement counter is local to a run segment.  Resetting it
        // here lets every segment start with a fresh stagnation allowance.
        stale = 0;
    }

    ++diagnostic.sppCalls;
    const double profitBeforeFinalSpp = best.profit;
    const std::size_t poolBeforeFinalSpp = routePool.size();
    double packedCpuTime = 0;
    double packedWallTime = 0;
    Solution packed = solveSetPacking(best, packedCpuTime, packedWallTime);
    if (better(packed, best)) {
        ++diagnostic.sppImprovements;
        best = packed;
        bestCpuTime = packedCpuTime;
        bestWallTime = packedWallTime;
    }
    if (diagnosticsEnabled)
        std::cerr << "HALNS_DIAG spp call=" << diagnostic.sppCalls
                  << " pool=" << poolBeforeFinalSpp
                  << " before=" << profitBeforeFinalSpp
                  << " after=" << best.profit << '\n';

    printDiagnostics(stale, temperature);

    Result result;
    result.profit = best.profit;
    result.distance = best.distance;
    result.routes = best.routes;
    result.timeToBestCpu = bestCpuTime;
    result.totalCpuTime = elapsedCpu();
    result.timeToBestWall = bestWallTime;
    result.totalWallTime = elapsedWall();
    result.routePoolSize = routePool.size();
    return result;
}
