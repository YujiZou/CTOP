#ifndef BIF_SOLVER_H
#define BIF_SOLVER_H

#include "Instance.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class BiFSolver
{
public:
    // 论文中的两个 BiF&F 版本：算法结构相同，主要区别是搜索预算。
    enum class Mode { Fast, Slow };

    // 对外返回的结果。routes 中不保存仓库 0，输出时由 main.cpp 补上。
    struct Result
    {
        double profit = 0.0;
        double distance = 0.0;
        double timeToBest = 0.0;
        double totalTime = 0.0;
        std::vector<std::vector<int>> routes;
    };

    BiFSolver(const Instance &instance, Mode mode, std::uint32_t seed);
    Result solve();

private:
    // 算法内部的 CTOP 解。路线改变后由 rebuild() 统一刷新全部缓存字段。
    struct Solution
    {
        std::vector<std::vector<int>> routes;
        std::vector<unsigned char> selected;
        std::vector<double> routeDistance;
        std::vector<double> routeLoad;
        double profit = 0.0;
        double distance = 0.0;
    };

    // 三个利润邻域：移除/插入客户数量分别为 1-1、2-1 和 1-2。
    enum class ProfitNeighborhood { Replace11, Replace21, Replace12 };

    // 轻量移动描述：只保存变动位置和增量结果，不为每个候选复制完整解。
    // 论文要求的 forward/reverse attributes 仍随移动一并记录。
    struct Move
    {
        std::vector<int> removed;
        std::vector<int> inserted;
        std::vector<std::size_t> removePositions;
        std::vector<std::size_t> insertEdges;
        bool firstBeforeSecond = true;
        int routeIndex = -1;
        double routeDistanceAfter = 0.0;
        double routeLoadAfter = 0.0;
        double candidateProfit = 0.0;
        double candidateDistance = 0.0;
        std::string forwardKey;
        std::string reverseKey;
    };

    // F&F 为每条路线和每个利润邻域保存一个有序候选前缀。
    // guaranteed 表示其中从头开始有多少个候选确定属于真实的最好前缀；
    // exhaustive=true 表示该路线的相应邻域已经被完整保存。
    struct MoveCache
    {
        std::vector<Move> moves;
        std::size_t guaranteed = 0;
        bool exhaustive = false;
    };

    // Filter-and-Fan 搜索树节点，forbidden 保存当前搜索路径上的禁用逆移动。
    struct TreeNode
    {
        Solution solution;
        std::unordered_set<std::string> forbidden;
        std::vector<std::array<MoveCache, 3>> routeCaches;
    };

    const Instance &data;
    Mode mode;
    std::mt19937 rng;
    int n;
    int vehicleCount;
    double capacity;
    double duration;
    int tabuTenure = 30;       // forward/reverse move attributes 的 tabu tenure
    int tabuNoImproveLimit;
    int fanWidth = 100;        // 每层最多保留的候选解数
    int fanDepth = 100;        // 最大展开深度
    std::chrono::steady_clock::time_point startTime;
    double bestTime = 0.0;

    // 基础评价、可行性判断及解的一致性检查。
    double routeDistance(const std::vector<int> &route) const;
    double routeLoad(const std::vector<int> &route) const;
    bool routeFeasible(const std::vector<int> &route) const;
    void rebuild(Solution &solution) const;
    bool better(const Solution &lhs, const Solution &rhs) const;
    bool sameQuality(const Solution &lhs, const Solution &rhs) const;
    bool better(const Move &lhs, const Move &rhs) const;
    std::string solutionKey(const Solution &solution) const;
    std::string moveKey(const std::vector<int> &removed,
                        const std::vector<int> &inserted) const;
    bool rebaseMove(Move &move, const Solution &solution) const;
    Solution applyMove(const Solution &solution, const Move &move) const;

    // 论文第 3.2 节的并行插入初始化：按利润优先，插入最小成本可行位置。
    Solution constructInitial();
    // VND 不降低利润，只缩短距离，为利润替换邻域释放时长余量。
    Solution variableNeighborhoodDescent(
            Solution solution, const std::vector<unsigned char> *initiallyAffected = nullptr);
    bool improveTwoOpt(Solution &solution, const std::vector<unsigned char> *affected);
    bool improveExchange(Solution &solution, const std::vector<unsigned char> *affected);
    bool improveRelocate(Solution &solution, const std::vector<unsigned char> *affected);

    // 枚举指定利润邻域，只保留质量最好的 limit 个可接受移动。
    std::vector<Move> enumerateMoves(const Solution &solution,
                                     ProfitNeighborhood neighborhood,
                                     std::size_t limit,
                                     const std::unordered_map<std::string, int> *tabuUntil,
                                     int iteration,
                                     double aspirationProfit,
                                     const std::unordered_set<std::string> *forbidden,
                                     int onlyRoute = -1,
                                     const std::unordered_set<int> *requiredInserted = nullptr);
    void considerMove(std::vector<Move> &moves, Move move, std::size_t limit) const;
    bool admissible(const Move &move,
                    const std::unordered_map<std::string, int> *tabuUntil,
                    int iteration,
                    double aspirationProfit,
                    const std::unordered_set<std::string> *forbidden) const;

    // BiF&F 的两个主要搜索阶段。
    Solution tabuSearch(const Solution &initial);
    Solution filterAndFan(const Solution &root);
    void recordBestTime();
};

#endif
