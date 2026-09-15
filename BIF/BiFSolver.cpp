#include "BiFSolver.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace
{
constexpr double EPS = 1e-9;

// vector::erase 使用有符号的 difference_type，这里集中处理下标类型转换。
template <typename T>
void eraseAt(std::vector<T> &values, std::size_t position)
{
    values.erase(values.begin() + static_cast<std::ptrdiff_t>(position));
}
}

BiFSolver::BiFSolver(const Instance &instance, Mode selectedMode, std::uint32_t seed)
    : data(instance), mode(selectedMode), rng(seed), n(instance.nbClients),
      vehicleCount(instance.nbVehicles), capacity(instance.vehicleCapacity),
      duration(instance.durationLimite)
{
    if(n <= 0 || vehicleCount <= 0)
        throw std::runtime_error("invalid customer or vehicle count returned by Instance");
    if(static_cast<int>(data.demands.size()) != n + 1 ||
       static_cast<int>(data.serviceTime.size()) != n + 1 ||
       static_cast<int>(data.profits.size()) != n + 1 ||
       static_cast<int>(data.dist_mtx.size()) != n + 1)
        throw std::runtime_error("incomplete instance data returned by Instance");
    if(!std::isfinite(capacity) || !std::isfinite(duration) ||
       capacity <= 0.0 || duration <= 0.0)
        throw std::runtime_error("invalid capacity or duration returned by Instance");

    // 快慢版本使用相同搜索结构，只改变连续未改进的停止预算。
    if(mode == Mode::Fast) tabuNoImproveLimit = 750;
    else tabuNoImproveLimit = n > 200 ? 5000 : 10000;
}

double BiFSolver::routeDistance(const std::vector<int> &route) const
{
    double value = 0.0;
    int previous = 0;
    for(int node : route)
    {
        value += data.dist_mtx[previous][node] + data.serviceTime[node];
        previous = node;
    }
    value += data.dist_mtx[previous][0];
    return value;
}

double BiFSolver::routeLoad(const std::vector<int> &route) const
{
    double value = 0.0;
    for(int node : route) value += data.demands[node];
    return value;
}

bool BiFSolver::routeFeasible(const std::vector<int> &route) const
{
    return routeLoad(route) <= capacity + EPS &&
           routeDistance(route) <= duration + EPS;
}

void BiFSolver::rebuild(Solution &solution) const
{
    // 路线变化后从 routes 完整重建评价值；同时检查重复客户和约束可行性。
    solution.selected.assign(n + 1, 0);
    solution.routeDistance.assign(vehicleCount, 0.0);
    solution.routeLoad.assign(vehicleCount, 0.0);
    solution.profit = 0.0;
    solution.distance = 0.0;

    if(static_cast<int>(solution.routes.size()) != vehicleCount)
        throw std::runtime_error("solution has an invalid number of routes");

    for(int r = 0; r < vehicleCount; ++r)
    {
        solution.routeDistance[r] = routeDistance(solution.routes[r]);
        solution.routeLoad[r] = routeLoad(solution.routes[r]);
        solution.distance += solution.routeDistance[r];
        if(solution.routeDistance[r] > duration + EPS ||
           solution.routeLoad[r] > capacity + EPS)
            throw std::runtime_error("an internal move produced an infeasible route");

        for(int node : solution.routes[r])
        {
            if(node <= 0 || node > n || solution.selected[node])
                throw std::runtime_error("an internal move produced duplicate/invalid customers");
            solution.selected[node] = 1;
            solution.profit += data.profits[node];
        }
    }
}

bool BiFSolver::better(const Solution &lhs, const Solution &rhs) const
{
    // CTOP 首先最大化利润；利润相同时选择总距离更短的解。
    if(lhs.profit > rhs.profit + EPS) return true;
    if(lhs.profit + EPS < rhs.profit) return false;
    return lhs.distance + EPS < rhs.distance;
}

bool BiFSolver::sameQuality(const Solution &lhs, const Solution &rhs) const
{
    return std::abs(lhs.profit - rhs.profit) <= EPS &&
           std::abs(lhs.distance - rhs.distance) <= EPS;
}

bool BiFSolver::better(const Move &lhs, const Move &rhs) const
{
    if(lhs.candidateProfit > rhs.candidateProfit + EPS) return true;
    if(lhs.candidateProfit + EPS < rhs.candidateProfit) return false;
    return lhs.candidateDistance + EPS < rhs.candidateDistance;
}

std::string BiFSolver::solutionKey(const Solution &solution) const
{
    std::ostringstream out;
    for(const auto &route : solution.routes)
    {
        out << '|';
        for(int node : route) out << node << ',';
    }
    return out.str();
}

std::string BiFSolver::moveKey(const std::vector<int> &removed,
                               const std::vector<int> &inserted) const
{
    // 排序后再编码，使同一客户集合产生的移动具有唯一键值。
    std::vector<int> first = removed;
    std::vector<int> second = inserted;
    std::sort(first.begin(), first.end());
    std::sort(second.begin(), second.end());
    std::ostringstream out;
    out << 'R';
    for(int node : first) out << node << ',';
    out << "I";
    for(int node : second) out << node << ',';
    return out.str();
}

bool BiFSolver::rebaseMove(Move &move, const Solution &solution) const
{
    if(move.routeIndex < 0 || move.routeIndex >= vehicleCount) return false;
    const auto &route = solution.routes[move.routeIndex];
    if(move.removePositions.size() != move.removed.size()) return false;
    for(std::size_t k = 0; k < move.removed.size(); ++k)
        if(move.removePositions[k] >= route.size() ||
           route[move.removePositions[k]] != move.removed[k]) return false;
    for(int node : move.inserted) if(solution.selected[node]) return false;
    move.candidateProfit = solution.profit;
    for(int node : move.removed) move.candidateProfit -= data.profits[node];
    for(int node : move.inserted) move.candidateProfit += data.profits[node];
    move.candidateDistance = solution.distance - solution.routeDistance[move.routeIndex] +
                             move.routeDistanceAfter;
    return true;
}

BiFSolver::Solution BiFSolver::applyMove(const Solution &solution, const Move &move) const
{
    Solution result = solution; // 每轮只对最终选中的一个移动复制完整解。
    auto &route = result.routes[move.routeIndex];
    for(auto it = move.removePositions.rbegin(); it != move.removePositions.rend(); ++it)
        eraseAt(route, *it);

    if(move.inserted.size() == 1)
    {
        route.insert(route.begin() + static_cast<std::ptrdiff_t>(move.insertEdges[0]),
                     move.inserted[0]);
    }
    else if(move.inserted.size() == 2)
    {
        const std::size_t e1 = move.insertEdges[0], e2 = move.insertEdges[1];
        if(e1 == e2)
        {
            if(move.firstBeforeSecond)
            {
                route.insert(route.begin() + static_cast<std::ptrdiff_t>(e1), move.inserted[1]);
                route.insert(route.begin() + static_cast<std::ptrdiff_t>(e1), move.inserted[0]);
            }
            else
            {
                route.insert(route.begin() + static_cast<std::ptrdiff_t>(e1), move.inserted[0]);
                route.insert(route.begin() + static_cast<std::ptrdiff_t>(e1), move.inserted[1]);
            }
        }
        else if(e1 < e2)
        {
            route.insert(route.begin() + static_cast<std::ptrdiff_t>(e2), move.inserted[1]);
            route.insert(route.begin() + static_cast<std::ptrdiff_t>(e1), move.inserted[0]);
        }
        else
        {
            route.insert(route.begin() + static_cast<std::ptrdiff_t>(e1), move.inserted[0]);
            route.insert(route.begin() + static_cast<std::ptrdiff_t>(e2), move.inserted[1]);
        }
    }

    rebuild(result); // 只在移动被选中后进行一次安全校验，不用于候选评价。
    if(std::abs(result.routeDistance[move.routeIndex] - move.routeDistanceAfter) > 1e-6 ||
       std::abs(result.routeLoad[move.routeIndex] - move.routeLoadAfter) > 1e-6)
        throw std::runtime_error("incremental profit-move evaluation mismatch");
    return result;
}

BiFSolver::Solution BiFSolver::constructInitial()
{
    // 从 m 条空路线开始，只考虑能够单独完成仓库往返的客户。
    Solution solution;
    solution.routes.resize(vehicleCount);
    rebuild(solution);

    std::vector<int> candidates;
    for(int node = 1; node <= n; ++node)
        if(data.dist_mtx[0][node] <= duration / 2.0 + EPS)
            candidates.push_back(node);

    // 高利润客户优先；利润相同则按编号排列，使排序结果确定。
    std::stable_sort(candidates.begin(), candidates.end(), [this](int lhs, int rhs)
    {
        if(std::abs(data.profits[lhs] - data.profits[rhs]) > EPS)
            return data.profits[lhs] > data.profits[rhs];
        return lhs < rhs;
    });

    for(int node : candidates)
    {
        // 在所有路线和位置中寻找距离增量最小的可行插入位置。
        double bestIncrease = std::numeric_limits<double>::infinity();
        int bestRoute = -1;
        std::size_t bestPosition = 0;
        for(int r = 0; r < vehicleCount; ++r)
        {
            if(solution.routeLoad[r] + data.demands[node] > capacity + EPS) continue;
            for(std::size_t p = 0; p <= solution.routes[r].size(); ++p)
            {
                const int previous = p == 0 ? 0 : solution.routes[r][p - 1];
                const int next = p == solution.routes[r].size() ? 0 : solution.routes[r][p];
                const double increase = data.dist_mtx[previous][node] + data.serviceTime[node] +
                                        data.dist_mtx[node][next] - data.dist_mtx[previous][next];
                if(solution.routeDistance[r] + increase <= duration + EPS &&
                   increase + EPS < bestIncrease)
                {
                    bestIncrease = increase;
                    bestRoute = r;
                    bestPosition = p;
                }
            }
        }
        if(bestRoute >= 0)
        {
            solution.routes[bestRoute].insert(
                    solution.routes[bestRoute].begin() +
                    static_cast<std::ptrdiff_t>(bestPosition), node);
            rebuild(solution);
        }
    }
    // 论文的初始化在尝试完圆内所有客户后直接结束。
    // 此处不能调用 VND；VND 只在 Algorithm 2 和 Algorithm 3 指定的位置执行。
    return solution;
}

bool BiFSolver::improveTwoOpt(Solution &solution, const std::vector<unsigned char> *affected)
{
    // 对称距离下 2-opt 只替换两条边，每个候选以 O(1) 计算距离增量。
    double bestDelta = -EPS;
    int bestRoute = -1;
    std::size_t bestFirst = 0, bestLast = 0;
    for(int r = 0; r < vehicleCount; ++r)
    {
        if(affected && !(*affected)[r]) continue;
        const auto &route = solution.routes[r];
        const std::size_t size = route.size();
        for(std::size_t first = 0; first + 1 < size; ++first)
            for(std::size_t last = first + 1; last < size; ++last)
            {
                const int a = first == 0 ? 0 : route[first - 1];
                const int b = route[first];
                const int c = route[last];
                const int d = last + 1 == size ? 0 : route[last + 1];
                const double delta = data.dist_mtx[a][c] + data.dist_mtx[b][d] -
                                     data.dist_mtx[a][b] - data.dist_mtx[c][d];
                if(delta < bestDelta)
                {
                    bestDelta = delta; bestRoute = r; bestFirst = first; bestLast = last;
                }
            }
    }
    if(bestRoute < 0) return false;
    const double expectedDistance = solution.distance + bestDelta;
    std::reverse(solution.routes[bestRoute].begin()+static_cast<std::ptrdiff_t>(bestFirst),
                 solution.routes[bestRoute].begin()+static_cast<std::ptrdiff_t>(bestLast+1));
    rebuild(solution);
    if(std::abs(solution.distance - expectedDistance) > 1e-6)
        throw std::runtime_error("incremental 2-opt evaluation mismatch");
    return true;
}

bool BiFSolver::improveExchange(Solution &solution,
                                const std::vector<unsigned char> *affected)
{
    // 交换只改变两个客户附近的有限条边；相邻节点通过去重边集合单独处理。
    double bestDelta = -EPS;
    int bestR1=-1,bestR2=-1; std::size_t bestP1=0,bestP2=0;
    auto edgeCost = [&](int a,int b){return data.dist_mtx[a][b]+(b?data.serviceTime[b]:0.0);};
    for(int r1 = 0; r1 < vehicleCount; ++r1)
        for(std::size_t p1 = 0; p1 < solution.routes[r1].size(); ++p1)
            for(int r2 = r1; r2 < vehicleCount; ++r2)
            {
                if(affected && !(*affected)[r1] && !(*affected)[r2]) continue;
                for(std::size_t p2 = r2 == r1 ? p1 + 1 : 0;
                    p2 < solution.routes[r2].size(); ++p2)
                {
                    double delta=0;
                    if(r1==r2)
                    {
                        const auto &route=solution.routes[r1];
                        std::vector<std::size_t> edges{p1,p1+1,p2,p2+1};
                        std::sort(edges.begin(),edges.end());edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
                        auto oldNode=[&](std::size_t pos){return pos<route.size()?route[pos]:0;};
                        auto newNode=[&](std::size_t pos){if(pos==p1)return route[p2];if(pos==p2)return route[p1];return oldNode(pos);};
                        for(std::size_t e:edges){int oa=e?oldNode(e-1):0,ob=oldNode(e);int na=e?newNode(e-1):0,nb=newNode(e);delta+=edgeCost(na,nb)-edgeCost(oa,ob);}
                        if(solution.routeDistance[r1]+delta>duration+EPS)continue;
                    }
                    else
                    {
                        const auto&rA=solution.routes[r1];const auto&rB=solution.routes[r2];int x=rA[p1],y=rB[p2];
                        int ax=p1?rA[p1-1]:0,bx=p1+1<rA.size()?rA[p1+1]:0;
                        int ay=p2?rB[p2-1]:0,by=p2+1<rB.size()?rB[p2+1]:0;
                        double d1=edgeCost(ax,y)+edgeCost(y,bx)-edgeCost(ax,x)-edgeCost(x,bx);
                        double d2=edgeCost(ay,x)+edgeCost(x,by)-edgeCost(ay,y)-edgeCost(y,by);
                        if(solution.routeDistance[r1]+d1>duration+EPS||solution.routeDistance[r2]+d2>duration+EPS)continue;
                        if(solution.routeLoad[r1]-data.demands[x]+data.demands[y]>capacity+EPS||solution.routeLoad[r2]-data.demands[y]+data.demands[x]>capacity+EPS)continue;
                        delta=d1+d2;
                    }
                    if(delta<bestDelta){bestDelta=delta;bestR1=r1;bestR2=r2;bestP1=p1;bestP2=p2;}
                }
            }
    if(bestR1<0)return false;
    const double expectedDistance=solution.distance+bestDelta;
    std::swap(solution.routes[bestR1][bestP1],solution.routes[bestR2][bestP2]);
    rebuild(solution);
    if(std::abs(solution.distance-expectedDistance)>1e-6)
        throw std::runtime_error("incremental exchange evaluation mismatch");
    return true;
}

bool BiFSolver::improveRelocate(Solution &solution,
                                const std::vector<unsigned char> *affected)
{
    // Relocate 是一次 O(1) 删除增量加一次 O(1) 插入增量。
    double bestDelta=-EPS;int bestFromRoute=-1,bestToRoute=-1;std::size_t bestFrom=0,bestTo=0;
    for(int fromRoute = 0; fromRoute < vehicleCount; ++fromRoute)
        for(std::size_t from = 0; from < solution.routes[fromRoute].size(); ++from)
            for(int toRoute = 0; toRoute < vehicleCount; ++toRoute)
            {
                if(affected && !(*affected)[fromRoute] && !(*affected)[toRoute]) continue;
                for(std::size_t to = 0; to <= solution.routes[toRoute].size(); ++to)
                {
                    if(fromRoute == toRoute && (to == from || to == from + 1)) continue;
                    const auto&fromR=solution.routes[fromRoute];int x=fromR[from];int a=from?fromR[from-1]:0,b=from+1<fromR.size()?fromR[from+1]:0;
                    double removeDelta=data.dist_mtx[a][b]-data.dist_mtx[a][x]-data.serviceTime[x]-data.dist_mtx[x][b];
                    double delta;
                    if(fromRoute==toRoute)
                    {
                        std::size_t adjusted=to>from?to-1:to,baseSize=fromR.size()-1;
                        auto baseNode=[&](std::size_t p){return p<from?fromR[p]:fromR[p+1];};
                        int u=adjusted?baseNode(adjusted-1):0,v=adjusted<baseSize?baseNode(adjusted):0;
                        double insertDelta=data.dist_mtx[u][x]+data.serviceTime[x]+data.dist_mtx[x][v]-data.dist_mtx[u][v];
                        delta=removeDelta+insertDelta;if(solution.routeDistance[fromRoute]+delta>duration+EPS)continue;
                    }
                    else
                    {
                        const auto&toR=solution.routes[toRoute];int u=to?toR[to-1]:0,v=to<toR.size()?toR[to]:0;
                        double insertDelta=data.dist_mtx[u][x]+data.serviceTime[x]+data.dist_mtx[x][v]-data.dist_mtx[u][v];
                        if(solution.routeDistance[fromRoute]+removeDelta>duration+EPS||solution.routeDistance[toRoute]+insertDelta>duration+EPS||solution.routeLoad[toRoute]+data.demands[x]>capacity+EPS)continue;
                        delta=removeDelta+insertDelta;
                    }
                    if(delta<bestDelta){bestDelta=delta;bestFromRoute=fromRoute;bestToRoute=toRoute;bestFrom=from;bestTo=to;}
                }
            }
    if(bestFromRoute<0)return false;
    const double expectedDistance=solution.distance+bestDelta;
    int node=solution.routes[bestFromRoute][bestFrom];eraseAt(solution.routes[bestFromRoute],bestFrom);std::size_t adjusted=bestTo;if(bestFromRoute==bestToRoute&&bestTo>bestFrom)--adjusted;solution.routes[bestToRoute].insert(solution.routes[bestToRoute].begin()+static_cast<std::ptrdiff_t>(adjusted),node);rebuild(solution);
    if(std::abs(solution.distance-expectedDistance)>1e-6)
        throw std::runtime_error("incremental relocate evaluation mismatch");
    return true;
}

BiFSolver::Solution BiFSolver::variableNeighborhoodDescent(
        Solution solution, const std::vector<unsigned char> *initiallyAffected)
{
    // 上一解已经是 VND 局部最优时，仅含修改路线的移动可能产生新改善。
    // 任一移动执行后，再把该移动实际影响的一条或两条路线作为下一轮活动集。
    std::vector<unsigned char> active;
    if(initiallyAffected) active = *initiallyAffected;
    int neighborhood = 0;
    while(neighborhood < 3)
    {
        const Solution before = solution;
        const auto *restriction = initiallyAffected ? &active : nullptr;
        bool improved = false;
        if(neighborhood == 0) improved = improveTwoOpt(solution, restriction);
        else if(neighborhood == 1) improved = improveExchange(solution, restriction);
        else improved = improveRelocate(solution, restriction);
        if(improved)
        {
            if(initiallyAffected)
            {
                std::fill(active.begin(), active.end(), 0);
                for(int r = 0; r < vehicleCount; ++r)
                    if(before.routes[r] != solution.routes[r]) active[r] = 1;
            }
            neighborhood = 0;
        }
        else ++neighborhood;
    }
    return solution;
}

void BiFSolver::considerMove(std::vector<Move> &moves, Move move, std::size_t limit) const
{
    // 在线维护最多 limit 个最好移动，避免保存完整的大规模邻域。
    if(limit == 0) return;
    if(moves.size() < limit)
    {
        moves.push_back(std::move(move));
        return;
    }
    std::size_t worst = 0;
    for(std::size_t i = 1; i < moves.size(); ++i)
        if(better(moves[worst], moves[i])) worst = i;
    if(better(move, moves[worst])) moves[worst] = std::move(move);
}

bool BiFSolver::admissible(const Move &move,
                           const std::unordered_map<std::string, int> *tabuUntil,
                           int iteration,
                           double aspirationProfit,
                           const std::unordered_set<std::string> *forbidden) const
{
    // 分支记忆同时禁止已经出现过的 forward/reverse attributes。
    if(forbidden && (forbidden->count(move.forwardKey) ||
                     forbidden->count(move.reverseKey))) return false;
    if(!tabuUntil || move.candidateProfit > aspirationProfit + EPS) return true;
    const auto forward = tabuUntil->find(move.forwardKey);
    const auto reverse = tabuUntil->find(move.reverseKey);
    if(forward != tabuUntil->end() && forward->second > iteration) return false;
    if(reverse != tabuUntil->end() && reverse->second > iteration) return false;
    return true;
}

std::vector<BiFSolver::Move> BiFSolver::enumerateMoves(
        const Solution &solution, ProfitNeighborhood neighborhood, std::size_t limit,
        const std::unordered_map<std::string, int> *tabuUntil, int iteration,
        double aspirationProfit, const std::unordered_set<std::string> *forbidden,
        int onlyRoute, const std::unordered_set<int> *requiredInserted)
{
    std::vector<int> unserved;
    for(int node = 1; node <= n; ++node)
        if(!solution.selected[node]) unserved.push_back(node);
    std::stable_sort(unserved.begin(), unserved.end(), [this](int lhs, int rhs)
    {
        if(std::abs(data.profits[lhs] - data.profits[rhs]) > EPS)
            return data.profits[lhs] > data.profits[rhs];
        return lhs < rhs;
    });

    // 堆顶始终是当前保留集合中最差的移动，因此每次候选插入由原来的
    // O(limit) 线性查找降为 O(log limit)。
    auto heapCompare = [this](const Move &lhs, const Move &rhs)
    {
        return better(lhs, rhs);
    };
    std::priority_queue<Move, std::vector<Move>, decltype(heapCompare)>
            bestMoves(heapCompare);
    auto mayCompete = [&](double candidateProfit)
    {
        if(bestMoves.size() < limit) return true;
        return candidateProfit + EPS >= bestMoves.top().candidateProfit;
    };
    auto submit = [&](Move move)
    {
        move.forwardKey = moveKey(move.removed, move.inserted);
        move.reverseKey = moveKey(move.inserted, move.removed);
        if(!admissible(move, tabuUntil, iteration, aspirationProfit, forbidden) || limit == 0)
            return;
        if(bestMoves.size() < limit) bestMoves.push(std::move(move));
        else if(better(move, bestMoves.top()))
        {
            bestMoves.pop();
            bestMoves.push(std::move(move));
        }
    };
    auto makeBase = [&](const std::vector<int> &route,
                        const std::vector<std::size_t> &positions)
    {
        std::vector<int> base=route;
        for(auto it=positions.rbegin();it!=positions.rend();++it)eraseAt(base,*it);
        return base;
    };
    auto removalDelta = [&](const std::vector<int>&route,
                            const std::vector<std::size_t>&positions)
    {
        if(positions.empty()) return 0.0;
        if(positions.size()==1){std::size_t p=positions[0];int x=route[p],a=p?route[p-1]:0,b=p+1<route.size()?route[p+1]:0;return data.dist_mtx[a][b]-data.dist_mtx[a][x]-data.serviceTime[x]-data.dist_mtx[x][b];}
        std::size_t p=positions[0],q=positions[1];int x=route[p],y=route[q];
        if(q==p+1){int a=p?route[p-1]:0,b=q+1<route.size()?route[q+1]:0;return data.dist_mtx[a][b]-data.dist_mtx[a][x]-data.serviceTime[x]-data.dist_mtx[x][y]-data.serviceTime[y]-data.dist_mtx[y][b];}
        int ax=p?route[p-1]:0,bx=route[p+1],ay=route[q-1],by=q+1<route.size()?route[q+1]:0;
        return data.dist_mtx[ax][bx]-data.dist_mtx[ax][x]-data.serviceTime[x]-data.dist_mtx[x][bx]+
               data.dist_mtx[ay][by]-data.dist_mtx[ay][y]-data.serviceTime[y]-data.dist_mtx[y][by];
    };
    auto bestInsertOne = [&](const std::vector<int> &base,double baseDistance,int node,
                             std::size_t &bestEdge,double &bestDistance)
    {
        bestDistance=std::numeric_limits<double>::infinity();bestEdge=0;
        for(std::size_t edge = 0; edge <= base.size(); ++edge)
        {
            const int previous = edge == 0 ? 0 : base[edge - 1];
            const int next = edge == base.size() ? 0 : base[edge];
            const double delta = data.dist_mtx[previous][node] + data.serviceTime[node] +
                                 data.dist_mtx[node][next] - data.dist_mtx[previous][next];
            double value=baseDistance+delta;
            if(value<=duration+EPS&&value+EPS<bestDistance){bestDistance=value;bestEdge=edge;}
        }
        return std::isfinite(bestDistance);
    };
    auto bestInsertTwo = [&](const std::vector<int>&base,double baseDistance,int firstNode,int secondNode,
                             std::size_t&firstEdge,std::size_t&secondEdge,bool&firstBeforeSecond,double&bestDistance)
    {
        const std::size_t edgeCount = base.size() + 1;
        bestDistance=std::numeric_limits<double>::infinity();firstEdge=secondEdge=0;firstBeforeSecond=true;
        for(std::size_t e1 = 0; e1 < edgeCount; ++e1)
            for(std::size_t e2 = 0; e2 < edgeCount; ++e2)
            {
                int a1=e1?base[e1-1]:0,b1=e1<base.size()?base[e1]:0;
                int a2=e2?base[e2-1]:0,b2=e2<base.size()?base[e2]:0;
                double distanceValue;bool order=true;
                if(e1 != e2)
                    distanceValue=baseDistance+data.dist_mtx[a1][firstNode]+data.serviceTime[firstNode]+data.dist_mtx[firstNode][b1]-data.dist_mtx[a1][b1]+data.dist_mtx[a2][secondNode]+data.serviceTime[secondNode]+data.dist_mtx[secondNode][b2]-data.dist_mtx[a2][b2];
                else
                {
                    const double firstThenSecond = baseDistance +
                            data.dist_mtx[a1][firstNode] + data.serviceTime[firstNode] +
                            data.dist_mtx[firstNode][secondNode] + data.serviceTime[secondNode] +
                            data.dist_mtx[secondNode][b1] - data.dist_mtx[a1][b1];
                    const double secondThenFirst = baseDistance +
                            data.dist_mtx[a1][secondNode] + data.serviceTime[secondNode] +
                            data.dist_mtx[secondNode][firstNode] + data.serviceTime[firstNode] +
                            data.dist_mtx[firstNode][b1] - data.dist_mtx[a1][b1];
                    if(firstThenSecond<=secondThenFirst){distanceValue=firstThenSecond;order=true;}else{distanceValue=secondThenFirst;order=false;}
                }
                if(distanceValue<=duration+EPS&&distanceValue+EPS<bestDistance){bestDistance=distanceValue;firstEdge=e1;secondEdge=e2;firstBeforeSecond=order;}
            }
        return std::isfinite(bestDistance);
    };

    auto finishMove=[&](Move &move,int r,double distanceAfter,double loadAfter){move.routeIndex=r;move.routeDistanceAfter=distanceAfter;move.routeLoadAfter=loadAfter;move.candidateProfit=solution.profit;for(int v:move.removed)move.candidateProfit-=data.profits[v];for(int v:move.inserted)move.candidateProfit+=data.profits[v];move.candidateDistance=solution.distance-solution.routeDistance[r]+distanceAfter;submit(std::move(move));};

    if(neighborhood == ProfitNeighborhood::Replace11)
    {
        // N1：纯插入，或者移除一个已访问客户再插入一个未访问客户。
        // The paper defines 1-1 Replace to perform a pure insertion whenever
        // the unserved customer fits without deleting a routed customer.
        for(int add : unserved)
        {
            if(requiredInserted && !requiredInserted->count(add)) continue;
            for(int r = 0; r < vehicleCount; ++r)
            {
                if(onlyRoute >= 0 && r != onlyRoute) continue;
                if(!mayCompete(solution.profit + data.profits[add])) continue;
                if(solution.routeLoad[r] + data.demands[add] > capacity + EPS) continue;
                std::size_t edge;double after;if(bestInsertOne(solution.routes[r],solution.routeDistance[r],add,edge,after)){Move move;move.inserted={add};move.insertEdges={edge};finishMove(move,r,after,solution.routeLoad[r]+data.demands[add]);}
            }
        }

        for(int r = 0; r < vehicleCount; ++r)
        {
            if(onlyRoute >= 0 && r != onlyRoute) continue;
            for(std::size_t removePos = 0; removePos < solution.routes[r].size(); ++removePos)
            {
                std::vector<std::size_t> removedPositions{removePos};auto base=makeBase(solution.routes[r],removedPositions);double baseDistance=solution.routeDistance[r]+removalDelta(solution.routes[r],removedPositions);int remove=solution.routes[r][removePos];
                for(int add : unserved)
                {
                    if(requiredInserted && !requiredInserted->count(add)) continue;
                    if(!mayCompete(solution.profit - data.profits[remove] +
                                   data.profits[add])) break;
                    if(solution.routeLoad[r] - data.demands[remove] + data.demands[add] >
                       capacity + EPS) continue;
                    std::size_t edge;double after;if(bestInsertOne(base,baseDistance,add,edge,after)){Move move;move.removed={remove};move.inserted={add};move.removePositions=removedPositions;move.insertEdges={edge};finishMove(move,r,after,solution.routeLoad[r]-data.demands[remove]+data.demands[add]);}
                }
            }
        }
    }
    else if(neighborhood == ProfitNeighborhood::Replace21)
    {
        // N2：从同一路线移除两个已访问客户，再插入一个未访问客户。
        for(int r = 0; r < vehicleCount; ++r)
        {
            if(onlyRoute >= 0 && r != onlyRoute) continue;
            for(std::size_t first = 0; first < solution.routes[r].size(); ++first)
                for(std::size_t second = first + 1; second < solution.routes[r].size(); ++second)
                {
                    std::vector<std::size_t>removedPositions{first,second};auto base=makeBase(solution.routes[r],removedPositions);double baseDistance=solution.routeDistance[r]+removalDelta(solution.routes[r],removedPositions);int remove1=solution.routes[r][first],remove2=solution.routes[r][second];
                    for(int add : unserved)
                    {
                        if(requiredInserted && !requiredInserted->count(add)) continue;
                        if(!mayCompete(solution.profit - data.profits[remove1] -
                                      data.profits[remove2] + data.profits[add])) break;
                        if(solution.routeLoad[r] - data.demands[remove1] - data.demands[remove2] +
                           data.demands[add] > capacity + EPS) continue;
                        std::size_t edge;double after;if(bestInsertOne(base,baseDistance,add,edge,after)){Move move;move.removed={remove1,remove2};move.inserted={add};move.removePositions=removedPositions;move.insertEdges={edge};finishMove(move,r,after,solution.routeLoad[r]-data.demands[remove1]-data.demands[remove2]+data.demands[add]);}
                    }
                }
        }
    }
    else
    {
        // N3：移除一个已访问客户，再插入两个不同的未访问客户。
        std::vector<std::pair<int, int>> unservedPairs;
        unservedPairs.reserve(unserved.size() * (unserved.size() - 1) / 2);
        for(std::size_t first = 0; first < unserved.size(); ++first)
            for(std::size_t second = first + 1; second < unserved.size(); ++second)
                if(!requiredInserted || requiredInserted->count(unserved[first]) ||
                   requiredInserted->count(unserved[second]))
                    unservedPairs.emplace_back(unserved[first], unserved[second]);
        std::stable_sort(unservedPairs.begin(), unservedPairs.end(), [this](const auto &lhs,
                                                                           const auto &rhs)
        {
            const double lhsProfit = data.profits[lhs.first] + data.profits[lhs.second];
            const double rhsProfit = data.profits[rhs.first] + data.profits[rhs.second];
            if(std::abs(lhsProfit - rhsProfit) > EPS) return lhsProfit > rhsProfit;
            return lhs < rhs;
        });

        for(int r = 0; r < vehicleCount; ++r)
        {
            if(onlyRoute >= 0 && r != onlyRoute) continue;
            for(std::size_t removePos = 0; removePos < solution.routes[r].size(); ++removePos)
            {
                std::vector<std::size_t>removedPositions{removePos};auto base=makeBase(solution.routes[r],removedPositions);double baseDistance=solution.routeDistance[r]+removalDelta(solution.routes[r],removedPositions);int remove=solution.routes[r][removePos];
                for(const auto &pair : unservedPairs)
                    {
                        if(requiredInserted && !requiredInserted->count(pair.first) &&
                           !requiredInserted->count(pair.second)) continue;
                        const int add1 = pair.first;
                        const int add2 = pair.second;
                        if(!mayCompete(solution.profit - data.profits[remove] +
                                      data.profits[add1] + data.profits[add2])) break;
                        if(solution.routeLoad[r] - data.demands[remove] + data.demands[add1] +
                           data.demands[add2] > capacity + EPS) continue;
                        std::size_t e1,e2;bool order;double after;if(bestInsertTwo(base,baseDistance,add1,add2,e1,e2,order,after)){Move move;move.removed={remove};move.inserted={add1,add2};move.removePositions=removedPositions;move.insertEdges={e1,e2};move.firstBeforeSecond=order;finishMove(move,r,after,solution.routeLoad[r]-data.demands[remove]+data.demands[add1]+data.demands[add2]);}
                    }
            }
        }
    }

    std::vector<Move> moves;
    moves.reserve(bestMoves.size());
    while(!bestMoves.empty())
    {
        moves.push_back(bestMoves.top());
        bestMoves.pop();
    }
    std::sort(moves.begin(), moves.end(), [this](const Move &lhs, const Move &rhs)
    {
        return better(lhs, rhs);
    });
    return moves;
}

BiFSolver::Solution BiFSolver::tabuSearch(const Solution &initial)
{
    // 每次随机选择一个利润邻域，并采用其中最好的可接受移动。
    // routeCache 保存每个邻域、每条路线当前最好的移动。执行移动后，只有
    // 发生变化的路线被完整重算；其他路线仅补充刚变成未访问客户产生的新移动。
    std::cout << "tabu_search starts, objective " << initial.profit
              << ", distance " << initial.distance << '\n';
    Solution current = initial;
    Solution best = initial;
    std::unordered_map<std::string, int> tabuUntil;
    std::uniform_int_distribution<int> selectNeighborhood(0, 2);
    int noImprove = 0;
    int iteration = 0;

    std::vector<std::vector<Move>> routeCache(
            3, std::vector<Move>(static_cast<std::size_t>(vehicleCount)));
    std::vector<std::vector<unsigned char>> cacheValid(
            3, std::vector<unsigned char>(static_cast<std::size_t>(vehicleCount), 0));

    auto refreshRoute = [&](int type, int route)
    {
        auto moves = enumerateMoves(current, static_cast<ProfitNeighborhood>(type), 1,
                                    nullptr, iteration, best.profit, nullptr, route, nullptr);
        cacheValid[type][route] = !moves.empty();
        if(!moves.empty()) routeCache[type][route] = std::move(moves.front());
    };
    auto rebase = [&](const Move &stored, Move &rebased)
    {
        rebased = stored;
        return rebaseMove(rebased, current);
    };

    // 第一次建立全部路线的邻域评价表。后续迭代采用增量更新。
    for(int type = 0; type < 3; ++type)
        for(int route = 0; route < vehicleCount; ++route)
            refreshRoute(type, route);

    while(noImprove < tabuNoImproveLimit)
    {
        const int type = selectNeighborhood(rng);
        const auto neighborhood = static_cast<ProfitNeighborhood>(type);
        std::vector<Move> moves;
        for(int route = 0; route < vehicleCount; ++route)
        {
            Move candidate;
            bool available = cacheValid[type][route] &&
                             rebase(routeCache[type][route], candidate) &&
                             admissible(candidate, &tabuUntil, iteration, best.profit, nullptr);
            if(!available)
            {
                // 缓存中的最好移动可能暂时 tabu；只重新扫描该路线寻找下一个允许移动。
                auto allowed = enumerateMoves(current, neighborhood, 1, &tabuUntil,
                                              iteration, best.profit, nullptr, route, nullptr);
                if(allowed.empty()) continue;
                candidate = std::move(allowed.front());
            }
            considerMove(moves, std::move(candidate), 1);
        }
        ++iteration;
        if(moves.empty())
        {
            ++noImprove;
            if(iteration % 100 == 0)
            {
                const double elapsed = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - startTime).count();
                std::cout << "tabu_search progress, iteration " << iteration
                          << ", current " << current.profit
                          << ", best " << best.profit
                          << ", no_improve " << noImprove
                          << ", time " << elapsed << '\n';
            }
            continue;
        }

        // 按论文记录局部移动的 forward 和 reverse attributes，而不是禁用客户。
        Move move = std::move(moves.front());
        tabuUntil[move.forwardKey] = iteration + tabuTenure;
        tabuUntil[move.reverseKey] = iteration + tabuTenure;
        const Solution previous = current;
        std::vector<unsigned char> upperAffected(static_cast<std::size_t>(vehicleCount), 0);
        if(move.routeIndex >= 0) upperAffected[move.routeIndex] = 1;
        // 第一次进入 VND 时初始化解尚未证明为下层局部最优，因此完整评价；
        // 后续仅评价涉及本次利润移动所修改路线的组合。
        current = variableNeighborhoodDescent(applyMove(current, move),
                                              iteration == 1 ? nullptr : &upperAffected);

        std::vector<unsigned char> modified(static_cast<std::size_t>(vehicleCount), 0);
        for(int route = 0; route < vehicleCount; ++route)
            if(previous.routes[route] != current.routes[route]) modified[route] = 1;
        std::unordered_set<int> newlyUnserved(move.removed.begin(), move.removed.end());

        // 论文第 3.3.1 节的增量评价：完整重算修改路线，保留其他路线的旧评价，
        // 并仅评价刚变成未访问客户所新增的移动组合。
        for(int cacheType = 0; cacheType < 3; ++cacheType)
            for(int route = 0; route < vehicleCount; ++route)
            {
                if(modified[route])
                {
                    refreshRoute(cacheType, route);
                    continue;
                }
                Move oldBest;
                if(cacheValid[cacheType][route] &&
                   !rebase(routeCache[cacheType][route], oldBest))
                    refreshRoute(cacheType, route);
                const bool hasCurrentBest = cacheValid[cacheType][route] &&
                                            rebase(routeCache[cacheType][route], oldBest);
                auto additions = enumerateMoves(current,
                        static_cast<ProfitNeighborhood>(cacheType), 1, nullptr,
                        iteration, best.profit, nullptr, route, &newlyUnserved);
                if(!additions.empty() &&
                   (!hasCurrentBest || better(additions.front(), oldBest)))
                {
                    routeCache[cacheType][route] = std::move(additions.front());
                    cacheValid[cacheType][route] = 1;
                }
            }
        if(better(current, best))
        {
            // 论文的时间指标以目标利润为准：只有利润严格提高时，才记录该
            // 新目标值第一次出现的时间；同利润的距离改进不能覆盖它。
            const bool profitImproved = current.profit > best.profit + EPS;
            best = current;
            noImprove = 0;
            if(profitImproved)
            {
                recordBestTime();
                std::cout << "tabu_search new objective, iteration " << iteration
                          << ", objective " << best.profit
                          << ", distance " << best.distance
                          << ", first_objective_time " << bestTime << '\n';
            }
            else
            {
                std::cout << "tabu_search same objective with shorter distance, iteration "
                          << iteration << ", objective " << best.profit
                          << ", distance " << best.distance
                          << ", first_objective_time " << bestTime << '\n';
            }
        }
        else ++noImprove;

        // 周期性输出只用于观察运行状态，不参与搜索决策。
        if(iteration % 100 == 0)
        {
            const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - startTime).count();
            std::cout << "tabu_search progress, iteration " << iteration
                      << ", current " << current.profit
                      << ", best " << best.profit
                      << ", no_improve " << noImprove
                      << ", time " << elapsed << '\n';
        }
    }
    std::cout << "tabu_search ends, iterations " << iteration
              << ", objective " << best.profit << '\n';
    return best;
}

BiFSolver::Solution BiFSolver::filterAndFan(const Solution &root)
{
    // 宽度受限的搜索树。除了分支禁忌属性，每个节点还维护按路线划分的
    // 利润邻域缓存；子节点只重算实际修改的路线，其余路线采用增量更新。
    std::cout << "filter_and_fan starts, objective " << root.profit << '\n';
    Solution best = root;
    const std::size_t cacheReserve = static_cast<std::size_t>(fanWidth) * 2;

    auto refreshCache = [&](const Solution &solution, int route, int type)
    {
        MoveCache cache;
        cache.moves = enumerateMoves(solution, static_cast<ProfitNeighborhood>(type),
                                     cacheReserve, nullptr, 0, best.profit,
                                     nullptr, route, nullptr);
        cache.guaranteed = cache.moves.size();
        cache.exhaustive = cache.moves.size() < cacheReserve;
        return cache;
    };

    auto initializeCaches = [&](TreeNode &node)
    {
        node.routeCaches.resize(static_cast<std::size_t>(vehicleCount));
        for(int route = 0; route < vehicleCount; ++route)
            for(int type = 0; type < 3; ++type)
                node.routeCaches[route][type] = refreshCache(node.solution, route, type);
    };

    // 从缓存读取一个节点在指定路线/邻域上的最好允许移动。缓存保留了
    // fanWidth 之外的后备候选；只有分支禁忌耗尽了保证前缀才回退到完整扫描。
    auto allowedFromCache = [&](const TreeNode &node, int route, int type)
    {
        const MoveCache &cache = node.routeCaches[route][type];
        std::vector<Move> allowed;
        allowed.reserve(static_cast<std::size_t>(fanWidth));
        const std::size_t prefix = std::min(cache.guaranteed, cache.moves.size());
        for(std::size_t i = 0; i < prefix; ++i)
        {
            Move candidate = cache.moves[i];
            if(rebaseMove(candidate, node.solution) &&
               admissible(candidate, nullptr, 0, best.profit, &node.forbidden))
            {
                // cache.moves 已按词典序质量排序；过滤非法移动不会改变剩余
                // 候选的相对次序，因此收集满 n1 个即可停止。
                allowed.push_back(std::move(candidate));
                if(allowed.size() == static_cast<std::size_t>(fanWidth)) break;
            }
        }
        if(allowed.size() < static_cast<std::size_t>(fanWidth) && !cache.exhaustive)
            return enumerateMoves(node.solution, static_cast<ProfitNeighborhood>(type),
                                  fanWidth, nullptr, 0, best.profit,
                                  &node.forbidden, route, nullptr);
        return allowed;
    };

    // 由父节点缓存构造子节点缓存。利润移动使 removed 客户成为新的未访问
    // 客户，因此未修改路线只需删除失效移动并补充包含这些客户的新组合。
    auto updateChildCaches = [&](TreeNode &child, const TreeNode &parent,
                                 const Move &selectedMove)
    {
        child.routeCaches.resize(static_cast<std::size_t>(vehicleCount));
        const std::unordered_set<int> newlyUnserved(selectedMove.removed.begin(),
                                                     selectedMove.removed.end());
        for(int route = 0; route < vehicleCount; ++route)
        {
            const bool modified = parent.solution.routes[route] != child.solution.routes[route];
            for(int type = 0; type < 3; ++type)
            {
                if(modified)
                {
                    child.routeCaches[route][type] = refreshCache(child.solution, route, type);
                    continue;
                }

                const MoveCache &oldCache = parent.routeCaches[route][type];
                std::vector<Move> surviving;
                const std::size_t oldPrefix = std::min(oldCache.guaranteed,
                                                       oldCache.moves.size());
                surviving.reserve(oldPrefix);
                for(std::size_t i = 0; i < oldPrefix; ++i)
                {
                    Move candidate = oldCache.moves[i];
                    if(rebaseMove(candidate, child.solution))
                        surviving.push_back(std::move(candidate));
                }
                const std::size_t survivingCoverage = surviving.size();

                auto additions = enumerateMoves(child.solution,
                        static_cast<ProfitNeighborhood>(type), cacheReserve,
                        nullptr, 0, best.profit, nullptr, route, &newlyUnserved);
                const bool additionsExhaustive = additions.size() < cacheReserve;
                const std::size_t additionsCoverage = additions.size();

                std::vector<Move> merged;
                merged.reserve(surviving.size() + additions.size());
                for(Move &move : surviving) merged.push_back(std::move(move));
                for(Move &move : additions) merged.push_back(std::move(move));
                std::sort(merged.begin(), merged.end(), [this](const Move &lhs,
                                                               const Move &rhs)
                {
                    return better(lhs, rhs);
                });
                if(merged.size() > cacheReserve) merged.resize(cacheReserve);

                MoveCache updated;
                updated.moves = std::move(merged);
                updated.exhaustive = oldCache.exhaustive && additionsExhaustive;
                if(updated.exhaustive)
                    updated.guaranteed = updated.moves.size();
                else if(oldCache.exhaustive)
                    updated.guaranteed = std::min(additionsCoverage, updated.moves.size());
                else if(additionsExhaustive)
                    updated.guaranteed = std::min(survivingCoverage, updated.moves.size());
                else
                    updated.guaranteed = std::min({survivingCoverage,
                                                   additionsCoverage,
                                                   updated.moves.size()});

                // 保证前缀不足100时不能安全地筛选本层，回退到该路线完整重算。
                if(updated.guaranteed < static_cast<std::size_t>(fanWidth) &&
                   !updated.exhaustive)
                    updated = refreshCache(child.solution, route, type);
                child.routeCaches[route][type] = std::move(updated);
            }
        }
    };

    TreeNode rootNode;
    rootNode.solution = root;
    initializeCaches(rootNode);
    std::vector<TreeNode> level;
    level.push_back(std::move(rootNode));

    for(int depth = 0; depth < fanDepth && !level.empty(); ++depth)
    {
        struct Trial
        {
            std::size_t parentIndex;
            Move move;
        };
        std::vector<Trial> trials;

        for(std::size_t parentIndex = 0; parentIndex < level.size(); ++parentIndex)
        {
            const TreeNode &parent = level[parentIndex];
            // 每条路线/邻域的缓存都已经有序。用18路归并从它们的并集精确
            // 取出最好 n1 个，结果与枚举后全排序完全相同。
            std::vector<std::vector<Move>> orderedLists;
            orderedLists.reserve(static_cast<std::size_t>(vehicleCount) * 3);
            for(int type = 0; type < 3; ++type)
                for(int route = 0; route < vehicleCount; ++route)
                    orderedLists.push_back(allowedFromCache(parent, route, type));

            struct Cursor
            {
                std::size_t list;
                std::size_t position;
            };
            auto cursorCompare = [&](const Cursor &lhs, const Cursor &rhs)
            {
                return better(orderedLists[rhs.list][rhs.position],
                              orderedLists[lhs.list][lhs.position]);
            };
            std::priority_queue<Cursor, std::vector<Cursor>, decltype(cursorCompare)>
                    mergeQueue(cursorCompare);
            for(std::size_t list = 0; list < orderedLists.size(); ++list)
                if(!orderedLists[list].empty()) mergeQueue.push(Cursor{list, 0});

            std::vector<Move> parentMoves;
            parentMoves.reserve(static_cast<std::size_t>(fanWidth));
            while(!mergeQueue.empty() &&
                  parentMoves.size() < static_cast<std::size_t>(fanWidth))
            {
                const Cursor cursor = mergeQueue.top();
                mergeQueue.pop();
                parentMoves.push_back(std::move(
                        orderedLists[cursor.list][cursor.position]));
                const std::size_t next = cursor.position + 1;
                if(next < orderedLists[cursor.list].size())
                    mergeQueue.push(Cursor{cursor.list, next});
            }
            for(Move &move : parentMoves)
                trials.push_back(Trial{parentIndex, std::move(move)});
        }

        // 先对轻量 Move 描述执行全局 Filter；不再为最多 n1^2 个试探移动
        // 全部复制并 rebuild 完整解。
        std::sort(trials.begin(), trials.end(), [this](const Trial &lhs,
                                                       const Trial &rhs)
        {
            return better(lhs.move, rhs.move);
        });

        struct SelectedNode
        {
            TreeNode node;
            std::size_t parentIndex;
            Move move;
        };
        std::vector<SelectedNode> selected;
        std::unordered_set<std::string> seen;
        for(Trial &trial : trials)
        {
            const TreeNode &parent = level[trial.parentIndex];
            Solution childSolution = applyMove(parent.solution, trial.move);
            const std::string key = solutionKey(childSolution);
            if(!seen.insert(key).second) continue;
            TreeNode child;
            child.solution = std::move(childSolution);
            child.forbidden = parent.forbidden;
            child.forbidden.insert(trial.move.forwardKey);
            child.forbidden.insert(trial.move.reverseKey);
            selected.push_back(SelectedNode{std::move(child), trial.parentIndex,
                                            std::move(trial.move)});
            if(selected.size() == static_cast<std::size_t>(fanWidth)) break;
        }

        std::vector<TreeNode> candidates;
        candidates.reserve(selected.size());
        for(SelectedNode &entry : selected)
        {
            // Fan 节点先用 VND 缩短路线，再与本轮最好解比较。
            TreeNode &node = entry.node;
            std::vector<unsigned char> affected(static_cast<std::size_t>(vehicleCount), 0);
            if(entry.move.routeIndex >= 0) affected[entry.move.routeIndex] = 1;
            node.solution = variableNeighborhoodDescent(std::move(node.solution), &affected);
            if(better(node.solution, best))
            {
                const bool profitImproved = node.solution.profit > best.profit + EPS;
                best = node.solution;
                if(profitImproved) recordBestTime();
            }
            candidates.push_back(std::move(node));
        }
        if(best.profit > root.profit + EPS)
        {
            // 论文规定利润提高后立即结束本次F&F；因此不构建永远不会使用的
            // 下一层缓存。recordBestTime() 已在首次遇到该利润时执行。
            const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - startTime).count();
            std::cout << "filter_and_fan depth " << depth + 1
                      << ", trials " << trials.size()
                      << ", candidates " << candidates.size()
                      << ", best " << best.profit
                      << ", time " << elapsed << '\n';
            std::cout << "filter_and_fan new best, objective " << best.profit
                      << ", distance " << best.distance
                      << ", time " << bestTime << '\n';
            return best;
        }

        // 只有确定还要进入下一层时，才为保留下来的节点更新邻域缓存。
        for(std::size_t i = 0; i < candidates.size(); ++i)
            updateChildCaches(candidates[i], level[selected[i].parentIndex],
                              selected[i].move);

        const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - startTime).count();
        std::cout << "filter_and_fan depth " << depth + 1
                  << ", trials " << trials.size()
                  << ", candidates " << candidates.size()
                  << ", best " << best.profit
                  << ", time " << elapsed << '\n';
        level = std::move(candidates);
    }
    std::cout << "filter_and_fan ends, objective " << best.profit << '\n';
    return best;
}

void BiFSolver::recordBestTime()
{
    bestTime = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - startTime).count();
}

BiFSolver::Result BiFSolver::solve()
{
    // 总流程：初始解 -> Tabu Search -> Filter-and-Fan。
    // F&F 改善利润时开始下一轮，否则当前解即为最终结果。
    startTime = std::chrono::steady_clock::now();
    std::cout << "initialization starts\n";
    Solution best = constructInitial();
    recordBestTime();
    std::cout << "initialization ends, objective " << best.profit
              << ", distance " << best.distance
              << ", time " << bestTime << '\n';

    int round = 0;
    while(true)
    {
        ++round;
        std::cout << "BiF round " << round << " starts\n";
        Solution afterTabu = tabuSearch(best);
        if(better(afterTabu, best)) best = std::move(afterTabu);

        Solution afterFan = filterAndFan(best);
        if(afterFan.profit > best.profit + EPS)
        {
            best = std::move(afterFan);
            // filterAndFan() 已经在首次得到该新利润时记录时间，不能在返回后覆盖。
            continue;
        }
        break;
    }

    const double total = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - startTime).count();
    std::cout << "BiF search ends, objective " << best.profit
              << ", distance " << best.distance
              << ", total_time " << total << '\n';
    return Result{best.profit, best.distance, bestTime, total, best.routes};
}
