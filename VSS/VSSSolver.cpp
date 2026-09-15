#include "VSSSolver.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace { constexpr double EPS = 1e-9; }

VSSSolver::VSSSolver(const Instance &i, Mode m, std::uint32_t seed)
    : data(i), mode(m), rng(seed), n(i.nbClients), vehicles(i.nbVehicles),
      capacity(i.vehicleCapacity), duration(i.durationLimite) {
    if (n <= 0 || vehicles <= 0 || capacity <= 0 || duration <= 0 ||
        !std::isfinite(capacity) || !std::isfinite(duration) ||
        data.profits.size() != static_cast<std::size_t>(n + 1) ||
        data.demands.size() != static_cast<std::size_t>(n + 1) ||
        data.serviceTime.size() != static_cast<std::size_t>(n + 1) ||
        data.dist_mtx.size() != static_cast<std::size_t>(n + 1))
        throw std::runtime_error("invalid or incomplete data returned by Instance");

    // Diagnostic output is opt-in so that the two server executables retain
    // their strict two-line normal output. Enable with VSS_DIAGNOSTICS=1.
    const char *diagnosticEnvironment=std::getenv("VSS_DIAGNOSTICS");
    diagnostics=diagnosticEnvironment!=nullptr
            &&std::string(diagnosticEnvironment)!="0";
}

double VSSSolver::routeTime(const std::vector<int> &r) const {
    double t = 0; int prev = 0;
    for (int v : r) { t += data.dist_mtx[prev][v] + data.serviceTime[v]; prev = v; }
    return t + data.dist_mtx[prev][0];
}
double VSSSolver::routeLoad(const std::vector<int> &r) const {
    double q = 0; for (int v : r) q += data.demands[v]; return q;
}
bool VSSSolver::feasible(const std::vector<int> &r) const {
    return routeLoad(r) <= capacity + EPS && routeTime(r) <= duration + EPS;
}
void VSSSolver::rebuild(Solution &s) const {
    if (s.routes.size() != static_cast<std::size_t>(vehicles))
        throw std::runtime_error("invalid number of routes");
    s.selected.assign(n + 1, 0); s.profit = s.distance = 0;
    for (std::size_t routeIndex = 0; routeIndex < s.routes.size(); ++routeIndex) {
        const auto &r = s.routes[routeIndex];
        double load=0.0;
        double time=0.0;
        int previous=0;
        for (int v : r) {
            if (v <= 0 || v > n)
                throw std::runtime_error("VSS generated invalid customer " +
                                         std::to_string(v) + " in route " +
                                         std::to_string(routeIndex));
            if (s.selected[v])
                throw std::runtime_error("VSS generated duplicate customer " +
                                         std::to_string(v) + " in route " +
                                         std::to_string(routeIndex));
            s.selected[v]=1;
            s.profit+=data.profits[v];
            load+=data.demands[v];
            time+=data.dist_mtx[previous][v]+data.serviceTime[v];
            previous=v;
        }
        time+=data.dist_mtx[previous][0];
        if (load>capacity+EPS || time>duration+EPS)
            throw std::runtime_error("VSS generated an infeasible route");
        s.distance+=time;
    }
}
bool VSSSolver::profitBetter(const Solution &a, const Solution &b) const {
    return a.profit > b.profit + EPS;
}
void VSSSolver::observeProfit(double profit) {
    // TTB按目标值记录：同利润但距离更短不会刷新第一次到达时间。
    if(profit>observedBestProfit+EPS) {
        observedBestProfit=profit;
        bestTime=std::chrono::duration<double>(
                std::chrono::steady_clock::now()-start).count();
    }
}

void VSSSolver::diagnostic(const char *stage, const Solution &solution,
                           int first, int second) const {
    if(!diagnostics) return;
    std::cerr << "VSS_DIAG " << stage;
    if(first>=0) std::cerr << " " << first;
    if(second>=0) std::cerr << " " << second;
    std::cerr << " profit=" << solution.profit
              << " distance=" << solution.distance << '\n';
}

VSSSolver::Solution VSSSolver::bestInsertion(Solution s, Parameters p) {
    rebuild(s);
    const double pmax = *std::max_element(data.profits.begin()+1, data.profits.end());
    while (true) {
        // 一轮BIA中路线不发生变化，因此载重和时间只计算一次。
        // 插入一个客户后进入下一轮，再刷新这些缓存。
        std::vector<double> routeLoads(static_cast<std::size_t>(vehicles));
        std::vector<double> routeTimes(static_cast<std::size_t>(vehicles));
        for (int r=0; r<vehicles; ++r) {
            routeLoads[r]=routeLoad(s.routes[r]);
            routeTimes[r]=routeTime(s.routes[r]);
        }
        double bestRatio = std::numeric_limits<double>::infinity();
        int bestNode=-1, bestRoute=-1; std::size_t bestPos=0;
        for (int v=1; v<=n; ++v) if (!s.selected[v]) {
            for (int r=0; r<vehicles; ++r) {
                const double old=routeTimes[r];
                if (routeLoads[r]+data.demands[v] > capacity+EPS) continue;
                for (std::size_t pos=0; pos<=s.routes[r].size(); ++pos) {
                    int a=pos? s.routes[r][pos-1]:0;
                    int b=pos<s.routes[r].size()?s.routes[r][pos]:0;
                    double delta=data.dist_mtx[a][v]+data.serviceTime[v]+data.dist_mtx[v][b]-data.dist_mtx[a][b];
                    if (old+delta > duration+EPS) continue;
                    double x=std::max(delta/duration, 1e-12);
                    double q=std::max(data.demands[v]/capacity, 1e-12);
                    double z=std::max(data.profits[v]/std::max(pmax,EPS), 1e-12);
                    double ratio=std::pow(x,p.beta)*std::pow(q,p.gamma)/std::pow(z,p.alpha);
                    if (ratio < bestRatio-EPS) { bestRatio=ratio; bestNode=v; bestRoute=r; bestPos=pos; }
                }
            }
        }
        if (bestNode<0) break;
        s.routes[bestRoute].insert(s.routes[bestRoute].begin()+static_cast<std::ptrdiff_t>(bestPos),bestNode);
        rebuild(s);
    }
    return s;
}

std::vector<VSSSolver::ConstructionCandidate>
VSSSolver::constructionCandidates(const Solution &partial,
                                  const Parameters &center) {
    std::uniform_real_distribution<double> u(0,1);
    std::vector<Parameters> ps;
    const double d=.05;
    ps.push_back({1,std::clamp(center.beta-d,0.0,1.0),std::clamp(center.gamma-d,0.0,1.0)});
    ps.push_back({1,std::clamp(center.beta-d,0.0,1.0),std::clamp(center.gamma+d,0.0,1.0)});
    ps.push_back({1,std::clamp(center.beta+d,0.0,1.0),std::clamp(center.gamma-d,0.0,1.0)});
    ps.push_back({1,std::clamp(center.beta+d,0.0,1.0),std::clamp(center.gamma+d,0.0,1.0)});
    ps.push_back({1,u(rng),u(rng)}); ps.push_back({u(rng),u(rng),u(rng)});
    std::vector<ConstructionCandidate> candidates;
    candidates.reserve(ps.size());
    for(const Parameters &parameters:ps) {
        Solution candidate=bestInsertion(partial,parameters);
        observeProfit(candidate.profit);
        candidates.push_back({std::move(candidate),parameters});
    }
    return candidates;
}

VSSSolver::Solution VSSSolver::adaptiveConstruction(
        const Solution &partial, Parameters &center, bool recordScores) {
    std::vector<ConstructionCandidate> candidates=
            constructionCandidates(partial,center);
    Solution best; best.routes.resize(vehicles); rebuild(best); Parameters bp=center;
    int tiedBest=0;
    for (ConstructionCandidate &candidate:candidates) {
        Solution &c=candidate.solution;
        if(profitBetter(c,best)) {
            best=std::move(c); bp=candidate.parameters; tiedBest=1;
        } else if(std::abs(c.profit-best.profit)<=EPS) {
            ++tiedBest;
            std::uniform_int_distribution<int> chooseTie(1,tiedBest);
            if(chooseTie(rng)==1) {
                best=std::move(c);
                bp=candidate.parameters;
            }
        }
    }
    center=bp;
    // score_min and score_max in Sect. 4.3.2 refer to the solutions retained
    // during AIDCH, rather than all six internal heuristic trials. Recording
    // raw trials can make Delta_p0 and therefore Temp0 unrealistically large.
    if(recordScores) {
        aidchScoreMin=std::min(aidchScoreMin,best.profit);
        aidchScoreMax=std::max(aidchScoreMax,best.profit);
    }
    return best;
}
void VSSSolver::destroy(Solution &s, int count) {
    std::vector<std::pair<int,int>> positions;
    for (int r=0;r<vehicles;++r) for (int p=0;p<(int)s.routes[r].size();++p) positions.push_back({r,p});
    std::shuffle(positions.begin(),positions.end(),rng);
    count=std::min(count,(int)positions.size());
    std::vector<int> remove; for(int i=0;i<count;++i) remove.push_back(s.routes[positions[i].first][positions[i].second]);
    for(auto &r:s.routes) r.erase(std::remove_if(r.begin(),r.end(),[&](int v){return std::find(remove.begin(),remove.end(),v)!=remove.end();}),r.end());
    rebuild(s);
}
void VSSSolver::twoOpt(Solution &s) {
    for(auto &r:s.routes) {
        bool improve=true;
        while(improve) {
            improve=false;

            // reverseDeltaPrefix允许在O(1)时间内计算反转区间内部
            // 所有有向边的变化；因此该评价对非对称距离也仍然成立。
            std::vector<double> reverseDeltaPrefix(r.size(),0.0);
            for(std::size_t k=0;k+1<r.size();++k)
                reverseDeltaPrefix[k+1]=reverseDeltaPrefix[k]
                        +data.dist_mtx[r[k+1]][r[k]]
                        -data.dist_mtx[r[k]][r[k+1]];

            for(std::size_t i=0;i+1<r.size()&&!improve;++i) for(std::size_t j=i+1;j<r.size();++j) {
                const int a=i?r[i-1]:0;
                const int b=r[i];
                const int c=r[j];
                const int d=j+1<r.size()?r[j+1]:0;
                const double delta=
                        data.dist_mtx[a][c]+data.dist_mtx[b][d]
                        -data.dist_mtx[a][b]-data.dist_mtx[c][d]
                        +reverseDeltaPrefix[j]-reverseDeltaPrefix[i];
                if(delta+EPS<0.0) {
                    std::reverse(r.begin()+static_cast<std::ptrdiff_t>(i),
                                 r.begin()+static_cast<std::ptrdiff_t>(j+1));
                    improve=true;
                    break;
                }
            }
        }
    }
    rebuild(s);
}
VSSSolver::Solution VSSSolver::fullAIDCH(Solution cur) {
    if(cur.routes.empty()) cur.routes.resize(vehicles);
    rebuild(cur);
    Solution best=cur;
    AdaptiveState state;
    int stale=0;
    while(stale<n) {
        cur=adaptiveConstruction(cur,state.center,true);
        if(profitBetter(cur,best)){
            best=cur;
            stale=0;
            state.destructionMax=3;
        } else {
            ++stale;
            state.destructionMax=std::min(
                    state.destructionMax+1,
                    std::max(3,n/std::max(1,vehicles)));
        }
        if(stale>=n) break;
        const int selectedCount=static_cast<int>(std::accumulate(
                cur.routes.begin(),cur.routes.end(),std::size_t(0),
                [](std::size_t total,const auto &route){
                    return total+route.size();
                }));
        if(selectedCount>0) {
            const int upper=std::min(state.destructionMax,selectedCount);
            std::uniform_int_distribution<int> removalCount(1,upper);
            destroy(cur,removalCount(rng));
            twoOpt(cur);
        }
    }
    return best;
}

VSSSolver::ConstructionCandidate
VSSSolver::fastAIDCHNeighbor(const Solution &solution,
                             AdaptiveState &state) {
    Solution partial=solution;
    const int selectedCount=static_cast<int>(std::accumulate(
            partial.routes.begin(),partial.routes.end(),std::size_t(0),
            [](std::size_t total,const auto &route){
                return total+route.size();
            }));
    if(selectedCount>0) {
        const int upper=std::min(state.destructionMax,selectedCount);
        std::uniform_int_distribution<int> removalCount(1,std::max(1,upper));
        destroy(partial,removalCount(rng));
        twoOpt(partial);
    }
    std::vector<ConstructionCandidate> candidates=
            constructionCandidates(partial,state.center);

    ConstructionCandidate best;
    best.solution.routes.resize(vehicles);
    rebuild(best.solution);
    best.parameters=state.center;
    int tiedBest=0;
    for(ConstructionCandidate &candidate:candidates) {
        if(profitBetter(candidate.solution,best.solution)) {
            best=std::move(candidate);
            tiedBest=1;
        } else if(std::abs(candidate.solution.profit-best.solution.profit)<=EPS) {
            ++tiedBest;
            std::uniform_int_distribution<int> chooseTie(1,tiedBest);
            if(chooseTie(rng)==1) best=std::move(candidate);
        }
    }
    // A fast-AIDCH application produces one retained solution S', together
    // with the parameter triplet that generated it. SA decides whether S'
    // replaces the current child S.
    // Sect. 3.3.1 additionally applies 2-opt to the solution produced by the
    // fast AIDCH. The earlier 2-opt on the destroyed partial solution belongs
    // to the destruction/reconstruction mechanism and does not replace this
    // post-construction route improvement.
    twoOpt(best.solution);
    state.center=best.parameters;
    return best;
}

std::vector<int> VSSSolver::concat(const Solution &s) {
    std::vector<int> order(vehicles); std::iota(order.begin(),order.end(),0); std::shuffle(order.begin(),order.end(),rng);
    std::vector<std::vector<int>> gaps(vehicles+1); std::uniform_int_distribution<int> g(0,vehicles);
    std::vector<int> unserved; for(int v=1;v<=n;++v)if(!s.selected[v])unserved.push_back(v);
    std::shuffle(unserved.begin(),unserved.end(),rng); for(int v:unserved)gaps[g(rng)].push_back(v);
    std::vector<int> tour; tour.reserve(n);
    for(int i=0;i<=vehicles;++i){tour.insert(tour.end(),gaps[i].begin(),gaps[i].end());if(i<vehicles)tour.insert(tour.end(),s.routes[order[i]].begin(),s.routes[order[i]].end());}
    return tour;
}
VSSSolver::Solution VSSSolver::split(const std::vector<int>&t) const {
    std::vector<unsigned char> tourSeen(static_cast<std::size_t>(n + 1), 0);
    for(std::size_t position = 0; position < t.size(); ++position) {
        const int node = t[position];
        if(node <= 0 || node > n)
            throw std::runtime_error("split received invalid customer " +
                                     std::to_string(node) + " at position " +
                                     std::to_string(position));
        if(tourSeen[node])
            throw std::runtime_error("split received duplicate customer " +
                                     std::to_string(node) + " at position " +
                                     std::to_string(position));
        tourSeen[node] = 1;
    }
    int sz=(int)t.size();
    std::vector<int> end(sz);
    std::vector<double> profit(sz);
    for(int i=0;i<sz;++i) {
        // end[i] < i 明确表示：连客户 t[i] 单独成线都不可行。
        // 原实现默认 end[i]=0，在 i>0 时会让DP跳回tour前部并重复取区间。
        end[i]=i-1;
        double load=0.0;
        double pathTime=0.0;
        int previous=0;
        for(int j=i;j<sz;++j) {
            const int customer=t[j];
            const double nextLoad=load+data.demands[customer];
            const double nextPathTime=pathTime
                    +data.dist_mtx[previous][customer]
                    +data.serviceTime[customer];
            const double nextRouteTime=nextPathTime+data.dist_mtx[customer][0];
            if(nextLoad>capacity+EPS || nextRouteTime>duration+EPS) break;

            // 客户只追加到路线尾部，载重和路线时间均可增量维护，
            // 无需每次重新遍历当前整条路线。
            load=nextLoad;
            pathTime=nextPathTime;
            previous=customer;
            end[i]=j;
            profit[i]+=data.profits[customer];
        }
    }
    std::vector<std::vector<double>>dp(vehicles+1,std::vector<double>(sz+1));
    std::vector<std::vector<unsigned char>>take(vehicles+1,std::vector<unsigned char>(sz));
    for(int k=1;k<=vehicles;++k)for(int i=sz-1;i>=0;--i){
        const double skip=dp[k][i+1];
        const bool hasFeasibleSegment=end[i]>=i;
        const double use=hasFeasibleSegment
                ? profit[i]+dp[k-1][end[i]+1]
                : -std::numeric_limits<double>::infinity();
        if(use>skip+EPS){dp[k][i]=use;take[k][i]=1;}else dp[k][i]=skip;
    }

    // In diagnostic mode, validate the saturated-route recurrence once
    // against a more general O(m*n^2) DP that enumerates every feasible
    // interval, including all non-saturated prefixes. This check is kept out
    // of normal experiments and therefore has no production-time overhead.
    if(diagnostics&&!splitValidated) {
        std::vector<std::vector<double>> exhaustive(
                vehicles+1,std::vector<double>(sz+1));
        for(int k=1;k<=vehicles;++k) {
            for(int position=sz-1;position>=0;--position) {
                double value=exhaustive[k][position+1];
                double intervalProfit=0.0;
                for(int last=position;last<=end[position];++last) {
                    intervalProfit+=data.profits[t[last]];
                    value=std::max(value,intervalProfit
                            +exhaustive[k-1][last+1]);
                }
                exhaustive[k][position]=value;
            }
        }
        if(std::abs(exhaustive[vehicles][0]-dp[vehicles][0])>EPS) {
            throw std::runtime_error(
                    "saturated-route Split disagrees with exhaustive interval DP");
        }
        splitValidated=true;
        std::cerr << "VSS_DIAG split_validation objective="
                  << dp[vehicles][0] << " status=ok\n";
    }
    Solution s;s.routes.resize(vehicles);int i=0,k=vehicles,r=0;while(i<sz&&k>0){if(take[k][i]){for(int j=i;j<=end[i];++j)s.routes[r].push_back(t[j]);i=end[i]+1;--k;++r;}else ++i;}rebuild(s);return s;
}
std::vector<int> VSSSolver::giantTourSearch(std::vector<int> tour) {
    Solution incumbent=split(tour);
    observeProfit(incumbent.profit);

    Parameters constructionCenter;
    int destructionMax=3;
    const int destructionLimit=std::max(3,n/std::max(1,vehicles));

    // 论文中的swap实际是：取出一个客户并枚举其所有重新插入位置。
    // 一旦利润改善，立即以新序列重新开始Giant Tour局部搜索。
    auto swapOperator=[&](){
        std::vector<int> chosen=tour;
        std::shuffle(chosen.begin(),chosen.end(),rng);
        if(chosen.size()>10) chosen.resize(10);

        for(int node:chosen) {
            const auto position=std::find(tour.begin(),tour.end(),node);
            if(position==tour.end())
                throw std::runtime_error("swap operator lost a customer");
            const std::size_t oldPosition=static_cast<std::size_t>(position-tour.begin());

            std::vector<int> reduced=tour;
            reduced.erase(reduced.begin()+static_cast<std::ptrdiff_t>(oldPosition));
            Solution bestForNode=incumbent;
            std::vector<int> bestTour=tour;
            for(std::size_t insertion=0;insertion<=reduced.size();++insertion) {
                std::vector<int> candidateTour=reduced;
                candidateTour.insert(candidateTour.begin()+static_cast<std::ptrdiff_t>(insertion),node);
                Solution candidate=split(candidateTour);
                observeProfit(candidate.profit);
                if(profitBetter(candidate,bestForNode)) {
                    bestForNode=std::move(candidate);
                    bestTour=std::move(candidateTour);
                }
            }
            if(profitBetter(bestForNode,incumbent)) {
                incumbent=std::move(bestForNode);
                tour=std::move(bestTour);
                destructionMax=3;
                return true;
            }
        }
        return false;
    };

    // Construction/Destruction复用AIDCH的自适应破坏规模和六组参数构造。
    auto constructionDestructionOperator=[&](){
        Solution candidate=incumbent;
        const int selectedCount=static_cast<int>(std::accumulate(
                candidate.routes.begin(),candidate.routes.end(),std::size_t(0),
                [](std::size_t total,const auto &route){return total+route.size();}));
        if(selectedCount==0) return false;

        const int upper=std::min(destructionMax,selectedCount);
        std::uniform_int_distribution<int> removalCount(1,std::max(1,upper));
        destroy(candidate,removalCount(rng));
        twoOpt(candidate);
        candidate=adaptiveConstruction(candidate,constructionCenter);

        std::vector<int> candidateTour=concat(candidate);
        Solution extracted=split(candidateTour);
        observeProfit(extracted.profit);
        if(profitBetter(extracted,incumbent)) {
            incumbent=std::move(extracted);
            tour=std::move(candidateTour);
            destructionMax=3;
            return true;
        }
        destructionMax=std::min(destructionMax+1,destructionLimit);
        return false;
    };

    std::uniform_int_distribution<int> firstOperator(0,1);
    while(true) {
        const bool swapFirst=firstOperator(rng)==0;
        // 论文第3.4.2节规定：每轮先随机确定两个算子的顺序，然后
        // 两个算子都要执行；只要至少一个改善，才开始下一轮。
        const bool firstImproved=swapFirst
                ? swapOperator()
                : constructionDestructionOperator();
        const bool secondImproved=swapFirst
                ? constructionDestructionOperator()
                : swapOperator();
        if(firstImproved||secondImproved) continue;

        // 一整轮中的两个算子均不能提高Split所得利润时停止。
        break;
    }
    return tour;
}
VSSSolver::Solution VSSSolver::annealingSearch(Solution s) {
    Solution best=s;
    const double initialDelta=std::max(0.0,aidchScoreMax-aidchScoreMin);
    double temp=initialDelta>EPS ? -initialDelta/std::log(.95) : EPS;
    std::uniform_real_distribution<double>u(0,1);
    AdaptiveState state;
    const int destructionLimit=std::max(3,n/std::max(1,vehicles));
    int round=0;
    while(true) {
        ++round;
        // One complete fast-AIDCH neighbourhood application: destroy once,
        // run all six adaptive construction heuristics, retain their best
        // result, and apply the route-level 2-opt prescribed in Sect. 3.3.1.
        ConstructionCandidate entry=fastAIDCHNeighbor(s,state);
        Solution &candidate=entry.solution;
        const bool distinct=candidate.routes!=s.routes;
        const bool improvement=distinct&&profitBetter(candidate,s);
        const double delta=s.profit-candidate.profit;
        const bool acceptNonImprovement=distinct&&!improvement
                &&u(rng)<std::exp(-std::max(0.0,delta)
                                 /std::max(temp,EPS));
        const bool replaced=improvement||acceptNonImprovement;

        temp*=.95;
        if(replaced) {
            s=std::move(candidate);
            observeProfit(s.profit);
            if(profitBetter(s,best)) {
                best=s;
            }
        }

        if(diagnostics) {
            std::cerr << "VSS_DIAG sa_round " << round
                      << " evaluated=" << (distinct?1:0)
                      << " replaced=" << (replaced?1:0)
                      << " temperature=" << temp
                      << " dmax=" << state.destructionMax
                      << " current=" << s.profit
                      << " best=" << best.profit << '\n';
        }

        // The paper stops SA when the complete neighbourhood fails to produce
        // a new solution that replaces the current child.
        if(!replaced) break;

        if(improvement) {
            state.destructionMax=3;
        } else {
            state.destructionMax=std::min(
                    state.destructionMax+1,destructionLimit);
        }
    }
    return best;
}
VSSSolver::Solution VSSSolver::tabuSearch(Solution s) {
    struct TabuAttribute {
        int customer;
        int predecessor;
        int successor;
        int expiresAfter;
    };
    enum class MoveType { None, Remove, Insert, Swap };
    struct Move {
        MoveType type=MoveType::None;
        // For Remove this is the removed customer; for Insert it is the
        // inserted customer; for Swap it is the removed customer.
        int customer=-1;
        // The unvisited customer inserted by a Swap move.
        int replacement=-1;
        int route=-1;
        std::size_t position=0;
        // Position in the route after `customer` has been removed.
        std::size_t insertionPosition=0;
        int predecessor=0;
        int successor=0;
        double profit=-std::numeric_limits<double>::infinity();
    };

    Solution best=s;
    std::vector<TabuAttribute> tabu;
    int stale=0,iter=0;
    int executedAdd=0,executedRemove=0,executedSwap=0;
    std::uniform_int_distribution<int>tenure(5,25);

    while(stale<10) {
        ++iter;
        tabu.erase(std::remove_if(tabu.begin(),tabu.end(),
                                  [&](const TabuAttribute &attribute){
                                      return attribute.expiresAfter<iter;
                                  }),tabu.end());

        std::vector<double> routeLoads(static_cast<std::size_t>(vehicles));
        std::vector<double> routeTimes(static_cast<std::size_t>(vehicles));
        for(int r=0;r<vehicles;++r) {
            routeLoads[r]=routeLoad(s.routes[r]);
            routeTimes[r]=routeTime(s.routes[r]);
        }

        Move chosen;
        int tiedBestMoves=0;
        auto retain=[&](Move candidate){
            if(candidate.profit>chosen.profit+EPS) {
                chosen=std::move(candidate);
                tiedBestMoves=1;
            } else if(std::abs(candidate.profit-chosen.profit)<=EPS) {
                // 对目前所有最高利润并列移动做蓄水池抽样，保证均匀随机。
                ++tiedBestMoves;
                std::uniform_int_distribution<int>chooseTie(1,tiedBestMoves);
                if(chooseTie(rng)==1) chosen=std::move(candidate);
            }
        };

        // Removal邻域：移除一个已访问客户。移动本身总是保持可行，
        // 并生成论文规定的(c, predecessor, successor) Tabu属性。
        for(int r=0;r<vehicles;++r) {
            const auto &route=s.routes[r];
            for(std::size_t position=0;position<route.size();++position) {
                const int customer=route[position];
                const int predecessor=position?route[position-1]:0;
                const int successor=position+1<route.size()?route[position+1]:0;
                retain({MoveType::Remove,customer,-1,r,position,0,
                        predecessor,successor,
                        s.profit-data.profits[customer]});
            }
        }

        // Insertion邻域：把一个未访问客户插入任意可行位置。
        // 单个候选的载重、时间和目标变化均由相邻边在O(1)时间得到。
        for(int customer=1;customer<=n;++customer) {
            if(s.selected[customer]) continue;
            for(int r=0;r<vehicles;++r) {
                if(routeLoads[r]+data.demands[customer]>capacity+EPS) continue;
                const auto &route=s.routes[r];
                for(std::size_t position=0;position<=route.size();++position) {
                    const int predecessor=position?route[position-1]:0;
                    const int successor=position<route.size()?route[position]:0;
                    const double delta=data.dist_mtx[predecessor][customer]
                            +data.serviceTime[customer]
                            +data.dist_mtx[customer][successor]
                            -data.dist_mtx[predecessor][successor];
                    if(routeTimes[r]+delta>duration+EPS) continue;

                    const double candidateProfit=s.profit+data.profits[customer];
                    bool forbidden=false;
                    for(const auto &attribute:tabu) {
                        if(attribute.customer==customer
                           &&attribute.predecessor==predecessor
                           &&attribute.successor==successor) {
                            forbidden=true;
                            break;
                        }
                    }
                    // Aspiration：只有产生全局新最好利润时才允许Tabu插入。
                    if(forbidden&&candidateProfit<=observedBestProfit+EPS) continue;
                    retain({MoveType::Insert,customer,-1,r,position,
                            position,predecessor,successor,candidateProfit});
                }
            }
        }

        // Swap邻域：从一条路线移除一个已访问客户，再把一个未访问
        // 客户插入该路线中的任意可行位置。这里的Swap是客户集合空间中
        // 的一对一Replace，而不是交换两个已访问客户的访问顺序。
        for(int r=0;r<vehicles;++r) {
            const auto &route=s.routes[r];
            if(route.empty()) continue;

            for(std::size_t removePosition=0;
                removePosition<route.size();++removePosition) {
                const int removedCustomer=route[removePosition];
                const int removedPredecessor=
                        removePosition?route[removePosition-1]:0;
                const int removedSuccessor=
                        removePosition+1<route.size()
                                ?route[removePosition+1]:0;
                const double removalSaving=
                        data.dist_mtx[removedPredecessor][removedCustomer]
                        +data.serviceTime[removedCustomer]
                        +data.dist_mtx[removedCustomer][removedSuccessor]
                        -data.dist_mtx[removedPredecessor][removedSuccessor];
                const double reducedTime=routeTimes[r]-removalSaving;
                const double reducedLoad=
                        routeLoads[r]-data.demands[removedCustomer];
                const std::size_t reducedSize=route.size()-1;

                // Access a customer by its index in the route after the
                // removed customer has been omitted, without copying it.
                const auto reducedNode=[&](std::size_t index) {
                    return route[index<removePosition?index:index+1];
                };

                for(int replacement=1;replacement<=n;++replacement) {
                    if(s.selected[replacement]) continue;
                    if(reducedLoad+data.demands[replacement]
                       >capacity+EPS) continue;

                    const double candidateProfit=
                            s.profit-data.profits[removedCustomer]
                            +data.profits[replacement];
                    for(std::size_t insertionPosition=0;
                        insertionPosition<=reducedSize;
                        ++insertionPosition) {
                        const int predecessor=insertionPosition
                                ?reducedNode(insertionPosition-1):0;
                        const int successor=insertionPosition<reducedSize
                                ?reducedNode(insertionPosition):0;
                        const double insertionDelta=
                                data.dist_mtx[predecessor][replacement]
                                +data.serviceTime[replacement]
                                +data.dist_mtx[replacement][successor]
                                -data.dist_mtx[predecessor][successor];
                        if(reducedTime+insertionDelta>duration+EPS) continue;

                        bool forbidden=false;
                        for(const auto &attribute:tabu) {
                            if(attribute.customer==replacement
                               &&attribute.predecessor==predecessor
                               &&attribute.successor==successor) {
                                forbidden=true;
                                break;
                            }
                        }
                        if(forbidden
                           &&candidateProfit<=observedBestProfit+EPS) continue;

                        retain({MoveType::Swap,removedCustomer,replacement,r,
                                removePosition,insertionPosition,
                                removedPredecessor,removedSuccessor,
                                candidateProfit});
                    }
                }
            }
        }

        if(chosen.type==MoveType::None) break;
        if(chosen.type==MoveType::Remove) {
            ++executedRemove;
            auto &route=s.routes[chosen.route];
            route.erase(route.begin()+static_cast<std::ptrdiff_t>(chosen.position));
            tabu.push_back({chosen.customer,chosen.predecessor,chosen.successor,
                            iter+tenure(rng)});
        } else if(chosen.type==MoveType::Insert) {
            ++executedAdd;
            auto &route=s.routes[chosen.route];
            route.insert(route.begin()+static_cast<std::ptrdiff_t>(chosen.position),
                         chosen.customer);
        } else {
            ++executedSwap;
            auto &route=s.routes[chosen.route];
            route.erase(route.begin()+
                        static_cast<std::ptrdiff_t>(chosen.position));
            route.insert(route.begin()+static_cast<std::ptrdiff_t>(
                                 chosen.insertionPosition),
                         chosen.replacement);
            tabu.push_back({chosen.customer,chosen.predecessor,chosen.successor,
                            iter+tenure(rng)});
        }

        // Tabu小节未规定2-opt；执行移动后只统一重建并校验解。
        rebuild(s);
        observeProfit(s.profit);
        if(profitBetter(s,best)) {
            best=s;
            stale=0;
        } else {
            ++stale;
        }
    }
    if(diagnostics) {
        std::cerr << "VSS_DIAG tabu_moves add=" << executedAdd
                  << " remove=" << executedRemove
                  << " swap=" << executedSwap << '\n';
    }
    return best;
}
VSSSolver::Solution VSSSolver::routeSearch(Solution s){return mode==Mode::Tabu?tabuSearch(std::move(s)):annealingSearch(std::move(s));}

VSSSolver::Result VSSSolver::solve(){
    start=std::chrono::steady_clock::now();
    bestTime=0.0;
    observedBestProfit=0.0;
    Solution global;global.routes.resize(vehicles);rebuild(global);
    // Algorithm 1 uses >= at all ELS selection points. Therefore a newly
    // generated equal-profit solution replaces the incumbent as written.
    auto selectCandidate=[&](const Solution &candidate,const Solution &incumbent){
        if(profitBetter(candidate,incumbent)) return true;
        if(profitBetter(incumbent,candidate)) return false;
        return true;
    };
    for(int init=0;init<5;++init){
        aidchScoreMin=std::numeric_limits<double>::infinity();
        aidchScoreMax=-std::numeric_limits<double>::infinity();
        Solution empty;empty.routes.resize(vehicles);rebuild(empty);
        Solution level=fullAIDCH(empty);
        diagnostic("initial",level,init+1);
        if(selectCandidate(level,global)) global=level;

        for(int l=0;l<10;++l){
            Solution childBest;childBest.routes.resize(vehicles);rebuild(childBest);
            for(int c=0;c<10;++c){
                const Solution parent=level;
                auto tour=giantTourSearch(concat(parent));
                Solution giantSolution=split(tour);
                observeProfit(giantSolution.profit);
                diagnostic("after_giant",giantSolution,init+1,l+1);
                Solution child=routeSearch(std::move(giantSolution));
                observeProfit(child.profit);
                diagnostic(mode==Mode::Tabu?"after_tabu":"after_sa",
                           child,l+1,c+1);
                if(selectCandidate(child,childBest))childBest=child;
            }
            if(selectCandidate(childBest,level))level=std::move(childBest);
            if(selectCandidate(level,global))global=level;
        }
    }
    Result out;out.profit=global.profit;out.distance=global.distance;
    out.routes=global.routes;out.timeToBest=bestTime;
    out.totalTime=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-start).count();
    return out;
}
