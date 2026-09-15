#include "Instance.h"
#include "VSSSolver.h"

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#ifndef VSS_LOCAL_SEARCH_MODE
#error "VSS_LOCAL_SEARCH_MODE must be defined as 0 (Tabu) or 1 (SA)"
#endif

namespace
{
#if VSS_LOCAL_SEARCH_MODE == 0
constexpr VSSSolver::Mode COMPILED_MODE = VSSSolver::Mode::Tabu;
#elif VSS_LOCAL_SEARCH_MODE == 1
constexpr VSSSolver::Mode COMPILED_MODE =
        VSSSolver::Mode::SimulatedAnnealing;
#else
#error "VSS_LOCAL_SEARCH_MODE must be 0 (Tabu) or 1 (SA)"
#endif
}

int main(int argc, char **argv)
{
    if(argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <instance-path>\n";
        return 1;
    }

    try
    {
        const std::string instancePath = argv[1];

        // 论文将VSS作为随机算法，并对每个算例执行10次独立运行。
        // 每次启动自动产生新seed，并输出该seed以便复现实验。
        std::random_device randomDevice;
        const std::uint32_t seed = randomDevice();
        std::cout << "Seed = " << seed << '\n' << std::flush;

        Instance instance(instancePath);
        VSSSolver solver(instance, COMPILED_MODE, seed);
        const VSSSolver::Result result = solver.solve();

        // Output columns: instance, best objective, time to best, total time.
        std::cout << std::fixed << std::setprecision(6)
                  << instancePath << ' ' << result.profit << ' '
                  << result.timeToBest << ' ' << result.totalTime << '\n';
    }
    catch(const std::exception &error)
    {
        std::cerr << "VSS error: " << error.what() << '\n';
        return 2;
    }

    return 0;
}
