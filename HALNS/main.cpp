#include "HALNSSolver.h"
#include "Instance.h"

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <instance-path>\n";
        return 1;
    }

    try {
        const std::string instancePath = argv[1];
        std::random_device rd;
        const std::uint32_t seed = rd();
        std::cout << "Seed = " << seed << '\n' << std::flush;

        Instance instance(instancePath);
        HALNSSolver solver(instance, seed);
        const auto result = solver.solve();

        // Output columns: instance, best objective, time to best, total time.
        // The HALNS article reports CPU time, so use the CPU measurements here.
        std::cout << std::fixed << std::setprecision(6)
                  << instancePath << ' ' << result.profit << ' '
                  << result.timeToBestCpu << ' ' << result.totalCpuTime << '\n';
    } catch (const std::exception &error) {
        std::cerr << "HALNS error: " << error.what() << '\n';
        return 2;
    }

    return 0;
}
