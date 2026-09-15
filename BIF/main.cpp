#include "BiFSolver.h"
#include "Instance.h"

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

namespace
{
// 服务器批量实验只接收算例路径。BiF 的快/慢版本在这里切换。
constexpr BiFSolver::Mode SERVER_MODE = BiFSolver::Mode::Slow;

void printUsage(const char *program)
{
    std::cerr << "Usage: " << program << " <instance-path>\n";
}

// 屏蔽求解器内部的进度信息，服务器实验的标准输出只保留最终一行。
class NullBuffer : public std::streambuf
{
public:
    int overflow(int character) override
    {
        return traits_type::not_eof(character);
    }
};

class ScopedStdoutSilencer
{
public:
    ScopedStdoutSilencer() : original(std::cout.rdbuf(&buffer)) {}
    ~ScopedStdoutSilencer() { std::cout.rdbuf(original); }

    ScopedStdoutSilencer(const ScopedStdoutSilencer &) = delete;
    ScopedStdoutSilencer &operator=(const ScopedStdoutSilencer &) = delete;

private:
    NullBuffer buffer;
    std::streambuf *original;
};

std::string instanceName(const std::string &path)
{
    // 保留父目录以区分 2set/b1.txt 和 3set/b1.txt 等同名算例。
    const std::size_t fileSeparator = path.find_last_of("/\\");
    if(fileSeparator == std::string::npos) return path;

    const std::size_t parentSeparator =
            fileSeparator == 0 ? std::string::npos
                               : path.find_last_of("/\\", fileSeparator - 1);
    std::string name = parentSeparator == std::string::npos
                       ? path
                       : path.substr(parentSeparator + 1);
    for(char &character : name)
        if(character == '\\') character = '/';
    return name;
}
}

int main(int argc, char **argv)
{
    if(argc != 2)
    {
        printUsage(argv[0]);
        return 1;
    }

    try
    {
        const std::string instancePath = argv[1];
        // 每次运行自动生成随机种子，但批量输出中不再增加seed列。
        std::random_device randomDevice;
        const std::uint32_t seed = randomDevice();

        // 复用项目原有且已验证的算例读取代码，不在 BiF 中重复解析数据。
        Instance instance(instancePath);
        BiFSolver solver(instance, SERVER_MODE, seed);
        const BiFSolver::Result result = [&solver]()
        {
            ScopedStdoutSilencer silenceProgressOutput;
            return solver.solve();
        }();

        // 成功输出固定为两行：seed；算例名、最好值、time-to-best、总时间。
        std::cout << "Seed = " << seed << '\n'
                  << std::fixed << std::setprecision(6)
                  << instanceName(instancePath) << ' '
                  << result.profit << ' '
                  << result.timeToBest << ' '
                  << result.totalTime << '\n';
    }
    catch(const std::exception &error)
    {
        std::cerr << "BiF error: " << error.what() << '\n';
        return 2;
    }
    return 0;
}
