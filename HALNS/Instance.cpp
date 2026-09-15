//
// Created by SIX6 on 2024/7/18.
//
#include "Instance.h"

Instance::Instance(std::string pathToInstance)
{
    std::ifstream inputFile(pathToInstance);

    std::string line;
    demands.push_back(0);
    serviceTime.push_back(0);
    profits.push_back(0);
    std::string dataType = "old";


    if (inputFile.is_open())
    {
        if(dataType == "old")
        {
            while (std::getline(inputFile, line))
            {
                if (line.find("NAME") == 0)
                {
                    name = line.substr(5); // Extract name after "NAME "
                } else if (line.find("MAXVEHICLES") == 0)
                {
                    nbVehicles = std::stoi(line.substr(12)); // Extract maxVehicles after "MAXVEHICLES "
                } else if (line.find("MAXCAPACITY") == 0)
                {
                    vehicleCapacity = std::stod(line.substr(12)); // Extract maxCapacity after "MAXCAPACITY "
                } else if (line.find("MAXDURATION") == 0)
                {
                    durationLimite = std::stod(line.substr(12)); // Extract maxTime after "MAXTIME "
                } else if (line.find("DEPOT") == 0)
                {
                    std::istringstream iss(line.substr(6)); // Extract coordinates after "DEPOT "
                    double depotX, depotY;
                    iss >> depotX >> depotY;
                    x_coords.push_back(depotX);
                    y_coords.push_back(depotY);
                }
                else if (line.find("CUSTOMERS") == 0)
                {
                    nbClients = std::stoi(line.substr(10)); // Extract customers count after "CUSTOMERS "
                    dist_mtx = std::vector < std::vector< double > > (nbClients + 1, std::vector <double>(nbClients+1));
                } else if (line.find("CUSTOMERDATA") == 0)
                {
                    // Start reading customer data
                    while (std::getline(inputFile, line)) {
                        std::istringstream iss(line);
                        double x, y, demand, serviceTime1;
                        double profit;
                        if (iss >> x >> y >> demand >> serviceTime1 >> profit) {
                            x_coords.push_back(x);
                            y_coords.push_back(y);
                            demands.push_back(demand);
                            serviceTime.push_back(serviceTime1);
                            profits.push_back(profit);
                        }
                    }
                    break; // End reading customer data
                }
            }
        }
        else if(dataType == "new")
        {
            while (std::getline(inputFile, line))
            {
                if(line.find("NAME :") == 0)
                {
                    name = line.substr(6); // Extract name after "NAME "
                }else if(line.find("MAXVEHICLES :") == 0)
                {
                    std::string substr1 = line.substr(13);
                    substr1 = substr1.substr(substr1.find_first_not_of(" \t"));
                    substr1 = substr1.substr(0, substr1.find_last_not_of(" \t\r\n") + 1);
                    nbVehicles = std::stoi(substr1); // Extract maxVehicles after "MAXVEHICLES "
                }else if(line.find("CAPACITY") == 0)
                {
                    vehicleCapacity = std::stod(line.substr(10)); // Extract maxCapacity after "MAXCAPACITY "
                }else if(line.find("DISTANCE") == 0)
                {
                    durationLimite = std::stod(line.substr(10)); // Extract maxTime after "MAXTIME "
                }else if(line.find("DIMENSION") == 0)
                {
                    nbClients = std::stoi(line.substr(11)); // Extract customers count after "CUSTOMERS "
                    nbClients--;
                    dist_mtx = std::vector < std::vector< double > > (nbClients + 1, std::vector <double>(nbClients+1));
                }else if(line.find("TOTALPROFITS :") == 0)
                {
                    totalP = std::stod(line.substr(14));
                }
                else if(line.find("NODE_COORD_SECTION") == 0)
                {
                    // Start reading customer data
                    int counter =0;
                    while(std::getline(inputFile, line))
                    {
                        counter++;
                        std::istringstream iss(line);
                        double x, y;
                        int index;
                        if(iss >>index >> x >> y )
                        {
                            x_coords.push_back(x);
                            y_coords.push_back(y);
                        }
                        if(counter == nbClients + 1)
                            break;
                    }// End reading customer data
                }
                else if(line.find("SERVICE_TIME :") == 0)
                {
                    double st = std::stod(line.substr(15));
                    while(serviceTime.size() < nbClients + 1)
                    {
                        serviceTime.push_back(st);
                    }
                }
                else if(line.find("DEMAND&PROFITS_SECTION") == 0)
                {
                    int counter = 0;
                    while(std::getline(inputFile, line))
                    {
                        counter++;
                        std::istringstream iss(line);
                        double demand1, profit1;
                        int index;
                        if(iss >>index >> demand1 >> profit1 )
                        {
                            demands.push_back(demand1);
                            profits.push_back(profit1);
                        }
                        if(counter == nbClients + 1)
                            break;
                    }
                    demands.erase(demands.begin());
                    profits.erase(profits.begin());
                }
            }
        }
        else if (dataType == "real")
        {
            while (std::getline(inputFile, line))
            {
                if (line.find("NAME") == 0)
                {
                    name = line.substr(5); // Extract name after "NAME "
                } else if (line.find("MAXVEHICLES") == 0)
                {
                    nbVehicles = std::stoi(line.substr(12)); // Extract maxVehicles after "MAXVEHICLES "
                } else if (line.find("MAXCAPACITY") == 0)
                {
                    vehicleCapacity = std::stod(line.substr(12)); // Extract maxCapacity after "MAXCAPACITY "
                } else if (line.find("MAXDURATION") == 0)
                {
                    durationLimite = std::stod(line.substr(12)); // Extract maxTime after "MAXTIME "
                } else if (line.find("DEPOT") == 0)
                {
                    std::istringstream iss(line.substr(6)); // Extract coordinates after "DEPOT "
                    double depotX, depotY;
                    iss >> depotX >> depotY;
                    x_coords.push_back(depotX);
                    y_coords.push_back(depotY);
                }
                else if (line.find("CUSTOMERS") == 0)
                {
                    nbClients = std::stoi(line.substr(10)); // Extract customers count after "CUSTOMERS "
                    dist_mtx = std::vector < std::vector< double > > (nbClients + 1, std::vector <double>(nbClients+1));
                }
                else if (line.find("SERVICETIME") == 0)
                {
                    std::string substr = line.substr(12);
                    double servicetime = std::stod(substr);
                    while(serviceTime.size() < nbClients + 1)
                        serviceTime.push_back(servicetime);
                }
                else if (line.find("CUSTOMERDATA") == 0)
                {
                    // Start reading customer data
                    int counter = 0;
                    while (std::getline(inputFile, line)) {
                        counter++;
                        std::istringstream iss(line);
                        double  demand, serviceTime1;
                        double profit;
                        int index;
                        if (iss >>index >> demand >> profit) {
                            if(counter != 1)
                            {
                                demands.push_back(demand);
                                profits.push_back(profit);
                            }
                        }
                        if(counter == nbClients + 1)
                            break;
                    }
                  //  break; // End reading customer data
                }
                else if(line.find("EDGE WEIGHT") == 0)
                {
                    std::string disVal;
                    int counter = 0;
                    while (std::getline(inputFile, line))
                    {
                        std::stringstream ss(line);
                        int colcounter = 0;
                        while(ss >> disVal)
                        {
                            //std::cout << counter << " " << colcounter << " " << disVal << std::endl;
                            dist_mtx[counter][colcounter++] = std::stod(disVal);

                        }
                        counter++;
                        if(counter == nbClients + 1)
                            break;
                    }

                }
                else if(line.find("EOF") == 0)
                    break;
            }

        }
    }
    inputFile.close();

    if(dataType != "real")
    {
        for(int i=0; i < nbClients+1; i++)
        {
            for(int j = 0; j < nbClients + 1; j++)
            {
                dist_mtx[i][j] = std::sqrt(
                        (x_coords[i] - x_coords[j]) * (x_coords[i] - x_coords[j])
                        + (y_coords[i] - y_coords[j]) * (y_coords[i] - y_coords[j])
                );
                //dist_mtx[i][j] = std::round( dist_mtx[i][j]);
                //dist_mtx[i][j] = std::floor( dist_mtx[i][j]);
            }
        }
    }

}
Instance::~Instance()
{

}
