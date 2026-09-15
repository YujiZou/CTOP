//
// Created by SIX6 on 2024/7/18.
//

#ifndef CTOP_INSTANCE_H
#define CTOP_INSTANCE_H

#include<string>
#include<vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "cmath"

class Instance
{
public:
    std::vector<double> x_coords;
    std::vector<double> y_coords;
    std::vector< std::vector<double> > dist_mtx;
    std::vector<double> demands;
    std::vector<double> serviceTime;
    std::vector<double> profits;
    double vehicleCapacity = 1.e30;
    int nbClients;
    int nbVehicles;
    double durationLimite;
    double totalP;
    std::string name;

    Instance(std::string pathToInstance);
    ~Instance();

};


#endif //CTOP_INSTANCE_H
