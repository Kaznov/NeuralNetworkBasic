#pragma once

#include "NNAliases.h"


class NNMomentum {
public:
    virtual void applyMomentum(std::vector<NNEdgeMatrix>&) = 0;    
};

class NNSteadyLearningRate : public NNMomentum {
public:
    NNSteadyLearningRate(float learning_rate = 0.05);
    void applyMomentum(std::vector<NNEdgeMatrix>& matrices) override {
        for (auto& m : matrices)
        for (auto& r : m)
        for (auto& c : r)
        c *= learning_rate;
    }

private:
    float learning_rate; 
};
