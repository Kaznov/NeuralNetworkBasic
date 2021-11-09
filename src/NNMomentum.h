#pragma once

#include "NNAliases.h"


class NNMomentum {
public:
    virtual void applyMomentum(std::vector<NNEdgeMatrix>& matrices) = 0;
    virtual std::string toString() =0;
};

class NNSteadyLearningRate : public NNMomentum {
public:
    NNSteadyLearningRate(float learning_rate = 0.05, float gradient_threshold = 1.0)
        : learning_rate{learning_rate},
          gradient_threshold{gradient_threshold}{}
    void applyMomentum(std::vector<NNEdgeMatrix>& matrices) override {
        // first, apply learning rate
        for (auto& m : matrices)
        for (auto& r : m)
        for (auto& c : r)
            c *= -learning_rate;

        // then, replace all NaNs with 0s
        for (auto& m : matrices)
        for (auto& r : m)
        for (auto& c : r)
            if (!std::isfinite(c))
                c = 0;

        float gradient_norm = 0.0f;
        for (auto& m : matrices)
        for (auto& r : m)
        for (auto& c : r)
            gradient_norm += c*c;

        // Then, try to clip gradient
        https://arxiv.org/pdf/1211.5063.pdf

        if (gradient_norm > gradient_threshold)
        for (auto& m : matrices)
        for (auto& r : m)
        for (auto& c : r)
            c = c * gradient_threshold / gradient_norm;

    }

    std::string toString() override {
        return "Learning rate: " + std::to_string(learning_rate)
        + ", gradient threshold: " + std::to_string(gradient_threshold);
    }

private:
    float learning_rate;
    float gradient_threshold;
};
