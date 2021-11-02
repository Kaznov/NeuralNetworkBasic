#pragma once

#include <vector>
#include "NNAliases.h"

class NNLossFun {
public:
    virtual float calculateError(NNLayerValues calculated, NNLayerValues expected) = 0;
    virtual NNLayerValues calculateDerivative(NNLayerValues calculated, NNLayerValues expected) = 0;
    virtual const char* getName() = 0;
};

class MeanSquaredLossFun : public NNLossFun{
public:
    const char* getName() override { return "Square mean"; }
    float calculateError(NNLayerValues calculated, NNLayerValues expected) override {
        float result = 0;
        for (size_t i = 0; i < calculated.size(); ++i) {
            float diff = calculated[i] - expected[i];
            result += diff * diff;
        }
        return result / 2;
    }
    NNLayerValues calculateDerivative(NNLayerValues calculated, NNLayerValues expected) override {
        NNLayerValues result( calculated.size() );
        for (size_t i = 0; i < calculated.size(); ++i) {
            result[i] = expected[i] - calculated[i];//TOTO;
        }

        return result;
    }
};

class LogLoss : public NNLossFun{
public:
    const char* getName() override { return "Log Loss"; }
    float calculateError(NNLayerValues calculated, NNLayerValues expected) override {
        return 0.0f;
    }
    NNLayerValues calculateDerivative(NNLayerValues calculated, NNLayerValues expected) override {
        return {};
    }
};