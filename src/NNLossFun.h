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
        NNLayerValues& ye = expected;
        NNLayerValues& yc = calculated;
        float result = 0;
        for (size_t i = 0; i < yc.size(); ++i) {
            float diff = ye[i] - yc[i];
            result += diff * diff;
        }
        return result / 2;
    }
    NNLayerValues calculateDerivative(NNLayerValues calculated, NNLayerValues expected) override {
        NNLayerValues& ye = expected;
        NNLayerValues& yc = calculated;
        NNLayerValues result( calculated.size() );
        for (size_t i = 0; i < calculated.size(); ++i) {
            result[i] = ye[i] - yc[i];//TOTO;
        }

        return result;
    }
};

class LogLoss : public NNLossFun{
public:
    const char* getName() override { return "Log Loss"; }

    static NNLayerValues applyTransform(const NNLayerValues& calculated, const NNLayerValues& expected) {
        NNLayerValues yc = calculated;
        std::transform(yc.begin(), yc.end(), yc.begin(), [](float f) { return std::exp(f); });
        float sum_of_exps = std::accumulate(yc.begin(), yc.end(), 0.0f);
        std::transform(yc.begin(), yc.end(), yc.begin(), [=](float f) { return f / sum_of_exps; });
        return yc;
    }

    float calculateError(NNLayerValues calculated, NNLayerValues expected) override {
        NNLayerValues& ye = expected;
        NNLayerValues& yc = calculated;

        yc = applyTransform(yc, {});

        NNLayerValues res_v(yc.size());
        for (size_t i = 0; i < yc.size(); ++i) {
            res_v[i] = -ye[i] * log(yc[i]) - (1 - ye[i]) * log(1 - yc[i]);
        }
        float result = std::accumulate(res_v.begin(), res_v.end(), 0.0f);
        return result;
    }
    NNLayerValues calculateDerivative(NNLayerValues calculated, NNLayerValues expected) override {
        NNLayerValues& ye = expected;
        NNLayerValues& yc = calculated;
        NNLayerValues res(yc.size());
        for (size_t i = 0; i < yc.size(); ++i) {
            res[i] = yc[i] - ye[i];
        }
        return res;
    }
};