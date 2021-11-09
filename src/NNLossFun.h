#pragma once

#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>
#include "NNAliases.h"

class NNLossFun {
public:
    virtual float calculateError(NNLayerValues calculated, NNLayerValues expected) = 0;
    virtual NNLayerValues calculateDerivative(NNLayerValues calculated, NNLayerValues expected) = 0;
    virtual const char* getName() = 0;
    virtual NNLayerValues normalize(const NNLayerValues&) = 0;
};

class MeanSquaredLossFun : public NNLossFun{
public:
    const char* getName() override { return "Square mean"; }
    float calculateError(NNLayerValues calculated, NNLayerValues expected) override {
        NNLayerValues& ye = expected;
        NNLayerValues& yc = calculated;
        float result = 0;
        for (size_t i = 0; i < yc.size(); ++i) {
            float diff = yc[i] - ye[i];
            result += diff * diff;
        }
        return result / 2;
    }
    NNLayerValues calculateDerivative(NNLayerValues calculated, NNLayerValues expected) override {
        NNLayerValues& ye = expected;
        NNLayerValues& yc = calculated;
        NNLayerValues result( calculated.size() );
        for (size_t i = 0; i < calculated.size(); ++i) {
            result[i] = yc[i] - ye[i];//TOTO;
        }

        return result;
    }

    NNLayerValues normalize(const NNLayerValues& v) override {
        return v;
    }
};

class LogLoss : public NNLossFun{
public:
    const char* getName() override { return "Log Loss"; }

    NNLayerValues normalize(const NNLayerValues& calculated) override{
        NNLayerValues yc = calculated;
        float D = - *std::max_element(yc.begin(), yc.end());
        std::transform(yc.begin(), yc.end(), yc.begin(), [=](float f) { return std::exp(f + D); });
        float sum_of_exps = std::accumulate(yc.begin(), yc.end(), 0.0f);
        std::transform(yc.begin(), yc.end(), yc.begin(), [=](float f) { return f / sum_of_exps; });
        return yc;
    }

    float calculateError(NNLayerValues calculated, NNLayerValues expected) override {
        NNLayerValues& ye = expected;
        NNLayerValues& yc = calculated;

        yc = normalize(yc);

        NNLayerValues res_v(yc.size());
        for (size_t i = 0; i < yc.size(); ++i) {
            res_v[i] = -ye[i] * log(yc[i]);
        }
        float result = std::accumulate(res_v.begin(), res_v.end(), 0.0f);
        if (!std::isfinite(result)) result = 0;
        return result;
    }
    NNLayerValues calculateDerivative(NNLayerValues calculated, NNLayerValues expected) override {
        NNLayerValues& ye = expected;
        NNLayerValues& yc = calculated;

        yc = normalize(yc);
        NNLayerValues res(yc.size());
        for (size_t i = 0; i < yc.size(); ++i) {
            res[i] = yc[i] - ye[i];
        }
        return res;
    }
};