#pragma once

#include <vector>
#include "NNAliases.h"

class NNLossFun {
public:
    virtual float calculateError(NNLayerValues calculated, NNLayerValues expected) = 0;
    virtual NNLayerValues calculateDerivative(NNLayerValues calculated, NNLayerValues expected) = 0;
};