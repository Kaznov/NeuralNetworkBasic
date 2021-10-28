#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <random>

#include "NNAliases.h"
#include "NNLayer.h"

class NeuralNetwork {
public:
    void addLayer(std::unique_ptr<NNLayer>);
    void initializeWithRandomData();
    void evaluateNetwork(const NNLayerValues& input);

    // before calling that, reassign all neurons!!!
    // returns gradient of edges
    std::vector<NNEdgeMatrix> gradientDescent(const NNLayerValues& last_layer_gradient);

    NNLayer& getNthLayerAfterEvaluation(size_t n);
    NNLayer& getLastLayerAfterEvaluation();
    NNEdgeMatrix& getNthLayerEdges(size_t n);

    std::vector<NNEdgeMatrix> connections;
    std::vector<NNLayer> layers;
};