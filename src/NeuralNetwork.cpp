#include "NeuralNetwork.h"
#include <algorithm>
#include <cassert>
#include <numeric>

#include "utils.h"

void NeuralNetwork::initializeWithRandomData() {
    std::uniform_real_distribution<float> dis{};
    for (auto& matrix : connections)
    for (auto& neuron_out : matrix)
    for (auto& neuron_in : neuron_out)
        neuron_in = dis(RNG);
}

void NeuralNetwork::evaluateNetwork(const std::vector<float>& input) {
    assert(input.size() == layers[0].getSize());
    layers[0].assignValues(input);
    for(size_t l = 1; l < layers.size(); ++l)
        layers[l].calculateValues(layers[l-1].values, connections[l-1]);
}

std::vector<NNEdgeMatrix> NeuralNetwork::gradientDescent(const NNLayerValues& last_layer_gradient) {
    std::vector<NNEdgeMatrix> edges_gradients;
    auto last_gradient = last_layer_gradient;
    for (size_t l = this->layers.size() - 1; l > 0; l--) {
        auto r = layers[l].backwardPropagation(last_gradient, layers[l - 1].values);
        last_gradient = r.second;
        edges_gradients.insert(edges_gradients.begin(), std::move(r.first));
    }
    return edges_gradients;
}


NNLayer& NeuralNetwork::getNthLayerAfterEvaluation(size_t n) {
    return layers[n];
}
NNLayer& NeuralNetwork::getLastLayerAfterEvaluation() {
    return layers.back();
}

NNEdgeMatrix& NeuralNetwork::getNthLayerEdges(size_t n) {
    return connections[n];
}