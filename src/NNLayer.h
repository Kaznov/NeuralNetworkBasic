#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <numeric>

#include "NNAliases.h"

// an abstract class for all kinds of layers
// manages forward and backward propagation
class NNLayer {
public:
    void assignValues(const NNLayerValues& new_values_pre) {
        assert(new_values_pre.size() == getSize()); // no bias in data, duh
        std::copy(new_values_pre.begin(), new_values_pre.end(), values.begin());
        applyActivationFunction();
    }

    // calculates value of the neurons, stores it inside the class
    void calculateValues(const NNLayerValues& prev_layer,
                                 const NNEdgeMatrix& edges) {
        for (size_t neuron_id = 0; neuron_id < size; ++neuron_id)
            values[neuron_id] = std::inner_product(
                                    std::begin(prev_layer),
                                    std::end(prev_layer),
                                    std::begin(edges[neuron_id]),
                                    0.0f);
        applyActivationFunction();
    }

    // given gradient of the layer, returns gradient of the previous edges
    // and gradient of the previous layer 
    virtual std::pair<NNEdgeMatrix, NNLayerValues> backwardPropagation(
        const NNLayerValues& gradient_of_activation, // gradient dCost/dA(Layer)
        const NNLayerValues& previous_layer
    ) {
        assert(gradient_of_activation.size() == getSize() + hasBias());
        NNLayerValues gradient_of_accumulation = // gradient of sum of edges dCost/dZ[layer] = dA[layer]/dZ[layer] * dCost / dA[layer]
                                                 // where Z is sum (w * x) (weighted input)
            calculateActivationToAccumulationGradient(gradient_of_activation);
        
        NNEdgeMatrix edges_gradient(getSize());
        for (auto& v : edges_gradient) v.resize(previous_layer.size());

        NNLayerValues gradient_of_prev_layer(previous_layer.size());

        for (size_t out_neuron_id  = 0; out_neuron_id < getSize(); ++out_neuron_id)
        for (size_t in_neuron_id = 0; in_neuron_id < previous_layer.size(); ++in_neuron_id) {
            float e_grad = previous_layer[in_neuron_id] * gradient_of_accumulation[out_neuron_id];
            edges_gradient[out_neuron_id][in_neuron_id] = e_grad;
            gradient_of_prev_layer[in_neuron_id] += e_grad;
        }

        return {edges_gradient, gradient_of_prev_layer};            
    };

    size_t getSize() const { return size; }
    size_t getFullSize() const { return values.size(); }

    bool hasBias() {
        return getSize() != values.size();
    }

    // oh no public data
    // anyway
    NNLayerValues pre_values; // without activation function, weighted inputs
    NNLayerValues values;     // with activation function

protected:
    NNLayer(size_t size, bool has_bias) 
        : pre_values(size), values(size + has_bias) {
            if (has_bias) values.back() = 1.0f;
        }
    
    virtual void applyActivationFunction() = 0;
    virtual NNLayerValues calculateActivationToAccumulationGradient(const NNLayerValues&) = 0;


private:
    size_t size;
};

class InputLayer : public NNLayer {
    InputLayer(size_t size, bool has_bias = true) : NNLayer(size, has_bias) { }
    void applyActivationFunction() override {
        std::copy(pre_values.begin(), pre_values.end(), values.begin());
    }
    NNLayerValues calculateActivationToAccumulationGradient(const NNLayerValues&) override {
        throw "Wrong usage"; 
    }
};

class SigmoidLayer : public NNLayer {
public:
    SigmoidLayer(size_t size, bool has_bias = true, float slope = 1.0f) : NNLayer(size, has_bias), slope{slope} { }
    void applyActivationFunction() override {
        std::transform(pre_values.begin(), pre_values.end(), values.begin(), [this](float x) { return this->f(x); });
    }

    NNLayerValues calculateActivationToAccumulationGradient(const NNLayerValues& in) override {
        NNLayerValues results(getSize());
        for(size_t i = 0; i < getSize(); ++i) {
            results[i] = fp(pre_values[i]) * in[i];
        }
        return results;
    }
private:
    float slope;
    float f(float x) { return 1.0f / (1.0f + expf(-slope * x)); }
    float fp(float x) { float fx = f(x); return slope * fx * (1 - fx); }
};


class TanHLayer : public NNLayer {
    TanHLayer(size_t size, bool has_bias = true) : NNLayer(size, has_bias) { }
    void applyActivationFunction() override {
        std::transform(pre_values.begin(), pre_values.end(), values.begin(), [this](float x) { return this->f(x); });
    }

    NNLayerValues calculateActivationToAccumulationGradient(const NNLayerValues& in) override {
        NNLayerValues results(getSize());
        for(size_t i = 0; i < getSize(); ++i) {
            results[i] = fp(pre_values[i]) * in[i];
        }
        return results;
    }

    float f(float x) { return tanhf(x); }
    float fp(float x) { float fx = f(x); return 1 - fx*fx; }
};


class LinearLayer : public NNLayer {
    LinearLayer(size_t size, bool has_bias = true) : NNLayer(size, has_bias) { }
    void applyActivationFunction() override {
        std::copy(pre_values.begin(), pre_values.end(), values.begin());
    }

    NNLayerValues calculateActivationToAccumulationGradient(const NNLayerValues& in) override {
        NNLayerValues results(getSize());
        for(size_t i = 0; i < getSize(); ++i) {
            results[i] = in[i];
        }
        return results;
    }
};

class RampLayer : public NNLayer {
    RampLayer(size_t size, bool has_bias = true, float t1 = -1.0, float t2 = 1.0)
        : NNLayer(size, has_bias), t1(t1), t2(t2) { }
    void applyActivationFunction() override {
        std::transform(pre_values.begin(), pre_values.end(), values.begin(), [this](float x) { return this->f(x); });
    }

    NNLayerValues calculateActivationToAccumulationGradient(const NNLayerValues& in) override {
        NNLayerValues results(getSize());
        for(size_t i = 0; i < getSize(); ++i) {
            results[i] = fp(pre_values[i]) * in[i];
        }
        return results;
    }

    float f(float x) { if (x < t1) return 0; if (x < t2) return (x - t1)/(t2 - t1); return 1; }
    float fp(float x) { if (x < t1) return 0; if (x < t2) return 1/(t2 - t1); return 0; }

    float t1, t2;
};

