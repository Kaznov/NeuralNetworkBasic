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
        std::copy(new_values_pre.begin(), new_values_pre.end(), pre_values.begin());
        applyActivationFunction();
    }

    virtual const char* getName() = 0;

    // calculates value of the neurons, stores it inside the class
    void calculateValues(const NNLayerValues& prev_layer,
                                 const NNEdgeMatrix& edges) {
        for (size_t neuron_id = 0; neuron_id < size; ++neuron_id)
            pre_values[neuron_id] = std::inner_product(
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
        const NNLayerValues& previous_layer,
        const NNEdgeMatrix& edges
    ) {
        assert(gradient_of_activation.size() == getSize() + hasBias());
        NNLayerValues gradient_of_accumulation = // gradient of sum of edges dCost/dZ[layer] = dA[layer]/dZ[layer] * dCost / dA[layer]
                                                 // where Z is sum (w * x) (weighted input)
            calculateActivationToAccumulationGradient(gradient_of_activation);

        NNEdgeMatrix edges_gradient(getSize());
        for (auto& v : edges_gradient) v.resize(previous_layer.size());

        NNLayerValues gradient_of_prev_layer(previous_layer.size());

        for (size_t out_neuron_id = 0; out_neuron_id < getSize(); ++out_neuron_id)
        for (size_t in_neuron_id = 0; in_neuron_id < previous_layer.size(); ++in_neuron_id) {
            float e_grad = previous_layer[in_neuron_id] * gradient_of_accumulation[out_neuron_id];
            edges_gradient[out_neuron_id][in_neuron_id] = e_grad;
            gradient_of_prev_layer[in_neuron_id] +=
                gradient_of_accumulation[out_neuron_id] * edges[out_neuron_id][in_neuron_id];
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
    const size_t size;
    NNLayerValues pre_values; // without activation function, weighted inputs
    NNLayerValues values;     // with activation function

protected:
    NNLayer(size_t size, bool has_bias)
        : size(size), pre_values(size), values(size + has_bias) {
            if (has_bias) values.back() = 1;
        }

    virtual void applyActivationFunction() = 0;
    virtual NNLayerValues calculateActivationToAccumulationGradient(const NNLayerValues&) = 0;

};

class InputLayer : public NNLayer {
    public:
    InputLayer(size_t size, bool has_bias = true) : NNLayer(size, has_bias) { }
    void applyActivationFunction() override {
        std::copy(pre_values.begin(), pre_values.end(), values.begin());
    }
    const char* getName() override { return "Input layer"; }
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
    const char* getName() override { return "Sigmoid layer"; }

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
    float fp(float x) {
        float fx = f(x);
        return slope * fx * (1 - fx);
    }
};


class TanHLayer : public NNLayer {
    public:
    TanHLayer(size_t size, bool has_bias = true) : NNLayer(size, has_bias) { }
    void applyActivationFunction() override {
        std::transform(pre_values.begin(), pre_values.end(), values.begin(), [this](float x) { return this->f(x); });
    }
    const char* getName() override { return "TanH layer"; }

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
    public:
    LinearLayer(size_t size, bool has_bias = true) : NNLayer(size, has_bias) { }
    void applyActivationFunction() override {
        std::copy(pre_values.begin(), pre_values.end(), values.begin());
    }
    const char* getName() override { return "Linear layer"; }

    NNLayerValues calculateActivationToAccumulationGradient(const NNLayerValues& in) override {
        NNLayerValues results(getSize());
        for(size_t i = 0; i < getSize(); ++i) {
            results[i] = in[i];
        }
        return results;
    }
};


class LeakyRelu : public NNLayer {
    public:
    LeakyRelu(size_t size, bool has_bias = true) : NNLayer(size, has_bias) { }
    void applyActivationFunction() override {
        std::transform(pre_values.begin(), pre_values.end(), values.begin(), [](float f) {return std::max(f, 0.01f*f);});
    }
    const char* getName() override { return "Leaky Relu layer"; }

    NNLayerValues calculateActivationToAccumulationGradient(const NNLayerValues& in) override {
        NNLayerValues results(getSize());
        for(size_t i = 0; i < getSize(); ++i) {
            results[i] = in[i] * (pre_values[i] > 0 ? 1.0f : 0.01f);
        }
        return results;
    }
};

class RampLayer : public NNLayer {
    public:
    RampLayer(size_t size, bool has_bias = true, float t1 = -1.0, float t2 = 1.0)
        : NNLayer(size, has_bias), t1(t1), t2(t2) { }
    void applyActivationFunction() override {
        std::transform(pre_values.begin(), pre_values.end(), values.begin(), [this](float x) { return this->f(x); });
    }
    const char* getName() override { return "Ramp layer"; }

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
