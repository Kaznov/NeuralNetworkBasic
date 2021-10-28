#pragma once

#include <algorithm>
#include <vector>
#include <memory>

#include "utils.h"
#include "DataPoint.h"
#include "NeuralNetwork.h"
#include "NNLossFun.h"
#include "NNMomentum.h"
#include "NNTerminator.h"

// Contains all the training cases and verification cases.
// Contains a scheduler, a momentum keeper and a terminator of NN. 
class NNTeacher {
public:
    void addNetwork(std::unique_ptr<NeuralNetwork> nn) {
        network = std::move(nn);
    }
    void addTerminator(std::unique_ptr<NNTerminator> term) {
        terminator = std::move(term);
    }
    void addLossFunction(std::unique_ptr<NNLossFun> loss) {
        loss_fun = std::move(loss);
    }
    void addTrainingDataSet(std::vector<DataPoint> data) {
        dataset = std::move(data);
    }
    void addMomentum(std::unique_ptr<NNMomentum> mom) {
        momentum = std::move(mom);
    }

    bool hasNextBatch() {
        return !batches.empty();
    }

    void learnBatch() {
        if (batches.empty()) throw "woopsie";
        if (finished()) return;
        std::vector<DataPoint> batch = std::move(batches.back());
        batches.pop_back();
        
        std::vector<std::vector<NNEdgeMatrix>> gradients;
        
        // backprop for all in batch
        for (auto&& dp : batch) {
            network->evaluateNetwork(dp.input);
            auto err_der = loss_fun->calculateDerivative(network->getLastLayerAfterEvaluation().values, dp.output);
            auto err = loss_fun->calculateError(network->getLastLayerAfterEvaluation().values, dp.output);
            error_values.push_back(err);
            auto grad = network->gradientDescent(err_der);
            gradients.push_back(grad);
        }

        auto addMatrices = [](const std::vector<NNEdgeMatrix>& v_in, std::vector<NNEdgeMatrix>& v_out) {
            for (size_t matrix_id = 0; matrix_id < v_in.size(); ++matrix_id) {
                const auto& m_in = v_in[matrix_id];
                auto& m_out = v_out[matrix_id];
                for (size_t row_id = 0; row_id < m_in.size(); ++ row_id) {
                    const auto& row_in = m_in[row_id];
                    auto& row_out = m_out[row_id];
                    for (size_t col_id = 0; col_id < row_in.size(); ++col_id) {
                        row_out[col_id] += row_in[col_id];
                    }
                }
            }
        };

        // Sum all gradients
        std::vector<NNEdgeMatrix> grad_mean = gradients[0];
        size_t M = batch.size();
        for (size_t i = 1; i < M; ++i) {
            addMatrices(gradients[i], grad_mean);
        } 
        
        // reduce by size of batch (calulate mean)
        for (auto& matrix : grad_mean)
            for (auto& row : matrix)
                for (auto& col : row)
                   col /= M; 

        // apply momentum and learning factor
        momentum->applyMomentum(grad_mean);

        // apply changes to the network
        addMatrices(grad_mean, network->connections);
    }

    // starts a new epoch
    void generateBatches() {
        checkFinish();
        if (finished()) return;
        batches.clear();
        std::shuffle(dataset.begin(), dataset.end(), RNG);
        size_t i;
        for (i = 0; i + batch_size < dataset.size(); i += batch_size) {
            batches.push_back(std::vector(dataset.begin() + i, dataset.begin() + i + batch_size));
        }
        if (dataset.begin() + i != dataset.end())
            batches.push_back(std::vector(dataset.begin() + i, dataset.end()));
    }

    void learnEpoch() {
        generateBatches();
        if (finished()) return;
        while (hasNextBatch()) learnBatch();

    }

    bool finished() {
        return stopped;
    }

    NeuralNetwork& getNetwork() {
        return *network;
    }

    void checkFinish() {
        
        float total_error;
        if (error_values.size() == 0)
            total_error = INFINITY;
        else
            total_error = std::accumulate(error_values.begin(), error_values.end(), 0.0) / error_values.size();

        stopped = terminator->shouldFinish(total_error);
    }

private:
    std::unique_ptr<NNMomentum> momentum;
    std::unique_ptr<NNTerminator> terminator;
    std::unique_ptr<NNLossFun> loss_fun;
    std::unique_ptr<NeuralNetwork> network;
    std::vector<DataPoint> dataset;
    std::vector<std::vector<DataPoint>> batches;
    std::vector<float> error_values;
    
    size_t next_to_take = 0;
    size_t batch_size = 0;
    bool stopped = false;

};