#pragma once

#include <algorithm>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>

#include <iostream>
#include <iomanip>

#include "utils.h"
#include "DataPoint.h"
#include "NeuralNetwork.h"
#include "NNLossFun.h"
#include "NNMomentum.h"
#include "NNTerminator.h"

extern bool debug;

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
        if (dataset.empty()) return;

        DataPoint min_dp;
        min_dp.input.resize(dataset[0].input.size());
        min_dp.output.resize(dataset[0].output.size());

        DataPoint max_dp;
        max_dp.input.resize(dataset[0].input.size());
        max_dp.output.resize(dataset[0].output.size());

        for (const auto& p : dataset) {
            for(size_t i = 0; i < p.input.size(); ++i) {
                min_dp.input[i] = std::min(min_dp.input[i], p.input[i]);
                max_dp.input[i] = std::max(max_dp.input[i], p.input[i]);
            }
            for(size_t i = 0; i < p.output.size(); ++i) {
                min_dp.output[i] = std::min(min_dp.output[i], p.output[i]);
                max_dp.output[i] = std::max(max_dp.output[i], p.output[i]);
            }
        }

        dataset_min = min_dp;
        dataset_max = max_dp;


        for (auto& p : dataset) {
            normalizeDatapoint(p);
        }
    }

    void normalizeDatapoint(DataPoint& p) {
        for (size_t i = 0; i < p.input.size(); ++i) {
            p.input[i] -= dataset_min.input[i];
            p.input[i] /= (dataset_max.input[i] - dataset_min.input[i]);
        }
        for (size_t i = 0; i < p.output.size(); ++i) {
            p.output[i] -= dataset_min.output[i];
            p.output[i] /= (dataset_max.output[i] - dataset_min.output[i]);
        }
    }

    void denormalizeDatapoint(DataPoint& p) {
        for (size_t i = 0; i < p.input.size(); ++i) {
            p.input[i] *= (dataset_max.input[i] - dataset_min.input[i]);
            p.input[i] += dataset_min.input[i];

        }
        for (size_t i = 0; i < p.output.size(); ++i) {
            p.output[i] *= (dataset_max.output[i] - dataset_min.output[i]);
            p.output[i] += dataset_min.output[i];
        }
    }

    void addMomentum(std::unique_ptr<NNMomentum> mom) {
        momentum = std::move(mom);
    }

    void updateLast() {
        std::lock_guard l(m);
        this->last_readable->connections = network->connections;
    }
    void updateLastChange(const std::vector<NNEdgeMatrix>& grad) {
        std::lock_guard l(m);
        this->last_readable_changes->connections = grad;
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
            if (debug) {

                std::cerr << "DP in : ";
                for (float f : dp.input) std::cerr << f << " ";
                std::cerr << std::endl;
            }

            network->evaluateNetwork(dp.input);

            if (debug) {

                std::cerr << "Network nodes:\n";
                for (int i = 0; i < network->layers.size(); ++i) {
                    auto && l = network->layers[i];
                    std::cerr << "Layer " << i << " pre act:\n";
                    for (float f : l->pre_values)
                        std::cerr << std::setw(7) << std::setprecision(4) << f;
                    std::cerr << "\nLayer " << i << " post act:\n";
                    for (float f : l->values)
                        std::cerr << std::setw(7) << std::setprecision(4) << f;
                        std::cerr << "\n\n";
                }


                std::cerr << "DP out : ";
                for (float f : dp.output) std::cerr << f << " ";
                std::cerr << "\n";
            }


            auto network_ans= network->getLastLayerAfterEvaluation().values;

            if (debug) {
                std::cerr << "NN out : ";
                for (float f : network_ans) std::cerr << f << " ";
                std::cerr << "\n";
            }

            auto network_ans_cal = loss_fun->normalize(network_ans);

            if (debug) {
                std::cerr << "NN out norm: ";
                for (float f : network_ans_cal) std::cerr << f << " ";
                std::cerr << "\n";
            }

            auto err_der = loss_fun->calculateDerivative(network_ans, dp.output);

            if (debug) {
                std::cerr << "Err der : ";
                for (float f : err_der) std::cerr << f << " ";
                std::cerr << "\n";
            }

            auto err = loss_fun->calculateError(network_ans, dp.output);
            error_history_epoch.push_back(err);
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
        // for (auto& matrix : grad_mean)
        //     for (auto& row : matrix)
        //         for (auto& col : row)
        //            col /= M;

        // apply momentum and learning factor
        momentum->applyMomentum(grad_mean);

        // apply changes to the network
        addMatrices(grad_mean, network->connections);
        updateLast();
        updateLastChange(grad_mean);
    }

    // starts a new epoch
    void generateBatches() {
        checkFinish();
        ++epoch;
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
        if (error_history_epoch.size() == 0)
            total_error = INFINITY;
        else
            total_error =
                static_cast<float>(
                    std::accumulate(error_history_epoch.begin(),
                                    error_history_epoch.end(),
                                    0.0));

        stopped = terminator->shouldFinish(total_error);

        std::lock_guard l{m};
        if (error_history_epoch.size() > 0)
            error_history.push_back(total_error);
        error_history_epoch.clear();
    }

public: // whatev im out of time
    std::unique_ptr<NNMomentum> momentum;
    std::unique_ptr<NNTerminator> terminator;
    std::unique_ptr<NNLossFun> loss_fun;
    std::unique_ptr<NeuralNetwork> network;
    std::vector<DataPoint> dataset;
    DataPoint dataset_min;
    DataPoint dataset_max;
    std::vector<std::vector<DataPoint>> batches;

    size_t next_to_take = 0;
    size_t batch_size = 0;
    bool stopped = false;
    std::atomic_int epoch = 0;

    std::mutex m;
    std::vector<float> error_history;
    std::vector<float> error_history_epoch;
    std::shared_ptr<NeuralNetwork> last_readable;
    std::shared_ptr<NeuralNetwork> last_readable_changes;

    std::shared_ptr<NeuralNetwork>  GetLastReadable() {
        std::lock_guard l{m};
        return last_readable;
    }
    std::shared_ptr<NeuralNetwork>  GetLastReadableChanges() {
        std::lock_guard l{m};
        return last_readable_changes;
    }


    size_t getCurrentEpoch() { return (size_t)epoch.load();}
    float getCurrentError() { std::lock_guard l {m}; return error_history.empty() ? NAN : error_history.back(); }
    std::vector<float> getErrorHistory() { std::lock_guard l {m}; return error_history; }
};