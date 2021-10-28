#pragma once

#include "NNAliases.h"
#include <cmath>

class NNTerminator {
public:
    virtual bool shouldFinish(float error) = 0;
};

class NNConstantTerminator : public NNTerminator {
public:
    NNConstantTerminator(size_t epoch_count) :
        epoch_count{epoch_count} {}
    
    bool shouldFinish(float error) override {
        if (epoch_count == 0) return true;
        --epoch_count;
        return false;
    }

private:
    size_t epoch_count;
};

class NNRelativeErrorTerminator : public NNTerminator {
public:
    NNRelativeErrorTerminator(float ratio) :
        ratio{ratio} {}
    
    bool shouldFinish(float error) override {
        if (error * (1 + ratio) > last_error
            && last_error * (1 + ratio) > pre_last_error) return true;
        pre_last_error = last_error;
        last_error = error;
        return false;
    }

private:
    float ratio;
    float last_error = INFINITY;
    float pre_last_error = INFINITY;
};



