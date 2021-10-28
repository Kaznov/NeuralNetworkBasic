#pragma once
#include <cassert>
#include <random>
#include <string>
#include <fstream>
#include <sstream>

#include "NNAliases.h"
#include "DataPoint.h"

extern std::minstd_rand RNG;

inline std::string slurpFile(const std::string& path) {
    constexpr size_t read_size = 4096;
    std::ifstream stream{path.c_str()};
    if (!stream) {
        return "";
    }

    std::string out{};
    std::string buf(read_size, '\0');
    while (stream.read(buf.data(), read_size)) {
        out.append(buf, 0, stream.gcount());
    }
    out.append(buf, 0, stream.gcount());
    return out;
}

inline void burpFile(const std::string& path, const std::string& text) {
    std::ofstream stream{path.c_str()};
    if (!stream) {
        return;
    }
    stream << text;
}

inline std::vector<std::string> splitText(std::string text, char sep = ' ') {
    size_t pos = text.find(sep);
    size_t initial_pos = 0;
    std::vector<std::string> result;

    while (pos != std::string::npos) {
        result.emplace_back(text.substr(initial_pos, pos - initial_pos));
        initial_pos = pos + 1;
        pos = text.find(sep, initial_pos);
    }

    // Add the last one
    result.push_back(text.substr(initial_pos));

    return result;
}

struct CSVData {
    std::vector<std::string> headers;
    std::vector<DataPoint> points;
};

inline CSVData parseCSV(const std::string& text) {
    std::istringstream in{text};
    std::string first_row;
    CSVData result;

    std::getline(in, first_row, '\n');
    result.headers = splitText(first_row, ',');

    std::string row;
    while (std::getline(in, row, '\n')) {
        DataPoint point;
        auto nums_text = splitText(row, ',');
        assert(nums_text.size() == result.headers.size());
        for (auto&& s : nums_text) {
            point.input.push_back(strtof(s.c_str(), nullptr));
        }
        // move last element as output
        point.output.push_back(point.input.back());
        point.input.pop_back();
        result.points.push_back(point);
    }
    return result;
}