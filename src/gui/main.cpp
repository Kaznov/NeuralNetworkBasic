
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include "ImGuiFileBrowser.h"
#include <stdio.h>
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include <memory>
#include "NNTeacher.h"

std::unique_ptr<NNTeacher> teacher = std::make_unique<NNTeacher>();
std::vector<std::string> set_labels;
std::vector<DataPoint> training_set;
std::vector<DataPoint> testing_set;


static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

bool show_intro_window = true;
bool regression = false;
bool classification = false;

bool show_train_data_text = false;
bool show_test_data_text = false;

bool show_train_data_visual = false;
bool show_test_data_visual = false;
bool show_nn_result_visual = false;

bool training_set_loaded = false;
bool testing_set_loaded = false;

bool show_nn_visual = false;
bool show_nn_changes_visual = false;
bool show_nn_error_plot = false;

bool show_network_configuration = false;
bool network_initialized = false;
bool continue_training = false;

bool show_error_plot = false;

int class_count = -1;

void initializeTeacher() {
    teacher = std::make_unique<NNTeacher>();
    teacher->addTrainingDataSet(training_set);
}

void showDataWindowText(std::string name, const std::vector<DataPoint>& data_set) {
    ImGui::Begin(("Input data - " + name).c_str());
    static ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;

    // When using ScrollX or ScrollY we need to specify a size for our table container!
    // Otherwise by default the table will fit all available space, like a BeginChild() call.
    ImVec2 outer_size = ImVec2(0.0f, 0.0f);
    if (ImGui::BeginTable("table_scrolly", set_labels.size(), flags, outer_size))
    {
        ImGui::TableSetupScrollFreeze(0, 1); // Make top row always visible
        for (auto&& l : set_labels)
            ImGui::TableSetupColumn(l.c_str(), ImGuiTableColumnFlags_None);
        ImGui::TableHeadersRow();

        // Demonstrate using clipper for large vertical lists
        ImGuiListClipper clipper;
        clipper.Begin(data_set.size());
        while (clipper.Step())
        {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++)
            {
                ImGui::TableNextRow();
                for (int column = 0; column < set_labels.size(); column++)
                {
                    ImGui::TableSetColumnIndex(column);
                    if (column < data_set[row].input.size())
                        ImGui::Text("%f", data_set[row].input[column]);
                    else
                        ImGui::Text("%f", data_set[row].output[column - data_set[row].input.size()]);
                }
            }
        }
        ImGui::EndTable();
    }
    ImGui::End();
}

void drawInputSampleRegressionData(std::string name, bool points, 
    std::vector<float> xs,
    std::vector<float> ys) {
    
    if(points)
        ImPlot::PlotScatter(name.c_str(), xs.data(), ys.data(), xs.size());
    else
        ImPlot::PlotLine(name.c_str(), xs.data(), ys.data(), xs.size());
}

void drawVisualRegression() {
    ImGui::Begin("Regression visualization training");
    if (ImPlot::BeginPlot("Regression Plot")) {
        ImPlot::SetupAxes("x","y");
        if (show_train_data_visual) {
            std::vector<float> xs;
            std::vector<float> ys;
            for (auto&& el : training_set) {
                xs.push_back(el.input[0]);
                ys.push_back(el.output[0]);
            }
            drawInputSampleRegressionData("Training Test cases", true, xs, ys);
        }
        if (show_test_data_visual) {
            std::vector<float> xs;
            std::vector<float> ys;
            for (auto&& el : testing_set) {
                xs.push_back(el.input[0]);
                ys.push_back(el.output[0]);
            }
            drawInputSampleRegressionData("Testing Test Cases", true, xs, ys);
        }
        
        if (show_nn_result_visual) {
            auto nn = teacher->GetLastReadable();
            {
                std::vector<float> xs;
                std::vector<float> ys;
                for (auto&& el : training_set) {
                    float x = el.input[0];
                    xs.push_back(x);
                    
                    std::vector<float> input;
                    input.push_back(x);
                    nn->evaluateNetwork(input);
                    ys.push_back(nn->layers.back()->values[0]);
                }

                drawInputSampleRegressionData("Neural network on training set", true, xs, ys);
            }
            if (testing_set.size() > 0) {
                std::vector<float> xs;
                std::vector<float> ys;
                for (auto&& el : testing_set) {
                    float x = el.input[0];
                    xs.push_back(x);
                    
                    std::vector<float> input;
                    input.push_back(x);
                    nn->evaluateNetwork(input);
                    ys.push_back(nn->layers.back()->values[0]);
                }

                drawInputSampleRegressionData("Neural network on testing set", true, xs, ys);
            }
        }

        ImPlot::EndPlot();
    }
    ImGui::End();
}

void drawVisualClassificationData(std::string title,
            const std::vector<DataPoint>& data_set) {
    ImGui::Begin(("Classification visualization - " + title).c_str());

    std::vector<std::vector<DataPoint>> classes(class_count);
    
    for (size_t i = 0; i < data_set.size(); ++i) {
        for (size_t j = 0; j < class_count; ++j)
            if (data_set[i].output[j] == 1) {
                classes[j].push_back(data_set[i]);
                break;
            }
    }
    if (ImPlot::BeginPlot(("Classification Plot -" + title).c_str())) {
        ImPlot::SetupAxes("x","y");
        
        for (int c = 0; c < class_count; ++c) {
            auto&& cc = classes[c];
            std::vector<float> xs;
            std::vector<float> ys;
            for (auto&& el : cc) {
                xs.push_back(el.input[0]);
                ys.push_back(el.input[1]);
            }
            ImPlot::PlotScatter(std::to_string(c).c_str(), xs.data(), ys.data(), xs.size());
        }

        ImPlot::EndPlot();
    }

    ImGui::End();
}

void drawVisualClassificationTraining() {
    drawVisualClassificationData("Training", training_set);
}

void drawVisualClassificationTesting() {
    drawVisualClassificationData("Testing", testing_set);
}

void drawVisualClassificationNN() {
    // TODO
    //drawVisualClassificationData("NN", NN_last_result);
}

void showNNResultVisual() {

}

void drawNN(std::shared_ptr<NeuralNetwork> nn) {
    // Demonstrate using the low-level ImDrawList to draw custom shapes.
    if (nn == nullptr)
    {
        return;
    }

    static float node_radius = 40;
    static float node_thicc = node_radius / 6;
    static float layer_width = 300;
    static float left_offset = 20;
    static float top_offset = 20;
    static float right_offset = 20;
    static float bottom_offset = 20;
    static float font_size = 14;

    ImGui::SliderFloat("Node radius", &node_radius, 10, 200, nullptr, ImGuiSliderFlags_AlwaysClamp);
    ImGui::SliderFloat("Font size", &font_size, 5, 30, nullptr, ImGuiSliderFlags_AlwaysClamp);
    ImGui::SliderFloat("Layer width", &layer_width, 50, 1000, nullptr, ImGuiSliderFlags_AlwaysClamp);
    
    static ImVector<ImVec2> points;
    static ImVec2 scrolling(0.0f, 0.0f);
    static bool adding_line = false;

    ImGui::Text("Mouse Right: drag to scroll");

    // Using InvisibleButton() as a convenience 1) it will advance the layout cursor and 2) allows us to use IsItemHovered()/IsItemActive()
    ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();      // ImDrawList API uses screen coordinates!
    ImVec2 canvas_sz = ImGui::GetContentRegionAvail();   // Resize canvas to what's available
    if (canvas_sz.x < 500.0f) canvas_sz.x = 500.0f;
    if (canvas_sz.y < 600.0f) canvas_sz.y = 600.0f;
    ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);

    // Draw border and background color
    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(50, 50, 50, 255));
    draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

    // This will catch our interactions
    ImGui::InvisibleButton("canvas", canvas_sz, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    const bool is_active = ImGui::IsItemActive();   // Held
    const ImVec2 origin(canvas_p0.x + scrolling.x, canvas_p0.y + scrolling.y); // Lock scrolled origin

    if (is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Right, 0.0f))
    {
        scrolling.x += io.MouseDelta.x;
        scrolling.y += io.MouseDelta.y;
    }

    // Context menu (under default mouse threshold)
    ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);

    // Draw grid + all lines in the canvas
    draw_list->PushClipRect(canvas_p0, canvas_p1, true);

    
    const ImU32 col_white = ImColor(0.9f, 1.0f, 1.0f);
    const ImU32 col_gray = ImColor(0.9f, 1.0f, 1.0f, 0.1f);
    const ImU32 col_green = ImColor(0.1f, 1.0f, 0.3f);
    const ImU32 col_red = ImColor(1.0f, 0.1f, 0.2f);
    const ImU32 col_blue = ImColor(0.7f, 0.7f, 1.0f);


    std::vector<std::pair<float, float>> last_positions;
    std::vector<std::pair<float, float>> current_positions;
    size_t layers_count = nn->layers.size();
    float layer_x = left_offset + node_radius + origin.x;
    float step_x = canvas_sz.x - left_offset - right_offset - 2 * node_radius;
    if (layers_count >= 2) step_x /= (layers_count - 1);
    step_x = std::max(step_x, layer_width);

    float old_font_size = ImGui::GetFontSize();
    ImGui::GetFont()->FontSize = font_size;
    for (size_t l = 0; l < layers_count; ++l) {
        auto&& layer = *(nn->layers[l]);
        int layer_size = layer.getFullSize();

        
        float layer_y = top_offset + node_radius + origin.y;
        float step_y = canvas_sz.y - top_offset - bottom_offset - 2 * node_radius ;
        if (layer_size >= 2) step_y /= (layer_size - 1);

        float cur_y = layer_y;
        for (size_t n = 0; n < layer_size; ++n) {
            current_positions.push_back(std::make_pair(layer_x, cur_y));
            if (n == layer_size - 1 && layer.hasBias())
                draw_list->AddCircleFilled({layer_x, cur_y}, node_radius, col_white);
            else
                draw_list->AddCircle({layer_x, cur_y}, node_radius, col_white, 0, node_thicc);
                cur_y += step_y;
        }
        if (l != 0) {
            for (size_t n1 = 0; n1 < nn->layers[l - 1]->getFullSize(); ++n1)
            for (size_t n2 = 0; n2 < nn->layers[l]->getSize(); ++n2) {
                auto n1p = last_positions[n1];
                auto n2p = current_positions[n2];
                draw_list->AddLine({n1p.first, n1p.second}, {n2p.first, n2p.second}, col_gray, 3.0f);
                ImVec2 text_p = {(n1p.first * 0.6f + n2p.first * 0.4f), (n1p.second * 0.6f + n2p.second * 0.4f)}; 
                float weight = nn->connections[l - 1][n2][n1];
                char text[16]{};
                sprintf(text, "%.5f", weight);
                ImU32 col;
                if (std::abs(weight) < 0.0001) col = col_blue;
                else if (weight < 0) col = col_red;
                else col = col_green;
                draw_list->AddText(text_p, col, text);
            }
        }

        layer_x += step_x;
        last_positions = current_positions;
        current_positions.clear();
    }
    
    ImGui::GetFont()->FontSize = old_font_size;

    draw_list->PopClipRect();
}

void showNNValues() {
    auto last = teacher->GetLastReadable();
    // now we hold a thing that won't be mutated, we can release the lock

    ImGui::Begin("Neural Network connections values");
    drawNN(std::move(last));
    ImGui::End();
}

void showNNChanges() {
    std::shared_ptr<NeuralNetwork> last_changes = teacher->GetLastReadableChanges();
    // now we hold a thing that won't be mutated, we can release the lock

    ImGui::Begin("Neural Network last batch changes");
    drawNN(std::move(last_changes));
    ImGui::End();
}

void showNNErrorPlot() {
    ImGui::Begin("Error plot");
    std::lock_guard l{teacher->m};
    if (ImPlot::BeginPlot("Error plot")) {
        ImPlot::SetupAxes("epoch", "error");
        std::vector<float> xs;
        
        for (size_t i = 0; i < teacher->error_history.size(); ++i) {
            xs.push_back(i);
        }

        ImPlot::PlotLine("NN Error Plot", xs.data(), teacher->error_history.data(), xs.size());
        ImPlot::EndPlot();
    }
    ImGui::End();
}

void showIntroWindow() {
    ImGui::Begin("Hello, world!");

    ImGui::Text("This is a simple program showing inner workings of simple perceptron.");
    ImGui::Text("");
    ImGui::Text("Choose below, whether you want to perform classification of regression: ");

    if (ImGui::Button("Regression")) {
        show_intro_window = false;
        regression = true;
    }
    if (ImGui::Button("Classification")) {
        show_intro_window = false;
        classification = true;
    }
    
    ImGui::End();
}

std::vector<DataPoint> loadDataSet(std::string path) {
    std::string data_text = slurpFile(path);
    auto csv = parseCSV(data_text);
    bool new_labels = false;
    if (set_labels.empty()) {
        set_labels = csv.headers;
        new_labels = true;
    }
    if (classification) {
        // look through outputs, change to one-hot-encoding
        int max_class_id = -1;
        for (const auto& p : csv.points) {
            max_class_id = std::max((int)p.output.back(), max_class_id);
        }
        class_count = max_class_id; // no +1
        for (auto& p : csv.points) {
            int id = (int)p.output.back();
            p.output.clear();
            p.output.resize(class_count);
            p.output[id - 1] = 1.;
        }

        if (new_labels) {
            // add indexed labels to 1-hot-enc
            std::string label = set_labels.back();
            set_labels.pop_back();
            for (int i = 0; i < class_count; ++i) {
                set_labels.push_back(label + std::to_string(i));
            }
        }
    }

    return csv.points;
}

void loadTrainingSet(std::string path) {
    if (training_set_loaded) return;
    training_set = loadDataSet(path);
    teacher->addTrainingDataSet(training_set);
    training_set_loaded = !training_set.empty();
}

void loadTestingSet(std::string path) {
    if (testing_set_loaded) return;
    testing_set = loadDataSet(path);
    testing_set_loaded = !training_set.empty();
}

imgui_addons::ImGuiFileBrowser file_dialog;

void showNetworkConfiguration() {
    ImGui::Begin("NN configuration");
    if (!teacher->network) {
        teacher->addNetwork(std::make_unique<NeuralNetwork>());
    }
    if (!teacher->last_readable) {
        teacher->last_readable  = std::make_unique<NeuralNetwork>();
    }
    if (!teacher->last_readable_changes) {
        teacher->last_readable_changes  = std::make_unique<NeuralNetwork>();
    }
    NeuralNetwork* nn = teacher->network.get();
    NeuralNetwork* lr = teacher->last_readable.get();
    NeuralNetwork* lrc = teacher->last_readable_changes.get();
    if (ImGui::Button("Reset")) {
        initializeTeacher();
    }

    if (ImGui::Button("Add input layer")) {
        if (nn->layers.size() == 0 && training_set.size() > 0) {
            nn->addLayer(std::make_unique<InputLayer>(
                training_set[0].input.size()));
            lr->addLayer(std::make_unique<InputLayer>(
                training_set[0].input.size()));
            lrc->addLayer(std::make_unique<InputLayer>(
                training_set[0].input.size()));
            teacher->updateLast();
        }
    }   

    ImGui::Separator();
    static int batch_size = 32;
    ImGui::InputInt("Batch size", &batch_size);
    if (batch_size < 1) batch_size = 1;
    ImGui::Separator();

    ImGui::Text("Next layer properties: ");

    static int next_layer_size = 2;
    ImGui::InputInt("Layer size", &next_layer_size);
    next_layer_size = std::max(1, next_layer_size);

    static bool next_layer_bias = true;
    ImGui::Checkbox("Has bias", &next_layer_bias);


    std::string act_str[] = {"Sigmoid", "TanH", "Linear", "Ramp"};

    static int next_layer_type = 0;
    const char* combo_preview_value = act_str[next_layer_type].c_str();
    if (ImGui::BeginCombo("Activation function", combo_preview_value)) {
        for (int n = 0; n < std::size(act_str); n++)
        {
            const bool is_selected = (next_layer_type == n);
            if (ImGui::Selectable(act_str[n].c_str(), is_selected))
                next_layer_type = n;
                    
            if (is_selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    
    auto add_layer = [&]() {
        switch (next_layer_type)
        {
        case 0:
            nn->addLayer(std::make_unique<SigmoidLayer>(
                next_layer_size, next_layer_bias));
            lr->addLayer(std::make_unique<SigmoidLayer>(
                next_layer_size, next_layer_bias));
            lrc->addLayer(std::make_unique<SigmoidLayer>(
                next_layer_size, next_layer_bias));
            break;
        
        case 1:
            nn->addLayer(std::make_unique<TanHLayer>(
                next_layer_size, next_layer_bias));
            lr->addLayer(std::make_unique<TanHLayer>(
                next_layer_size, next_layer_bias));
            lrc->addLayer(std::make_unique<TanHLayer>(
                next_layer_size, next_layer_bias));
            break;
        
        case 2:
            nn->addLayer(std::make_unique<LinearLayer>(
                next_layer_size, next_layer_bias));
            lr->addLayer(std::make_unique<LinearLayer>(
                next_layer_size, next_layer_bias));
            lrc->addLayer(std::make_unique<LinearLayer>(
                next_layer_size, next_layer_bias));
            break;
        
        case 3:
            nn->addLayer(std::make_unique<RampLayer>(
                next_layer_size, next_layer_bias));
            lr->addLayer(std::make_unique<RampLayer>(
                next_layer_size, next_layer_bias));
            lrc->addLayer(std::make_unique<RampLayer>(
                next_layer_size, next_layer_bias));
            break;
        
        default:
            break;
        }
        teacher->updateLast();
    };

    if (ImGui::Button("Add new layer") && nn->layers.size() > 0) {
        add_layer();
    }

    
    if (ImGui::Button("Add last layer") && training_set.size() > 0) {
        next_layer_size = training_set[0].output.size();
        next_layer_bias = false;
        add_layer();

        teacher->batch_size = batch_size;
        if (regression)
            teacher->addLossFunction(std::make_unique<MeanSquaredLossFun>());
        else
            teacher->addLossFunction(std::make_unique<LogLoss>());
        teacher->addMomentum(std::make_unique<NNSteadyLearningRate>());
        teacher->addTerminator(std::make_unique<NNConstantTerminator>(200));
        network_initialized = true;
        show_network_configuration = false;
        nn->initializeWithRandomData();
        teacher->updateLast();
    }
    
    ImGui::End();
}

void showMainWindow() {
    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_MenuBar;
    ImGui::Begin(regression ? "Regression" : "Classification", nullptr, window_flags);
    bool open_training = false, open_testing = false;
    if(ImGui::BeginMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Load training set", nullptr, nullptr, !training_set_loaded))
                open_training = true;
            if (ImGui::MenuItem("Load testing set", nullptr, nullptr, !testing_set_loaded))
                open_testing = true;

            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }
    
    //Remember the name to ImGui::OpenPopup() and showFileDialog() must be same...
    if(open_training)
        ImGui::OpenPopup("Open File - training set");
    if(open_testing)
        ImGui::OpenPopup("Open File - testing set");
        
    /* Optional third parameter. Support opening only compressed rar/zip files. 
     * Opening any other file will show error, return false and won't close the dialog.
     */
    if(file_dialog.showFileDialog("Open File - training set", imgui_addons::ImGuiFileBrowser::DialogMode::OPEN, ImVec2(700, 310), ".csv"))
    {
        loadTrainingSet(file_dialog.selected_path);
    }
    if(file_dialog.showFileDialog("Open File - testing set", imgui_addons::ImGuiFileBrowser::DialogMode::OPEN, ImVec2(700, 310), ".csv"))
    {
        loadTestingSet(file_dialog.selected_path);
    }

    if (training_set_loaded) {
        ImGui::TextColored(ImVec4{0.3f, 0.9f, 0.5f, 1.0f}, "Successfully loaded training set!");
        ImGui::Checkbox("Training set visualization", &show_train_data_visual);
        ImGui::Checkbox("Training set text", &show_train_data_text);
    } else {
        ImGui::TextColored(ImVec4{0.9f, 0.6f, 0.7f, 1.0f}, "Load a training set from CSV File!");
    }

    if (testing_set_loaded) {
        ImGui::TextColored(ImVec4{0.3f, 0.9f, 0.5f, 1.0f}, "Successfully loaded testing set!");
        ImGui::Checkbox("Testing set visualization", &show_test_data_visual);
        ImGui::Checkbox("Testing set text", &show_test_data_text);
    } else {
        ImGui::TextColored(ImVec4{0.9f, 0.6f, 0.7f, 1.0f}, "Load a testing set from CSV File!");
    }

    ImGui::Separator();

    if (!network_initialized) {
        if (ImGui::Button("Configure network")) {
            show_network_configuration = true;
        }
        ImGui::Separator();
    }
    ImGui::Text("Layer info: ");

    static ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;

    // When using ScrollX or ScrollY we need to specify a size for our table container!
    // Otherwise by default the table will fit all available space, like a BeginChild() call.
    if (set_labels.size() && ImGui::BeginTable("table_scrolly", 4, flags, {0.f, 300.f}))
    {
        ImGui::TableSetupScrollFreeze(0, 1); // Make top row always visible
        ImGui::TableSetupColumn(".", ImGuiTableColumnFlags_None);
        ImGui::TableSetupColumn("Layer size", ImGuiTableColumnFlags_None);
        ImGui::TableSetupColumn("Layer type", ImGuiTableColumnFlags_None);
        ImGui::TableSetupColumn("Has bias", ImGuiTableColumnFlags_None);
        ImGui::TableHeadersRow();


        auto nn = teacher->GetLastReadable();
        if (nn)
        for (int row = 0; row < nn->layers.size(); row++)
        {
            auto&& l = *nn->layers[row];
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%d", row + 1);
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%d", l.getSize());
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%s", l.getName());
            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%s", l.hasBias() ? "true" : "false");
            
        }
        
        ImGui::EndTable();
    }

    ImGui::Text("Loss type: %s", teacher->loss_fun ? teacher->loss_fun->getName() : "?");
    ImGui::Text("Batch size: %d", teacher->batch_size);
    ImGui::Checkbox("Show NN visualization", &show_nn_visual);
    
    ImGui::Separator();

    if (network_initialized) {
        float cur_err = teacher->getCurrentError();
        ImGui::Text("Current error: %.5f", cur_err);
        if (classification && testing_set.size() != 0) {
            ImGui::Text("Correctly classified: %.3f%", 0.0);
        }
        ImGui::Text("Current epoch: %d", teacher->getCurrentEpoch());

        if (!continue_training && !teacher->finished()) {
            if (ImGui::Button("Next batch")) {
                if (!teacher->hasNextBatch()) teacher->generateBatches();
                teacher->learnBatch();
            }
            if (ImGui::Button("Next epoch")) {
                teacher->learnEpoch();
            }
            if (ImGui::Button("Continue training")) {
                continue_training = true;
            }
        }

        if (teacher->finished()) {
            ImGui::Text("Network finished learning");
        }
        
        ImGui::Checkbox("Show NN results", &show_nn_result_visual);
        ImGui::Checkbox("Show NN error plot", &show_nn_error_plot);
        ImGui::Checkbox("Show NN changes", &show_nn_changes_visual);

    
    }

    ImGui::End();

}


int main(int, char**)
{
    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100
    const char* glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(1600, 900, "Simple Neural Network Program", NULL, NULL);
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load Fonts
    // - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
    // - If the file cannot be loaded, the function will return NULL. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
    // - Read 'docs/FONTS.md' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
    //io.Fonts->AddFontDefault();
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/ProggyTiny.ttf", 10.0f);
    //ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
    //IM_ASSERT(font != NULL);

    initializeTeacher();
    ImVec4 clear_color = ImVec4(0.0f, 0.30f, 0.50f, 1.00f);

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
        //ImGui::ShowDemoWindow();
        //ImPlot::ShowDemoWindow();

        if (show_intro_window)        
            showIntroWindow();
        if (regression || classification)        
            showMainWindow();

        if (show_train_data_text)
            if (classification) showDataWindowText("Classification train data", training_set);
            else showDataWindowText("Regression train data", training_set);

        if (show_test_data_text)
            if (classification) showDataWindowText("Classification test data", testing_set);
            else showDataWindowText("Regression test data", testing_set);

        if (show_train_data_visual || show_test_data_visual || show_nn_result_visual)
            if (classification) {
                if (show_train_data_visual) drawVisualClassificationTraining();
                if (show_test_data_visual) drawVisualClassificationTesting();
            }
            else drawVisualRegression();
        
        if (show_network_configuration) {
            showNetworkConfiguration();
        }

        if (show_nn_visual) {
            showNNValues();
        }

        if (show_nn_changes_visual) {
            showNNChanges();
        }

        if (show_nn_error_plot) {
            showNNErrorPlot();
        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
