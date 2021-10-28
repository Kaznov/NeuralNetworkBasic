
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "ImGuiFileBrowser.h"
#include <stdio.h>
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include <memory>
#include "NNTeacher.h"

std::unique_ptr<NNTeacher> teacher;
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

bool training_set_loaded = false;
bool testing_set_loaded = false;

bool show_nn_visual = false;

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
    if (set_labels.empty()) set_labels = csv.headers;

    if (classification) {
        // look through outputs, change to one-hot-encoding
        int max_class_id = -1;
        for (const auto& p : csv.points) {
            max_class_id = std::max((int)p.output.back(), max_class_id);
        }
        int class_count = max_class_id + 1;
        for (auto& p : csv.points) {
            int id = (int)p.output.back();
            p.output.clear();
            p.output.resize(class_count);
            p.output[id] = 1.;
        }

        // add indexed labels to 1-hot-enc
        std::string label = set_labels.back();
        set_labels.pop_back();
        for (int i = 0; i < class_count; ++i) {
            set_labels.push_back(label + std::to_string(i));
        }
    }

    return csv.points;
}

void loadTrainingSet(std::string path) {
    if (training_set_loaded) return;
    training_set = loadDataSet(path);
    training_set_loaded = !training_set.empty();
}

void loadTestingSet(std::string path) {
    if (testing_set_loaded) return;
    testing_set = loadDataSet(path);
    testing_set_loaded = !training_set.empty();
}

imgui_addons::ImGuiFileBrowser file_dialog;

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
    else {
        ImGui::Text("Boo");
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
        ImGui::Checkbox("Testing set test", &show_test_data_text);
    } else {
        ImGui::TextColored(ImVec4{0.9f, 0.6f, 0.7f, 1.0f}, "Load a testing set from CSV File!");
    }

    ImGui::Separator();

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
        ImGui::ShowDemoWindow();

        if (show_intro_window)        
            showIntroWindow();
        if (regression || classification)        
            showMainWindow();

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
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
