#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp> 
#include <glm/gtc/type_ptr.hpp>
#include <shader.h>
#include <camera.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include<vector>
#include <array>
#include <map>
#include <string>
#include<thread>
#include<random>
#include "../raytracer.cuh"
#include "../platform.h"
#include<cuda_runtime.h>    

//#define STB_IMAGE_IMPLEMENTATION
//#include "stb_image.h"
//R*0.299 + G*0.587 + B*0.11
#define M_PI 3.141592653589793
// settings
#define RESOLUTION 1024
const unsigned int SCR_WIDTH = RESOLUTION;
const unsigned int SCR_HEIGHT = RESOLUTION;
std::map<std::string, glm::vec3> material_colors;
platform plane[1100]; // 預留 150 面三角形


int plane_count = 0;
unsigned char* textureData;
int texWidth;
int texHeight;

#define PICS 1
bool multpics = false;
unsigned int textName[3];
//unsigned char now_pixel[RESOLUTION][RESOLUTION][4];
//unsigned char all_pixel[RESOLUTION][RESOLUTION][4];
//unsigned char rt_tex[RESOLUTION][RESOLUTION][4];

unsigned char* rt_tex = new unsigned char[SCR_WIDTH * SCR_HEIGHT * 4];
unsigned char* all_pixel = new unsigned char[SCR_WIDTH * SCR_HEIGHT * 4];

// camera
//Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
Camera camera(glm::vec3(0.0f, 105.0f, 450.0f));

// lighting
glm::vec3 lightPos(0.0f, 180.0f,-20.0f);
//glm::vec3 lightPos;
lighthit lightHit[510];
int lightHitcnt = 0;
int lightray_num = 0;


float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;
// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

float find_max(glm::vec3 a) {
    float max = a.x;
    if (a.x < a.y)max = a.y;
    if (max < a.z)max = a.z;
    return max;
}
unsigned int refractLineVAO, refractLineVBO;
void cast_lightray(int ray_num);
void load_mtl_file(const std::string& filename);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
//void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);
void try_raytrace();
void load_obj_to_planes(const std::string& filename);
bool point_in_triangle2(glm::vec3 point, platform plane) {
    //barycentric method
    glm::vec3 vec1, vec2, vec3;
    vec1 = glm::vec3(
        plane.v[0][0] - point.x,
        plane.v[0][1] - point.y,
        plane.v[0][2] - point.z);
    vec2 = glm::vec3(
        plane.v[1][0] - point.x,
        plane.v[1][1] - point.y,
        plane.v[1][2] - point.z);
    vec3 = glm::vec3(
        plane.v[2][0] - point.x,
        plane.v[2][1] - point.y,
        plane.v[2][2] - point.z);

    float area1 = glm::length(glm::cross(vec1, vec2)) / 2;
    float area2 = glm::length(glm::cross(vec2, vec3)) / 2;
    float area3 = glm::length(glm::cross(vec1, vec3)) / 2;
    glm::vec3 vec11, vec22;
    vec11 = glm::vec3(
        plane.v[1][0] - plane.v[0][0],
        plane.v[1][1] - plane.v[0][1],
        plane.v[1][2] - plane.v[0][2]);
    vec22 = glm::vec3(
        plane.v[2][0] - plane.v[0][0],
        plane.v[2][1] - plane.v[0][1],
        plane.v[2][2] - plane.v[0][2]);
    float totalarea = glm::length(glm::cross(vec11, vec22)) / 2;
    //std::cout << "area:" << abs(totalarea) << "\n";
    if (abs(totalarea - area1 - area2 - area3) < 0.01f)return true;
    else return false;
}
void printVec3(glm::vec3 a) {
    std::cout << a.x << " " << a.y << " " << a.z << std::endl;
}
int main()
{
    srand(time(NULL));

    camera.Front = glm::vec3(0, 0, -1);
    //std::cout << camera.Front.x << camera.Front.y << camera.Front.z << "\n";
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(800, 800, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    //glfwSetCursorPosCallback(window, mouse_callback);
    //glfwSetScrollCallback(window, scroll_callback);

    // tell GLFW to capture our mouse
    //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST); 
    // build and compile our shader zprogram
    // ------------------------------------
    Shader lightingShader("raytrace/basic_lighting.vs", "raytrace/basic_lighting.fs");
    Shader lightCubeShader("raytrace/light_cube.vs", "raytrace/light_cube.fs");
    Shader rtShader("raytrace/myshader.vs", "raytrace/myshader.fs");
    float rt_vertices[] = {
         2.0f,  2.0f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f, // top right
         2.0f, -2.0f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, // bottom right
        -2.0f, -2.0f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
        -2.0f,  2.0f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f  // top left 
    };

    unsigned int indices[] = {
    0, 1, 3, // first triangle
    1, 2, 3  // second triangle
    };
    unsigned int rtVBO, rtVAO, EBO;
    glGenVertexArrays(1, &rtVAO);
    glGenBuffers(1, &rtVBO);
    glBindVertexArray(rtVAO);
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ARRAY_BUFFER, rtVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(rt_vertices), rt_vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);


    //texture
    int nrChannels;
    //textureData = stbi_load("checkerboard1.jpg", &texWidth, &texHeight, &nrChannels, 0);

    //unsigned int texture;
    
    //load_stl_to_planes("newroom.stl");
    //load_stl_to_planes("room.stl");
    //load_obj_to_planes("CornellBox-Original.obj");
    load_mtl_file("raytrace/TransparentBall.mtl");
    //load_mtl_file("raytrace/onlyball.mtl");
    //load_obj_to_planes("raytrace/Untitled.obj");
    //load_obj_to_planes("C:/Users/user/source/repos/CudaRuntime1/CudaRuntime1/raytrace/ball.obj");
    load_obj_to_planes("raytrace/Surface_scene.obj");

    lightHitcnt = 0;
    std::cout << "ok";
        //隨機光源
    float lightmagnitude = 10.0f;
    glm::vec3 minP(0-lightmagnitude,180,-20-lightmagnitude), maxP(0+lightmagnitude,180,-20+lightmagnitude);
    static std::mt19937 rng{ 123456u };               // 可換成時間種子
    std::uniform_real_distribution<float> dx(minP.x, maxP.x);
    //std::uniform_real_distribution<float> dy(minP.y, maxP.y);
    std::uniform_real_distribution<float> dz(minP.z, maxP.z);
    glm::vec3 Lpos{ dx(rng), 180.0f, dz(rng) };
    //cuda_set_info(camera.Position,lightPos,camera.Front,camera.Up,camera.Right,plane_count);
    cuda_set_info(camera.Position, camera.Front, camera.Up, camera.Right, plane_count);
    cuda_upload_planes(plane, plane_count);
    secondary_info(lightHit, lightHitcnt);
    //std::cout << plane_count << std::endl;
    double startTime = glfwGetTime();
    //cast_lightray(lightray_num);
    try_raytrace(); // 或 try_raytrace_parallel()
    double endTime = glfwGetTime();
    double elapsed = endTime - startTime;

    std::cout << elapsed << std::endl;

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        //try_raytrace();
        // per-frame time logic
        // --------------------
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        // input
        // -----
        processInput(window);
        // render
        // ------
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);



        glBindTexture(GL_TEXTURE_2D, textName[0]);
        glm::mat4 view;
        glm::mat4 projection;
        //view = camera.GetViewMatrix();
        //projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);

        rtShader.use();
        rtShader.setInt("ourTexture", 0);
        //rtShader.setMat4("projection", projection);
        //rtShader.setMat4("view", view);
        //rtShader.set
        glm::mat4 model = glm::mat4(1.0f);

        //model = glm::translate(model, glm::vec3(camera.Position.x, camera.Position.y, camera.Position.z - 2.0f));
        //rtShader.setMat4("model", model);
        glBindVertexArray(rtVAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);


        // ---------------)----------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

void try_raytrace() {    
    //for (int i = 0; i < 16; i++) {
    //    for (int j = 0; j < 16; j++) {
    //        glm::vec3 rayDir = glm::normalize(
    //            camera.Front +
    //            camera.Right * (2.0f * i / 16 - 1.0f) +
    //            camera.Up   * (2.0f * j / 16 - 1.0f)
    //        );
    //        cuda_try_raytrace(rt_tex, SCR_WIDTH, SCR_HEIGHT, rayDir);
    //        break;
    //        std::cout << "l23\n";
    //    }
    //    break;
    //}
    cuda_try_raytrace(rt_tex, SCR_WIDTH, SCR_HEIGHT, glm::vec3(0.0f));
    
    glGenTextures(1, &textName[0]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textName[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, RESOLUTION, RESOLUTION, 0, GL_RGBA, GL_UNSIGNED_BYTE, rt_tex);
}
// 回傳值為 float[3]，隨機單位向量在 -y 半球
void sample_direction_in_negative_y_hemisphere(float dir[3]) {
    float r1 = (float)rand() / RAND_MAX;
    float r2 = (float)rand() / RAND_MAX;

    float theta = 2.0f * M_PI * r1;
    float z = sqrtf(1.0f - r2);           // 餘弦加權，如果你要均勻分布，改成 z = r2
    //z = r2;
    //float z = sqrt(1-r2);
    float x = cosf(theta) * sqrtf(r2);
    float y = sinf(theta) * sqrtf(r2);

    // 目前的方向是以 +Z 為法向量的半球，轉到 -Y 方向

    // 建立 TBN 矩陣（以 -Y 為 normal）
    float N[3] = { 0, -1, 0 };
    float up[3] = { 0, 0, 1 };  // 用來生成 tangent

    // T = up × N
    float T[3] = {
        up[1] * N[2] - up[2] * N[1],
        up[2] * N[0] - up[0] * N[2],
        up[0] * N[1] - up[1] * N[0]
    };

    // normalize T
    float len = sqrtf(T[0] * T[0] + T[1] * T[1] + T[2] * T[2]);
    T[0] /= len; T[1] /= len; T[2] /= len;

    // B = N × T
    float B[3] = {
        N[1] * T[2] - N[2] * T[1],
        N[2] * T[0] - N[0] * T[2],
        N[0] * T[1] - N[1] * T[0]
    };

    // 最終隨機方向 dir = x*T + y*B + z*N
    dir[0] = x * T[0] + y * B[0] + z * N[0];
    dir[1] = x * T[1] + y * B[1] + z * N[1];
    dir[2] = x * T[2] + y * B[2] + z * N[2];
}
void load_obj_to_planes(const std::string& filename) {
    std::string current_mtl = "default";
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open OBJ file!" << std::endl;
        return;
    }
    int vertex_cnt = 0;
    std::vector<std::array<float, 3>> vertices;
    std::vector< std::array<float, 3>> normal;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "v") {
            float x, y, z;
            iss >> x >> y >> z;
            float v[3] = { x, y, z };
            vertices.push_back({ v[0], v[1], v[2] });
        }
        else if (type == "vn") {
            float nx, ny, nz;
            iss >> nx >> ny >> nz;
            float vn[3] = { nx, ny, nz };
            normal.push_back({ vn[0], vn[1], vn[2] });
        }
        else if (type == "usemtl") {
            iss >> current_mtl;
        }
        else if (type == "f") {// -4 -3 -2 -1 | f 14/13/4 15/14/4 13/15/4
            std::vector<int> indices, normalIndices;
            std::string token;
            while (iss >> token) {
                size_t firstSlash = token.find('/');
                size_t secondSlash = token.find('/', firstSlash + 1);

                int index = std::stoi(token.substr(0, firstSlash));
                if (index < 0) index = vertices.size() + index; // handle negative index
                else index -= 1;  // OBJ is 1-based
                indices.push_back(index);

                if (secondSlash != std::string::npos) {
                    std::string vnStr = token.substr(secondSlash + 1);
                    int vnIdx = std::stoi(vnStr);
                    //if (vnIdx < 0) vnIdx = normals.size() + vnIdx;
                    //else vnIdx -= 1; // OBJ 是 1-based
                    vnIdx -= 1;
                    normalIndices.push_back(vnIdx);
                }

            }

            // Convert quad face into two triangles (f a b c d → f a b c and f a c d)
            if (indices.size() >= 3 && plane_count < 1200) {
                //for (size_t i = 1; i < indices.size() && plane_count < 1200; i++) {
                if (indices.size() == 3 && plane_count < 1200) {
                    int i = 1;
                    const auto& v1 = vertices[indices[0]];
                    const auto& v2 = vertices[indices[i]];
                    const auto& v3 = vertices[indices[i + 1]];

                    platform p;
                    p.vn[0] = glm::vec3(
                        normal[normalIndices[0]][0],
                        normal[normalIndices[0]][1],
                        normal[normalIndices[0]][2]
                    );
                    p.vn[1] = glm::vec3(
                        normal[normalIndices[0 + 1]][0],
                        normal[normalIndices[0 + 1]][1],
                        normal[normalIndices[0 + 1]][2]
                    );
                    p.vn[2] = glm::vec3(
                        normal[normalIndices[0 + 2]][0],
                        normal[normalIndices[0 + 2]][1],
                        normal[normalIndices[0 + 2]][2]
                    );

                    memcpy(p.v[0], v1.data(), 3 * sizeof(float));
                    memcpy(p.v[1], v2.data(), 3 * sizeof(float));
                    memcpy(p.v[2], v3.data(), 3 * sizeof(float));

                    // 平面係數
                    float vec1[3] = {
                        v2[0] - v1[0],
                        v2[1] - v1[1],
                        v2[2] - v1[2]
                    };
                    float vec2[3] = {
                        v3[0] - v1[0],
                        v3[1] - v1[1],
                        v3[2] - v1[2]
                    };
                    float n[3] = {
                        vec1[1] * vec2[2] - vec1[2] * vec2[1],
                        vec1[2] * vec2[0] - vec1[0] * vec2[2],
                        vec1[0] * vec2[1] - vec1[1] * vec2[0]
                    };
                    float length = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                    if (length > 0.0f) {
                        n[0] /= length; n[1] /= length; n[2] /= length;
                    }
                    //float n[3] = { normal[0],normal[1],normal[2] };

                    p.coef[0] = n[0];
                    p.coef[1] = n[1];
                    p.coef[2] = n[2];
                    p.coef[3] = -(n[0] * v1[0] + n[1] * v1[1] + n[2] * v1[2]);

                    //memcpy(p.normal, n, 3 * sizeof(float));
                    glm::vec3 avgNormal = (p.vn[0] + p.vn[1] + p.vn[2]) / 3.0f;
                    avgNormal = glm::normalize(avgNormal);
                    memcpy(p.normal, glm::value_ptr(avgNormal), 3 * sizeof(float));

                    if (material_colors.find(current_mtl) != material_colors.end()) {
                        //std::cout << "color\n";
                        glm::vec3 kd = material_colors[current_mtl];
                        p.kd[0] = kd.r;
                        p.kd[1] = kd.g;
                        p.kd[2] = kd.b;
                    }
                    else {
                        p.kd[0] = p.kd[1] = p.kd[2] = 1.0f; // fallback to white
                    }
                    // UV (dummy)

                    plane[plane_count++] = p;

                }
            }
        }

    }
    file.close();
}
void load_mtl_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open MTL file!" << std::endl;
        return;
    }

    std::string line, current_material;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string word;
        iss >> word;

        if (word == "newmtl") {
            iss >> current_material;
        }
        else if (word == "Cd" && !current_material.empty()) {
            float r, g, b;
            iss >> r >> g >> b;
            material_colors[current_material] = glm::vec3(r, g, b);
        }
    }
}
// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}
// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}
