#include "raytracer.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include "platform.h"
#include<iostream>
#include <curand_kernel.h>
#include<complex>
#define RESOLUTION 1024
#define M_PI 3.141592653589793
__device__ trace_info store_trace[RESOLUTION][RESOLUTION];

// === GPU 上的資料 ===
__device__ glm::vec3 device_camera_pos;
//__device__ glm::vec3 lightPos;
__device__ glm::vec3 device_camera_front;
__device__ glm::vec3 device_camera_up;
__device__ glm::vec3 device_camera_right;
__device__ int plane_count;
__device__ int lightHitcnt;

__device__ platform* plane;
__device__ lighthit* lightHit;

__device__ void printVec3c(glm::vec3 a) {
    printf("Vec3: (%f, %f, %f)\n", a.x, a.y, a.z);
}

__device__ bool point_in_triangle(glm::vec3 point, platform plane) {
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

//(rgb to grayscale, distance, 0)
__device__ glm::vec3 trace(glm::vec3 origin, glm::vec3 dir, int depth,bool in_trans) {
    //printVec3c(origin);
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    curandState rand_state;
    int seed = threadIdx.x;
    int idx = x * RESOLUTION + y;
    curand_init(seed, idx, 0, &rand_state);

    glm::vec3 ran_light(0, 195, -20);
    glm::vec3 lightPos(ran_light.x + (curand_uniform(&rand_state)-0.5f)*40.0f, ran_light.y, ran_light.z + (curand_uniform(&rand_state) - 0.5f) * 40.0f);
    //test
    if (depth > 4) return glm::vec3(0.1f, 0.1f, 0.5f); // 背景色

    //ray-triangle 相交、Phong shading 等邏輯
    float mindist = 1e9;
    int hitIndex = -1, s_hitIndex = -1;
    glm::vec3 hitPoint, hitNormal, texColor;
    bool inShadow = false;

    for (int k = 0; k < plane_count; ++k) {
        glm::vec3 N = glm::vec3(plane[k].coef[0], plane[k].coef[1], plane[k].coef[2]);
        float denom = glm::dot(N, dir);
        if (fabs(denom) < 1e-5) continue;

        float t = -(glm::dot(N, origin) + plane[k].coef[3]) / denom;
        if (t < 0) continue;

        glm::vec3 point = origin + t * dir;
        if (!point_in_triangle(point, plane[k])) continue;

        float dist = glm::length(point - origin);
        if (dist < mindist) {

            mindist = dist;
            hitIndex = k;
            hitPoint = point;
            //hitNormal = glm::normalize(glm::vec3(plane[k].normal[0], plane[k].normal[1], plane[k].normal[2]));

            //if ((hitIndex >= 12 && hitIndex <= 971)) {
          
                glm::vec3 vec1, vec2, vec3;
                vec1 = glm::vec3(
                    plane[k].v[0][0] - point.x,
                    plane[k].v[0][1] - point.y,
                    plane[k].v[0][2] - point.z);
                vec2 = glm::vec3(
                    plane[k].v[1][0] - point.x,
                    plane[k].v[1][1] - point.y,
                    plane[k].v[1][2] - point.z);
                vec3 = glm::vec3(
                    plane[k].v[2][0] - point.x,
                    plane[k].v[2][1] - point.y,
                    plane[k].v[2][2] - point.z);

                float area1 = glm::length(glm::cross(vec1, vec2)) / 2;
                float area2 = glm::length(glm::cross(vec2, vec3)) / 2;
                float area3 = glm::length(glm::cross(vec1, vec3)) / 2;
                glm::vec3 vec11, vec22;
                vec11 = glm::vec3(
                    plane[k].v[1][0] - plane[k].v[0][0],
                    plane[k].v[1][1] - plane[k].v[0][1],
                    plane[k].v[1][2] - plane[k].v[0][2]);
                vec22 = glm::vec3(
                    plane[k].v[2][0] - plane[k].v[0][0],
                    plane[k].v[2][1] - plane[k].v[0][1],
                    plane[k].v[2][2] - plane[k].v[0][2]);
                float totalarea = glm::length(glm::cross(vec11, vec22)) / 2;

                float w0 = area1;
                float w1 = area2;
                float w2 = area3;

                float sum = area1 + area2 + area3;
                w0 /= sum;
                w1 /= sum;
                w2 /= sum;

                hitNormal = glm::normalize(
                    w1 * plane[k].vn[0] +
                    w2 * plane[k].vn[1] +
                    w0 * plane[k].vn[2]
                );


            //texColor.r = plane[hitIndex].kd[0];
            //texColor.g = plane[hitIndex].kd[1];
            //texColor.b = plane[hitIndex].kd[2];

            texColor.r = 0.5f;
            texColor.g = 0.5f;
            texColor.b = 0.5f;


            glm::vec3 shadowRayDir = glm::normalize(lightPos - hitPoint);
            float lightDist = glm::length(lightPos - hitPoint);
            glm::vec3 shadowRayOrigin = hitPoint + shadowRayDir * 0.001f;
            for (int s = 0; s < plane_count; s++) {
                glm::vec3 N_s = glm::vec3(plane[s].coef[0], plane[s].coef[1], plane[s].coef[2]);
                float denom = glm::dot(N_s, shadowRayDir);
                if (abs(denom) < 0.00001f) continue;

                float t_shadow = -(glm::dot(N_s, shadowRayOrigin) + plane[s].coef[3]) / denom;
                if (t_shadow < 0) continue;

                glm::vec3 shadowPoint = shadowRayOrigin + t_shadow * shadowRayDir;
                float distToShadowPoint = glm::length(shadowPoint - shadowRayOrigin);

                //判斷是否在三角形內
                if (distToShadowPoint < lightDist && point_in_triangle(shadowPoint, plane[s])) {
                    inShadow = true;
                    s_hitIndex = s;
                    break;
                }
            }
        }
    }
    //if (hitIndex == -1) return glm::vec3(0.1f, 0.1f, 0.3f); // 背景色
    //if (hitIndex == 9 || hitIndex == 8)return glm::vec3(1.0f);
    // Phong shading
    glm::vec3 lightColor(0.7f);
    glm::vec3 ambient = 0.3f * lightColor;
    glm::vec3 L = glm::normalize(lightPos - hitPoint);
    glm::vec3 V = glm::normalize(device_camera_pos - hitPoint);
    glm::vec3 H = glm::normalize(L + V);
    float distance = glm::length(lightPos - hitPoint);
    if (distance < 1.0f)distance = 1.0f;
    float constant = 0.0f;
    //std::random_device rd;
    //std::mt19937 gen(rd());
    //std::uniform_real_distribution<float> di(1.5f, 5.5f);
    //float linear = di(gen);
    float linear = 0.01f;
    //float quadratic = 0.1f;
    float attu = 1.0f / (linear * distance);

    float diff = fmaxf(glm::dot(hitNormal, L), 0.0f);
    float spec = powf(fmaxf(glm::dot(hitNormal, H), 0.0f), 32);
    glm::vec3 color = ambient * texColor;
    color = (ambient + attu * (diff * lightColor)) * texColor;
    //color = (ambient + attu * (diff * lightColor+spec *lightColor)) * texColor;
    //color = (ambient + 1.0f * (diff * lightColor)) * texColor;
    if (inShadow) {
        color = ambient * texColor;
    }
    //return color;
    //R*0.299 + G*0.587 + B*0.11
    //float gray = color.r * 0.299f + color.g * 0.587f + color.b * 0.11f;//dosen't work
    float ray_distance = glm::length(hitPoint - origin);
    glm::vec3 result_of_ray = glm::vec3(color.r, ray_distance, 1.0f); //1.0 for no reflect
    //printf("%f\n", color.r);
    if (hitIndex == -1) return glm::vec3(0.0f, 0.0f, 0.3f); // 背景色
    if (hitIndex == 9 || hitIndex == 8)color = glm::vec3(1.0f);//adjust
    return result_of_ray;
    
    //計算場景中的光源(secondary ray)

    // (hitIndex >= 972 && hitIndex <= 984)
    bool hitTrans = 0&&(hitIndex >= 12 && hitIndex <= 971);
    bool hitMirror = 0&&(hitIndex >= 972 && hitIndex <= 984);
    if (hitMirror) {
        //printf("reflect\n");
        glm::vec3 reflectDir = glm::reflect(dir, hitNormal);//反射方向
        //glm::vec3 reflectColor = trace(hitPoint + 1.0f * reflectDir, reflectDir, depth + 1, false);
        //return reflectColor;
        store_trace[x][y].r_tree[(1 + depth) * 2].do_trace = true;
        store_trace[x][y].r_tree[(1 + depth) * 2].cur_pos = hitPoint + reflectDir * 0.01f;
        store_trace[x][y].r_tree[(1 + depth) * 2].next_dir = reflectDir;
        store_trace[x][y].r_tree[(1 + depth) * 2].inside_trans = false;
            //color = 回傳0.0 等下再處理
            return glm::vec3(0.0f);
    }
    if (hitTrans&&!in_trans) {
        glm::vec3 reflectDir = glm::reflect(dir, hitNormal);//反射方向
        //refraction
        float eta = 1.0f / 1.5f;//折射率
        glm::vec3 refracted_dir = glm::refract(dir, hitNormal, eta);
        //total internal reflection
        if (glm::length(refracted_dir) == 0.0f) {
            store_trace[x][y].r_tree[(1 + depth) * 2].do_trace = true;
            store_trace[x][y].r_tree[(1 + depth) * 2].cur_pos = hitPoint + reflectDir * 0.01f;
            store_trace[x][y].r_tree[(1 + depth) * 2].next_dir = reflectDir;
            store_trace[x][y].r_tree[(1 + depth) * 2].inside_trans = false;
            //color = 回傳0.0 等下再處理
            return glm::vec3(0.0f);
            //does offset effect the result?
        }
        //trace
        //glm::vec3 refractColor = trace(hitPoint + 0.01f * refracted_dir, refracted_dir, depth + 1, true);
        store_trace[x][y].r_tree[((1 + depth) * 2) + 1].do_trace = true;
        store_trace[x][y].r_tree[((1 + depth) * 2) + 1].cur_pos = hitPoint + refracted_dir * 0.01f;
        store_trace[x][y].r_tree[((1 + depth) * 2) + 1].next_dir = refracted_dir;
        store_trace[x][y].r_tree[((1 + depth) * 2) + 1].inside_trans = true;
        //color = 回傳0.0 等下再處理
        return glm::vec3(0.0f);
    }
    else if (in_trans && hitTrans) {//在箱子裡面的情況(離開玻璃)
        float eta = 1.5f / 1.0f;//折射率
        glm::vec3 reflectDir = glm::reflect(dir, -hitNormal);//反射方向
        glm::vec3 refracted_dir = glm::refract(dir, -hitNormal, eta);//進到箱子裡面後，normal應該要是反的
        if (glm::length(refracted_dir) == 0.0f) {//全反射           
            //return glm::vec3(1.0f);
            //refracted_dir = glm::reflect(dir, -hitNormal);
            //return trace(hitPoint + 0.01f * refracted_dir, refracted_dir, depth + 1, true); // 還在玻璃內
            store_trace[x][y].r_tree[(1 + depth) * 2].do_trace = true;
            store_trace[x][y].r_tree[(1 + depth) * 2].cur_pos = hitPoint + reflectDir * 0.01f;
            store_trace[x][y].r_tree[(1 + depth) * 2].next_dir = reflectDir;
            store_trace[x][y].r_tree[(1 + depth) * 2].inside_trans = true;
            //color = 回傳0.0 等下再處理
            return glm::vec3(0.0f);
        }        
        //return trace(hitPoint + 0.01f * refracted_dir, refracted_dir, depth + 1, false);
        store_trace[x][y].r_tree[((1 + depth) * 2) + 1].do_trace = true;
        store_trace[x][y].r_tree[((1 + depth) * 2) + 1].cur_pos = hitPoint + refracted_dir * 0.01f;
        store_trace[x][y].r_tree[((1 + depth) * 2) + 1].next_dir = refracted_dir;
        store_trace[x][y].r_tree[((1 + depth) * 2) + 1].inside_trans = false;
        //color = 回傳0.0 等下再處理


        //???????
        //return glm::vec3(0.5f,0.5f,0.0f);
        return glm::vec3(0.2f);
    }   
    else {
        //return color;
    }

}

// === CUDA kernel ===
//__global__ void raytraceKernel(unsigned char* output, int width, int height, glm::vec3 raydir) {
//__global__ void raytraceKernel(std::complex<float>* output, int width, int height, glm::vec3 raydir) {
__global__ void raytraceKernel(glm::vec2* output, int width, int height, glm::vec3 raydir) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    //int idx = (y * width + x) * 4; // RGBA
    int idx = (y * width + x);

    glm::vec3 Dir = glm::normalize(
        device_camera_front +
        device_camera_right * (2.0f * x / width - 1.0f) +
        device_camera_up * (2.0f * y / height - 1.0f)
    );
    glm::vec3 N = glm::vec3(0,0,1);
    float denom = glm::dot(N, Dir);

    float t = -(glm::dot(N, device_camera_pos) + -120) / denom;

    glm::vec3 origin = device_camera_pos + t * Dir;
    glm::vec3 rayDir = raydir;

    glm::vec3 color = trace(origin, raydir, 0,false);
    bool f;
    for (int i = 0; i < 16; i++) {
        if (store_trace[x][y].r_tree[i].do_trace == true) {
            //bool trans = (i % 2);
            
            color = trace(store_trace[x][y].r_tree[i].cur_pos, store_trace[x][y].r_tree[i].next_dir, ((int)(logf((float)i))) + 1, store_trace[x][y].r_tree[i].inside_trans);
        }
    }
    for (int i = 0; i < 16; i++) {
        if (store_trace[x][y].r_tree[i].do_trace == true) {
            //bool trans = (i % 2);
            //if (((int)(logf((float)i) / 2)) + 1 >= 1)printf("depth : %d\n", ((int)(logf((float)i) / 2)) + 1);
            color = trace(store_trace[x][y].r_tree[i].cur_pos, store_trace[x][y].r_tree[i].next_dir, ((int)(logf((float)i))) + 1, store_trace[x][y].r_tree[i].inside_trans);
        }
    }
    //
    //output[idx] = static_cast<unsigned char>(fminf(color.r, 1.0f) * 255);
    //output[idx + 1] = static_cast<unsigned char>(fminf(color.g, 1.0f) * 255);
    //output[idx + 2] = static_cast<unsigned char>(fminf(color.b, 1.0f) * 255);
    //output[idx + 3] = 255; // Alpha
    
    //color x = amplitude y = R(distance)
    float distance = color.y;
    //printf("%f\n", distance);
    float amplitude = color.x;
    float lambda = 0.006f;
    float phi = 2 * M_PI * distance / lambda;
    //std::complex<float> I(0.0f, 1.0f);
    /*glm::vec2 I(0.0f, 1.0f);
    if (distance != 0)output[idx] += (amplitude / distance) * exp(I * phi);
    printf("%f, %f", output[idx].x, output[idx].y);*/
    const float eps = 1e-6f;
    if (fabsf(distance) > 0) {
        float scale = amplitude / distance;

        // e^{i phi} = (cos phi, sin phi)
        float c = cosf(phi);
        float s = sinf(phi);

        // output[idx] 與 (c, s) 做純量縮放後相加
        output[idx].x += scale * c;  // 實部
        output[idx].y += scale * s;  // 虛部
    }
    //printf("%f, %f\n", output[idx].x, output[idx].y);

}
//__global__ void kinoformKernel(std::complex<float>* input, unsigned char* output, int width, int height) {
__global__ void kinoformKernel(glm::vec2* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;
    int idx = y * width + x;
    int outidx = (y * width + x)*4;

    //float a = input[idx].real();
    //float b = input[idx].imag();
    float a = input[idx].x;
    float b = input[idx].y;
    float phi = atan2(b, a) + M_PI;
    //printf("%f", phi);
    unsigned char gray = static_cast<unsigned char>(phi / (2 * M_PI) * 255.0f);
    //printf("%d  ", gray); 

    output[outidx] = gray;
    output[outidx + 1] = gray;
    output[outidx + 2] = gray;
    output[outidx + 3] = 255; // Alpha
}
// === Host 函式給 C++ 調用 ===
void cuda_try_raytrace(unsigned char* output, int width, int height, glm::vec3 raydir) {
    
    //unsigned char* dev_output;
    //size_t size = width * height * 4; // RGBA

    //std::complex<float>* dev_output;
    glm::vec2* dev_output;
    size_t size = width * height * sizeof(glm::vec2);
    size_t c_size = width * height *4;

    unsigned char* kino_output;

    //cudaMalloc(&kino_output, c_size);
    //cudaMalloc(&dev_output, size);
    cudaMalloc(&dev_output, size);
    cudaMemset(dev_output, 0, size);    // 把所有 amplitude, phase 都清成 0

    cudaMalloc(&kino_output, c_size);
    cudaMemset(kino_output, 0, c_size); // 習慣上也清一下，避免之後意外用到


    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + 15) / 16, (height + 15) / 16);

    glm::vec3 camFront_h, camRight_h, camUp_h;
    cudaMemcpyFromSymbol(&camFront_h, reinterpret_cast<const void*>(&device_camera_front), sizeof(glm::vec3));
    cudaMemcpyFromSymbol(&camRight_h, reinterpret_cast<const void*>(&device_camera_right), sizeof(glm::vec3));
    cudaMemcpyFromSymbol(&camUp_h, reinterpret_cast<const void*>(&device_camera_up), sizeof(glm::vec3));
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            glm::vec3 rayDir = camFront_h + camRight_h * (2.0f * i / 16 - 1.0f) + camUp_h * (2.0f * j / 16 - 1.0f);
            //cuda_try_raytrace(rt_tex, SCR_WIDTH, SCR_HEIGHT, rayDir);
            raytraceKernel << <numBlocks, threadsPerBlock >> > (dev_output, width, height, rayDir);
           // break;
        }
        break;
    }
    //calculate dev_output(the accumulate result)
    kinoformKernel << <numBlocks, threadsPerBlock >> > (dev_output, kino_output, width, height);

    //raytraceKernel << <numBlocks, threadsPerBlock >> > (dev_output, width, height, camFront_h);
    //raytraceKernel << <numBlocks, threadsPerBlock >> > (dev_output, width, height,raydir);
    cudaDeviceSynchronize();

    //cudaMemcpy(output, dev_output, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output, kino_output, c_size, cudaMemcpyDeviceToHost);
    cudaFree(dev_output);
}
void cuda_upload_planes(const platform* planes, int count) {
    platform* temp;
    cudaMalloc(&temp, count * sizeof(platform));
    cudaMemcpy(temp, planes, count * sizeof(platform), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(plane, &temp, sizeof(platform*));
}
void secondary_info(const lighthit* lightHits, int lighthit_count) {
    lighthit* temp;
    cudaMalloc(&temp, lighthit_count * sizeof(lighthit));
    cudaMemcpy(temp, lightHits, lighthit_count * sizeof(lighthit), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(lightHit, &temp, sizeof(lighthit*));
    cudaMemcpyToSymbol(reinterpret_cast<const void*>(&lightHitcnt), &lighthit_count, sizeof(int));
}
void cuda_set_info(glm::vec3 cpos,  glm::vec3 cfront, glm::vec3 cup, glm::vec3 cright, int plane_cnt) {
    cudaMemcpyToSymbol(reinterpret_cast<const void*>(&device_camera_pos), &cpos, sizeof(glm::vec3));
    //cudaMemcpyToSymbol(reinterpret_cast<const void*>(&lightPos), &Lpos, sizeof(glm::vec3));
    cudaMemcpyToSymbol(reinterpret_cast<const void*>(&device_camera_front), &cfront, sizeof(glm::vec3));
    cudaMemcpyToSymbol(reinterpret_cast<const void*>(&device_camera_up), &cup, sizeof(glm::vec3));
    cudaMemcpyToSymbol(reinterpret_cast<const void*>(&device_camera_right), &cright, sizeof(glm::vec3));
    cudaMemcpyToSymbol(reinterpret_cast<const void*>(&plane_count), &plane_cnt, sizeof(int));
}
