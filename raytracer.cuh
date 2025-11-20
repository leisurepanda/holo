#pragma once
#include "platform.h"
#include<glm/glm.hpp>

#ifdef __cplusplus
extern "C" {
#endif

	//setter
	void cuda_set_info(glm::vec3 cpos, glm::vec3 cfront, glm::vec3 cup, glm::vec3 cright, int plane_cnt);
	void cuda_upload_planes(const platform* planes, int count);
	void secondary_info(const lighthit* lightHits, int lighthit_count);
	// CPU 調用這個函式啟動 CUDA 渲染
	void cuda_try_raytrace(unsigned char* output, int width, int height,glm::vec3 raydir);

#ifdef __cplusplus
}
#endif
