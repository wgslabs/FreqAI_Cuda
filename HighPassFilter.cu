#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// #include "Timer.h"
#include "HighPassFilter.h"
#include <iostream>

bool isCudaError(cudaError_t status)
{
	// printf("[%d] %s\n", status, cudaGetErrorString(status));
	return status != cudaSuccess;
}

__global__ void kernel(const uint8_t* src, const int loopCnt,const uint8_t * before_data_1, const uint8_t * before_data_2, float* max_1, float* max_2,float* min_1, float* min_2,const double AMPLFAC_1,const double AMPLFAC_2,const double Y1C_1,const double Y1C_2, const int OFFSET_1, const float SCALE_1, const int OFFSET_2, const float SCALE_2)
{
	const UINT taskIdx = threadIdx.x;
	float output_1 = 0, output_2 = 0;
	float x1_1 = 0, x1_2 = 0;
	float _max_1 = 0, _max_2 = 0;
	float _min_1 = (float)OFFSET_1;
	float _min_2 = (float)OFFSET_2;

	for(UINT index = 0; index < loopCnt; index++)
	{
		const UINT realIdx = taskIdx * loopCnt + index;
		
		// 진폭 알고리즘
		if((realIdx % 4) == 0 && realIdx != 0)
		{
			float _srcMax_1 = 0, _srcMax_2 = 0;
			for(UINT srcIndex = realIdx - 4; srcIndex <realIdx; srcIndex++)
			{
				float __src_1 =  abs((OFFSET_1 - src[srcIndex*2])*SCALE_1);
				float __src_2 =  abs((OFFSET_2 - src[srcIndex*2 + 1])*SCALE_2);
				if(__src_1 > _srcMax_1) _srcMax_1 = __src_1;
				if(__src_2 > _srcMax_2) _srcMax_2 = __src_2;
			}
			if(_srcMax_1 < _min_1) _min_1=_srcMax_1;
			if(_srcMax_2 < _min_2) _min_2=_srcMax_2;
		}
		
		// 2500만개의 index 0 에서 바로 전 2500만개의 뒷부분 가져와서 output_1, x1_1
		if(realIdx == 0)
		{
			// 여기에 2500만개의 before data 100개를 넣으면 됨
			for (UINT beforeDataIndex = 0; beforeDataIndex < BEFORE_DATA_COUNT; beforeDataIndex++)
			{
				output_1 = AMPLFAC_1 * ((float)before_data_1[beforeDataIndex] - x1_1 - output_1 * Y1C_1);
				output_2 = AMPLFAC_2 * ((float)before_data_2[beforeDataIndex] - x1_2 - output_2 * Y1C_2);;
				x1_1 = (float)before_data_1[beforeDataIndex];
				x1_2 = (float)before_data_2[beforeDataIndex];
			}
		}

		// 125056 의 배수들에서 그 앞 데이터 -100번째에서 output_1, x1_1
		if((realIdx % loopCnt) == 0 && realIdx != 0)
		{
			for (UINT beforeIndex = realIdx - BEFORE_DATA_COUNT; beforeIndex < realIdx; beforeIndex++)
			{
				output_1 = AMPLFAC_1 * ((float)src[beforeIndex*2] - x1_1 - output_1 * Y1C_1);
				output_2 = AMPLFAC_2 * ((float)src[beforeIndex*2+1] - x1_2 - output_2 * Y1C_2);
				x1_1 = (float)src[beforeIndex*2];
				x1_2 = (float)src[beforeIndex*2+1];
			}
		}

		output_1 = AMPLFAC_1 * ((float)src[realIdx*2] - x1_1 - output_1 * Y1C_1);
		output_2 = AMPLFAC_2 * ((float)src[realIdx*2+1] - x1_2 - output_2 * Y1C_2);
		
		x1_1 =  (float)src[realIdx*2];
		x1_2 =  (float)src[realIdx*2+1];

		if(output_1 > _max_1) _max_1 = output_1;
		if(output_2 > _max_2) _max_2 = output_2;
	}

	max_1[taskIdx] = _max_1;
	max_2[taskIdx] = _max_2;
	min_1[taskIdx] = _min_1;
	min_2[taskIdx] = _min_2;
}

EXPORT int cudaHighPassFilter(const uint8_t* src, const int cnt, const uint8_t * before_data_1, const uint8_t * before_data_2,float* max_1, float* max_2, float* min_1, float* min_2,const double hf_st1,const double hf_cf1,const double hf_st2, const double hf_cf2, const int offset_1, const double scale_1, const int offset_2, const double scale_2)
{
	uint8_t *dev_before_data_1 = 0;
	uint8_t *dev_before_data_2 = 0;
	uint8_t *dev_src = 0;
	float *dev_max_1 = 0, *dev_max_2 = 0;
	float *dev_min_1 = 0, *dev_min_2 = 0;
	const int OFFSET_1 = offset_1;
	const int OFFSET_2 = offset_2;
	const float SCALE_1 = scale_1;
	const float SCALE_2 = scale_2;
	const double OMEGA_C_1 = 2 * M_PI * hf_cf2; // 각주파수?
	const double OMEGA_C_2 = 2 * M_PI * hf_cf2;
	const double AMPLFAC_1 = 1 / ((hf_st2 * OMEGA_C_1 / 2) + 1);
	const double AMPLFAC_2 = 1 / ((hf_st2 * OMEGA_C_2 / 2) + 1);
	const double Y1C_1 = (hf_st2 * OMEGA_C_1 / 2) - 1;
	const double Y1C_2 = (hf_st2 * OMEGA_C_2 / 2) - 1;

	cudaError_t status;

	if (hf_cf1 < hf_st1 || hf_cf2 < hf_st2)
	{
		printf("Cuda Algorithm Value Error");
		goto Exit;
	}
	// printf("start checkVersion\n");
	// int runtimeVer = 0, driverVer = 0;
	// status = cudaRuntimeGetVersion(&runtimeVer);
	// if(isCudaError(status)) goto Exit;
	// status = cudaDriverGetVersion(&driverVer);
	// if(isCudaError(status)) goto Exit;

	// printf("cuda runtime ver.%d / cuda driver ver.%d\n", runtimeVer, driverVer);
	status = cudaSetDevice(0);
	if(isCudaError(status)) goto Exit;

	// cuda에 데이터 malloc
	status = cudaMalloc((void**)&dev_src, (cnt*2)* sizeof(uint8_t));
	if (isCudaError(status)) goto Exit;
	status = cudaMalloc((void**)&dev_before_data_1, BEFORE_DATA_COUNT* sizeof(uint8_t));
	if (isCudaError(status)) goto Exit;
	status = cudaMalloc((void**)&dev_before_data_2, BEFORE_DATA_COUNT * sizeof(uint8_t));
	if (isCudaError(status)) goto Exit;
	status = cudaMalloc((void**)&dev_max_1, UNIT_COUNT * sizeof(float));
	if (isCudaError(status)) goto Exit;
	status = cudaMalloc((void**)&dev_max_2, UNIT_COUNT * sizeof(float));
	if (isCudaError(status)) goto Exit;
	status = cudaMalloc((void**)&dev_min_1, UNIT_COUNT * sizeof(float));
	if (isCudaError(status)) goto Exit;
	status = cudaMalloc((void**)&dev_min_2, UNIT_COUNT * sizeof(float));
	if (isCudaError(status)) goto Exit;

	// cuda로 데이터 memcpy
	status = cudaMemcpy(dev_src, src, (cnt*2) * sizeof(uint8_t), cudaMemcpyHostToDevice);
	if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(dev_before_data_1, before_data_1, BEFORE_DATA_COUNT* sizeof(uint8_t), cudaMemcpyHostToDevice);
	if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(dev_before_data_2, before_data_2, BEFORE_DATA_COUNT * sizeof(uint8_t), cudaMemcpyHostToDevice);
	if (isCudaError(status)) goto Exit;

	// cuda로 작동하는 function
	kernel<<<1, UNIT_COUNT>>> (dev_src, cnt / UNIT_COUNT, dev_before_data_1, dev_before_data_2, dev_max_1, dev_max_2,dev_min_1, dev_min_2, AMPLFAC_1, AMPLFAC_2, Y1C_1, Y1C_2, OFFSET_1, SCALE_1, OFFSET_2, SCALE_2);
	if (isCudaError(cudaGetLastError())) goto Exit;
	// cuda 동기화
	status = cudaDeviceSynchronize();
	if (isCudaError(status)) goto Exit;

	// cuda데이터를 Host로 memcpy
	status = cudaMemcpy(max_1, dev_max_1, UNIT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);
	if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(max_2, dev_max_2, UNIT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);
	if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(min_1, dev_min_1, UNIT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);
	if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(min_2, dev_min_2, UNIT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);
	if (isCudaError(status)) goto Exit;

Exit:
	cudaFree(dev_src);
	cudaFree(dev_before_data_1);
	cudaFree(dev_before_data_2);
	cudaFree(dev_max_1);
	cudaFree(dev_max_2);
	cudaFree(dev_min_1);
	cudaFree(dev_min_2);

	return status;
}