#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "HighPassFilter.h"
#include <iostream>

bool isCudaError(cudaError_t status)
{
	// printf("[%d] %s\n", status, cudaGetErrorString(status));
	return status != cudaSuccess;
}

// __global__ void kernel(const uint8_t* src, const int loopCnt, uint8_t* dest_1, uint8_t* dest_2, uint8_t* filter_1, uint8_t* filter_2, float* max_1, float* max_2)
// __global__ void kernel(const uint8_t* src, const int loopCnt, uint8_t* dest_1, uint8_t* dest_2, float* max_1, float* max_2)
__global__ void kernel(const uint8_t* src_1,const uint8_t* src_2, const int loopCnt, float* max_1, float* max_2, float* filter_1, float* filter_2, float hf_st1,float hf_cf1,float hf_st2,float hf_cf2)
{
	double HF_ST1 = hf_st1;
	double HF_CF1 = hf_cf1;
	double HF_ST2 = hf_st2;
	double HF_CF2 = hf_cf2;
	double IDT_1 = HF_ST2;
	double IDT_2 = HF_ST2;
	double OMEGA_C_1 = 2 * M_PI * HF_CF2;
	double OMEGA_C_2 = 2 * M_PI * HF_CF2;
	double AMPLFAC_1 = 1 / ((IDT_2 * OMEGA_C_1 / 2) + 1);
	double AMPLFAC_2 = 1 / ((IDT_2 * OMEGA_C_2 / 2) + 1);
	double Y1C_1 = (IDT_2 * OMEGA_C_1 / 2) - 1;
	double Y1C_2 = (IDT_2 * OMEGA_C_2 / 2) - 1;
	// double DT = IDT;

	// printf("HF_ST2 %f",HF_ST2 );
	// printf("HF_CF2 %f",HF_CF2 );
	const UINT taskIdx = threadIdx.x;
	double output_1 = 0, output_2 = 0, x1_1 = 0, x1_2 = 0;
	double _max_1 = 0, _max_2 = 0;
	// if (taskIdx == 0)
	// {
	// 	index = 0;
	// } else {

	// }
	for(UINT index = 0; index < loopCnt; index++)
	{
		const UINT realIdx = taskIdx * loopCnt + index;
		


		// dest_1[realIdx] = src[realIdx * 2];
		// dest_2[realIdx] = src[realIdx * 2 + 1];

		if((realIdx % loopCnt) == 0)
		{
			if(realIdx > 0)
			{
				for (UINT beforeIndex = realIdx - 500; beforeIndex < realIdx; beforeIndex++)
				{
					output_1 = AMPLFAC_1 * (src_1[beforeIndex] - x1_1 - output_1 * Y1C_1);
					output_2 = AMPLFAC_2 * (src_2[beforeIndex] - x1_2 - output_2 * Y1C_2);
					x1_1 = src_1[beforeIndex];
					x1_2 = src_2[beforeIndex];
				}
				// x1_1 = src_1[realIdx-1];
				// x1_2 = src_2[realIdx-1];
				// output_1 = AMPLFAC_1 * (src_1[realIdx-1] - src_1[realIdx-2] - output_1 * Y1C_1);
				// output_2 = AMPLFAC_2 * (src_2[realIdx-1] - src_2[realIdx-2] - output_2 * Y1C_2);
				// printf("x1_1 \n" , x1_1);
				// printf("loopcnt : %d, %d \n", taskIdx,realIdx);
			} else {
				// 여기에 2500만개 before data 100개를 넣으면 됨
				x1_1 = 0;
				x1_2 = 0;
				output_1= 0;
				output_2 = 0;
				// printf("한번만 탈텐데 :%d, %d \n", taskIdx, realIdx);
			}
			// printf("loopcnt : %d, %d", taskIdx,realIdx);
		}
			
		// if((realIdx % loopCnt) == 0)
		// {
			// x1_1 = src[realIdx*2-2];
			// x1_2 = src[realIdx*2-1];
		// }
		output_1 = AMPLFAC_1 * (src_1[realIdx] - x1_1 - output_1 * Y1C_1);
		output_2 = AMPLFAC_2 * (src_2[realIdx] - x1_2 - output_2 * Y1C_2);
		
		x1_1 = src_1[realIdx];
		x1_2 = src_2[realIdx];

		// filter_1[realIdx] = floor(output_1*1000) /1000;
		// filter_2[realIdx] = floor(output_2*1000) /1000;
		filter_1[realIdx] = output_1;
		filter_2[realIdx] = output_2;
		
		if(filter_1[realIdx] > _max_1) _max_1 = filter_1[realIdx];
		if(filter_2[realIdx] > _max_2) _max_2 = filter_2[realIdx];

		// if(output_1 > _max_1) _max_1 = output_1;
		// if(output_2 > _max_2) _max_2 = output_2;
	}
	max_1[taskIdx] = _max_1;
	max_2[taskIdx] = _max_2;
	//printf("[%f]", max_1[taskIdx]);
	// printf("[%d] \n", taskIdx);
}

// EXPORT int cudaHighPassFilter(const uint8_t* src, const int cnt, uint8_t* dest_1, uint8_t* dest_2, uint8_t* filter_1, uint8_t* filter_2, float* max_1, float* max_2)
// EXPORT int cudaHighPassFilter(const uint8_t* src, const int cnt, uint8_t* dest_1, uint8_t* dest_2, float* max_1, float* max_2)
EXPORT int cudaHighPassFilter(const uint8_t* src_1,const uint8_t* src_2, const int cnt, float* max_1, float* max_2, float* filter_1, float* filter_2, float hf_st1,float hf_cf1,float hf_st2, float hf_cf2)
// EXPORT int cudaHighPassFilter(const uint8_t* src, const int cnt, float* max_1, float* max_2, )
{
	// float _hf_st1 =0;
	// float _hf_cf1 =0;
	// float _hf_st2 =0;
	// float _hf_cf2 =0;
	// _hf_st1 = hf_st1;
	// _hf_cf1 = hf_cf1;
	// _hf_st2 = hf_st2;
	// _hf_cf2 = hf_cf2;
	// HF_ST1 = _hf_st1;
	// HF_CF1 = _hf_cf1;
	// HF_ST2 = _hf_st2;
	// HF_CF2 = _hf_cf2;
	// printf("in cudaHighPassFilter\n");
	uint8_t *dev_src_1 = 0;
	uint8_t *dev_src_2 = 0;
	// uint8_t *dev_dest_1 = 0, *dev_dest_2 = 0;
	float*dev_filter_1 = 0, *dev_filter_2 = 0;
	float *dev_max_1 = 0, *dev_max_2 = 0;

	cudaError_t status;

	// printf("start checkVersion\n");
	// int runtimeVer = 0, driverVer = 0;
	// status = cudaRuntimeGetVersion(&runtimeVer);
	// if(isCudaError(status)) goto Exit;
	// status = cudaDriverGetVersion(&driverVer);
	// if(isCudaError(status)) goto Exit;

	// printf("start cuda\n");
	// printf("cuda runtime ver.%d / cuda driver ver.%d\n", runtimeVer, driverVer);
	status = cudaSetDevice(0);
	if(isCudaError(status)) goto Exit;
	// printf("success cudaSetDevice\n");

	status = cudaMalloc((void**)&dev_src_1, cnt* sizeof(uint8_t));
	if (isCudaError(status)) goto Exit;
	status = cudaMalloc((void**)&dev_src_2, cnt * sizeof(uint8_t));
	if (isCudaError(status)) goto Exit;
	// status = cudaMalloc((void**)&dev_dest_1, cnt * sizeof(uint8_t));
	// if (isCudaError(status)) goto Exit;
	// status = cudaMalloc((void**)&dev_dest_2, cnt * sizeof(uint8_t));
	// if (isCudaError(status)) goto Exit;
	status = cudaMalloc((void**)&dev_filter_1, cnt * sizeof(float));
	if (isCudaError(status)) goto Exit;
	status = cudaMalloc((void**)&dev_filter_2, cnt * sizeof(float));
	if (isCudaError(status)) goto Exit;
	status = cudaMalloc((void**)&dev_max_1, UNIT_COUNT * sizeof(float));
	if (isCudaError(status)) goto Exit;
	status = cudaMalloc((void**)&dev_max_2, UNIT_COUNT * sizeof(float));
	if (isCudaError(status)) goto Exit;
	// printf("success cudaMalloc\n");

	status = cudaMemcpy(dev_src_1, src_1, cnt* sizeof(uint8_t), cudaMemcpyHostToDevice);
	if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(dev_src_2, src_2, cnt * sizeof(uint8_t), cudaMemcpyHostToDevice);
	if (isCudaError(status)) goto Exit;
	// printf("success cudaMemcpy\n");

	kernel<<<1, UNIT_COUNT>>> (dev_src_1,dev_src_2, cnt / UNIT_COUNT, dev_max_1, dev_max_2,dev_filter_1, dev_filter_2,hf_st1,hf_cf1,hf_st2,hf_cf2);
	// kernel<<<1, UNIT_COUNT>>> (dev_src, cnt / UNIT_COUNT, dev_dest_1, dev_dest_2, dev_max_1, dev_max_2);
	// kernel<<<1, UNIT_COUNT>>> (dev_src, cnt / UNIT_COUNT, dev_dest_1, dev_dest_2, dev_filter_1, dev_filter_2, dev_max_1, dev_max_2);
	if (isCudaError(cudaGetLastError())) goto Exit;
	// printf("success kernel\n");

	status = cudaDeviceSynchronize();
	if (isCudaError(status)) goto Exit;
	// printf("success cudaDeviceSynchronize\n");

	// status = cudaMemcpy(dest_1, dev_dest_1, cnt * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	// if (isCudaError(status)) goto Exit;
	// status = cudaMemcpy(dest_2, dev_dest_2, cnt * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	// if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(filter_1, dev_filter_1, cnt * sizeof(float), cudaMemcpyDeviceToHost);
	if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(filter_2, dev_filter_2, cnt * sizeof(float), cudaMemcpyDeviceToHost);
	if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(max_1, dev_max_1, UNIT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);
	if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(max_2, dev_max_2, UNIT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);
	if (isCudaError(status)) goto Exit;
	// printf("success cudaMemcpy\n");

Exit:
	cudaFree(dev_src_1);
	cudaFree(dev_src_2);
	// cudaFree(dev_dest_1);
	// cudaFree(dev_dest_2);
	cudaFree(dev_filter_1);
	cudaFree(dev_filter_2);
	cudaFree(dev_max_1);
	cudaFree(dev_max_2);

	return status;
}