#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Timer.h"
#include "HighPassFilter.h"
#include <iostream>

bool isCudaError(cudaError_t status)
{
	// printf("[%d] %s\n", status, cudaGetErrorString(status));
	return status != cudaSuccess;
}

__global__ void kernel(double* times,const uint8_t * before_data_1, const uint8_t * before_data_2,const uint8_t* src, const int loopCnt, float* max_1, float* max_2,const double AMPLFAC_1,const double AMPLFAC_2,const double Y1C_1,const double Y1C_2)
{

	clock_t start = clock(); 
	const UINT taskIdx = threadIdx.x;
	double output_1 = 0, output_2 = 0;
	double x1_1 = 0, x1_2 = 0;
	float _max_1 = 0, _max_2 = 0;
	// const UINT beforeIdx = taskIdx * loopCnt;

	// if((beforeIdx % loopCnt) == 0)
	// {
	// 	if(beforeIdx == 0)
	// 	{
	// 		// 여기에 2500만개 before data 100개를 넣으면 됨
	// 		for (UINT beforeDataIndex = 0; beforeDataIndex < BEFORE_DATA_COUNT; beforeDataIndex++)
	// 		{
	// 			output_1 = AMPLFAC_1 * ((float)before_data_1[beforeDataIndex] - x1_1 - output_1 * Y1C_1);
	// 			output_2 = AMPLFAC_2 * ((float)before_data_2[beforeDataIndex] - x1_2 - output_2 * Y1C_2);;
	// 			x1_1 = (float)before_data_1[beforeDataIndex];
	// 			x1_2 = (float)before_data_2[beforeDataIndex];
	// 		}
	// 	} else {
	// 		for (UINT beforeIndex = beforeIdx - BEFORE_DATA_COUNT; beforeIndex < beforeIdx; beforeIndex++)
	// 		{
	// 			output_1 = AMPLFAC_1 * ((float)src[beforeIndex*2] - x1_1 - output_1 * Y1C_1);
	// 			output_2 = AMPLFAC_2 * ((float)src[beforeIndex*2+1] - x1_2 - output_2 * Y1C_2);
	// 			x1_1 = (float)src[beforeIndex*2];
	// 			x1_2 = (float)src[beforeIndex*2+1];
	// 		}
	// 	}
	// }

	for(UINT index = 0; index < loopCnt; index++)
	{
		const UINT realIdx = taskIdx * loopCnt + index;

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
		if((realIdx % loopCnt) == 0 && realIdx !=0)
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
		
		x1_1 = (float) src[realIdx*2];
		x1_2 = (float) src[realIdx*2+1];

		if(output_1 > _max_1) _max_1 = output_1;
		if(output_2 > _max_2) _max_2 = output_2;
	}

	max_1[taskIdx] = _max_1;
	max_2[taskIdx] = _max_2;
	clock_t stop = clock(); 
    times[taskIdx] = (double)((double)(stop - start) / 1000);
}

EXPORT int cudaHighPassFilter(double * times,const uint8_t * before_data_1, const uint8_t * before_data_2,const uint8_t* src, const int cnt, float* max_1, float* max_2, double hf_st1,double hf_cf1,double hf_st2, double hf_cf2)
{
	// clock_t clock();
	// long long int clock64();
	// WGSTest::Timer timer;
	uint8_t *dev_before_data_1 = 0;
	uint8_t *dev_before_data_2 = 0;
	uint8_t *dev_src = 0;
	float *dev_max_1 = 0, *dev_max_2 = 0;
	const double OMEGA_C_1 = 2 * M_PI * hf_cf2;
	const double OMEGA_C_2 = 2 * M_PI * hf_cf2;
	const double AMPLFAC_1 = 1 / ((hf_st2 * OMEGA_C_1 / 2) + 1);
	const double AMPLFAC_2 = 1 / ((hf_st2 * OMEGA_C_2 / 2) + 1);
	const double Y1C_1 = (hf_st2 * OMEGA_C_1 / 2) - 1;
	const double Y1C_2 = (hf_st2 * OMEGA_C_2 / 2) - 1;
	double * dev_times=0; 
	// 알고리즘

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

	// printf("start cuda\n");
	// printf("cuda runtime ver.%d / cuda driver ver.%d\n", runtimeVer, driverVer);
	status = cudaSetDevice(0);
	if(isCudaError(status)) goto Exit;

	// timer.Reset();
    // timer.Start();
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
	status = cudaMalloc((void**)&dev_times, UNIT_COUNT * sizeof(double));
	if (isCudaError(status)) goto Exit;
	// timer.End();
	// timer.Print("cudaMalloc");

	// timer.Reset();
    // timer.Start();
	// cuda로 데이터 memcpy
	status = cudaMemcpy(dev_src, src, (cnt*2) * sizeof(uint8_t), cudaMemcpyHostToDevice);
	if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(dev_before_data_1, before_data_1, BEFORE_DATA_COUNT* sizeof(uint8_t), cudaMemcpyHostToDevice);
	if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(dev_before_data_2, before_data_2, BEFORE_DATA_COUNT * sizeof(uint8_t), cudaMemcpyHostToDevice);
	if (isCudaError(status)) goto Exit;
	// timer.End();
	// timer.Print("Device로 cudaMemcpy");

	// timer.Reset();
    // timer.Start();

	// cuda로 작동하는 function
	// float timess;
	// cudaEvent_t start, stop;
	// cudaEventCreate(&start);
	// cudaEventCreate(&stop);
	// cudaEventRecord(start, 0);

	// kernel<<<1, UNIT_COUNT>>> (dev_times,dev_before_data_1, dev_before_data_2,dev_src, cnt / UNIT_COUNT, dev_max_1, dev_max_2,AMPLFAC_1,AMPLFAC_2,Y1C_1,Y1C_2);
	// cudaEventRecord(stop);
	// cudaEventSynchronize(stop);
	// cudaEventElapsedTime(&timess, start, stop);
	// printf("Elapsed time : %f ms\n" ,timess);

	if (isCudaError(cudaGetLastError())) goto Exit;

	// cuda 동기화
	status = cudaDeviceSynchronize();
	if (isCudaError(status)) goto Exit;
	// timer.End();
	// timer.Print("kernel");



	// timer.Reset();
    // timer.Start();
	// cuda데이터를 Host로 memcpy
	status = cudaMemcpy(max_1, dev_max_1, UNIT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);
	if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(max_2, dev_max_2, UNIT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);
	if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(times, dev_times, UNIT_COUNT * sizeof(double), cudaMemcpyDeviceToHost);
	if (isCudaError(status)) goto Exit;
	// timer.End();
	// timer.Print("Host로 cudaMemcpy");

	// for(UINT i = 0; i <200; i++)
	// {
	// 	printf("times %d %d",i, times[i]);
	// }
	// printf("times %d ",times[0]);

Exit:
	cudaFree(dev_src);
	cudaFree(dev_before_data_1);
	cudaFree(dev_before_data_2);
	cudaFree(dev_max_1);
	cudaFree(dev_max_2);
	cudaFree(dev_times);

	return status;
}