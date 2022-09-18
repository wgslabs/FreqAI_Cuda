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
__global__ void kernel(const uint8_t* src, const int loopCnt,const uint8_t * before_data_1, const uint8_t * before_data_2, float* max_1, float* max_2,uint8_t* min_1, uint8_t* min_2,const double AMPLFAC_1,const double AMPLFAC_2,const double Y1C_1,const double Y1C_2, const int OFFSET_1, const float SCALE_1, const int OFFSET_2, const float SCALE_2, int* ampResults1, int* ampResults2, int ampThreshold, uint8_t *ampMaxs1,uint8_t *ampMaxs2)
// __global__ void kernel(const uint8_t* src, const int loopCnt,const uint8_t * before_data_1, const uint8_t * before_data_2, float* max_1, float* max_2,uint8_t* min_1, uint8_t* min_2,const double AMPLFAC_1,const double AMPLFAC_2,const double Y1C_1,const double Y1C_2, const int OFFSET_1, const float SCALE_1, const int OFFSET_2, const float SCALE_2)
{
	const UINT taskIdx = threadIdx.x;
	float output_1 = 0, output_2 = 0;
	float x1_1 = 0, x1_2 = 0;
	float _max_1 = 0, _max_2 = 0;
	uint8_t _min_1 = 255;
	uint8_t _min_2 = 255;
	int ampMaxTotal1 = 0;
	int ampMaxTotal2 = 0;
	int ampMaxsIdx = taskIdx * loopCnt/ AMP_DEVIDE_COUNT; // taskIdx 0~199, loopCnt =125056, 5
	
	for(UINT index = 0; index < loopCnt; index++)
	{
		if(index < AMP_RESULT_DATA_COUNT){
			const int ampResultsIdx = taskIdx * AMP_RESULT_DATA_COUNT + index;
			ampResults1[ampResultsIdx] = -1;
			ampResults2[ampResultsIdx] = -1;
		}
		const UINT realIdx = taskIdx * loopCnt + index;
		
		// 진폭 알고리즘
		if((realIdx % AMP_DEVIDE_COUNT) == 0 && realIdx != 0)
		{
			ampMaxs1[ampMaxsIdx] = 0;
			ampMaxs2[ampMaxsIdx] = 0;
			// uint8_t _srcMax_1 = 0, _srcMax_2 = 0;
			for(UINT srcIndex = realIdx - AMP_DEVIDE_COUNT; srcIndex <realIdx; srcIndex++)
			{
				
				const uint8_t __src_1 =  abs((OFFSET_1 - src[srcIndex*2]));
				const uint8_t __src_2 =  abs((OFFSET_2 - src[srcIndex*2 + 1]));
				if(__src_1 > ampMaxs1[ampMaxsIdx]) ampMaxs1[ampMaxsIdx] = __src_1;
				if(__src_2 > ampMaxs2[ampMaxsIdx]) ampMaxs2[ampMaxsIdx] = __src_2;
				// if(__src_1 > _srcMax_1) _srcMax_1 = __src_1;
				// if(__src_2 > _srcMax_2) _srcMax_2 = __src_2;
			}
			// if(_srcMax_1 < _min_1) _min_1=_srcMax_1;
			// if(_srcMax_1 < _min_1) _min_1=_srcMax_1;
			ampMaxTotal1 = ampMaxTotal1 + ampMaxs1[ampMaxsIdx];
			ampMaxTotal2 = ampMaxTotal2 + ampMaxs2[ampMaxsIdx];
			if(ampMaxs1[ampMaxsIdx] < _min_1) _min_1=ampMaxs1[ampMaxsIdx];
			if(ampMaxs2[ampMaxsIdx] < _min_2) _min_2=ampMaxs2[ampMaxsIdx];
			ampMaxsIdx++;
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
	
	const float ampMaxAvg1 = ampMaxTotal1/ (loopCnt/AMP_DEVIDE_COUNT);
	const float ampMaxAvg2 = ampMaxTotal2/ (loopCnt/AMP_DEVIDE_COUNT);
	int ampResultIdx1 = 0;
	int ampResultIdx2 = 0;
	for(UINT index = 0; index < loopCnt/AMP_DEVIDE_COUNT; index++){
		const UINT ampMaxsIdx = taskIdx * loopCnt/AMP_DEVIDE_COUNT + index;
		const uint8_t diff1 = 100-((ampMaxs1[ampMaxsIdx] / ampMaxAvg1)  *100);
		const uint8_t diff2 = 100-((ampMaxs2[ampMaxsIdx] / ampMaxAvg2)  *100);
		const UINT ampResultIdx = taskIdx * AMP_RESULT_DATA_COUNT;
		if( ampThreshold < diff1 && ampResultIdx1 < 100){
			ampResults1[ampResultIdx+ampResultIdx1++] = ampMaxsIdx * AMP_DEVIDE_COUNT;
			// printf("1 %d_%d_%d_%f,%d\n",ampResultIdx, ampResultIdx1, ampMaxsIdx * AMP_DEVIDE_COUNT, ampMaxAvg1 , ampMaxs1[ampMaxsIdx]);
		}
		if( ampThreshold < diff2 && ampResultIdx2 < 100){
			ampResults2[ampResultIdx+ampResultIdx2++] = ampMaxsIdx * AMP_DEVIDE_COUNT;
			// printf("2 %d_%d_%d_%f,%d\n",ampResultIdx, ampResultIdx2, ampMaxsIdx * AMP_DEVIDE_COUNT, ampMaxAvg2 , ampMaxs2[ampMaxsIdx]);
		}

	}
	max_1[taskIdx] = _max_1;
	max_2[taskIdx] = _max_2;
	min_1[taskIdx] = _min_1;
	min_2[taskIdx] = _min_2;
}
EXPORT int cudaHighPassFilter(const uint8_t* src, const int cnt, const uint8_t * before_data_1, const uint8_t * before_data_2,float* max_1, float* max_2, uint8_t* min_1, uint8_t* min_2,const double hf_st1,const double hf_cf1,const double hf_st2, const double hf_cf2, const int offset_1, const double scale_1, const int offset_2, const double scale_2, int* ampResult1, int* ampResult2, const int ampThreshold)
// EXPORT int cudaHighPassFilter(const uint8_t* src, const int cnt, const uint8_t * before_data_1, const uint8_t * before_data_2,float* max_1, float* max_2, uint8_t* min_1, uint8_t* min_2,const double hf_st1,const double hf_cf1,const double hf_st2, const double hf_cf2, const int offset_1, const double scale_1, const int offset_2, const double scale_2)
{
	uint8_t *dev_before_data_1 = 0;
	uint8_t *dev_before_data_2 = 0;
	uint8_t *dev_src = 0;
	float *dev_max_1 = 0, *dev_max_2 = 0;
	uint8_t *dev_min_1 = 0, *dev_min_2 = 0;
	int *dev_ampResults1 = 0, *dev_ampResults2 = 0;
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
	
	uint8_t *dev_ampMaxs1 =0;
	uint8_t *dev_ampMaxs2 =0;
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
	// printf("1 \n");
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
	status = cudaMalloc((void**)&dev_min_1, UNIT_COUNT * sizeof(uint8_t));
	if (isCudaError(status)) goto Exit;
	status = cudaMalloc((void**)&dev_min_2, UNIT_COUNT * sizeof(uint8_t));
	if (isCudaError(status)) goto Exit;
	
	status = cudaMalloc((void**)&dev_ampResults1, UNIT_COUNT * AMP_RESULT_DATA_COUNT* sizeof(uint32_t));
	if (isCudaError(status)) goto Exit;
	status = cudaMalloc((void**)&dev_ampResults2, UNIT_COUNT * AMP_RESULT_DATA_COUNT* sizeof(uint32_t));
	if (isCudaError(status)) goto Exit;

	status = cudaMalloc((void**)&dev_ampMaxs1,  cnt / AMP_DEVIDE_COUNT * sizeof(uint8_t)); //2500만 /5
	if (isCudaError(status)) goto Exit;
	status = cudaMalloc((void**)&dev_ampMaxs2,  cnt / AMP_DEVIDE_COUNT * sizeof(uint8_t));
	if (isCudaError(status)) goto Exit;

	// cuda로 데이터 memcpy
	status = cudaMemcpy(dev_src, src, (cnt*2) * sizeof(uint8_t), cudaMemcpyHostToDevice);
	if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(dev_before_data_1, before_data_1, BEFORE_DATA_COUNT * sizeof(uint8_t), cudaMemcpyHostToDevice);
	if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(dev_before_data_2, before_data_2, BEFORE_DATA_COUNT* sizeof(uint8_t), cudaMemcpyHostToDevice);
	if (isCudaError(status)) goto Exit;

	// cuda로 작동하는 function
	// kernel<<<1, UNIT_COUNT>>> (dev_src, cnt / UNIT_COUNT, dev_before_data_1, dev_before_data_2, dev_max_1, dev_max_2,dev_min_1, dev_min_2, AMPLFAC_1, AMPLFAC_2, Y1C_1, Y1C_2, OFFSET_1, SCALE_1, OFFSET_2, SCALE_2);
	kernel<<<1, UNIT_COUNT>>> (dev_src, cnt / UNIT_COUNT, dev_before_data_1, dev_before_data_2, dev_max_1, dev_max_2,dev_min_1, dev_min_2, AMPLFAC_1, AMPLFAC_2, Y1C_1, Y1C_2, OFFSET_1, SCALE_1, OFFSET_2, SCALE_2, dev_ampResults1, dev_ampResults2, ampThreshold, dev_ampMaxs1, dev_ampMaxs2);
	if (isCudaError(cudaGetLastError())) goto Exit;
	// cuda 동기화
	status = cudaDeviceSynchronize();
	if (isCudaError(status)) goto Exit;

	// cuda데이터를 Host로 memcpy
	status = cudaMemcpy(max_1, dev_max_1, UNIT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);
	if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(max_2, dev_max_2, UNIT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);
	if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(min_1, dev_min_1, UNIT_COUNT * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(min_2, dev_min_2, UNIT_COUNT * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	if (isCudaError(status)) goto Exit;

Exit:
	cudaFree(dev_src);
	cudaFree(dev_before_data_1);
	cudaFree(dev_before_data_2);
	cudaFree(dev_max_1);
	cudaFree(dev_max_2);
	cudaFree(dev_min_1);
	cudaFree(dev_min_2);
	cudaFree(dev_ampResults1);
	cudaFree(dev_ampResults2);
	cudaFree(dev_ampMaxs1);
	cudaFree(dev_ampMaxs2);

	return status;
}



__global__ void kernel2(const uint8_t* src, const int loopCnt,const uint8_t * before_data_1, const uint8_t * before_data_2, float* max_1, float* max_2,uint8_t* min_1, uint8_t* min_2,const double AMPLFAC_1,const double AMPLFAC_2,const double Y1C_1,const double Y1C_2, const int OFFSET_1, const float SCALE_1, const int OFFSET_2, const float SCALE_2)
{
	const UINT taskIdx = threadIdx.x;
	float output_1 = 0, output_2 = 0;
	float x1_1 = 0, x1_2 = 0;
	float _max_1 = 0, _max_2 = 0;
	uint8_t _min_1 = 255;
	uint8_t _min_2 = 255;
	int a = 0;

	// 2500만개의 index 0 에서 바로 전 2500만개의 뒷부분 가져와서 output_1, x1_1
	if(taskIdx == 0)
	{
		// 여기에 2500만개의 before data 100개를 넣으면 됨
		for (UINT beforeDataIndex = 0; beforeDataIndex < BEFORE_DATA_COUNT; beforeDataIndex++)
		{
			
			const float value1 = OFFSET_1 - before_data_1[beforeDataIndex];
			const float value2 = OFFSET_2 - before_data_2[beforeDataIndex];
			output_1 = AMPLFAC_1 * (value1 - x1_1 - output_1 * Y1C_1);
			output_2 = AMPLFAC_2 * (value2 - x1_2 - output_2 * Y1C_2);;
			x1_1 = value1;
			x1_2 = value2;
		}
	}
	else{//125056 의 배수들에서 그 앞 데이터 -100번째에서 output_1, x1_1
		for (UINT beforeIndex = 0; beforeIndex < BEFORE_DATA_COUNT; beforeIndex++)
		{
			const UINT realIdx = taskIdx * loopCnt + beforeIndex - BEFORE_DATA_COUNT;
			const float value1 = OFFSET_1 - src[realIdx*2];
			const float value2 = OFFSET_2 - src[realIdx*2+1];
			output_1 = AMPLFAC_1 * (value1 - x1_1 - output_1 * Y1C_1);
			output_2 = AMPLFAC_2 * (value2 - x1_2 - output_2 * Y1C_2);
			x1_1 = value1;
			x1_2 = value2;
		}
	}

	for(UINT index = 0; index < loopCnt; index++)
	{
		const UINT realIdx = taskIdx * loopCnt + index;
		
		// if(taskIdx == 1){
		// 	if(index==0){
		// 		printf("hpf data %f %f",output_1, x1_1);

		// 	}
		// }

		if((realIdx % AMP_DEVIDE_COUNT) == AMP_DEVIDE_COUNT - 1)
		{
			uint8_t _ampMax1 = 0;
			uint8_t _ampMax2 = 0;
			for(UINT srcIndex = realIdx - AMP_DEVIDE_COUNT + 1; srcIndex <= realIdx; srcIndex++)
			{
				const uint8_t __src_1 =  abs((OFFSET_1 - src[srcIndex*2]));
				const uint8_t __src_2 =  abs((OFFSET_2 - src[srcIndex*2 + 1]));
				if(__src_1 > _ampMax1) _ampMax1 = __src_1;
				if(__src_2 > _ampMax2) _ampMax2 = __src_2;
			}
			if(_ampMax1 < _min_1) _min_1=_ampMax1;
			if(_ampMax2 < _min_2) _min_2=_ampMax2;
			a++;
		}

		const float value1 = OFFSET_1 - src[realIdx*2];
		const float value2 = OFFSET_2 - src[realIdx*2+1];
		output_1 = AMPLFAC_1 * (value1 - x1_1 - output_1 * Y1C_1);
		output_2 = AMPLFAC_2 * (value2 - x1_2 - output_2 * Y1C_2);
		
		x1_1 =  (float)value1;
		x1_2 =  (float)value2;
		
		// H ALGO
		const float absO1 = abs(output_1);
		const float absO2 = abs(output_2);
		// if(taskIdx == 1){
		// 	if(index < 10){
		// 		printf("value %d %f %f %f %f",realIdx, value1,value2 ,absO1, absO2);
		// 		printf("aaaaaaaa %f %f %f %f \n",AMPLFAC_1, x1_1,output_1 ,Y1C_1);
		// 	}
		// }
		if(absO1 > _max_1) _max_1 = absO1;
		if(absO2 > _max_2) _max_2 = absO2;
	}
	
	max_1[taskIdx] = _max_1;
	max_2[taskIdx] = _max_2;
	min_1[taskIdx] = _min_1;
	min_2[taskIdx] = _min_2;
}
EXPORT int cudaHighPassFilter2(const uint8_t* src, const int cnt, const uint8_t * before_data_1, const uint8_t * before_data_2,float* max_1, float* max_2, uint8_t* min_1, uint8_t* min_2,const double hf_st1,const double hf_cf1,const double hf_st2, const double hf_cf2, const int offset_1, const double scale_1, const int offset_2, const double scale_2)
{
	uint8_t *dev_before_data_1 = 0;
	uint8_t *dev_before_data_2 = 0;
	uint8_t *dev_src = 0;
	float *dev_max_1 = 0, *dev_max_2 = 0;
	uint8_t *dev_min_1 = 0, *dev_min_2 = 0;
	const int OFFSET_1 = offset_1;
	const int OFFSET_2 = offset_2;
	const float SCALE_1 = scale_1;
	const float SCALE_2 = scale_2;
	const double OMEGA_C_1 = 2 * M_PI * hf_cf1;
	const double OMEGA_C_2 = 2 * M_PI * hf_cf2;
	const double AMPLFAC_1 = 1 / ((hf_st1 * OMEGA_C_1 / 2) + 1);
	const double AMPLFAC_2 = 1 / ((hf_st2 * OMEGA_C_2 / 2) + 1);
	const double Y1C_1 = (hf_st1 * OMEGA_C_1 / 2) - 1;
	const double Y1C_2 = (hf_st2 * OMEGA_C_2 / 2) - 1;
	
	cudaError_t status;

	if (hf_cf1 < hf_st1 || hf_cf2 < hf_st2)
	{
		printf("Cuda Algorithm Value Error");
		goto Exit;
	}
	// int runtimeVer = 0, driverVer = 0;
	// status = cudaRuntimeGetVersion(&runtimeVer);
	// if(isCudaError(status)) goto Exit;
	// status = cudaDriverGetVersion(&driverVer);
	// if(isCudaError(status)) goto Exit;

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
	status = cudaMalloc((void**)&dev_min_1, UNIT_COUNT * sizeof(uint8_t));
	if (isCudaError(status)) goto Exit;
	status = cudaMalloc((void**)&dev_min_2, UNIT_COUNT * sizeof(uint8_t));
	if (isCudaError(status)) goto Exit;
	
	// cuda로 데이터 memcpy
	status = cudaMemcpy(dev_src, src, (cnt*2) * sizeof(uint8_t), cudaMemcpyHostToDevice);
	if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(dev_before_data_1, before_data_1, BEFORE_DATA_COUNT * sizeof(uint8_t), cudaMemcpyHostToDevice);
	if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(dev_before_data_2, before_data_2, BEFORE_DATA_COUNT* sizeof(uint8_t), cudaMemcpyHostToDevice);
	if (isCudaError(status)) goto Exit;

	// cuda로 작동하는 function
	kernel2<<<1, UNIT_COUNT>>> (dev_src, cnt / UNIT_COUNT, dev_before_data_1, dev_before_data_2, dev_max_1, dev_max_2,dev_min_1, dev_min_2, AMPLFAC_1, AMPLFAC_2, Y1C_1, Y1C_2, OFFSET_1, SCALE_1, OFFSET_2, SCALE_2);
	if (isCudaError(cudaGetLastError())) goto Exit;
	// cuda 동기화
	status = cudaDeviceSynchronize();
	if (isCudaError(status)) goto Exit;

	
	// cuda데이터를 Host로 memcpy
	status = cudaMemcpy(max_1, dev_max_1, UNIT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);
	if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(max_2, dev_max_2, UNIT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);
	if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(min_1, dev_min_1, UNIT_COUNT * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	if (isCudaError(status)) goto Exit;
	status = cudaMemcpy(min_2, dev_min_2, UNIT_COUNT * sizeof(uint8_t), cudaMemcpyDeviceToHost);
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