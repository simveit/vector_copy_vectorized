#include <cuda_runtime.h>

#include <iostream>
#include <random>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const *func, char const *file, int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(char const *file, int line) {
  cudaError_t const err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

template <int threadsPerBlock>
__global__ void vectorCopyVectorized(float4 *input, float4 *output, int N) {
  const int i = threadIdx.x + blockIdx.x * threadsPerBlock;

  if (i < (N >> 2)) {
    output[i] = input[i];
  }
}

template <int threadsPerBlock>
void launchVectorCopyVectorized(float *input, float *output, int N) {
  const int blocksPerGrid = (N / 4 + threadsPerBlock - 1) / threadsPerBlock;
  vectorCopyVectorized<threadsPerBlock><<<blocksPerGrid, threadsPerBlock>>>(
      reinterpret_cast<float4 *>(input), reinterpret_cast<float4 *>(output), N);
}

bool checkCorrectness(float *input, float *output, int N) {
  for (int i = 0; i < N; i++) {
    if (fabs(input[i] - output[i]) > 1e-5) {
      std::cout << "Verification failed" << std::endl;
      return false;
    }
  }
  std::cout << "Verification passed" << std::endl;
  return true;
}

int main() {
  const int N = 1 << 30;
  const size_t size = N * sizeof(float);
  const int threadsPerBlock = 1 << 10;

  float *inputHost = new float[N];
  float *outputHost = new float[N];

  std::default_random_engine generator(42);
  std::normal_distribution<float> distribution(0.0, 1.0);

  for (int i = 0; i < N; i++) {
    inputHost[i] = distribution(generator);
  }

  float *inputDevice;
  float *outputDevice;

  CHECK_CUDA_ERROR(cudaMalloc(&inputDevice, size));
  CHECK_CUDA_ERROR(cudaMalloc(&outputDevice, size));

  CHECK_CUDA_ERROR(
      cudaMemcpy(inputDevice, inputHost, size, cudaMemcpyHostToDevice));

  launchVectorCopyVectorized<threadsPerBlock>(inputDevice, outputDevice, N);

  CHECK_LAST_CUDA_ERROR();

  CHECK_CUDA_ERROR(
      cudaMemcpy(outputHost, outputDevice, size, cudaMemcpyDeviceToHost));

  if (!checkCorrectness(inputHost, outputHost, N)) {
    return -1;
  }

  cudaEvent_t start, stop;
  int numWarmup = 1000;
  int numRounds = 10000;
  size_t numCrossMemoryBounds = 2 * size;
  float time;

  for (int i = 0; i < numWarmup; i++) {
    launchVectorCopyVectorized<threadsPerBlock>(inputDevice, outputDevice, N);
  }

  CHECK_LAST_CUDA_ERROR();

  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

  CHECK_CUDA_ERROR(cudaEventRecord(start));
  for (int i = 0; i < numRounds; i++) {
    launchVectorCopyVectorized<threadsPerBlock>(inputDevice, outputDevice, N);
  }
  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));

  float latency = time / numRounds;
  float bandwidth = (numCrossMemoryBounds / latency) / 1e6;

  std::cout << "Latency = " << latency << " ms" << std::endl;
  std::cout << "Bandwidth = " << bandwidth << " GB/s" << std::endl;
  std::cout << "% of max = " << bandwidth / 3300 * 100 << " %" << std::endl;

  CHECK_CUDA_ERROR(cudaFree(inputDevice));
  CHECK_CUDA_ERROR(cudaFree(outputDevice));

  free(inputHost);
  free(outputHost);

  return 0;
}