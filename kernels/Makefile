NVCC_FLAGS = -std=c++17 -O3 -DNDEBUG -w
NVCC_LDFLAGS = -lcublas -lcuda
OUT_DIR = out
PROFILE_DIR = profile

CUDA_OUTPUT_FILE = -o $(OUT_DIR)/$@
NCU_PATH := $(shell which ncu)
NCU_COMMAND = $(NCU_PATH) --set full --import-source yes

NVCC_FLAGS += --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
NVCC_FLAGS += -arch=sm_90a

NVCC_BASE = nvcc $(NVCC_FLAGS) $(NVCC_LDFLAGS) -lineinfo

vector_copy: vector_copy.cu
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

vector_copy_for_profile: vector_copy_for_profile.cu
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

vector_copy_profile: vector_copy_for_profile
	$(NCU_COMMAND) -o $(PROFILE_DIR)/$@ -f $(OUT_DIR)/$^

vector_copy_vectorized: vector_copy_vectorized.cu
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

vector_copy_vectorized_for_profile: vector_copy_vectorized_for_profile.cu
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

vector_copy_vectorized_profile: vector_copy_vectorized_for_profile
	$(NCU_COMMAND) -o $(PROFILE_DIR)/$@ -f $(OUT_DIR)/$^

compile_all:
	make vector_copy
	make vector_copy_vectorized
	make vector_copy_for_profile
	make vector_copy_vectorized_for_profile

profile_all:
	make vector_copy_profile
	make vector_copy_vectorized_profile

run_all:
	./$(OUT_DIR)/vector_copy
	./$(OUT_DIR)/vector_copy_vectorized

clean:
	rm $(OUT_DIR)/* 