CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17
PCL_FLAGS := $(shell pkg-config --cflags --libs pcl_io)

.PHONY: all clean

all: decode_lidar

decode_lidar: decode_lidar.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(PCL_FLAGS)

clean:
	rm -f decode_lidar
