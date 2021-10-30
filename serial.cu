/*
 * 2019074 Jayraj Rathod
 * 2019203 Aniket Choudhari
 * 2019200 Aman Kumar
 */
#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <thrust/scan.h>                                                        
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "cudaTriangleCounter.h"

#define BLOCK_SIZE 32

struct GlobalConstants {

    int *NodeList;
    int *ListLen;
    int numNodes;
    int numEdges;
};

__constant__ GlobalConstants cuConstCounterParams;

void
CudaTriangleCounter::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

  

    // By this time the graph should be loaded.  Copying graph to 
    // data structures into device memory so that it is accessible to
    // CUDA kernels
    //

    cudaMalloc(&cudaDeviceListLen, sizeof(int ) * numNodes);
    cudaMemcpy(cudaDeviceListLen, list_len, sizeof(int) * numNodes, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&cudaDeviceNodeList, node_list_size * sizeof(int));
    cudaMemcpy(cudaDeviceNodeList, node_list, sizeof(int) * node_list_size, cudaMemcpyHostToDevice);

    GlobalConstants params;
    params.ListLen = cudaDeviceListLen;
    params.NodeList = cudaDeviceNodeList;
    params.numNodes = numNodes;
    params.numEdges = numEdges;
    cudaMemcpyToSymbol(cuConstCounterParams, &params, sizeof(GlobalConstants));
}

CudaTriangleCounter::CudaTriangleCounter(char *fileName) {
    clock_t start, diff, malloc_diff;
    int node, edge_id, temp = 0;
    int total_nodes = 0;
    int total_edges = 0;
    int msec;

    std::string line;
    std::ifstream myfile;
    myfile.open(fileName);

    std::string token;                                                             
    if (strstr(fileName,"new_orkut") != NULL) {                                    
        printf("This is the NEW_ORKUT FILE **\n");                             
        total_nodes = 3072600;                                                     
        total_edges = 117185083 + 1;                                               
    } else {                                                                       
        std::getline(myfile,line);                                                 
        std::stringstream lineStream(line);                                        
        while (lineStream >> token) {                                              
            if (temp == 0) {                                                       
                total_nodes = std::stoi(token, NULL, 10) + 1;                      
            } else if (temp == 1) {                                                
                total_edges = std::stoi(token, NULL, 10) + 1;                      
            } else {                                                               
                printf("!!!!!!!!!!!! TEMP IS %d\n ", temp);                        
                break;                                                             
            }                                                                      
            temp++;                                                                
        }                                                                          
    }

    start = clock();

    numNodes = total_nodes;
    node_list_size = total_edges * 2;
    numEdges = total_edges;

    printf("total_nodes %d\n", total_nodes);
    printf("node_list_size %d\n", node_list_size);
    printf("numEdges %d\n", numEdges);

    list_len = (int *)calloc(total_nodes, sizeof(int));
    start_addr = (int *)calloc(total_nodes, sizeof(int));
    node_list = (int *)calloc(node_list_size, sizeof(int));

    malloc_diff = clock() - start;
    msec = malloc_diff * 1000 / CLOCKS_PER_SEC;

    printf("memory allocated ......\n");
    node = 1;
    temp = 1;
    int neighbors;
    while(std::getline(myfile, line)) {
        neighbors = 0;
        std::stringstream lineStream(line);
        std::string token;
        while(lineStream >> token)
        {
            edge_id = std::stoi(token, NULL, 10);
            if (edge_id > node) {
                node_list[temp++] = edge_id;
                neighbors++;
            }
        }

        list_len[node] = neighbors;
        node++;
    }

    printf("graph created......\n");
    diff = clock() - start;
    msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);

    myfile.close();
}

CudaTriangleCounter::~CudaTriangleCounter() {

    free(node_list);
    free(list_len);
}

void CudaTriangleCounter::countTriangles() {
    int i, j, k, m, count=0;

    for (i=1; i<numNodes; i++) {

        int *list = node_list + start_addr[i-1] + 1;

        int len = list_len[i];

        if (len < 2) {
            continue;
        }

        for (j=0; j<len-1; j++) {
            for (k=j+1; k<len; k++) {

                int idx1;
                int idx2;
                idx1 = list[j];
                idx2 = list[k];
                int *list1 = node_list + start_addr[idx1-1] + 1;
                int len1 = list_len[idx1];

                for (m=0; m<len1; m++) {

                    if (list1[m] == idx2) {
                        count++;
                    }
                }
            }

        }

    }
        printf("count for %d -> %d\n", i, count);

}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        printf("usage: ./a.out <input_file>");
        exit(-1);
    }

    int msec;
    clock_t start, diff;

    CudaTriangleCounter *tCounter = new CudaTriangleCounter(argv[1]);
   
    tCounter->setup();
    start = clock();
    tCounter->countTriangles();
    diff = clock() - start;
    msec = diff * 1000 / CLOCKS_PER_SEC;
    // printf("counting taken %d seconds %d milliseconds\n", msec/1000, msec%1000);

    return 0;
}
