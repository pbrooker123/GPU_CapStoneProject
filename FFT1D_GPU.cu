/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

// #include <opencv2/opencv.hpp>
#include <cufft.h>

#include "device_launch_parameters.h"
#include <helper_functions.h>
#include <cmath>

#include <iostream>
#include <cufft.h>

#include <cstdlib> // For system()
#include <cstring> // Add this line to include the <cstring> header

#define prLN std::cout << "Made it to line number: " << __LINE__ << std::endl;

using namespace std;

__global__ void applyFilter(cufftComplex *d_freq, const cufftComplex *filter)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Multiply the real and imaginary parts of FFT data by the corresponding parts of the filter
  d_freq[idx].x *= filter[idx].x;
  d_freq[idx].y *= filter[idx].x; // Using filter[idx].x since filter[idx].y is expected to be 0
}

void writeComplexToFile(const cufftComplex *signal, int N, const std::string &fileName)
{
  // Open the file for writing
  std::ofstream outFile(fileName);

  // Check if the file was opened successfully
  if (!outFile.is_open())
  {
    std::cerr << "Error: Unable to open file " << fileName << " for writing." << std::endl;
    return;
  }

  for (int i = 0; i < N; ++i)
  {
    outFile << signal[i].x << " " << signal[i].y << std::endl;
  }
  outFile.close();

}

void TopHat(int width, cufftComplex *h_data, int N)
{
  for (int i = 0; i < N; ++i)
  {
    if (i >= N / 2 - width / 2 && i < N / 2 + width / 2)
    {
      h_data[i].x = 1;
      h_data[i].y = 0;
    }
    else
    {
      h_data[i].x = 0;
      h_data[i].y = 0;
    }
  }
}

  int main(int argc, char *argv[])
  {
    printf("%s Starting...\n\n", argv[0]);

    try
    {

      // ************ FFT processing *************************

      const int N = 40; // all test have same spatial grid axis

      //Allocate memory for the input data and copy to device
      cufftComplex *h_data = new cufftComplex[N];
      cufftComplex *d_data;
      cudaMalloc((void **)&d_data, N * sizeof(cufftComplex));
      std::string label = "xx";

            // Create a 1D FFT plan
      cufftHandle plan;
      cufftPlan1d(&plan, N, CUFFT_C2C, 1);

      cufftComplex *result = new cufftComplex[N];

      std::string command = "XX";

      int f = 0;


      /* *****************************************************
       ******************** TEST freq = 1 ***************************
       **************************************************** */
      // Test1: Explore FFT output for known test case.
      // By design, f(x_j,p) is exp(2*PI*i*(f/N)*j)
      // Only non zero freq will be j=f
      // f gets redefined for each test
      // f is the number of cycles over the span
      // N is the same for all test cases

      f = 1; // Lambda = N/f = 40/1 = 40 <== 1 cycle over the range

      // Define the input data on the host
      for (int i = 0; i < N; ++i)
      {
        h_data[i].x = cos(-2 * M_PI * f * i / N);
        h_data[i].y = sin(-2 * M_PI * f * i / N);
      }

      //prLN; // prints the current line number. Defined using #define

      // need to save complex array to file in order to plot with gnuplot
      // unable to figure out how to use gnuplot without saving to a file.
      writeComplexToFile(h_data, N, "temp.dat");

      // gp_script accepts 5 arguments
      command = "gnuplot -c gp_script.gp 'f=1 Input Data:Lambda = 40' 'index' 'amp' 'temp.dat' 'plot_f01a_InputData.png'";
      system(command.c_str());

      // copy h_data to data on GPU
      cudaMemcpy(d_data, h_data, N * sizeof(cufftComplex), cudaMemcpyHostToDevice);

      // Perform the FFT
      // plan already defined at top
      cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

      // Copy the result back to host memory
      cudaMemcpy(result, d_data, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

      // Write FFT data to a file and create png
      writeComplexToFile(result, N, "temp.dat");

      command = "gnuplot -c gp_script.gp 'f=1 FFT of Input Data: Lambda=40' 'Cycles over range:freq' 'amp' 'temp.dat' 'plot_f01b_FFTofData.png'";
      system(command.c_str());

      /* *****************************************************
       ******************** TEST freq = 20********************
       **************************************************** */
      f = 20; // Lambda = N/p = 40/20 = 2 <== 1

      // Define the input data on the host
      for (int i = 0; i < N; ++i)
      {
        h_data[i].x = cos(-2 * M_PI * f * i / N);
        h_data[i].y = sin(-2 * M_PI * f * i / N);
      }

      writeComplexToFile(h_data, N, "file_input_data.dat");

      command = "gnuplot -c gp_script.gp 'freq=20 Input Data' 'index' 'amp' 'file_input_data.dat' 'plot_f20a_InputData.png'";
      system(command.c_str());

      cudaMemcpy(d_data, h_data, N * sizeof(cufftComplex), cudaMemcpyHostToDevice);
      cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
      cudaMemcpy(result, d_data, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

      // Write FFT data to a file and create png
      writeComplexToFile(result, N, "file_input_data.dat");

      command = "gnuplot -c gp_script.gp 'freq=20 FFT Input Data' 'Cycles over range:freq' 'amp' 'file_input_data.dat' 'plot_f20b_FFTofData.png'";
      system(command.c_str());

      // Perform the inverse FFT
      cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

      // Copy the result back to host memory
      cudaMemcpy(result, d_data, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

      // Write invFFT data to a file and create png
      writeComplexToFile(result, N, "file_input_data.dat");

      command = "gnuplot -c gp_script.gp 'freq=20 invFFT' 'index' 'amp' 'file_input_data.dat' 'plot_f20c_invFFT.png'";
      system(command.c_str());

      /* *****************************************************
       ******************** TEST freq = 15********************
       **************************************************** */
      f = 15; // Lambda = N/p = 40/20 = 2 <== 1

      // Define the input data on the host
      for (int i = 0; i < N; ++i)
      {
        h_data[i].x = cos(-2 * M_PI * f * i / N);
        h_data[i].y = sin(-2 * M_PI * f * i / N);
      }

      writeComplexToFile(h_data, N, "file_input_data.dat");

      command = "gnuplot -c gp_script.gp 'freq=15 Input Data' 'index' 'amp' 'file_input_data.dat' 'plot_f15a_InputData.png'";
      system(command.c_str());

      cudaMemcpy(d_data, h_data, N * sizeof(cufftComplex), cudaMemcpyHostToDevice);
      cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
      cudaMemcpy(result, d_data, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

      // Write FFT data to a file and create png
      writeComplexToFile(result, N, "file_input_data.dat");

      command = "gnuplot -c gp_script.gp 'freq=15 FFT Input Data' 'freq' 'amp' 'file_input_data.dat' 'plot_f15b_FFTofData.png'";
      system(command.c_str());

      // Perform the inverse FFT
      cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

      // Copy the result back to host memory
      cudaMemcpy(result, d_data, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

      // Write invFFT data to a file and create png
      writeComplexToFile(result, N, "file_input_data.dat");

      command = "gnuplot -c gp_script.gp 'freq=15 invFFT' 'index' 'amp' 'file_input_data.dat' 'plot_f15c_invFFT.png'";
      system(command.c_str());

      /* *****************************************************
       ******************** TEST Top Hat***************************
       **************************************************** */

      // want to also create the freq filter
      // Define the input data on the host
      // Data is a top hat source
      for (int i = 0; i < N; ++i)
      {
        if (i > 9 && i < 31 )
        {
          h_data[i].x = 1;
          h_data[i].y = 0;
        } else
        {
          h_data[i].x = 0;
          h_data[i].y = 0;
        } 
      }

      

            // Write TopHat data to a file and create png
      writeComplexToFile(h_data, N, "file_input_data.dat");

      command = "gnuplot -c gp_script.gp 'Top Hat' 'index' 'amp' 'file_input_data.dat' 'plot_fTopHata_InputData.png'";
      system(command.c_str());

      cudaMemcpy(d_data, h_data, N * sizeof(cufftComplex), cudaMemcpyHostToDevice);

      // Perform the FFT
      cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

      // Copy the result back to host memory
      cudaMemcpy(result, d_data, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

      // Write FFT data to a file and create png
      writeComplexToFile(result, N, "file_input_data.dat");

      command = "gnuplot -c gp_script.gp 'freq=20 FFT Input Data' 'freq' 'amp' 'file_input_data.dat' 'plot_fTopHatb_FFT.png'";
      system(command.c_str());

      // Perform the inverse FFT
      cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

      // Copy the result back to host memory
      cudaMemcpy(result, d_data, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

      cufftComplex *result_norm = new cufftComplex[N];
      for (int i = 0; i < N; i++)
      {
        result_norm[i].x = result[i].x / N;
        result_norm[i].y = result[i].y / N;
      }

      // Write FFT data to a file and create png
      writeComplexToFile(result, N, "file_input_data.dat");
      command = "gnuplot -c gp_script.gp 'freq=20 invFFT' 'index' 'amp' 'file_input_data.dat' 'plot_fTopHatc_invFFT.png'";
      system(command.c_str());

      //Let's redo but chop out the higher frequencies before the invFFT
      //will need to reset d_data to h_data

      cudaMemcpy(d_data, h_data, N * sizeof(cufftComplex), cudaMemcpyHostToDevice);

      // Perform the FFT. d_data will contains the unfiltered fft
      cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

      // Let's chop higher frequencies out of FFT
      cufftComplex h_filter[N];

      // Initialize filter array
      for (int i = 0; i < N; ++i)
      {
        if (i <= 5 || i >= 35)
        {
          h_filter[i].x = 1.0f;
          h_filter[i].y = 0.0f;
        }
        else
        {
          h_filter[i].x = 0.0f;
          h_filter[i].y = 0.0f;
        }
      }

      // Copy filter array to device memory
      cufftComplex *d_filter;
      cudaMalloc(&d_filter, N * sizeof(cufftComplex));
      cudaMemcpy(d_filter, h_filter, N * sizeof(cufftComplex), cudaMemcpyHostToDevice);

      // Define grid and block sizes
      //int blockSize = 256;
      //int numBlocks = (N + blockSize - 1) / blockSize;

      int threadsPerBlock = 256;
      int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

      //vectorMult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

      // Call the kernel function
      applyFilter<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_filter);

      //let's now copy back the filterred spectrum
      //d_data has been modified..let's copy and look

      // Copy the result back to host memory
      cudaMemcpy(result, d_data, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

      // Write FFT filtered data to a file and create png
      writeComplexToFile(result, N, "file_input_data.dat");

      command = "gnuplot -c gp_script.gp 'TopHat filtered freq spectrum' 'freq' 'amp' 'file_input_data.dat' 'plot_fTopHatd_filteredFFT.png'";
      system(command.c_str());

      // Perform the inverse FFT
      cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

      // Copy the result back to host memory
      cudaMemcpy(result, d_data, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

      // Write FFT data to a file and create png
      writeComplexToFile(result, N, "file_input_data.dat");
      command = "gnuplot -c gp_script.gp 'Top Hat filtered InvFFT ' 'index' 'amp' 'file_input_data.dat' 'plot_fTopHate_filteredInvFFT.png'";
      system(command.c_str());



      // Check for kernel launch errors
      cudaError_t launchError = cudaGetLastError();
      if (launchError != cudaSuccess)
      {
        printf("Kernel launch error: %s\n", cudaGetErrorString(launchError));
        // Further error handling if needed
      }

      /* *****************************************************
       ******************** TEST Many FFT***************************
       ******************** Do FFT 2 TopHat signals at once **********
       **************************************************** */

      // Define and populate topHat_30 array
      cufftComplex topHat_30[N];
      TopHat(30, topHat_30, N);

      // Define and populate topHat_20 array
      cufftComplex topHat_20[N];
      TopHat(20, topHat_20, N);

      // Calculate total size for the large array
      const int totalSize = 2 * N;
      cufftComplex largeArray[totalSize];

      // Copy topHat_30 array into the beginning of largeArray
      std::memcpy(largeArray, topHat_30, N * sizeof(cufftComplex));

      // Copy topHat_20 array into the remaining space of largeArray
      std::memcpy(largeArray + N, topHat_20, N * sizeof(cufftComplex));

      // Write TopHat largeArray data to a file and create png
      writeComplexToFile(largeArray, totalSize, "file_input_data.dat");
      command = "gnuplot -c gp_script.gp 'topHat=30 & topHat=20 together ' 'index' 'amp' 'file_input_data.dat' 'plot_ManyFFTa_InputData.png'";
      system(command.c_str());

      // Allocate memory for d_largeArray on the device
      cufftComplex *d_largeArray;
      cudaMalloc(&d_largeArray, 2 * N * sizeof(cufftComplex));

      // Copy largeArray to d_largeArray on the device
      cudaMemcpy(d_largeArray, largeArray, 2 * N * sizeof(cufftComplex), cudaMemcpyHostToDevice);

      const int batch = 2;              // Number of FFTs to perform
      
      // Create a cufftHandle for FFT plan
      int n[1] = {N}; // Size of each dimension of the input data
      cufftHandle planM;
      cufftPlanMany(&planM, 1, n, nullptr, 1, N, nullptr, 1, N, CUFFT_C2C, batch); // Create a 1D complex-to-complex FFT plan for a batch of signals

      // Execute FFT on the batch of signals
      cufftExecC2C(planM, d_largeArray, d_largeArray, CUFFT_FORWARD); // Forward FFT

      cufftComplex *resultMany = new cufftComplex[totalSize];

      // Copy the result back to host memory
      cudaMemcpy(resultMany, d_largeArray, totalSize * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

      // Write FFT data to a file and create png
      writeComplexToFile(resultMany, totalSize, "file_input_data.dat");
      command = "gnuplot -c gp_script.gp 'Top Hat Many FFT ' 'freq every 40' 'amp' 'file_input_data.dat' 'plot_ManyFFTb_ManyFFT.png'";
      system(command.c_str());

      // Free memory and destroy plan
      delete[] result, result_norm, h_data;
      cudaFree(d_data);
      cudaFree(d_filter);
      cufftDestroy(plan);
      cufftDestroy(planM);
      cudaFree(d_largeArray);

      // ************end FFT processing ******************

      exit(EXIT_SUCCESS);
    }
  catch (npp::Exception &rException)
  {
    std::cerr << "Program error! The following exception occurred: \n";
    std::cerr << rException << std::endl;
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
  }
  catch (...)
  {
    std::cerr << "Program error! An unknow type of exception occurred. \n";
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
    return -1;
  }

  return 0;
}
