# 1D FFT exporations with cuFFT
Please read the ppt first to understand the project
Project starting point was the NPP Box Filter Lab located in week5 of the Cuda at Scale course

## Project Description
Please read the ppt first to understand the project background.

In this project, I probe the fft frequency spectrum output with various 1D arry test cases. In particular, I define input signals of fixed frequencies. In the ppt, I outline what expected solution is for a fixed frequency signal.

Also, I invesigate the spectrum obtained from a tophat signal as input.

Next, I figure out how to block all but the lowest 5 (out of 20)  frequencies and show that the tophat turns into the ringing tophat as expected from the invFFT of the filtered frequency spectrum.

Finally, I show how to use the cufftPlanMany option where more that 1 signals can be loaded into one array, and the FFTs be done at once in batch. THis shows the methodology of creating a 1D array of say >10,000 signals and sending them all at once for FFT processing.

## Code Organization

```bin/```
Empty. Main level has *.cu & *.cpp

```data/```
*.png files produced by the program are stored here

```lib/```
The program runs in the coursera environment for NPP Box Filter Lab located in week5 of the Cuda at Scale course. The Makefile contains a number of modifications to allow the program to run.

```src/```
Empty

```README.md```
This file should hold the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

```INSTALL```
Just need to  make sure all libraries referenced in the makefile are available.

```Makefile or CMAkeLists.txt or build.sh```
There should be some rudimentary scripts for building your project's code in an automatic fashion.

```run.sh```
No run.sh. Code has no inputs. Just run with ./FFT1D_GPU
