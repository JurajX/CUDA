# Parallel Programming with CUDA

This project is based on the Udacity course called *Intro to Parallel Programming*. At the time I took the course the provided [code](https://github.com/udacity/cs344), was obsolete (did not work with new versions of openCV and other libraries). Hence, I decided to reimplement the problem sets from scratch. The code implements
 * four main GPU methods
   * reduce,
   * scan,
   * histogram, and
   * sort

 * four homework assignments where the above methods are used
   * converting colour pictures to greyscale pictures
   * blurring pictures
   * tone mapping (HDR) operation on pictures
   * red-eye removal from pictures

The code is divided into two folders Libraries and Projects.

In the Libraries folder, one can find implementation of all methods further divided into sub-folders. The sub-folders Reduce, Scan, Histogram, and Sort contain the implementation of the respective GPU methods. The folder SupportingFiles contains implementations of functions and classes used by all files. Finally, the folder HW contains implementations of the homework assignments.

The Projects folder contains main functions for executing the methods and assignments, again divided into sub-folders. The sub-folders Reduce, Scan, Histogram, and Sort contain main functions to execute the methods on ![formula](https://render.githubusercontent.com/render/math?math=2^27) element arrays and compare the execution times with serial and parallel library implementations. The sub-folder ProblemSets contain the main functions for the homework assignments.

In order to compile and run a project one needs to `cd` to the desired project, create a new directory and `cd` there, call cmake, then make, and finally run the executable. The executables are named *reduce_test, scan_test, histogram_test, sort_test,* and *HomeWork*. For example, to compile and run the homework assignments one does the following
```terminal
% cd Projects/ProblemSets/
% mkdir build
% cd build
% cmake ..
% make
% ./HomeWork
```
The prerequisites are *opencv4*, *tbb* (used by std parallel CPU algorithms, for non-Intel CPUs one needs to modify CMakeLists.txt in Libraries folder), and a *cuda* capable GPU with drivers (tested on nvcc V10.1.243).
