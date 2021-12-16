/*****************************************************************************
 *   Number Plate Recognition using SVM and Neural Networks
 ******************************************************************************
 *   by David Mill�n Escriv�, 5th Dec 2012
 *   http://blog.damiles.com
 ******************************************************************************
 *   Ch5 of the book "Mastering OpenCV with Practical Computer Vision Projects"
 *   Copyright Packt Publishing 2012.
 *   http://www.packtpub.com/cool-projects-with-opencv/book
 *****************************************************************************/

#ifndef DetectRegions_h
#define DetectRegions_h

#include <string.h>
#include <vector>

#include "Plate.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

class DetectRegions {
public:
    DetectRegions();
    string filename;
    void setFilename(string f);
    bool saveRegions;
    bool showSteps;
    vector<Plate> run(Mat input);

private:
    vector<Plate> segment(Mat input);
    bool verifySizes(RotatedRect mr);
    Mat histeq(Mat in);
};

#endif
